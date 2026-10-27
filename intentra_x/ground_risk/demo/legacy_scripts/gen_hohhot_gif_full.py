#!/usr/bin/env python
"""
呼和浩特每3小时风险评估GIF - 完整版 (实时OSM查询)
严格使用项目API + 完整NHAPS时间模型 + 实时OSM查询
"""
import os
os.environ['USE_PYGEOS'] = '0'
os.environ['SPGR_REGION'] = 'CHINA'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from scipy.ndimage import zoom
from shapely.geometry import box, Polygon
import imageio
import warnings
from time import time
warnings.filterwarnings('ignore')

# 项目API
from seedpod_ground_risk.api.api import make_strike_grid, make_fatality_grid
from seedpod_ground_risk.layers.osm_tag_layer import query_osm_polygons
from seedpod_ground_risk.core.utils import make_bounds_polygon
from seedpod_ground_risk.data import nhaps_data_filepath
from casex import AircraftSpecs
from casex.enums import AircraftType

print("=" * 70)
print("呼和浩特风险评估GIF - 完整版 (实时OSM查询)")
print("=" * 70)

# ============ 参数配置 ============
MIN_LAT, MAX_LAT = 40.78, 40.84
MIN_LON, MAX_LON = 111.62, 111.78
RESOLUTION = 50  # 米

# 飞行参数
AIRSPEED = 20   # m/s
ALTITUDE = 120  # m  
FAILURE_PROB = 1e-3
WIND_SPEED = 5  # m/s
WIND_DIR = 180  # 度

# 时间点 (每3小时)
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

print(f"\n区域: 呼和浩特 ({MIN_LAT}°N-{MAX_LAT}°N, {MIN_LON}°E-{MAX_LON}°E)")
print(f"分辨率: {RESOLUTION}m")
print(f"时间点: {HOURS}")

# ============ NHAPS 分类 (与项目 temporal_population_estimate_layer 完全一致) ============
nhaps_category_groupings = [
    [0, 1],      # 住宅区
    [5, 9],      # 工业/商业
    [7],         # 学校
    [6, 8],      # 商场/零售
]

# 对应的OSM标签 (与项目 nhaps_group_tags 完全一致)
nhaps_group_tags = [
    ['landuse=residential'],
    ['landuse=industrial', 'landuse=commercial'],
    ['building=school', 'building=college', 'building=university', 'building=public', 'building=government',
     'building=civic', 'building=hospital'],
    ['landuse=retail']
]

# 创建bounds polygon (使用项目的方式)
bounds_poly = make_bounds_polygon((MIN_LON, MAX_LON), (MIN_LAT, MAX_LAT))
print(f"Bounds polygon: {bounds_poly.bounds}")

# ============ 1. 加载NHAPS时间数据 ============
print("\n[1/6] 加载NHAPS时间模式数据...")
nhaps_df = pd.read_json(nhaps_data_filepath())
print(f"  NHAPS数据: {nhaps_df.shape[0]} 类别 × {nhaps_df.shape[1]} 小时")

# ============ 2. 加载基础人口密度 ============
print("\n[2/6] 加载WorldPop基础人口密度...")
pop_file = './static_data/china/hohhot_pop.tif'
with rasterio.open(pop_file) as src:
    pop_raw = src.read(1)
    print(f"  WorldPop原始: {pop_raw.shape}")

# 计算目标网格尺寸
lon_dist = (MAX_LON - MIN_LON) * 111 * np.cos(np.radians((MIN_LAT + MAX_LAT) / 2))
lat_dist = (MAX_LAT - MIN_LAT) * 111
target_cols = int(lon_dist * 1000 / RESOLUTION)
target_rows = int(lat_dist * 1000 / RESOLUTION)

# 上采样
zoom_y = target_rows / pop_raw.shape[0]
zoom_x = target_cols / pop_raw.shape[1]
pop_density_base = zoom(pop_raw, (zoom_y, zoom_x), order=1)
print(f"  上采样后: {pop_density_base.shape}")
print(f"  基础人口密度: {pop_density_base.min():.0f} ~ {pop_density_base.max():.0f} 人/km²")

# ============ 3. 实时查询OSM数据 ============
print("\n[3/6] 实时查询OSM土地利用数据...")
print("  (使用项目的 query_osm_polygons 函数)")

category_gdfs = []
category_masks = []

for group_idx, tags in enumerate(nhaps_group_tags):
    print(f"\n  --- 类别 {group_idx}: {tags} ---")
    group_gdfs = []
    
    for tag in tags:
        print(f"    查询: {tag}...")
        t0 = time()
        try:
            gdf = query_osm_polygons(tag, bounds_poly)
            print(f"    OSM query took {time()-t0:.1f}s, 返回 {len(gdf)} 个多边形")
            if not gdf.empty:
                group_gdfs.append(gdf)
        except Exception as e:
            print(f"    查询失败: {e}")
    
    # 合并该类别的所有GeoDataFrame
    if group_gdfs:
        group_gdf = gpd.GeoDataFrame(pd.concat(group_gdfs, ignore_index=True), crs='EPSG:4326')
    else:
        group_gdf = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:4326')
    
    category_gdfs.append(group_gdf)
    
    # 栅格化多边形
    group_mask = np.zeros(pop_density_base.shape, dtype=bool)
    for _, row in group_gdf.iterrows():
        geom = row.geometry
        if geom.is_valid:
            minx, miny, maxx, maxy = geom.bounds
            col_start = max(0, int((minx - MIN_LON) / (MAX_LON - MIN_LON) * target_cols))
            col_end = min(target_cols, int((maxx - MIN_LON) / (MAX_LON - MIN_LON) * target_cols))
            row_start = max(0, int((MAX_LAT - maxy) / (MAX_LAT - MIN_LAT) * target_rows))
            row_end = min(target_rows, int((MAX_LAT - miny) / (MAX_LAT - MIN_LAT) * target_rows))
            group_mask[row_start:row_end, col_start:col_end] = True
    
    category_masks.append(group_mask)
    coverage = group_mask.sum() / group_mask.size * 100
    print(f"  类别{group_idx} 总计: {len(group_gdf)} 多边形, 覆盖率: {coverage:.1f}%")

# ============ 4. 加载中国人口普查数据 ============
print("\n[4/6] 加载中国区域人口密度数据...")
from seedpod_ground_risk.data import england_wa_2011_clipped_filepath, density_filepath

census_wards_file = england_wa_2011_clipped_filepath()
density_file = density_filepath()
print(f"  人口边界: {census_wards_file}")
print(f"  人口密度: {density_file}")

try:
    census_wards_df = gpd.read_file(census_wards_file)
    if not census_wards_df.crs:
        census_wards_df = census_wards_df.set_crs('EPSG:4326')
    else:
        census_wards_df = census_wards_df.to_crs('EPSG:4326')
    
    density_df = pd.read_csv(density_file, header=0)
    census_wards = census_wards_df.merge(density_df, on='code')
    print(f"  加载成功: {len(census_wards)} 个区域")
    
    # 获取区域内的人口密度
    bounded_census = census_wards.cx[MIN_LON:MAX_LON, MIN_LAT:MAX_LAT]
    if not bounded_census.empty:
        total_population = (bounded_census['density'] * bounded_census.to_crs('EPSG:3395').area * 1e-6).sum()
        print(f"  区域总人口估计: {total_population:.0f}")
    else:
        print(f"  警告: 区域内无人口数据，使用WorldPop")
except Exception as e:
    print(f"  加载失败: {e}, 使用WorldPop数据")

# ============ 5. 配置飞行器 ============
print("\n[5/6] 配置飞行器 (casex.AircraftSpecs)...")
aircraft = AircraftSpecs(AircraftType.FIXED_WING, 3.0, 2.0, 4.0)
aircraft.set_ballistic_drag_coefficient(0.8)
aircraft.set_glide_speed_ratio(12, 15)
aircraft.set_ballistic_frontal_area(0.1)
print(f"  固定翼无人机: 质量={aircraft.mass}kg, 翼展={aircraft.width}m")

# ============ 6. 为每个时间点计算风险 ============
print("\n[6/6] 计算各时间点风险 (完整NHAPS模型)...")
output_dir = './output_hohhot/gif_frames_full'
os.makedirs(output_dir, exist_ok=True)

frame_files = []
extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

for hour in HOURS:
    print(f"\n{'='*60}")
    print(f"  时间: {hour:02d}:00")
    print(f"{'='*60}")
    t_hour_start = time()
    
    # 获取该小时的NHAPS比例
    hour_key = hour * 100
    hour_proportions = nhaps_df[hour_key]
    
    # 计算各类别的人口权重
    pop_grid = np.zeros_like(pop_density_base)
    
    for group_idx, categories in enumerate(nhaps_category_groupings):
        # 该类别在该时间的人口比例
        group_proportion = sum(hour_proportions.iloc[cat] for cat in categories)
        
        # 将人口分配到对应区域
        mask = category_masks[group_idx]
        if mask.any():
            pop_grid[mask] += pop_density_base[mask] * group_proportion
        
        print(f"    类别{group_idx} (NHAPS {categories}): {group_proportion*100:.1f}%")
    
    # 未分类区域使用住宅区比例
    unclassified = ~np.any(category_masks, axis=0)
    residential_prop = sum(hour_proportions.iloc[cat] for cat in [0, 1])
    pop_grid[unclassified] += pop_density_base[unclassified] * residential_prop
    
    print(f"    未分类区域: {unclassified.sum()/unclassified.size*100:.1f}%, 使用住宅比例 {residential_prop*100:.1f}%")
    print(f"    最终人口密度: {pop_grid.min():.0f} ~ {pop_grid.max():.0f} 人/km²")
    
    # 调用项目API计算碰撞风险
    print(f"    调用 make_strike_grid...")
    t0 = time()
    strike_grid, v_is = make_strike_grid(
        aircraft, AIRSPEED, ALTITUDE, FAILURE_PROB,
        pop_grid, RESOLUTION, WIND_DIR, WIND_SPEED
    )
    print(f"    Strike grid took {time()-t0:.1f}s")
    print(f"    碰撞风险: {strike_grid.min():.2e} ~ {strike_grid.max():.2e}")
    print(f"    撞击速度: ballistic={v_is[0]:.1f}m/s, glide={v_is[1]:.1f}m/s")
    
    # 调用项目API计算致命风险
    print(f"    调用 make_fatality_grid...")
    t0 = time()
    fatality_grid = make_fatality_grid(aircraft, strike_grid, v_is)
    print(f"    Fatality grid took {time()-t0:.1f}s")
    print(f"    致命风险: {fatality_grid.min():.2e} ~ {fatality_grid.max():.2e}")
    
    # 生成可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 人口密度图
    im1 = axes[0].imshow(pop_grid, cmap='YlOrRd', extent=extent, origin='upper',
                          vmin=0, vmax=pop_density_base.max() * 0.8)
    axes[0].set_title(f'Population Density\n(NHAPS Hour={hour})', fontsize=11)
    plt.colorbar(im1, ax=axes[0], label='people/km²', shrink=0.8)
    
    # 碰撞风险图
    im2 = axes[1].imshow(np.log10(strike_grid + 1e-15), cmap='hot', extent=extent, origin='upper',
                          vmin=-9, vmax=-6)
    axes[1].set_title('Strike Probability\n(Ballistic + Glide)', fontsize=11)
    plt.colorbar(im2, ax=axes[1], label='log₁₀(P)', shrink=0.8)
    
    # 致命风险图
    im3 = axes[2].imshow(np.log10(fatality_grid + 1e-15), cmap='jet', extent=extent, origin='upper',
                          vmin=-10, vmax=-7)
    axes[2].set_title('Fatality Risk\n(Impact Energy)', fontsize=11)
    plt.colorbar(im3, ax=axes[2], label='log₁₀(Risk)', shrink=0.8)
    
    for ax in axes:
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
    
    plt.suptitle(f'Hohhot UAV Ground Risk Assessment - {hour:02d}:00\n'
                 f'(Full NHAPS Model + Real-time OSM Query)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 保存帧
    frame_file = f'{output_dir}/frame_{hour:02d}.png'
    plt.savefig(frame_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    frame_files.append(frame_file)
    
    print(f"    Hour {hour:02d}:00 total took {time()-t_hour_start:.1f}s")
    print(f"    已保存: {frame_file}")

# ============ 生成GIF ============
print("\n" + "=" * 70)
print("生成GIF动画...")
gif_file = './output_hohhot/hohhot_risk_24h_full.gif'

images = [imageio.imread(f) for f in frame_files]
imageio.mimsave(gif_file, images, duration=1.5, loop=0)

print(f"\n✓ GIF已保存: {gif_file}")
print(f"  帧数: {len(images)}")
print(f"  时间间隔: 每3小时")

print("\n" + "=" * 70)
print("使用的项目组件:")
print("  - query_osm_polygons: 实时OSM Overpass API查询")
print("  - make_bounds_polygon: 区域边界创建")
print("  - casex.AircraftSpecs: 飞行器物理模型")
print("  - api.make_strike_grid: 碰撞概率 (BallisticModel + GlideDescentModel)")  
print("  - api.make_fatality_grid: 致命风险 (FatalityModel)")
print("  - NHAPS时间模式: static_data/nhaps.json")
print("=" * 70)
