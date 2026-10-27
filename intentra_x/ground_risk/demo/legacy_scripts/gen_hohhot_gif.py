#!/usr/bin/env python
"""
生成呼和浩特每3小时风险评估GIF动画
严格使用项目API: make_strike_grid, make_fatality_grid
"""
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import zoom
import imageio
import warnings
warnings.filterwarnings('ignore')

# 导入项目的风险计算API
from seedpod_ground_risk.api.api import make_strike_grid, make_fatality_grid
from casex import AircraftSpecs
from casex.enums import AircraftType

print("=" * 60)
print("呼和浩特风险评估GIF - 使用项目API")
print("=" * 60)

# ============ 参数配置 ============
MIN_LAT, MAX_LAT = 40.78, 40.84
MIN_LON, MAX_LON = 111.62, 111.78
RESOLUTION = 50  # 米

# 飞行参数
AIRSPEED = 20  # m/s
ALTITUDE = 120  # m  
FAILURE_PROB = 1e-3  # 失效概率
WIND_SPEED = 5  # m/s
WIND_DIR = 180  # 度 (南风)

# 时间点 (每3小时)
HOURS = [0, 3, 6, 9, 12, 15, 18, 21]

# NHAPS 住宅区室内比例 (类别0)
NHAPS_RESIDENTIAL = {
    0: 0.94, 1: 0.95, 2: 0.96, 3: 0.96, 4: 0.96, 5: 0.93,
    6: 0.85, 7: 0.71, 8: 0.55, 9: 0.45, 10: 0.39, 11: 0.36,
    12: 0.37, 13: 0.37, 14: 0.39, 15: 0.43, 16: 0.50, 17: 0.59,
    18: 0.65, 19: 0.70, 20: 0.77, 21: 0.84, 22: 0.89, 23: 0.92
}

# NHAPS 室外人口比例 (类别9)
NHAPS_OUTDOOR = {
    0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01,
    6: 0.02, 7: 0.02, 8: 0.03, 9: 0.03, 10: 0.03, 11: 0.04,
    12: 0.04, 13: 0.03, 14: 0.03, 15: 0.03, 16: 0.03, 17: 0.02,
    18: 0.03, 19: 0.02, 20: 0.02, 21: 0.01, 22: 0.01, 23: 0.01
}

print(f"\n区域: 呼和浩特 ({MIN_LAT}N-{MAX_LAT}N, {MIN_LON}E-{MAX_LON}E)")
print(f"分辨率: {RESOLUTION}m")
print(f"时间点: {HOURS}")

# ============ 1. 加载人口数据 ============
print("\n[1/4] 加载WorldPop人口密度数据...")
pop_file = './static_data/china/hohhot_pop.tif'
with rasterio.open(pop_file) as src:
    pop_raw = src.read(1)
    print(f"  WorldPop原始: {pop_raw.shape}")

# 计算目标网格尺寸
lon_dist = (MAX_LON - MIN_LON) * 111 * np.cos(np.radians((MIN_LAT + MAX_LAT) / 2))
lat_dist = (MAX_LAT - MIN_LAT) * 111
target_cols = int(lon_dist * 1000 / RESOLUTION)
target_rows = int(lat_dist * 1000 / RESOLUTION)

# 上采样到目标分辨率
zoom_y = target_rows / pop_raw.shape[0]
zoom_x = target_cols / pop_raw.shape[1]
pop_density_base = zoom(pop_raw, (zoom_y, zoom_x), order=1)
print(f"  目标网格: {pop_density_base.shape}")

# ============ 2. 配置飞行器 ============
print("\n[2/4] 配置飞行器 (casex.AircraftSpecs)...")
aircraft = AircraftSpecs(AircraftType.FIXED_WING, 3.0, 2.0, 4.0)
aircraft.set_ballistic_drag_coefficient(0.8)
aircraft.set_glide_speed_ratio(12, 15)
aircraft.set_ballistic_frontal_area(0.1)
print(f"  类型: 固定翼无人机, 质量: {aircraft.mass}kg, 翼展: {aircraft.width}m")

# ============ 3. 为每个时间点计算风险 ============
print("\n[3/4] 计算各时间点风险...")
output_dir = './output_hohhot/gif_frames'
os.makedirs(output_dir, exist_ok=True)

frame_files = []
extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

for hour in HOURS:
    print(f"\n  === {hour:02d}:00 ===")
    
    # 计算室外人口 (1 - 住宅室内比例 + 室外活动比例)
    outdoor_frac = (1 - NHAPS_RESIDENTIAL[hour]) + NHAPS_OUTDOOR[hour]
    pop_grid = pop_density_base * outdoor_frac
    print(f"    室外人口比例: {outdoor_frac*100:.1f}%")
    print(f"    人口密度: {pop_grid.min():.0f} ~ {pop_grid.max():.0f} 人/km2")
    
    # 调用项目API计算碰撞风险
    strike_grid, v_is = make_strike_grid(
        aircraft, AIRSPEED, ALTITUDE, FAILURE_PROB,
        pop_grid, RESOLUTION, WIND_DIR, WIND_SPEED
    )
    print(f"    碰撞风险: {strike_grid.min():.2e} ~ {strike_grid.max():.2e}")
    
    # 调用项目API计算致命风险
    fatality_grid = make_fatality_grid(aircraft, strike_grid, v_is)
    print(f"    致命风险: {fatality_grid.min():.2e} ~ {fatality_grid.max():.2e}")
    
    # 生成可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 人口密度图
    im1 = axes[0].imshow(pop_grid, cmap='YlOrRd', extent=extent, origin='upper',
                          vmin=0, vmax=pop_density_base.max() * 0.7)
    axes[0].set_title(f'Population Density\n(Outdoor {outdoor_frac*100:.0f}%)', fontsize=10)
    plt.colorbar(im1, ax=axes[0], label='people/km2', shrink=0.8)
    
    # 碰撞风险图
    im2 = axes[1].imshow(np.log10(strike_grid + 1e-15), cmap='hot', extent=extent, origin='upper',
                          vmin=-9, vmax=-6)
    axes[1].set_title('Strike Probability\n(log10)', fontsize=10)
    plt.colorbar(im2, ax=axes[1], label='log10(P)', shrink=0.8)
    
    # 致命风险图
    im3 = axes[2].imshow(np.log10(fatality_grid + 1e-15), cmap='jet', extent=extent, origin='upper',
                          vmin=-10, vmax=-7)
    axes[2].set_title('Fatality Risk\n(log10)', fontsize=10)
    plt.colorbar(im3, ax=axes[2], label='log10(Risk)', shrink=0.8)
    
    for ax in axes:
        ax.set_xlabel('Longitude (E)')
        ax.set_ylabel('Latitude (N)')
    
    plt.suptitle(f'Hohhot UAV Ground Risk - {hour:02d}:00\n(NHAPS Temporal Model)', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # 保存帧
    frame_file = f'{output_dir}/frame_{hour:02d}.png'
    plt.savefig(frame_file, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    frame_files.append(frame_file)
    print(f"    已保存: {frame_file}")

# ============ 4. 生成GIF ============
print("\n[4/4] 生成GIF动画...")
gif_file = './output_hohhot/hohhot_risk_24h.gif'

images = [imageio.imread(f) for f in frame_files]
imageio.mimsave(gif_file, images, duration=1.0, loop=0)

print(f"\n已保存GIF: {gif_file}")
print(f"  帧数: {len(images)}")
print(f"  时间点: {HOURS}")

print("\n" + "=" * 60)
print("使用的项目API:")
print("  - casex.AircraftSpecs")
print("  - api.make_strike_grid (BallisticModel + GlideDescentModel + StrikeModel)")
print("  - api.make_fatality_grid (FatalityModel)")
print("  - NHAPS时间模式 (EPA National Human Activity Pattern Survey)")
print("=" * 60)
