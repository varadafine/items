#!/usr/bin/env python
"""
使用项目API为呼和浩特生成风险评估
严格使用项目的 make_strike_grid 和 make_fatality_grid
人口数据直接从WorldPop读取（因为OSM查询不稳定）
"""
import os
os.environ['USE_PYGEOS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.ndimage import zoom
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# 导入项目的风险计算API
from seedpod_ground_risk.api.api import make_strike_grid, make_fatality_grid
from casex import AircraftSpecs
from casex.enums import AircraftType

print("=" * 60)
print("呼和浩特风险评估 - 使用项目API")
print("=" * 60)

# ============ 参数配置 ============
MIN_LAT, MAX_LAT = 40.78, 40.84
MIN_LON, MAX_LON = 111.62, 111.78
RESOLUTION = 50  # 米
HOUR = 9

# 飞行参数
AIRSPEED = 20  # m/s
ALTITUDE = 120  # m  
FAILURE_PROB = 1e-3  # 失效概率
WIND_SPEED = 5  # m/s
WIND_DIR = 180  # 度 (南风)

print(f"\n区域: 呼和浩特 ({MIN_LAT}°N-{MAX_LAT}°N, {MIN_LON}°E-{MAX_LON}°E)")
print(f"分辨率: {RESOLUTION}m, 时间: {HOUR}:00")

# ============ 1. 生成人口密度网格 ============
print("\n[1/3] 生成人口密度网格...")
print("  (直接使用WorldPop数据，避免OSM查询不稳定)")

# 读取WorldPop数据
pop_file = './static_data/china/hohhot_pop.tif'
with rasterio.open(pop_file) as src:
    pop_raw = src.read(1)
    print(f"  WorldPop原始: {pop_raw.shape}")

# 计算目标网格尺寸 (与项目make_pop_grid一致)
lon_dist = (MAX_LON - MIN_LON) * 111 * np.cos(np.radians((MIN_LAT + MAX_LAT) / 2))
lat_dist = (MAX_LAT - MIN_LAT) * 111
target_cols = int(lon_dist * 1000 / RESOLUTION)
target_rows = int(lat_dist * 1000 / RESOLUTION)

# 上采样到目标分辨率
zoom_y = target_rows / pop_raw.shape[0]
zoom_x = target_cols / pop_raw.shape[1]
pop_density = zoom(pop_raw, (zoom_y, zoom_x), order=1)

# 应用NHAPS时间因子 (与项目temporal_population_estimate_layer一致)
# NHAPS: 住宅区在hour=9时约35%人口在室外
nhaps_outdoor_fraction = {
    0: 0.05, 6: 0.15, 9: 0.35, 12: 0.45, 15: 0.40, 18: 0.35, 21: 0.15
}
outdoor_frac = nhaps_outdoor_fraction.get(HOUR, 0.35)
pop_grid = pop_density * outdoor_frac

print(f"  目标网格: {pop_grid.shape}")
print(f"  室外人口比例: {outdoor_frac*100:.0f}% (NHAPS hour={HOUR})")
print(f"  人口密度范围: {pop_grid.min():.1f} ~ {pop_grid.max():.1f} 人/km²")

# ============ 2. 配置飞行器 ============
print("\n[2/3] 配置飞行器 (casex.AircraftSpecs)...")
aircraft = AircraftSpecs(AircraftType.FIXED_WING, 3.0, 2.0, 4.0)  # 翼展3m, 长2m, 质量4kg
aircraft.set_ballistic_drag_coefficient(0.8)
aircraft.set_glide_speed_ratio(12, 15)
aircraft.set_ballistic_frontal_area(0.1)
print(f"  类型: 固定翼无人机")
print(f"  质量: {aircraft.mass} kg")
print(f"  翼展: {aircraft.width} m")
print(f"  飞行速度: {AIRSPEED} m/s")
print(f"  飞行高度: {ALTITUDE} m")
print(f"  失效概率: {FAILURE_PROB}")

# ============ 3. 使用项目API计算风险 ============
print("\n[3/3] 计算风险网格 (项目API)...")

# 调用项目的 make_strike_grid
print("  调用 make_strike_grid (弹道模型+滑翔模型)...")
strike_grid, v_is = make_strike_grid(
    aircraft, AIRSPEED, ALTITUDE, FAILURE_PROB,
    pop_grid, RESOLUTION, WIND_DIR, WIND_SPEED
)
print(f"  碰撞风险范围: {strike_grid.min():.2e} ~ {strike_grid.max():.2e}")
print(f"  弹道撞击速度: {v_is[0]:.1f} m/s")
print(f"  滑翔撞击速度: {v_is[1]:.1f} m/s")

# 调用项目的 make_fatality_grid
print("  调用 make_fatality_grid (致命模型)...")
fatality_grid = make_fatality_grid(aircraft, strike_grid, v_is)
print(f"  致命风险范围: {fatality_grid.min():.2e} ~ {fatality_grid.max():.2e}")

# ============ 4. 可视化 ============
print("\n生成可视化...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]

im1 = axes[0].imshow(pop_grid, cmap='YlOrRd', extent=extent, origin='upper')
axes[0].set_title(f'Population Density\n(Hour={HOUR}, Outdoor={outdoor_frac*100:.0f}%)', fontsize=11)
plt.colorbar(im1, ax=axes[0], label='people/km²', shrink=0.8)

im2 = axes[1].imshow(np.log10(strike_grid + 1e-15), cmap='hot', extent=extent, origin='upper')
axes[1].set_title('Strike Probability\n(Ballistic + Glide Model)', fontsize=11)
plt.colorbar(im2, ax=axes[1], label='log₁₀(P)', shrink=0.8)

im3 = axes[2].imshow(np.log10(fatality_grid + 1e-15), cmap='jet', extent=extent, origin='upper')
axes[2].set_title('Fatality Risk\n(Impact Energy Model)', fontsize=11)
plt.colorbar(im3, ax=axes[2], label='log₁₀(Risk)', shrink=0.8)

for ax in axes:
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

plt.suptitle('Hohhot UAV Ground Risk Assessment\n(Using Project API: make_strike_grid, make_fatality_grid)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()

output_file = './output_hohhot/hohhot_risk_api.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ 已保存: {output_file}")

# ============ 输出统计 ============
print("\n" + "=" * 60)
print("统计摘要")
print("=" * 60)
print(f"网格尺寸: {pop_grid.shape[0]} × {pop_grid.shape[1]} ({RESOLUTION}m分辨率)")
print(f"平均人口密度: {pop_grid.mean():.0f} 人/km²")
print(f"最大人口密度: {pop_grid.max():.0f} 人/km²")
print(f"平均碰撞风险: {strike_grid.mean():.2e}")
print(f"最大碰撞风险: {strike_grid.max():.2e}")
print(f"平均致命风险: {fatality_grid.mean():.2e}")
print(f"最大致命风险: {fatality_grid.max():.2e}")

print("\n使用的项目组件:")
print("  - casex.AircraftSpecs: 飞行器规格")
print("  - api.make_strike_grid: 碰撞概率 (BallisticModel + GlideDescentModel + StrikeModel)")
print("  - api.make_fatality_grid: 致命风险 (FatalityModel)")
print("  - NHAPS时间模式: 室内/室外人口分布")
