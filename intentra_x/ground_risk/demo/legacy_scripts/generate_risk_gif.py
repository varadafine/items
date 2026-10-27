#!/usr/bin/env python
"""
基于已有hour 9数据 + NHAPS时间模式生成24小时GIF动画
完全离线，无需OSM查询
"""

import os
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import rasterio

OUTPUT_DIR = './output_gif'
H9_DIR = './output_h9'
HOURS = [0, 6, 12, 18]  # 6小时间隔
GIF_DURATION = 1000

# NHAPS住宅区人口比例
NHAPS = {
    0: 0.95, 6: 0.80, 12: 0.40, 18: 0.70,
    9: 0.45  # hour 9 基准
}

def load_tif(path):
    with rasterio.open(path) as src:
        return src.read(1), src.bounds

def main():
    print("Loading hour 9 base data...")
    pop9, bounds = load_tif(f"{H9_DIR}/pop_density_9h.tif")
    strike9, _ = load_tif(f"{H9_DIR}/strike_9h.tif")
    fatality9, _ = load_tif(f"{H9_DIR}/fatality_9h.tif")
    print(f"  Shape: {pop9.shape}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    # 计算所有时间点的最大值
    all_max = {'pop': [], 'strike': [], 'fatality': []}
    for h in HOURS:
        scale = NHAPS[h] / NHAPS[9]
        all_max['pop'].append(pop9.max() * scale)
        all_max['strike'].append(strike9.max() * scale)
        all_max['fatality'].append(fatality9.max() * scale)
    
    vmax = {
        'pop': max(all_max['pop']) * 1.1,
        'strike': max(all_max['strike']) * 1.1,
        'fatality': max(all_max['fatality']) * 1.1
    }
    
    print(f"\nGenerating {len(HOURS)} frames...")
    frame_paths = []
    
    for h in HOURS:
        scale = NHAPS[h] / NHAPS[9]
        pop = pop9 * scale
        strike = strike9 * scale
        fatality = fatality9 * scale
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=80)
        
        im1 = axes[0].imshow(pop, cmap='viridis', extent=extent, vmin=0, vmax=vmax['pop'], aspect='auto')
        plt.colorbar(im1, ax=axes[0], label='people/km²')
        axes[0].set_title(f'Population Density\nt={h:02d}:00')
        
        im2 = axes[1].imshow(strike, cmap='jet', extent=extent, vmin=0, vmax=vmax['strike'], aspect='auto')
        plt.colorbar(im2, ax=axes[1], label='Strike Risk', format='%.1e')
        axes[1].set_title(f'Strike Risk\nt={h:02d}:00')
        
        im3 = axes[2].imshow(fatality, cmap='jet', extent=extent, vmin=0, vmax=vmax['fatality'], aspect='auto')
        plt.colorbar(im3, ax=axes[2], label='Fatality Risk', format='%.1e')
        axes[2].set_title(f'Fatality Risk\nt={h:02d}:00')
        
        plt.suptitle(f'Southampton Ground Risk - {h:02d}:00 (NHAPS scale: {NHAPS[h]:.0%})', fontsize=12)
        plt.tight_layout()
        
        path = f'{OUTPUT_DIR}/frame_{h:02d}.png'
        plt.savefig(path, dpi=80)
        plt.close()
        frame_paths.append(path)
        print(f"  Hour {h:02d}: scale={scale:.2f}")
        gc.collect()
    
    # 创建GIF
    print("\nCreating GIF...")
    frames = [Image.open(p).copy() for p in frame_paths if os.path.exists(p)]
    if frames:
        gif_path = f'{OUTPUT_DIR}/risk_animation.gif'
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=GIF_DURATION, loop=0)
        print(f"\nSuccess! {gif_path}")
        print(f"  {len(frames)} frames, {GIF_DURATION}ms each")

if __name__ == '__main__':
    main()
