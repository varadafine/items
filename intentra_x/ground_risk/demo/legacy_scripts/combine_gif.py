#!/usr/bin/env python
"""
将生成的TIF文件合成为GIF动画
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
HOURS = [3, 7, 10, 14, 18, 22]  # 与shell脚本一致
GIF_DURATION = 800  # 每帧800ms

def load_tif(filepath):
    """加载TIF文件"""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            bounds = src.bounds
        return data, bounds
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return np.zeros((100, 100)), None

def main():
    print("Combining TIF files to GIF...")
    
    # 收集所有数据的范围
    all_pop, all_strike, all_fatality = [], [], []
    
    for hour in HOURS:
        hour_dir = f"{OUTPUT_DIR}/h{hour}"
        p, _ = load_tif(f"{hour_dir}/pop_density_{hour}h.tif")
        s, _ = load_tif(f"{hour_dir}/strike_{hour}h.tif")
        f, _ = load_tif(f"{hour_dir}/fatality_{hour}h.tif")
        all_pop.append(p.max())
        all_strike.append(s.max())
        all_fatality.append(f.max())
        del p, s, f
        gc.collect()
    
    vmax = {
        'pop': max(max(all_pop), 1),
        'strike': max(max(all_strike), 1e-10),
        'fatality': max(max(all_fatality), 1e-10)
    }
    print(f"Value ranges: pop={vmax['pop']:.0f}, strike={vmax['strike']:.2e}, fatality={vmax['fatality']:.2e}")
    
    # 生成每帧
    frame_paths = []
    for hour in HOURS:
        print(f"  Creating frame for hour {hour}...")
        hour_dir = f"{OUTPUT_DIR}/h{hour}"
        
        pop, bounds = load_tif(f"{hour_dir}/pop_density_{hour}h.tif")
        strike, _ = load_tif(f"{hour_dir}/strike_{hour}h.tif")
        fatality, _ = load_tif(f"{hour_dir}/fatality_{hour}h.tif")
        
        if bounds:
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        else:
            extent = [-1.5, -1.3, 50.87, 51.0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=80)
        
        im1 = axes[0].imshow(pop, cmap='viridis', extent=extent, vmin=0, vmax=vmax['pop'], aspect='auto')
        plt.colorbar(im1, ax=axes[0], label='people/km²')
        axes[0].set_title(f'Population Density\nt={hour:02d}:00')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        
        im2 = axes[1].imshow(strike, cmap='jet', extent=extent, vmin=0, vmax=vmax['strike'], aspect='auto')
        plt.colorbar(im2, ax=axes[1], label='Strike Risk', format='%.1e')
        axes[1].set_title(f'Strike Risk\nt={hour:02d}:00')
        axes[1].set_xlabel('Longitude')
        
        im3 = axes[2].imshow(fatality, cmap='jet', extent=extent, vmin=0, vmax=vmax['fatality'], aspect='auto')
        plt.colorbar(im3, ax=axes[2], label='Fatality Risk', format='%.1e')
        axes[2].set_title(f'Fatality Risk\nt={hour:02d}:00')
        axes[2].set_xlabel('Longitude')
        
        plt.suptitle(f'Southampton Risk Assessment - {hour:02d}:00', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        frame_path = f"{OUTPUT_DIR}/frame_{hour:02d}.png"
        plt.savefig(frame_path, dpi=80, bbox_inches='tight')
        plt.close(fig)
        frame_paths.append(frame_path)
        
        del pop, strike, fatality, fig, axes
        gc.collect()
    
    # 合成GIF
    print("\nCreating GIF...")
    frames = []
    for p in frame_paths:
        if os.path.exists(p):
            img = Image.open(p)
            frames.append(img.copy())
            img.close()
    
    if frames:
        gif_path = f"{OUTPUT_DIR}/risk_animation.gif"
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                      duration=GIF_DURATION, loop=0)
        print(f"\nSuccess! GIF saved: {gif_path}")
        print(f"  - {len(frames)} frames")
        print(f"  - {GIF_DURATION}ms per frame")
    else:
        print("No frames to combine!")

if __name__ == '__main__':
    main()
