#!/usr/bin/env python
"""生成单个小时的风险图"""
import sys
import os
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import casex

from seedpod_ground_risk.api.api import make_pop_grid, make_strike_grid, make_fatality_grid
from seedpod_ground_risk.core.utils import make_bounds_polygon

MIN_LAT, MAX_LAT = 50.87, 51.0
MIN_LON, MAX_LON = -1.5, -1.3
RESOLUTION = 60
OUTPUT_DIR = './output_gif'

def main():
    if len(sys.argv) < 2:
        print("Usage: python gen_one_hour.py <hour>")
        sys.exit(1)
    
    hour = int(sys.argv[1])
    print(f"Generating hour {hour:02d}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    bounds = make_bounds_polygon(MIN_LAT, MAX_LAT, MIN_LON, MAX_LON)
    
    aircraft = casex.AircraftSpecs(casex.enums.AircraftType.FIXED_WING, 2, 1.5, 7)
    aircraft.set_ballistic_drag_coefficient(0.8)
    aircraft.set_ballistic_frontal_area(0.1)
    aircraft.set_glide_speed_ratio(15, 12)
    aircraft.set_glide_drag_coefficient(0.1)
    
    try:
        pop_grid = make_pop_grid(bounds, hour, RESOLUTION)
        if pop_grid is None or np.all(pop_grid == 0):
            pop_grid = np.zeros((100, 100))
        
        strike_grid, v_is = make_strike_grid(aircraft, 20, 120, 5e-3, pop_grid, RESOLUTION, 90, 5)
        fatality_grid = make_fatality_grid(aircraft, strike_grid, v_is)
    except Exception as e:
        print(f"Error: {e}")
        pop_grid = np.zeros((100, 100))
        strike_grid = np.zeros((100, 100))
        fatality_grid = np.zeros((100, 100))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=80)
    extent = [MIN_LON, MAX_LON, MIN_LAT, MAX_LAT]
    vmax = {'pop': 100000, 'strike': 1e-6, 'fatality': 3e-7}
    
    im1 = axes[0].imshow(pop_grid, cmap='viridis', extent=extent, vmin=0, vmax=vmax['pop'], aspect='auto')
    plt.colorbar(im1, ax=axes[0], label='people/km2')
    axes[0].set_title(f'Population Density t={hour:02d}:00')
    
    im2 = axes[1].imshow(strike_grid, cmap='jet', extent=extent, vmin=0, vmax=vmax['strike'], aspect='auto')
    plt.colorbar(im2, ax=axes[1], label='Strike Risk', format='%.1e')
    axes[1].set_title(f'Strike Risk t={hour:02d}:00')
    
    im3 = axes[2].imshow(fatality_grid, cmap='jet', extent=extent, vmin=0, vmax=vmax['fatality'], aspect='auto')
    plt.colorbar(im3, ax=axes[2], label='Fatality Risk', format='%.1e')
    axes[2].set_title(f'Fatality Risk t={hour:02d}:00')
    
    plt.suptitle(f'Southampton Ground Risk - {hour:02d}:00', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    frame_path = f'{OUTPUT_DIR}/frame_{hour:02d}.png'
    plt.savefig(frame_path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"Saved: {frame_path}")

if __name__ == '__main__':
    main()
