#!/usr/bin/env python
"""合并PNG为GIF"""
import os
from PIL import Image

OUTPUT_DIR = './output_gif'
HOURS = [0, 6, 12, 18]  # 6小时间隔

frames = []
for h in HOURS:
    path = f'{OUTPUT_DIR}/frame_{h:02d}.png'
    if os.path.exists(path):
        img = Image.open(path)
        frames.append(img.copy())
        img.close()
        print(f"Loaded: {path}")
    else:
        print(f"Missing: {path}")

if frames:
    gif_path = f'{OUTPUT_DIR}/risk_animation.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=1000, loop=0)
    print(f"\nGIF saved: {gif_path} ({len(frames)} frames, 1s each)")
else:
    print("No frames found!")
