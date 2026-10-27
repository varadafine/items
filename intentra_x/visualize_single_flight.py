"""Visualize wind speed/direction and trajectory for one representative flight
from the manual gust-collection session.

Reuses the plotting style of visualize_wind_data.py, but targets a single
flight_id so every figure tells the story of one continuous flight.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import visualize_wind_data as vw

# SimSun lacks the Unicode minus glyph (U+2212); use ASCII hyphen on axis ticks.
vw.matplotlib.rcParams["axes.unicode_minus"] = False


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "UAV_gust_manual_sessions.jsonl"
OUTPUT_DIR = HERE / "demo_output" / "single_flight_visualization"

# Chosen from _compare_consistency.py: wind direction reads most 'normal' — long
# stretches of parallel arrows (two coherent regimes). openness 0.85 (clean
# traverse), 52% gust samples (strongest wind), 15 collisions.
FLIGHT_ID = "WIND_003_FLIGHT_1789880454191"


def plot_uav_speed(rows, output_path: Path, title: str):
    """Plot the UAV's own speed over one flight: total |v|, horizontal, vertical.

    NED convention: +vz is downward, so climb rate = -vz (positive = ascending).
    """
    ts = np.asarray([r["timestamp"] for r in rows], dtype=float)
    elapsed = ts - ts[0]
    vel = np.asarray([r["velocity"] for r in rows], dtype=float)
    total = np.linalg.norm(vel, axis=1)
    horizontal = np.linalg.norm(vel[:, :2], axis=1)
    climb = -vel[:, 2]

    fig, ax = vw.plt.subplots(figsize=(10.8, 5.3), dpi=220)

    ax.plot(elapsed, total, color="#2F5D8C", linewidth=1.9, alpha=0.95, label="总速度 |v|")
    ax.plot(elapsed, horizontal, color="#4A8C68", linewidth=1.4, alpha=0.9, label="水平速度")
    ax.plot(elapsed, climb, color="#C35A3A", linewidth=1.4, alpha=0.9, label="爬升率(正为上升/负为下降)")
    ax.axhline(0.0, color="#9AA5B1", linewidth=0.8, linestyle="--", alpha=0.7)

    vw.style_axis(ax)
    ax.set_title(title, fontproperties=vw.CN, fontsize=15, pad=12)
    ax.set_xlabel("单次飞行经过时间（s）", fontproperties=vw.CN, fontsize=11)
    ax.set_ylabel("速度（m/s）", fontproperties=vw.CN, fontsize=11)
    ax.legend(prop=vw.CN, fontsize=9, ncol=3, frameon=False, loc="upper right")
    fig.text(
        0.5, 0.015,
        f"数据：{len(rows)}个采样点；总速度峰值{total.max():.1f} m/s，均值{total.mean():.1f} m/s。",
        ha="center", va="bottom", fontproperties=vw.CN, fontsize=9, color="#4A5560",
    )
    fig.tight_layout(rect=(0.025, 0.05, 0.99, 0.98))
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    vw.plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_flights = vw.load_flights(DATA_PATH)
    if FLIGHT_ID not in all_flights:
        raise SystemExit(f"flight_id {FLIGHT_ID} not found in {DATA_PATH}")

    flight = {FLIGHT_ID: all_flights[FLIGHT_ID]}
    n = len(flight[FLIGHT_ID])
    print(f"Rendering flight {FLIGHT_ID}: {n} samples")

    speed_path = OUTPUT_DIR / "single_wind_speed_timeseries.png"
    direction_path = OUTPUT_DIR / "single_wind_direction_timeseries.png"
    rose_path = OUTPUT_DIR / "single_wind_direction_rose.png"
    spatial_path = OUTPUT_DIR / "single_wind_trajectory_vectors.png"
    overview_path = OUTPUT_DIR / "single_flight_overview.png"

    tail = FLIGHT_ID[-6:]
    uav_speed_path = OUTPUT_DIR / "single_uav_speed_timeseries.png"
    vw.plot_speed(flight, speed_path, title=f"单次飞行风速时序（{tail}）")
    vw.plot_direction(flight, direction_path, title=f"单次飞行风向时序（{tail}）")
    vw.plot_wind_rose(flight, rose_path)
    vw.plot_spatial_vectors(flight, spatial_path)
    plot_uav_speed(flight[FLIGHT_ID], uav_speed_path, title=f"单次飞行无人机速度时序（{tail}）")
    vw.make_overview([speed_path, direction_path, rose_path, spatial_path], overview_path)

    for path in (speed_path, direction_path, rose_path, spatial_path, uav_speed_path, overview_path):
        print(path)


if __name__ == "__main__":
    main()
