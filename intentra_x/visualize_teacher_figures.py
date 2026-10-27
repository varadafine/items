"""Teacher-requested visualizations for a single representative flight.

All figures use flight WIND_003_FLIGHT_1789880454191
(UAV_gust_manual_sessions.jsonl, lines 675-865, 191 samples @10Hz, 18.8 s).

Figures:
  1. wind speed vs time
  2. wind direction vs wind speed (relationship)
  3. 3D trajectory colored by wind speed, with wind-direction arrows
  4. UAV speed vs time, split into four charts:
       (a) total speed |v|      (magnitude, >=0)
       (b) horizontal speed     (magnitude, >=0)
       (c) vertical speed        (height-axis magnitude |vz|, >=0)
       (d) climb rate = -vz      (signed: + up / - down)

Speed magnitudes are always >=0 by definition; only the climb-rate chart
carries a sign, which keeps the "speed" charts free of confusing negatives.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "UAV_gust_manual_sessions.jsonl"
OUTPUT_DIR = HERE / "demo_output" / "single_flight_visualization"
FONT_PATH = Path(r"C:\Windows\Fonts\simsun.ttc")
CN = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else FontProperties(family="SimSun")

# Chosen by eye from _compare_3d.py: cleanest single-path 3D story — fly out
# near ground, climb to ~53 m at the far end, arc back and descend to the end
# point. 53 m altitude range, 10 wind segments, 31% gust, 12 collisions.
FLIGHT_ID = "WIND_003_FLIGHT_1790153665040"

import json


def load_flight(flight_id):
    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["flight_id"] == flight_id:
                rows.append(r)
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def flight_arrays(rows):
    ts = np.asarray([r["timestamp"] for r in rows], dtype=float)
    elapsed = ts - ts[0]
    vel = np.asarray([r["velocity"] for r in rows], dtype=float)
    pos = np.asarray([r["position"] for r in rows], dtype=float)
    wind_speed = np.asarray([r.get("wind_speed", 0.0) or 0.0 for r in rows], dtype=float)
    sin_v = np.asarray([r.get("wind_direction_sin", 0.0) or 0.0 for r in rows], dtype=float)
    cos_v = np.asarray([r.get("wind_direction_cos", 1.0) or 1.0 for r in rows], dtype=float)
    wind_from = np.degrees(np.arctan2(sin_v, cos_v)) % 360.0
    return elapsed, vel, pos, wind_speed, wind_from


def style_axis(ax):
    ax.grid(True, color="#D8DEE6", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, colors="#2F3740")


def save(fig, path):
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(path)


TAIL = FLIGHT_ID[-6:]


# ---------- Figure 1: wind speed vs time ----------
def fig_wind_speed(elapsed, wind_speed, path):
    fig, ax = plt.subplots(figsize=(10.8, 5.3), dpi=220)
    ax.plot(elapsed, wind_speed, color="#2F5D8C", linewidth=1.8, alpha=0.95)
    ax.fill_between(elapsed, 0, wind_speed, color="#2F5D8C", alpha=0.10)
    style_axis(ax)
    ax.set_title("风速随时间变化", fontproperties=CN, fontsize=15, pad=12)
    ax.set_xlabel("飞行经过时间（s）", fontproperties=CN, fontsize=11)
    ax.set_ylabel("风速（m/s）", fontproperties=CN, fontsize=11)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, path)


# ---------- Figure 2: wind rose (direction distribution + mean speed) ----------
def fig_dir_speed(wind_from, wind_speed, path):
    bin_count = 16
    edges = np.linspace(0.0, 360.0, bin_count + 1)
    bin_index = np.digitize(wind_from, edges, right=False) - 1
    bin_index[bin_index == bin_count] = 0
    counts = np.bincount(bin_index, minlength=bin_count).astype(float)
    share = counts / counts.sum() * 100.0
    mean_speed = np.asarray(
        [wind_speed[bin_index == i].mean() if np.any(bin_index == i) else 0.0
         for i in range(bin_count)]
    )
    centers = np.deg2rad((edges[:-1] + edges[1:]) / 2.0)
    width = np.deg2rad(360.0 / bin_count * 0.88)

    fig = plt.figure(figsize=(7.8, 7.4), dpi=220)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    norm = colors.Normalize(vmin=float(wind_speed.min()), vmax=float(wind_speed.max()))
    cmap = plt.get_cmap("viridis")
    ax.bar(centers, share, width=width, bottom=0.0, color=cmap(norm(mean_speed)),
           edgecolor="white", linewidth=0.9, alpha=0.94)
    ax.set_title("风向分布与对应平均风速", fontproperties=CN, fontsize=15, pad=22)
    ax.set_thetagrids(np.arange(0, 360, 45),
                      labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.set_rlabel_position(68)
    ax.set_ylabel("样本占比（%）", fontproperties=CN, fontsize=10, labelpad=26)
    ax.grid(color="#D8DEE6", linewidth=0.7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.72)
    cbar.set_label("各方向区间的平均风速（m/s）", fontproperties=CN, fontsize=10)
    fig.text(0.5, 0.02, "扇区角度=风向（来向），柱高=样本占比，颜色=该方向平均风速。",
             ha="center", fontproperties=CN, fontsize=9, color="#4A5560")
    save(fig, path)


# ---------- Figure 3: 3D trajectory + wind ----------
def fig_3d_trajectory(pos, wind_speed, wind_from, path):
    x = pos[:, 0]
    y = pos[:, 1]
    alt = -pos[:, 2]  # NED: z is +down, altitude = -z

    fig = plt.figure(figsize=(10.5, 8.5), dpi=220)
    ax = fig.add_subplot(111, projection="3d")

    norm = colors.Normalize(vmin=float(wind_speed.min()), vmax=float(wind_speed.max()))
    cmap = plt.get_cmap("turbo")

    # colored trajectory segments by wind speed
    for i in range(1, len(x)):
        ax.plot(x[i-1:i+1], y[i-1:i+1], alt[i-1:i+1],
                color=cmap(norm(wind_speed[i])), linewidth=2.0, alpha=0.9)

    # wind-direction arrows (wind blows toward from+180), scaled to plot size
    step = max(1, len(x) // 22)
    sel = np.arange(0, len(x), step)
    dir_to = np.deg2rad((wind_from + 180.0) % 360.0)
    span = max(np.ptp(x), np.ptp(y), np.ptp(alt), 1.0)
    L = span * 0.06
    u = wind_speed * np.sin(dir_to)
    v = wind_speed * np.cos(dir_to)
    wmax = max(wind_speed.max(), 1.0)
    ax.quiver(x[sel], y[sel], alt[sel],
              u[sel] / wmax * L, v[sel] / wmax * L, np.zeros(len(sel)),
              color="#333333", alpha=0.7, linewidth=1.0, arrow_length_ratio=0.4)

    ax.scatter(x[0], y[0], alt[0], s=70, color="#22805A", marker="o", label="起点")
    ax.scatter(x[-1], y[-1], alt[-1], s=90, color="#A94032", marker="X", label="终点")

    ax.set_title("三维飞行轨迹与风速风向关系", fontproperties=CN, fontsize=15, pad=14)
    ax.set_xlabel("X（m）", fontproperties=CN, fontsize=10, labelpad=8)
    ax.set_ylabel("Y（m）", fontproperties=CN, fontsize=10, labelpad=8)
    ax.set_zlabel("高度（m）", fontproperties=CN, fontsize=10, labelpad=6)
    ax.legend(prop=CN, fontsize=9, loc="upper left")
    ax.view_init(elev=22, azim=-60)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.6)
    cbar.set_label("风速（m/s）", fontproperties=CN, fontsize=10)
    fig.text(0.5, 0.02, "轨迹颜色=风速；箭头=风吹向方向（水平面内，已按轨迹尺度缩放）。",
             ha="center", fontproperties=CN, fontsize=9, color="#4A5560")
    save(fig, path)


# ---------- Figure 4a-d: UAV speed components ----------
def _speed_chart(elapsed, series, title, ylabel, color, path, signed=False):
    fig, ax = plt.subplots(figsize=(10.8, 5.0), dpi=220)
    ax.plot(elapsed, series, color=color, linewidth=1.9, alpha=0.95)
    if signed:
        ax.axhline(0.0, color="#9AA5B1", linewidth=0.9, linestyle="--", alpha=0.8)
    else:
        ax.fill_between(elapsed, 0, series, color=color, alpha=0.10)
        ax.set_ylim(bottom=0)
    style_axis(ax)
    ax.set_title(title, fontproperties=CN, fontsize=15, pad=12)
    ax.set_xlabel("飞行经过时间（s）", fontproperties=CN, fontsize=11)
    ax.set_ylabel(ylabel, fontproperties=CN, fontsize=11)
    stat = (f"峰值{series.max():.1f} / 均值{series.mean():.1f} m/s"
            if not signed else
            f"范围 {series.min():.1f} ~ {series.max():.1f} m/s（正为上升，负为下降）")
    fig.text(0.5, 0.015, f"数据：{len(elapsed)}个采样点；{stat}。",
             ha="center", va="bottom", fontproperties=CN, fontsize=9, color="#4A5560")
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.98))
    save(fig, path)


def fig_speed_stacked(elapsed, total, horizontal, vertical, path):
    """Three speed components stacked vertically, sharing one time axis:
    total on top, horizontal in the middle, vertical (height axis) at the bottom.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 9.6), dpi=220, sharex=True)
    panels = [
        (total, "合速度 |v|", "#2F5D8C"),
        (horizontal, "水平速度", "#4A8C68"),
        (vertical, "垂直速度 |vz|", "#8669A6"),
    ]
    for ax, (series, label, color) in zip(axes, panels):
        ax.plot(elapsed, series, color=color, linewidth=1.8, alpha=0.95)
        ax.fill_between(elapsed, 0, series, color=color, alpha=0.12)
        ax.set_ylim(bottom=0)
        style_axis(ax)
        ax.set_ylabel(f"{label}\n（m/s）", fontproperties=CN, fontsize=10.5)
        ax.text(0.012, 0.86, f"峰值{series.max():.1f} / 均值{series.mean():.1f} m/s",
                transform=ax.transAxes, fontproperties=CN, fontsize=9, color="#4A5560")
    axes[-1].set_xlabel("飞行经过时间（s）", fontproperties=CN, fontsize=11)
    fig.suptitle(f"无人机速度分量随时间变化（{TAIL}）", fontproperties=CN, fontsize=15, y=0.98)
    fig.text(0.5, 0.02, f"数据：{len(elapsed)}个采样点；上=合速度，中=水平速度，下=垂直速度（高度方向），共享同一时间轴。",
             ha="center", fontproperties=CN, fontsize=9, color="#4A5560")
    fig.tight_layout(rect=(0.02, 0.035, 0.99, 0.96))
    save(fig, path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_flight(FLIGHT_ID)
    if not rows:
        raise SystemExit(f"flight {FLIGHT_ID} not found")
    print(f"Flight {FLIGHT_ID}: {len(rows)} samples")

    elapsed, vel, pos, wind_speed, wind_from = flight_arrays(rows)
    total = np.linalg.norm(vel, axis=1)
    horizontal = np.linalg.norm(vel[:, :2], axis=1)
    vertical = np.abs(vel[:, 2])
    climb = -vel[:, 2]

    fig_wind_speed(elapsed, wind_speed, OUTPUT_DIR / "fig1_wind_speed_time.png")
    fig_dir_speed(wind_from, wind_speed, OUTPUT_DIR / "fig2_wind_dir_vs_speed.png")
    fig_3d_trajectory(pos, wind_speed, wind_from, OUTPUT_DIR / "fig3_trajectory_3d.png")
    _speed_chart(elapsed, total, f"合速度随时间变化（{TAIL}）", "合速度 |v|（m/s）",
                 "#2F5D8C", OUTPUT_DIR / "fig4a_speed_total.png")
    _speed_chart(elapsed, horizontal, f"水平速度随时间变化（{TAIL}）", "水平速度（m/s）",
                 "#4A8C68", OUTPUT_DIR / "fig4b_speed_horizontal.png")
    _speed_chart(elapsed, vertical, f"垂直速度随时间变化（高度方向）（{TAIL}）", "垂直速度 |vz|（m/s）",
                 "#8669A6", OUTPUT_DIR / "fig4c_speed_vertical.png")
    _speed_chart(elapsed, climb, f"爬升率随时间变化（{TAIL}）", "爬升率（m/s）",
                 "#C35A3A", OUTPUT_DIR / "fig4d_climb_rate.png", signed=True)
    fig_speed_stacked(elapsed, total, horizontal, vertical,
                      OUTPUT_DIR / "fig4e_speed_stacked.png")


if __name__ == "__main__":
    main()
