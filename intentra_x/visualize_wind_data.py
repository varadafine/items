"""Visualize wind speed and direction in the AirSim gust-flight dataset."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "UAV_gust_all.jsonl"
OUTPUT_DIR = HERE / "demo_output" / "wind_visualization"
FONT_PATH = Path(r"C:\Windows\Fonts\simsun.ttc")

CN = FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else FontProperties(family="SimSun")
COLORS = ["#2F5D8C", "#C35A3A", "#4A8C68", "#8669A6", "#B98A30"]


def load_flights(path: Path):
    flights = defaultdict(list)
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            flights[row["flight_id"]].append(row)
    for rows in flights.values():
        rows.sort(key=lambda item: item["timestamp"])
    return dict(sorted(flights.items()))


def arrays(rows):
    timestamp = np.asarray([row["timestamp"] for row in rows], dtype=float)
    elapsed = timestamp - timestamp[0]
    speed = np.asarray([row.get("wind_speed", 0.0) or 0.0 for row in rows], dtype=float)
    sin_value = np.asarray([row.get("wind_direction_sin", 0.0) or 0.0 for row in rows], dtype=float)
    cos_value = np.asarray([row.get("wind_direction_cos", 1.0) or 1.0 for row in rows], dtype=float)
    direction_from = np.degrees(np.arctan2(sin_value, cos_value)) % 360.0
    position = np.asarray([row.get("position") or [0.0, 0.0, 0.0] for row in rows], dtype=float)
    return elapsed, speed, direction_from, position


def style_axis(ax):
    ax.grid(True, color="#D8DEE6", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, colors="#2F3740")


def add_summary(fig, sample_count, flight_count, speed_min, speed_max):
    fig.text(
        0.5,
        0.015,
        f"数据：{sample_count}条记录，{flight_count}次飞行；风速范围{speed_min:.2f}～{speed_max:.2f} m/s；典型采样间隔约0.108 s",
        ha="center",
        va="bottom",
        fontproperties=CN,
        fontsize=9,
        color="#4A5560",
    )


def plot_speed(flights, output_path: Path, title: str | None = None):
    fig, ax = plt.subplots(figsize=(10.8, 5.3), dpi=220)
    all_speed = []
    for index, (flight_id, rows) in enumerate(flights.items(), 1):
        elapsed, speed, _, _ = arrays(rows)
        all_speed.extend(speed.tolist())
        ax.plot(
            elapsed,
            speed,
            linewidth=1.55,
            color=COLORS[index - 1],
            alpha=0.92,
            label=f"飞行{index}（{flight_id[-6:]}）",
        )
    style_axis(ax)
    ax.set_title(title or "AirSim五次飞行的风速时序", fontproperties=CN, fontsize=15, pad=12)
    ax.set_xlabel("单次飞行经过时间（s）", fontproperties=CN, fontsize=11)
    ax.set_ylabel("风速（m/s）", fontproperties=CN, fontsize=11)
    ax.set_ylim(bottom=0)
    ax.legend(prop=CN, fontsize=9, ncol=3, frameon=False, loc="upper right")
    add_summary(fig, sum(map(len, flights.values())), len(flights), min(all_speed), max(all_speed))
    fig.tight_layout(rect=(0.025, 0.05, 0.99, 0.98))
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_direction(flights, output_path: Path, title: str | None = None):
    fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=220)
    for index, (flight_id, rows) in enumerate(flights.items(), 1):
        elapsed, _, direction, _ = arrays(rows)
        ax.scatter(
            elapsed,
            direction,
            s=5,
            color=COLORS[index - 1],
            alpha=0.58,
            edgecolors="none",
            label=f"飞行{index}（{flight_id[-6:]}）",
        )
    style_axis(ax)
    ax.set_title(title or "AirSim五次飞行的风向时序", fontproperties=CN, fontsize=15, pad=12)
    ax.set_xlabel("单次飞行经过时间（s）", fontproperties=CN, fontsize=11)
    ax.set_ylabel("风向角（°，按当前项目编码恢复）", fontproperties=CN, fontsize=11)
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.legend(prop=CN, fontsize=9, ncol=3, frameon=False, loc="upper right")
    fig.text(
        0.5,
        0.018,
        "说明：0°与360°表示同一方向，图中跨越边界的跳变不一定代表风向发生突变。",
        ha="center",
        va="bottom",
        fontproperties=CN,
        fontsize=9,
        color="#4A5560",
    )
    fig.tight_layout(rect=(0.025, 0.055, 0.99, 0.98))
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_wind_rose(flights, output_path: Path):
    all_speed, all_direction = [], []
    for rows in flights.values():
        _, speed, direction, _ = arrays(rows)
        all_speed.append(speed)
        all_direction.append(direction)
    speed = np.concatenate(all_speed)
    direction = np.concatenate(all_direction)

    bin_count = 16
    edges = np.linspace(0.0, 360.0, bin_count + 1)
    bin_index = np.digitize(direction, edges, right=False) - 1
    bin_index[bin_index == bin_count] = 0
    counts = np.bincount(bin_index, minlength=bin_count).astype(float)
    share = counts / counts.sum() * 100.0
    mean_speed = np.asarray(
        [speed[bin_index == i].mean() if np.any(bin_index == i) else 0.0 for i in range(bin_count)]
    )
    centers = np.deg2rad((edges[:-1] + edges[1:]) / 2.0)
    width = np.deg2rad(360.0 / bin_count * 0.88)

    fig = plt.figure(figsize=(7.5, 7.2), dpi=220)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    norm = colors.Normalize(vmin=float(speed.min()), vmax=float(speed.max()))
    cmap = plt.get_cmap("viridis")
    ax.bar(centers, share, width=width, bottom=0.0, color=cmap(norm(mean_speed)),
           edgecolor="white", linewidth=0.9, alpha=0.94)
    ax.set_title("风向分布与对应平均风速", fontproperties=CN, fontsize=15, pad=22)
    ax.set_thetagrids(np.arange(0, 360, 45), labels=["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.set_rlabel_position(68)
    ax.set_ylabel("样本占比（%）", fontproperties=CN, fontsize=10, labelpad=26)
    ax.grid(color="#D8DEE6", linewidth=0.7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.72)
    cbar.set_label("各方向区间的平均风速（m/s）", fontproperties=CN, fontsize=10)
    fig.text(0.5, 0.02, "角度按当前项目中 atan2(sin, cos) 的恢复方式统计。",
             ha="center", fontproperties=CN, fontsize=9, color="#4A5560")
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_vectors(flights, output_path: Path):
    flight_id, rows = next(iter(flights.items()))
    elapsed, speed, direction_from, position = arrays(rows)
    x, y = position[:, 0], position[:, 1]
    direction_to = (direction_from + 180.0) % 360.0
    theta_to = np.deg2rad(direction_to)
    wind_east = speed * np.sin(theta_to)
    wind_north = speed * np.cos(theta_to)

    step = max(1, len(rows) // 28)
    select = np.arange(0, len(rows), step)
    max_speed = max(float(speed.max()), 1.0)
    arrow_scale = max(np.ptp(x), np.ptp(y), 1.0) * 0.065
    u = wind_east[select] / max_speed * arrow_scale
    v = wind_north[select] / max_speed * arrow_scale

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=220)
    ax.plot(x, y, color="#56616D", linewidth=1.5, alpha=0.8, label="无人机轨迹")
    norm = colors.Normalize(vmin=float(speed.min()), vmax=float(speed.max()))
    q = ax.quiver(
        x[select], y[select], u, v, speed[select],
        cmap="turbo", norm=norm, angles="xy", scale_units="xy", scale=1.0,
        width=0.0048, headwidth=3.7, headlength=4.8, alpha=0.92,
    )
    ax.scatter(x[0], y[0], s=58, color="#22805A", marker="o", zorder=4, label="起点")
    ax.scatter(x[-1], y[-1], s=72, color="#A94032", marker="X", zorder=4, label="终点")
    style_axis(ax)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("代表性飞行轨迹与风漂移方向", fontproperties=CN, fontsize=15, pad=12)
    ax.set_xlabel("AirSim局部X坐标（m）", fontproperties=CN, fontsize=11)
    ax.set_ylabel("AirSim局部Y坐标（m）", fontproperties=CN, fontsize=11)
    ax.legend(prop=CN, fontsize=9, frameon=False, loc="best")
    cbar = fig.colorbar(q, ax=ax, pad=0.025, shrink=0.84)
    cbar.set_label("风速（m/s）", fontproperties=CN, fontsize=10)
    fig.text(
        0.5,
        0.018,
        f"代表飞行：{flight_id}。箭头表示按当前地面模型约定换算的风吹向方向，已按轨迹尺度缩放。",
        ha="center",
        fontproperties=CN,
        fontsize=9,
        color="#4A5560",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_overview(paths, output_path: Path):
    from PIL import Image, ImageOps, ImageDraw

    images = [Image.open(path).convert("RGB") for path in paths]
    tile_w, tile_h = 1050, 650
    canvas = Image.new("RGB", (tile_w * 2 + 40, tile_h * 2 + 40), "white")
    for index, image in enumerate(images):
        fitted = ImageOps.contain(image, (tile_w - 20, tile_h - 20))
        x0 = 20 + (index % 2) * tile_w + (tile_w - fitted.width) // 2
        y0 = 20 + (index // 2) * tile_h + (tile_h - fitted.height) // 2
        canvas.paste(fitted, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.line((tile_w + 20, 10, tile_w + 20, tile_h * 2 + 30), fill="#D8DEE6", width=2)
    draw.line((10, tile_h + 20, tile_w * 2 + 30, tile_h + 20), fill="#D8DEE6", width=2)
    canvas.save(output_path, quality=95)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flights = load_flights(DATA_PATH)
    speed_path = OUTPUT_DIR / "wind_speed_timeseries.png"
    direction_path = OUTPUT_DIR / "wind_direction_timeseries.png"
    rose_path = OUTPUT_DIR / "wind_direction_rose.png"
    spatial_path = OUTPUT_DIR / "wind_trajectory_vectors.png"
    overview_path = OUTPUT_DIR / "wind_visualization_overview.png"
    plot_speed(flights, speed_path)
    plot_direction(flights, direction_path)
    plot_wind_rose(flights, rose_path)
    plot_spatial_vectors(flights, spatial_path)
    make_overview([speed_path, direction_path, rose_path, spatial_path], overview_path)
    for path in (speed_path, direction_path, rose_path, spatial_path, overview_path):
        print(path)


if __name__ == "__main__":
    main()
