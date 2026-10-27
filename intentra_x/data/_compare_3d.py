"""Render 3D trajectory thumbnails for top 3D-suitable candidates."""
import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.font_manager import FontProperties

matplotlib.rcParams["axes.unicode_minus"] = False
HERE = Path(__file__).resolve().parent
PATH = HERE / "UAV_gust_manual_sessions.jsonl"
OUT = HERE.parent / "demo_output" / "single_flight_visualization" / "_3d_compare.png"
FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
CN = FontProperties(fname=str(FONT)) if FONT.exists() else FontProperties(family="SimSun")

CANDS = [
    ("WIND_003_FLIGHT_1789880754923", "高度变化最大(69m)"),
    ("WIND_003_FLIGHT_1789883104583", "风段最多(17)"),
    ("WIND_003_FLIGHT_1790153665040", "较立体+阵风31%"),
    ("WIND_003_FLIGHT_1789881134818", "碰撞最少(6)"),
]

by_flight = defaultdict(list)
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            by_flight[r["flight_id"]].append(r)
for rs in by_flight.values():
    rs.sort(key=lambda r: r["timestamp"])

fig = plt.figure(figsize=(13, 11), dpi=150)
for k, (fid, tag) in enumerate(CANDS, 1):
    rs = by_flight[fid]
    x = np.array([r["position"][0] for r in rs])
    y = np.array([r["position"][1] for r in rs])
    alt = np.array([-r["position"][2] for r in rs])
    ws = np.array([r["wind_speed"] for r in rs])
    ax = fig.add_subplot(2, 2, k, projection="3d")
    norm = colors.Normalize(vmin=ws.min(), vmax=ws.max())
    cmap = plt.get_cmap("turbo")
    for i in range(1, len(x)):
        ax.plot(x[i-1:i+1], y[i-1:i+1], alt[i-1:i+1], color=cmap(norm(ws[i])), lw=1.8, alpha=0.9)
    ax.scatter(x[0], y[0], alt[0], s=50, color="#22805A", marker="o")
    ax.scatter(x[-1], y[-1], alt[-1], s=65, color="#A94032", marker="X")
    ax.set_title(f"{fid[-6:]}  {tag}\n高度范围{alt.max()-alt.min():.0f}m", fontproperties=CN, fontsize=11)
    ax.set_xlabel("X(m)", fontsize=8); ax.set_ylabel("Y(m)", fontsize=8)
    ax.set_zlabel("高度(m)", fontproperties=CN, fontsize=8)
    ax.view_init(elev=22, azim=-60)
    ax.tick_params(labelsize=7)
fig.suptitle("三维适配候选飞行对比（颜色=风速，绿=起点 红=终点）", fontproperties=CN, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("saved:", OUT)
