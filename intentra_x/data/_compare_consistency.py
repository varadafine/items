"""Render trajectory + wind arrows for the most direction-consistent candidates
so we can pick the one whose wind looks most 'normal' by eye."""
import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
PATH = HERE / "UAV_gust_manual_sessions.jsonl"
OUT = HERE.parent / "demo_output" / "single_flight_visualization" / "_consistency_compare.png"
FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
CN = FontProperties(fname=str(FONT)) if FONT.exists() else FontProperties(family="SimSun")

CANDS = [
    ("WIND_003_FLIGHT_1789880296443", "最稳(段最少)"),
    ("WIND_003_FLIGHT_1789880524943", "轨迹最清晰"),
    ("WIND_003_FLIGHT_1789881756133", "稳+较清晰"),
    ("WIND_003_FLIGHT_1789880454191", "阵风最足"),
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

fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=160)
for ax, (fid, tag) in zip(axes.flat, CANDS):
    rs = by_flight[fid]
    x = np.array([r["position"][0] for r in rs])
    y = np.array([r["position"][1] for r in rs])
    spd = np.array([r["wind_speed"] for r in rs])
    df = np.degrees(np.arctan2([r["wind_direction_sin"] for r in rs],
                               [r["wind_direction_cos"] for r in rs])) % 360
    dto = np.radians((df + 180) % 360)
    ue = spd * np.sin(dto); vn = spd * np.cos(dto)
    step = max(1, len(rs)//24); sel = np.arange(0, len(rs), step)
    scale = max(np.ptp(x), np.ptp(y), 1.0) * 0.07
    u = ue[sel]/max(spd.max(), 1)*scale; v = vn[sel]/max(spd.max(), 1)*scale
    ax.plot(x, y, color="#56616D", lw=1.3, alpha=0.8)
    norm = colors.Normalize(vmin=float(spd.min()), vmax=float(spd.max()))
    ax.quiver(x[sel], y[sel], u, v, spd[sel], cmap="turbo", norm=norm,
              angles="xy", scale_units="xy", scale=1.0, width=0.006,
              headwidth=3.6, headlength=4.6, alpha=0.92)
    ax.scatter(x[0], y[0], s=55, color="#22805A", marker="o", zorder=4)
    ax.scatter(x[-1], y[-1], s=70, color="#A94032", marker="X", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#D8DEE6", lw=0.6, alpha=0.8)
    ax.set_title(f"{fid[-6:]}  {tag}", fontproperties=CN, fontsize=12)
fig.suptitle("风向较规整的候选飞行（箭头=风吹向，颜色=风速）", fontproperties=CN, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("saved:", OUT)
