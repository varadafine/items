"""Render trajectory thumbnails for several candidate flights so we can pick
the most readable one by eye. Also reports an 'openness' metric
(spatial extent / path length): high = clean A->B traverse, low = loopy.
"""
import json, math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
PATH = HERE / "UAV_gust_manual_sessions.jsonl"
OUT = HERE.parent / "demo_output" / "single_flight_visualization" / "_candidate_compare.png"
FONT = Path(r"C:\Windows\Fonts\simsun.ttc")
CN = FontProperties(fname=str(FONT)) if FONT.exists() else FontProperties(family="SimSun")

by_flight = defaultdict(list)
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            by_flight[r["flight_id"]].append(r)
for rs in by_flight.values():
    rs.sort(key=lambda r: r["timestamp"])

rows = []
for fid, rs in by_flight.items():
    xs = [r["position"][0] for r in rs]
    ys = [r["position"][1] for r in rs]
    plen = sum(math.dist(rs[i]["position"][:2], rs[i-1]["position"][:2]) for i in range(1, len(rs)))
    extent = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
    start_end = math.dist([xs[0], ys[0]], [xs[-1], ys[-1]])
    gust = sum(1 for r in rs if r["wind_speed"] >= 8)/len(rs)
    coll = sum(1 for r in rs if r.get("obstacle_collision") or r.get("ground_collision"))
    rows.append({"fid": fid, "n": len(rs), "plen": plen, "extent": extent,
                 "openness": extent/plen if plen else 0,
                 "start_end": start_end, "gust": gust, "coll": coll})

# candidates: current loopy one + most-open + a couple mid, all low-collision
rows.sort(key=lambda r: r["openness"], reverse=True)
open_cands = [r for r in rows if r["n"] >= 250 and r["coll"] <= 20][:3]
current = next(r for r in rows if r["fid"].endswith("704464"))
picks = []
seen = set()
for r in [current] + open_cands:
    if r["fid"] not in seen:
        picks.append(r); seen.add(r["fid"])
picks = picks[:4]

print(f"{'flight_id':<32}{'n':>5}{'path':>7}{'ext':>6}{'open':>6}{'s-e':>7}{'gust%':>7}{'coll':>5}")
for r in picks:
    tag = "  <- current" if r["fid"].endswith("704464") else ""
    print(f"{r['fid']:<32}{r['n']:>5}{r['plen']:>7.0f}{r['extent']:>6.0f}"
          f"{r['openness']:>6.2f}{r['start_end']:>7.0f}{100*r['gust']:>6.0f}%{r['coll']:>5}{tag}")

fig, axes = plt.subplots(2, 2, figsize=(11, 10), dpi=160)
for ax, r in zip(axes.flat, picks):
    rs = by_flight[r["fid"]]
    xs = [p["position"][0] for p in rs]
    ys = [p["position"][1] for p in rs]
    ax.plot(xs, ys, color="#56616D", lw=1.4, alpha=0.85)
    ax.scatter(xs[0], ys[0], s=55, color="#22805A", marker="o", zorder=4)
    ax.scatter(xs[-1], ys[-1], s=70, color="#A94032", marker="X", zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="#D8DEE6", lw=0.6, alpha=0.8)
    cur = "（当前）" if r["fid"].endswith("704464") else ""
    ax.set_title(f"{r['fid'][-6:]}{cur}\n展开度={r['openness']:.2f}  阵风={100*r['gust']:.0f}%  碰撞={r['coll']}",
                 fontproperties=CN, fontsize=11)
fig.suptitle("候选飞行轨迹对比（绿=起点  红=终点）", fontproperties=CN, fontsize=15)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
print("\nsaved:", OUT)
