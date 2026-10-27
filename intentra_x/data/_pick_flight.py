"""Score each flight and pick the most representative one for visualization."""
import json, math
from collections import defaultdict

PATH = "UAV_gust_manual_sessions.jsonl"
by_flight = defaultdict(list)
with open(PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by_flight[r["flight_id"]].append(r)

rows = []
for fid, rs in by_flight.items():
    ts = [r["timestamp"] for r in rs]
    dur = max(ts) - min(ts)
    xs = [r["position"][0] for r in rs]
    ys = [r["position"][1] for r in rs]
    zs = [r["position"][2] for r in rs]
    ws = [r["wind_speed"] for r in rs]
    # path length
    plen = sum(math.dist(rs[i]["position"], rs[i-1]["position"]) for i in range(1, len(rs)))
    extent = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
    alt_range = max(zs) - min(zs)
    wind_std = (sum((w-sum(ws)/len(ws))**2 for w in ws)/len(ws))**0.5
    gust_frac = sum(1 for w in ws if w >= 8)/len(rs)
    coll = sum(1 for r in rs if r.get("obstacle_collision") or r.get("ground_collision"))
    rows.append({
        "fid": fid, "n": len(rs), "dur": dur, "path_len": plen,
        "xy_extent": extent, "alt_range": alt_range,
        "wind_std": wind_std, "gust_frac": gust_frac, "collisions": coll,
    })

# normalize a few metrics for a composite "richness" score
def norm(vals):
    lo, hi = min(vals), max(vals)
    return [(v-lo)/(hi-lo) if hi > lo else 0 for v in vals]

dur_n = norm([r["dur"] for r in rows])
ext_n = norm([r["xy_extent"] for r in rows])
alt_n = norm([r["alt_range"] for r in rows])
gust_n = norm([r["gust_frac"] for r in rows])
for r, a, b, c, d in zip(rows, dur_n, ext_n, alt_n, gust_n):
    r["score"] = 0.35*a + 0.30*b + 0.15*c + 0.20*d

rows.sort(key=lambda r: r["score"], reverse=True)
print(f"{'flight_id':<32}{'n':>5}{'dur':>7}{'path_m':>8}{'xy_ext':>8}{'alt':>7}{'gust%':>7}{'coll':>6}{'score':>7}")
for r in rows[:10]:
    print(f"{r['fid']:<32}{r['n']:>5}{r['dur']:>7.1f}{r['path_len']:>8.0f}"
          f"{r['xy_extent']:>8.0f}{r['alt_range']:>7.0f}{100*r['gust_frac']:>6.0f}%"
          f"{r['collisions']:>6}{r['score']:>7.3f}")
