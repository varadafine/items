"""Rank flights by wind-direction consistency.

Because gust mode re-randomizes direction every 3 s over the full compass,
no flight is truly steady. But some, by chance, have fewer/smaller direction
switches. We score each flight by:
  - R: circular resultant length of wind direction (0=scattered, 1=all one way)
  - n_seg: number of distinct 3s wind segments (fewer = steadier)
  - big_jumps: consecutive-sample direction changes > 90 deg
and cross-reference openness/gust/collisions so the pick is still drawable.
"""
import json, math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = HERE / "UAV_gust_manual_sessions.jsonl"

by_flight = defaultdict(list)
with PATH.open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            r = json.loads(line)
            by_flight[r["flight_id"]].append(r)
for rs in by_flight.values():
    rs.sort(key=lambda r: r["timestamp"])

def dir_to(r):
    df = math.degrees(math.atan2(r["wind_direction_sin"], r["wind_direction_cos"])) % 360
    return (df + 180) % 360

rows = []
for fid, rs in by_flight.items():
    dirs = [dir_to(r) for r in rs]
    # circular resultant length R
    sx = sum(math.cos(math.radians(d)) for d in dirs) / len(dirs)
    sy = sum(math.sin(math.radians(d)) for d in dirs) / len(dirs)
    R = math.hypot(sx, sy)
    mean_dir = math.degrees(math.atan2(sy, sx)) % 360
    # count distinct wind segments (direction changes between consecutive samples)
    n_seg = 1 + sum(1 for i in range(1, len(dirs))
                    if abs((dirs[i]-dirs[i-1]+180) % 360 - 180) > 5)
    big_jumps = sum(1 for i in range(1, len(dirs))
                    if abs((dirs[i]-dirs[i-1]+180) % 360 - 180) > 90)
    # readability + richness
    xs = [r["position"][0] for r in rs]; ys = [r["position"][1] for r in rs]
    plen = sum(math.dist(rs[i]["position"][:2], rs[i-1]["position"][:2]) for i in range(1, len(rs)))
    extent = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
    openness = extent/plen if plen else 0
    gust = sum(1 for r in rs if r["wind_speed"] >= 8)/len(rs)
    coll = sum(1 for r in rs if r.get("obstacle_collision") or r.get("ground_collision"))
    rows.append(dict(fid=fid, n=len(rs), R=R, mean_dir=mean_dir, n_seg=n_seg,
                     big_jumps=big_jumps, openness=openness, gust=gust, coll=coll))

# rank by direction consistency (R), require enough samples to be meaningful
cand = [r for r in rows if r["n"] >= 150]
cand.sort(key=lambda r: r["R"], reverse=True)

print(f"{'flight_id':<32}{'n':>5}{'R':>6}{'meanDir':>8}{'segs':>6}{'jmp>90':>7}{'open':>6}{'gust%':>7}{'coll':>5}")
print("-- most direction-consistent flights --")
for r in cand[:12]:
    print(f"{r['fid']:<32}{r['n']:>5}{r['R']:>6.2f}{r['mean_dir']:>8.0f}"
          f"{r['n_seg']:>6}{r['big_jumps']:>7}{r['openness']:>6.2f}{100*r['gust']:>6.0f}%{r['coll']:>5}")
