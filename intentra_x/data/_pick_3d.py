"""Pick a flight that is good for 3D viz: large altitude range + spatial extent,
and rich wind data (many distinct wind segments = more points in dir/speed plot).
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

def wdir(r):
    return math.degrees(math.atan2(r["wind_direction_sin"], r["wind_direction_cos"])) % 360

rows = []
for fid, rs in by_flight.items():
    if len(rs) < 200:  # want enough samples
        continue
    xs = [r["position"][0] for r in rs]
    ys = [r["position"][1] for r in rs]
    alt = [-r["position"][2] for r in rs]
    alt_range = max(alt) - min(alt)
    xy_ext = math.hypot(max(xs)-min(xs), max(ys)-min(ys))
    # distinct wind segments (how many points fig2 will actually show)
    wsegs = 1
    for i in range(1, len(rs)):
        if (abs(rs[i]["wind_speed"]-rs[i-1]["wind_speed"]) > 0.1 or
                abs((wdir(rs[i])-wdir(rs[i-1])+180) % 360 - 180) > 3):
            wsegs += 1
    gust = sum(1 for r in rs if r["wind_speed"] >= 8)/len(rs)
    coll = sum(1 for r in rs if r.get("obstacle_collision") or r.get("ground_collision"))
    # 3D score: altitude range matters most, plus xy extent, plus wind richness
    rows.append(dict(fid=fid, n=len(rs), alt=alt_range, xy=xy_ext,
                     wsegs=wsegs, gust=gust, coll=coll,
                     alt_ratio=alt_range/max(xy_ext, 1)))

# rank by altitude range (the thing missing in current pick), require decent wind variety
cand = [r for r in rows if r["wsegs"] >= 8]
cand.sort(key=lambda r: (r["alt"], r["wsegs"]), reverse=True)

print(f"{'flight_id':<32}{'n':>5}{'altRange':>9}{'xyExt':>7}{'alt/xy':>7}{'wSegs':>6}{'gust%':>7}{'coll':>5}")
print("-- best altitude variation + rich wind --")
for r in cand[:12]:
    print(f"{r['fid']:<32}{r['n']:>5}{r['alt']:>9.0f}{r['xy']:>7.0f}"
          f"{r['alt_ratio']:>7.2f}{r['wsegs']:>6}{100*r['gust']:>6.0f}%{r['coll']:>5}")
