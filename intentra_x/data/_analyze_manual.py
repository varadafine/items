"""Quick analysis of UAV_gust_manual_sessions.jsonl"""
import json, math
from collections import defaultdict, Counter

PATH = "UAV_gust_manual_sessions.jsonl"

records = []
bad = 0
with open(PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1

n = len(records)
print(f"records={n}  bad_lines={bad}")

# per-flight breakdown
by_flight = defaultdict(list)
for r in records:
    by_flight[r.get("flight_id", "?")].append(r)

print(f"\n=== flights: {len(by_flight)} ===")
for fid, rs in by_flight.items():
    ts = [r["timestamp"] for r in rs if r.get("timestamp") is not None]
    dur = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
    hz = (len(ts) - 1) / dur if dur > 0 else float("nan")
    print(f"  {fid}: {len(rs)} rec, dur={dur:.1f}s, ~{hz:.1f}Hz")

# sampling gaps (per flight, detect pauses)
print("\n=== sampling gaps > 0.5s (pauses/resets) ===")
for fid, rs in by_flight.items():
    ts = sorted(r["timestamp"] for r in rs if r.get("timestamp") is not None)
    gaps = [(ts[i]-ts[i-1]) for i in range(1, len(ts))]
    big = [g for g in gaps if g > 0.5]
    if big:
        print(f"  {fid}: {len(big)} gaps, max={max(big):.1f}s")

def stats(vals):
    vals = [v for v in vals if v is not None and isinstance(v, (int, float)) and math.isfinite(v)]
    if not vals:
        return None
    vals.sort()
    m = sum(vals)/len(vals)
    return {"n": len(vals), "min": min(vals), "max": max(vals), "mean": m,
            "p50": vals[len(vals)//2]}

# wind
print("\n=== wind_speed ===")
ws = stats([r.get("wind_speed") for r in records])
print(f"  {ws}")

# gust classification: base=3, gust 8-12
gust_cnt = sum(1 for r in records if (r.get("wind_speed") or 0) >= 8)
print(f"  gust samples (>=8 m/s): {gust_cnt} ({100*gust_cnt/n:.1f}%)")

# position / altitude (z is NED-ish; report range)
print("\n=== position ===")
for i, ax in enumerate(["x", "y", "z"]):
    print(f"  pos_{ax}: {stats([r['position'][i] for r in records if r.get('position')])}")

# speed magnitude
speeds = []
for r in records:
    v = r.get("velocity")
    if v:
        speeds.append(math.sqrt(sum(c*c for c in v)))
print("\n=== speed (|velocity|) ===")
print(f"  {stats(speeds)}")

# attitude ranges
print("\n=== attitude ===")
for k in ["roll", "pitch"]:
    print(f"  {k}: {stats([r['attitude'].get(k) for r in records if r.get('attitude')])}")

# obstacle distances
print("\n=== obstacle ===")
print(f"  sensor_obstacle_distance: {stats([r.get('sensor_obstacle_distance') for r in records])}")
print(f"  nearest_obstacle_distance: {stats([r.get('nearest_obstacle_distance') for r in records])}")
print("  obstacle source:", dict(Counter(r.get("nearest_obstacle_source") for r in records)))

# collisions
gc = sum(1 for r in records if r.get("ground_collision"))
oc = sum(1 for r in records if r.get("obstacle_collision"))
print("\n=== collisions ===")
print(f"  ground_collision: {gc}  obstacle_collision: {oc}")
objs = Counter(r.get("collision_object_name") for r in records if r.get("collision_object_name"))
if objs:
    print(f"  collision objects: {dict(objs)}")

# null-field coverage
print("\n=== null coverage ===")
for k in ["planned_route_error", "signal_strength", "sensor_obstacle_distance", "nearest_obstacle_distance"]:
    nulls = sum(1 for r in records if r.get(k) is None)
    print(f"  {k}: {nulls}/{n} null ({100*nulls/n:.0f}%)")
