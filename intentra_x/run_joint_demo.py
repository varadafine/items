"""Run the first integrated air-to-ground Hohhot risk demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .air_risk_predictor import SEQUENCE_LENGTH, row_to_features
    from .ground_risk.population import load_hohhot_population
    from .ground_risk.visualization import (
        generate_hohhot_risk_gif,
        save_ground_risk_figure,
    )
    from .joint_risk_engine import JointRiskEngine
except ImportError:
    from air_risk_predictor import SEQUENCE_LENGTH, row_to_features
    from ground_risk.population import load_hohhot_population
    from ground_risk.visualization import generate_hohhot_risk_gif, save_ground_risk_figure
    from joint_risk_engine import JointRiskEngine


HERE = Path(__file__).resolve().parent


def load_collision_demo_window(data_path=None):
    """Select a deterministic 20-step window ending at the first logged collision."""

    data_path = Path(data_path or (HERE / "data" / "UAV_gust_all.jsonl"))
    flights = {}
    with data_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            flights.setdefault(row["flight_id"], []).append(row)
    for rows in flights.values():
        rows.sort(key=lambda item: item["timestamp"])

    for flight_id in sorted(flights):
        rows = flights[flight_id]
        for index, row in enumerate(rows):
            if index >= SEQUENCE_LENGTH - 1 and (
                row.get("ground_collision") or row.get("obstacle_collision")
            ):
                window = rows[index - SEQUENCE_LENGTH + 1 : index + 1]
                sequence = np.stack([row_to_features(item) for item in window])
                return flight_id, window, sequence
    raise RuntimeError("no collision window was found in the AirSim demonstration data")


def run_demo(output_dir=None, make_gif=False):
    output_dir = Path(output_dir or (HERE / "demo_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    flight_id, rows, sequence = load_collision_demo_window()
    last = rows[-1]
    velocity = np.asarray(last.get("velocity") or [0.0, 0.0, 0.0], dtype=float)
    position = last.get("position") or [0.0, 0.0, -120.0]
    wind_sin = float(last.get("wind_direction_sin", 0.0) or 0.0)
    wind_cos = float(last.get("wind_direction_cos", 1.0) or 1.0)
    wind_direction = float(np.rad2deg(np.arctan2(wind_sin, wind_cos)) % 360.0)
    hour = 9

    engine = JointRiskEngine()
    result = engine.evaluate(
        sequence,
        hour=hour,
        altitude_m=max(1.0, -float(position[2])),
        airspeed_mps=max(0.1, float(np.linalg.norm(velocity))),
        wind_speed_mps=float(last.get("wind_speed", 0.0) or 0.0),
        wind_direction_deg=wind_direction,
    )

    arrays_path = output_dir / "joint_risk_arrays.npz"
    np.savez_compressed(
        arrays_path,
        impact_probability=result.ground.impact_probability,
        strike_risk=result.ground.strike_risk,
        fatality_risk=result.ground.fatality_risk,
    )
    population = load_hohhot_population(
        hour=hour,
        resolution_m=engine.ground_engine.config.resolution_m,
    )
    figure_path = save_ground_risk_figure(
        output_dir / "joint_ground_risk.png",
        population,
        result.ground,
        hour=hour,
    )

    payload = result.summary()
    payload["demo"] = {
        "flight_id": flight_id,
        "sequence_length": SEQUENCE_LENGTH,
        "figure": str(figure_path),
        "arrays": str(arrays_path),
    }
    if make_gif:
        gif_path = generate_hohhot_risk_gif(
            engine.ground_engine,
            output_dir / "joint_ground_risk_24h.gif",
            accident_probability=result.air.risk_probability,
            altitude_m=result.ground.metadata["altitude_m"],
            airspeed_mps=result.ground.metadata["airspeed_mps"],
            wind_speed_mps=result.ground.metadata["wind_speed_mps"],
            wind_direction_deg=result.ground.metadata["wind_direction_deg"],
        )
        payload["demo"]["gif"] = str(gif_path)

    json_path = output_dir / "joint_risk_result.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gif", action="store_true", help="also regenerate the 24-hour GIF")
    args = parser.parse_args()
    json_path, payload = run_demo(args.output_dir, make_gif=args.gif)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()

