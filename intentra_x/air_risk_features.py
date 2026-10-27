"""Shared AirSim feature definitions for controlled 9-vs-12 input experiments."""

from __future__ import annotations

import math

import numpy as np


LEGACY_FEATURE_NAMES = (
    "wind_speed", "drone_speed", "roll", "pitch", "altitude",
    "sensor_obstacle_distance", "obstacle_detected",
    "wind_direction_sin", "wind_direction_cos",
)

FEATURE_NAMES = (
    "altitude", "velocity_x", "velocity_y", "velocity_z", "drone_speed",
    "roll", "pitch", "wind_speed", "wind_direction_sin",
    "wind_direction_cos", "sensor_obstacle_distance", "obstacle_detected",
)


def row_to_features(row: dict, feature_names=FEATURE_NAMES) -> np.ndarray:
    """Convert a telemetry row to a fixed-order 12- or legacy 9-feature vector.

    Collision flags, absolute x/y position, and postprocessed nearest-obstacle
    distances are deliberately excluded: they are not pre-event model inputs.
    """
    position = row.get("position") or (0.0, 0.0, 0.0)
    velocity = row.get("velocity") or (0.0, 0.0, 0.0)
    attitude = row.get("attitude") or {}
    vx, vy, vz = (float(v) for v in velocity)
    raw_distance = row.get("sensor_obstacle_distance")
    distance = 100.0 if raw_distance is None else float(raw_distance)
    values = {
        "altitude": max(0.0, -float(position[2])),
        "velocity_x": vx,
        "velocity_y": vy,
        "velocity_z": vz,
        "drone_speed": math.sqrt(vx * vx + vy * vy + vz * vz),
        "roll": float(attitude.get("roll") or 0.0),
        "pitch": float(attitude.get("pitch") or 0.0),
        "wind_speed": float(row.get("wind_speed") or 0.0),
        "wind_direction_sin": float(row.get("wind_direction_sin") or 0.0),
        "wind_direction_cos": (
            1.0 if row.get("wind_direction_cos") is None
            else float(row["wind_direction_cos"])
        ),
        "sensor_obstacle_distance": distance,
        "obstacle_detected": float(distance < 100.0),
    }
    return np.asarray([values[name] for name in feature_names], dtype=np.float32)
