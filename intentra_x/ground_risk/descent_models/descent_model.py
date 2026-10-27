"""Shared transformations for ground-impact distributions.

Adapted from the MIT-licensed SEEDPOD Ground Risk Model.  The original
implementation fits a one-component Gaussian mixture; a one-component model
is equivalently represented here by the sample mean and covariance.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


Distribution2D = Tuple[np.ndarray, np.ndarray]


def bearing_to_math_angle(heading_deg: np.ndarray) -> np.ndarray:
    """Convert a clockwise-from-north bearing in degrees to an x/y angle."""

    return np.deg2rad((90.0 - np.asarray(heading_deg, dtype=float)) % 360.0)


def primitives_to_distribution(
    impact_angle_rad,
    horizontal_distance_m,
    heading_deg,
    origin_x_m,
    origin_y_m,
    impact_time_s,
    impact_speed_mps,
    wind_east_mps,
    wind_north_mps,
) -> Tuple[Distribution2D, float, float]:
    """Turn sampled descent primitives into a 2-D Gaussian impact model."""

    distance = np.atleast_1d(np.asarray(horizontal_distance_m, dtype=float))
    heading = np.broadcast_to(np.asarray(heading_deg, dtype=float), distance.shape)
    impact_time = np.broadcast_to(np.asarray(impact_time_s, dtype=float), distance.shape)
    wind_east = np.broadcast_to(np.asarray(wind_east_mps, dtype=float), distance.shape)
    wind_north = np.broadcast_to(np.asarray(wind_north_mps, dtype=float), distance.shape)

    theta = bearing_to_math_angle(heading)
    east = distance * np.cos(theta) + wind_east * impact_time + float(origin_x_m)
    north = distance * np.sin(theta) + wind_north * impact_time + float(origin_y_m)
    points = np.column_stack((east, north))
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 2:
        raise ValueError("descent model produced fewer than two finite impact samples")

    mean = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    covariance = np.atleast_2d(covariance).astype(float)
    covariance += np.eye(2) * 1e-6

    impact_speed = float(np.nanmean(np.asarray(impact_speed_mps, dtype=float)))
    impact_angle = float(np.nanmean(np.asarray(impact_angle_rad, dtype=float)))
    return (mean, covariance), impact_speed, impact_angle


class DescentModel:
    """Base class for models that map a flight state to an impact distribution."""

    def transform(
        self,
        altitude_m,
        velocity_mps,
        heading_deg,
        wind_north_mps,
        wind_east_mps,
        origin_x_m=0.0,
        origin_y_m=0.0,
    ):
        raise NotImplementedError

