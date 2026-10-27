"""Glide descent model adapted from SEEDPOD Ground Risk."""

from __future__ import annotations

import numpy as np

from ..config import AircraftConfig
from .descent_model import DescentModel, primitives_to_distribution


class GlideDescentModel(DescentModel):
    def __init__(self, aircraft_config: AircraftConfig):
        self.aircraft = aircraft_config

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
        altitude = np.asarray(altitude_m, dtype=float)
        distance = self.aircraft.glide_ratio * altitude
        impact_time = np.sqrt(distance**2 + altitude**2) / self.aircraft.glide_speed_mps
        impact_angle = np.full_like(distance, np.arctan(1.0 / self.aircraft.glide_ratio))
        impact_speed = distance / impact_time
        return primitives_to_distribution(
            impact_angle,
            distance,
            heading_deg,
            origin_x_m,
            origin_y_m,
            impact_time,
            impact_speed,
            wind_east_mps,
            wind_north_mps,
        )

