"""Population exposure and person-strike model."""

from __future__ import annotations

import numpy as np


def get_lethal_area(impact_angle_rad: float, uas_width_m: float) -> float:
    """Effective strike area from Smith (2000), as used by SEEDPOD."""

    person_radius_m = 1.0
    person_height_m = 1.8
    uas_radius_m = uas_width_m / 2.0
    safe_angle = max(abs(float(impact_angle_rad)), 1e-3)
    return (
        (2.0 * (person_radius_m + uas_radius_m) * person_height_m)
        / np.tan(safe_angle)
        + np.pi * (uas_radius_m + person_radius_m) ** 2
    )


class StrikeModel:
    """Convert population density into expected exposed persons per impact."""

    def __init__(self, population_people_km2, uas_width_m: float, impact_angle_rad: float):
        density_people_m2 = np.asarray(population_people_km2, dtype=float) * 1e-6
        density_people_m2 = np.nan_to_num(density_people_m2, nan=0.0, posinf=0.0, neginf=0.0)
        density_people_m2 = np.clip(density_people_m2, 0.0, None)
        self.lethal_area_m2 = get_lethal_area(impact_angle_rad, uas_width_m)
        self.exposure_grid = density_people_m2 * self.lethal_area_m2

