"""Configuration objects shared by the ground-risk modules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AircraftConfig:
    """Fixed aircraft parameters used by the first integrated prototype."""

    width_m: float = 3.0
    length_m: float = 2.0
    mass_kg: float = 4.0
    ballistic_drag_coefficient: float = 0.8
    ballistic_frontal_area_m2: float = 0.1
    glide_speed_mps: float = 12.0
    glide_ratio: float = 15.0


@dataclass(frozen=True)
class GroundRiskConfig:
    """Numerical defaults for a ground-risk evaluation."""

    resolution_m: float = 50.0
    monte_carlo_samples: int = 5000
    ballistic_mode_weight: float = 0.5
    glide_mode_weight: float = 0.5
    shelter_probability: float = 0.3
    fatality_alpha_j: float = 1e6
    fatality_beta_j: float = 34.0
    random_seed: int = 42

    def validate(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.monte_carlo_samples < 10:
            raise ValueError("monte_carlo_samples must be at least 10")
        if self.ballistic_mode_weight < 0 or self.glide_mode_weight < 0:
            raise ValueError("descent-mode weights cannot be negative")
        total = self.ballistic_mode_weight + self.glide_mode_weight
        if abs(total - 1.0) > 1e-9:
            raise ValueError("ballistic_mode_weight + glide_mode_weight must equal 1")

