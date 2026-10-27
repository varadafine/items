"""Impact-energy fatality probability model."""

from __future__ import annotations

import numpy as np


def kinetic_energy_j(mass_kg: float, speed_mps: float) -> float:
    return 0.5 * float(mass_kg) * float(speed_mps) ** 2


def probability_of_fatality(
    impact_energy_j: float,
    alpha_j: float,
    beta_j: float,
    shelter_probability: float,
) -> float:
    """Dalamagkidis impact-energy fatality curve used by SEEDPOD."""

    energy = max(float(impact_energy_j), 1e-12)
    shelter = max(float(shelter_probability), 1e-6)
    probability = 1.0 / (
        1.0
        + np.sqrt(float(alpha_j) / float(beta_j))
        * np.power(float(beta_j) / energy, 1.0 / (4.0 * shelter))
    )
    return float(np.clip(probability, 0.0, 1.0))


class FatalityModel:
    def __init__(self, shelter_probability: float, alpha_j: float, beta_j: float):
        self.shelter_probability = shelter_probability
        self.alpha_j = alpha_j
        self.beta_j = beta_j

    def probability(self, impact_energy_j: float) -> float:
        return probability_of_fatality(
            impact_energy_j,
            self.alpha_j,
            self.beta_j,
            self.shelter_probability,
        )

    def transform(self, strike_risk, impact_energy_j: float):
        return np.asarray(strike_risk, dtype=float) * self.probability(impact_energy_j)

