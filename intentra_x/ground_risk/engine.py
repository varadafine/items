"""Unified ground-impact and fatality-risk calculation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import fftconvolve

from .config import AircraftConfig, GroundRiskConfig
from .descent_models.ballistic_model import BallisticModel
from .descent_models.glide_model import GlideDescentModel
from .harm_models.fatality_model import FatalityModel, kinetic_energy_j
from .harm_models.strike_model import StrikeModel


@dataclass
class GroundRiskResult:
    impact_probability: np.ndarray
    strike_risk: np.ndarray
    fatality_risk: np.ndarray
    ballistic_strike_risk: np.ndarray
    glide_strike_risk: np.ndarray
    ballistic_fatality_risk: np.ndarray
    glide_fatality_risk: np.ndarray
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        max_fatality = float(np.max(self.fatality_risk))
        if max_fatality < 1e-7:
            level = "LOW"
        elif max_fatality < 1e-6:
            level = "MEDIUM"
        else:
            level = "HIGH"
        return {
            "mean_strike_risk": float(np.mean(self.strike_risk)),
            "max_strike_risk": float(np.max(self.strike_risk)),
            "mean_fatality_risk": float(np.mean(self.fatality_risk)),
            "max_fatality_risk": max_fatality,
            "risk_level": level,
            **self.metadata,
        }


class GroundRiskEngine:
    """Evaluate conditional ground consequences on a population raster."""

    def __init__(
        self,
        aircraft: Optional[AircraftConfig] = None,
        config: Optional[GroundRiskConfig] = None,
    ):
        self.aircraft = aircraft or AircraftConfig()
        self.config = config or GroundRiskConfig()
        self.config.validate()
        self.ballistic_model = BallisticModel(self.aircraft)
        self.glide_model = GlideDescentModel(self.aircraft)
        self.fatality_model = FatalityModel(
            self.config.shelter_probability,
            self.config.fatality_alpha_j,
            self.config.fatality_beta_j,
        )

    @staticmethod
    def _wind_components(wind_speed_mps, wind_from_direction_deg):
        # Meteorological wind direction is where wind comes from; drift is opposite.
        wind_to_bearing = (np.asarray(wind_from_direction_deg, dtype=float) + 180.0) % 360.0
        angle = np.deg2rad((90.0 - wind_to_bearing) % 360.0)
        return wind_speed_mps * np.cos(angle), wind_speed_mps * np.sin(angle)

    def _sample_state(
        self,
        altitude_m: float,
        airspeed_mps: float,
        wind_speed_mps: float,
        wind_direction_deg: float,
        heading_deg: Optional[float],
    ):
        rng = np.random.default_rng(self.config.random_seed)
        n = self.config.monte_carlo_samples
        altitude = np.clip(rng.normal(altitude_m, 5.0, n), 1.0, None)
        speed = np.clip(rng.normal(airspeed_mps, 2.5, n), 0.1, None)
        wind_speed = np.clip(rng.normal(wind_speed_mps, 1.0, n), 0.0, None)
        wind_direction = rng.normal(wind_direction_deg, 5.0, n)
        if heading_deg is None:
            heading = rng.uniform(0.0, 360.0, n)
        else:
            heading = rng.normal(heading_deg, 5.0, n)
        wind_east, wind_north = self._wind_components(wind_speed, wind_direction)
        return altitude, speed, heading, wind_north, wind_east

    def _distribution_grid(self, shape, distribution):
        mean_m, covariance_m2 = distribution
        rows, cols = shape
        mean_cells = np.asarray(mean_m, dtype=float) / self.config.resolution_m
        covariance_cells = np.asarray(covariance_m2, dtype=float) / self.config.resolution_m**2
        covariance_cells += np.eye(2) * 1e-6
        center = np.array([(cols - 1) / 2.0, (rows - 1) / 2.0])

        yy, xx = np.mgrid[0:rows, 0:cols]
        mean = center + mean_cells
        # Explicit 2-D Gaussian evaluation avoids starting a second BLAS/OpenMP
        # runtime after PyTorch inference on Windows.
        a = float(covariance_cells[0, 0])
        b = float(covariance_cells[0, 1])
        c = float(covariance_cells[1, 0])
        d = float(covariance_cells[1, 1])
        determinant = max(a * d - b * c, 1e-12)
        dx = xx - mean[0]
        dy = yy - mean[1]
        mahalanobis = (
            d * dx**2 - (b + c) * dx * dy + a * dy**2
        ) / determinant
        pdf = np.exp(-0.5 * mahalanobis) / (2.0 * np.pi * np.sqrt(determinant))
        total = float(pdf.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("impact distribution could not be represented on the population grid")
        return pdf / total

    @staticmethod
    def _expected_exposure(exposure_grid, impact_probability):
        # Correlation: for every possible aircraft position, integrate exposure at displaced impacts.
        return fftconvolve(exposure_grid, impact_probability[::-1, ::-1], mode="same")

    def evaluate(
        self,
        population_grid,
        accident_probability: float,
        altitude_m: float,
        airspeed_mps: float,
        wind_speed_mps: float,
        wind_direction_deg: float,
        heading_deg: Optional[float] = None,
    ) -> GroundRiskResult:
        population = np.asarray(population_grid, dtype=float)
        if population.ndim != 2 or population.size == 0:
            raise ValueError("population_grid must be a non-empty two-dimensional array")
        event_probability = float(np.clip(accident_probability, 0.0, 1.0))

        state = self._sample_state(
            altitude_m,
            airspeed_mps,
            wind_speed_mps,
            wind_direction_deg,
            heading_deg,
        )
        ballistic_dist, ballistic_speed, ballistic_angle = self.ballistic_model.transform(*state)
        glide_dist, glide_speed, glide_angle = self.glide_model.transform(*state)

        ballistic_pdf = self._distribution_grid(population.shape, ballistic_dist)
        glide_pdf = self._distribution_grid(population.shape, glide_dist)
        ballistic_strike_model = StrikeModel(population, self.aircraft.width_m, ballistic_angle)
        glide_strike_model = StrikeModel(population, self.aircraft.width_m, glide_angle)

        ballistic_strike = (
            event_probability
            * self.config.ballistic_mode_weight
            * self._expected_exposure(ballistic_strike_model.exposure_grid, ballistic_pdf)
        )
        glide_strike = (
            event_probability
            * self.config.glide_mode_weight
            * self._expected_exposure(glide_strike_model.exposure_grid, glide_pdf)
        )

        ballistic_energy = kinetic_energy_j(self.aircraft.mass_kg, ballistic_speed)
        glide_energy = kinetic_energy_j(self.aircraft.mass_kg, glide_speed)
        ballistic_fatality = self.fatality_model.transform(ballistic_strike, ballistic_energy)
        glide_fatality = self.fatality_model.transform(glide_strike, glide_energy)

        impact_probability = (
            self.config.ballistic_mode_weight * ballistic_pdf
            + self.config.glide_mode_weight * glide_pdf
        )
        strike_risk = np.clip(ballistic_strike + glide_strike, 0.0, None)
        fatality_risk = np.clip(ballistic_fatality + glide_fatality, 0.0, None)
        metadata = {
            "accident_probability": event_probability,
            "altitude_m": float(altitude_m),
            "airspeed_mps": float(airspeed_mps),
            "wind_speed_mps": float(wind_speed_mps),
            "wind_direction_deg": float(wind_direction_deg),
            "heading_mode": "uniform" if heading_deg is None else "provided",
            "ballistic_impact_speed_mps": float(ballistic_speed),
            "glide_impact_speed_mps": float(glide_speed),
            "ballistic_impact_energy_j": float(ballistic_energy),
            "glide_impact_energy_j": float(glide_energy),
        }
        return GroundRiskResult(
            impact_probability=impact_probability,
            strike_risk=strike_risk,
            fatality_risk=fatality_risk,
            ballistic_strike_risk=ballistic_strike,
            glide_strike_risk=glide_strike,
            ballistic_fatality_risk=ballistic_fatality,
            glide_fatality_risk=glide_fatality,
            metadata=metadata,
        )
