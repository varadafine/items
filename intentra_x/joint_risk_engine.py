"""One-way integration of the existing air model and migrated ground model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from .air_risk_predictor import AirRiskPredictor, AirRiskResult
    from .ground_risk import GroundRiskEngine, GroundRiskResult
    from .ground_risk.population import load_hohhot_population
except ImportError:  # Allow direct execution from the intentra_x directory.
    from air_risk_predictor import AirRiskPredictor, AirRiskResult
    from ground_risk import GroundRiskEngine, GroundRiskResult
    from ground_risk.population import load_hohhot_population


@dataclass
class JointRiskResult:
    air: AirRiskResult
    ground: GroundRiskResult

    def summary(self):
        return {
            "air_risk": self.air.to_dict(),
            "ground_risk": self.ground.summary(),
            "joint_assessment": {
                "air_probability_used_once": self.air.risk_probability,
                "risk_level": self.ground.summary()["risk_level"],
                "requires_intervention": bool(
                    self.air.risk_detected
                    or self.ground.summary()["risk_level"] in {"MEDIUM", "HIGH"}
                ),
            },
        }


class JointRiskEngine:
    def __init__(
        self,
        air_predictor: Optional[AirRiskPredictor] = None,
        ground_engine: Optional[GroundRiskEngine] = None,
    ):
        self.air_predictor = air_predictor or AirRiskPredictor()
        self.ground_engine = ground_engine or GroundRiskEngine()

    def evaluate(
        self,
        air_sequence,
        *,
        hour: float,
        altitude_m: float,
        airspeed_mps: float,
        wind_speed_mps: float,
        wind_direction_deg: float,
        heading_deg=None,
    ) -> JointRiskResult:
        air_result = self.air_predictor.predict(air_sequence)
        population = load_hohhot_population(
            hour=hour,
            resolution_m=self.ground_engine.config.resolution_m,
        )
        ground_result = self.ground_engine.evaluate(
            population_grid=population,
            accident_probability=air_result.risk_probability,
            altitude_m=altitude_m,
            airspeed_mps=airspeed_mps,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
            heading_deg=heading_deg,
        )
        return JointRiskResult(air=air_result, ground=ground_result)

