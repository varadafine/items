"""Second-order drag ballistic descent model backed by CASEx."""

from __future__ import annotations

from ..config import AircraftConfig
from .descent_model import DescentModel, primitives_to_distribution


def build_casex_aircraft(config: AircraftConfig):
    """Build the CASEx aircraft object used by the source SEEDPOD model."""

    try:
        import casex
    except ImportError as exc:
        raise RuntimeError(
            "CASEx is required for ballistic ground-risk evaluation. "
            "Install the dependencies from intentra_x/requirements.txt."
        ) from exc

    aircraft = casex.AircraftSpecs(
        casex.enums.AircraftType.FIXED_WING,
        config.width_m,
        config.length_m,
        config.mass_kg,
    )
    aircraft.set_ballistic_drag_coefficient(config.ballistic_drag_coefficient)
    aircraft.set_ballistic_frontal_area(config.ballistic_frontal_area_m2)
    aircraft.set_glide_speed_ratio(config.glide_speed_mps, config.glide_ratio)
    return aircraft


class BallisticModel(DescentModel):
    def __init__(self, aircraft_config: AircraftConfig):
        try:
            import casex
        except ImportError as exc:
            raise RuntimeError(
                "CASEx is required for ballistic ground-risk evaluation."
            ) from exc

        self.aircraft = build_casex_aircraft(aircraft_config)
        self.model = casex.BallisticDescent2ndOrderDragApproximation()
        self.model.set_aircraft(self.aircraft)

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
        distance, impact_speed, impact_angle, impact_time = (
            self.model.compute_ballistic_distance(altitude_m, velocity_mps, 0)
        )
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

