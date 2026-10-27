"""Generate a Hohhot time-varying ground-risk GIF with the integrated engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..engine import GroundRiskEngine
from ..visualization import generate_hohhot_risk_gif


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--altitude", type=float, default=120.0)
    parser.add_argument("--speed", type=float, default=20.0)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-direction", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "hohhot_ground_risk.gif",
    )
    args = parser.parse_args()

    engine = GroundRiskEngine()
    output = generate_hohhot_risk_gif(
        engine,
        args.output,
        accident_probability=args.probability,
        altitude_m=args.altitude,
        airspeed_mps=args.speed,
        wind_speed_mps=args.wind_speed,
        wind_direction_deg=args.wind_direction,
    )
    print(output)


if __name__ == "__main__":
    main()

