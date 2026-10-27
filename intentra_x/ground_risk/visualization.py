"""Static-map and GIF rendering for the integrated Hohhot demonstration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import GroundRiskEngine, GroundRiskResult
from .population import HOHHOT_BOUNDS, load_hohhot_population, outdoor_population_fraction


def _safe_log10(values):
    return np.log10(np.clip(np.asarray(values, dtype=float), 1e-15, None))


def make_ground_risk_figure(
    population_grid,
    result: GroundRiskResult,
    hour: float = 9,
    bounds=HOHHOT_BOUNDS,
    title: str = "Hohhot UAV Ground Risk Assessment",
):
    """Create the population/strike/fatality three-panel demonstration figure."""

    min_lon, min_lat, max_lon, max_lat = bounds
    extent = [min_lon, max_lon, min_lat, max_lat]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))

    population_image = axes[0].imshow(
        population_grid,
        cmap="YlOrRd",
        extent=extent,
        origin="upper",
        aspect="auto",
    )
    axes[0].set_title(
        f"Population Density\n(hour={hour:g}, outdoor={outdoor_population_fraction(hour):.0%})"
    )
    figure.colorbar(population_image, ax=axes[0], label="people/km²", shrink=0.8)

    strike_image = axes[1].imshow(
        _safe_log10(result.strike_risk),
        cmap="hot",
        extent=extent,
        origin="upper",
        aspect="auto",
    )
    axes[1].set_title("Person Strike Risk\n(ballistic + glide)")
    figure.colorbar(strike_image, ax=axes[1], label="log10(risk)", shrink=0.8)

    fatality_image = axes[2].imshow(
        _safe_log10(result.fatality_risk),
        cmap="jet",
        extent=extent,
        origin="upper",
        aspect="auto",
    )
    axes[2].set_title("Fatality Risk\n(impact-energy model)")
    figure.colorbar(fatality_image, ax=axes[2], label="log10(risk)", shrink=0.8)

    for axis in axes:
        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Latitude (°N)")

    figure.suptitle(
        f"{title}\nAir accident probability={result.metadata['accident_probability']:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    return figure


def save_ground_risk_figure(
    path,
    population_grid,
    result: GroundRiskResult,
    hour: float = 9,
    bounds=HOHHOT_BOUNDS,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = make_ground_risk_figure(population_grid, result, hour=hour, bounds=bounds)
    figure.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def generate_hohhot_risk_gif(
    engine: GroundRiskEngine,
    output_path,
    accident_probability: float,
    altitude_m: float = 120.0,
    airspeed_mps: float = 20.0,
    wind_speed_mps: float = 5.0,
    wind_direction_deg: float = 180.0,
    hours: Sequence[int] = (0, 3, 6, 9, 12, 15, 18, 21),
    frame_duration_seconds: float = 1.0,
) -> Path:
    """Recompute and render a time-varying Hohhot ground-risk GIF."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = output_path.parent / f"{output_path.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    for hour in hours:
        population = load_hohhot_population(
            hour=hour,
            resolution_m=engine.config.resolution_m,
        )
        result = engine.evaluate(
            population_grid=population,
            accident_probability=accident_probability,
            altitude_m=altitude_m,
            airspeed_mps=airspeed_mps,
            wind_speed_mps=wind_speed_mps,
            wind_direction_deg=wind_direction_deg,
        )
        frame_path = frame_dir / f"frame_{int(hour):02d}.png"
        save_ground_risk_figure(frame_path, population, result, hour=hour)
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(
        output_path,
        frames,
        duration=float(frame_duration_seconds),
        loop=0,
    )
    return output_path

