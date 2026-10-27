"""Offline Hohhot population-grid loading for ground-risk demonstrations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import zoom


DATA_DIR = Path(__file__).resolve().parent / "data" / "hohhot"
HOHHOT_BOUNDS = (111.62, 40.78, 111.78, 40.84)  # min lon, min lat, max lon, max lat

# First integrated demo follows the time factors used by the existing Hohhot scripts.
_OUTDOOR_FRACTIONS = {
    0: 0.05,
    3: 0.08,
    6: 0.15,
    9: 0.35,
    12: 0.45,
    15: 0.40,
    18: 0.35,
    21: 0.15,
    24: 0.05,
}


def outdoor_population_fraction(hour: float) -> float:
    """Interpolate the demonstration outdoor-population fraction by hour."""

    hour = float(hour) % 24.0
    keys = np.asarray(sorted(_OUTDOOR_FRACTIONS), dtype=float)
    values = np.asarray([_OUTDOOR_FRACTIONS[int(key)] for key in keys], dtype=float)
    return float(np.interp(hour, keys, values))


def target_grid_shape(resolution_m: float):
    min_lon, min_lat, max_lon, max_lat = HOHHOT_BOUNDS
    mean_lat = (min_lat + max_lat) / 2.0
    width_m = (max_lon - min_lon) * 111_000.0 * np.cos(np.deg2rad(mean_lat))
    height_m = (max_lat - min_lat) * 111_000.0
    rows = max(2, int(height_m / resolution_m))
    cols = max(2, int(width_m / resolution_m))
    return rows, cols


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError:
        import tifffile

        return np.asarray(tifffile.imread(path), dtype=np.float64)

    with rasterio.open(path) as dataset:
        return np.asarray(dataset.read(1), dtype=np.float64)


def load_hohhot_population(
    hour: float = 9,
    resolution_m: float = 50.0,
    apply_outdoor_fraction: bool = True,
) -> np.ndarray:
    """Load and resample the offline Hohhot WorldPop demonstration raster."""

    source_path = DATA_DIR / "hohhot_pop.tif"
    if not source_path.exists():
        raise FileNotFoundError(f"Hohhot population raster not found: {source_path}")

    raw = np.squeeze(_read_tiff(source_path))
    if raw.ndim != 2:
        raise ValueError(f"expected a 2-D population raster, got shape {raw.shape}")
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0.0, None)

    rows, cols = target_grid_shape(resolution_m)
    population = zoom(raw, (rows / raw.shape[0], cols / raw.shape[1]), order=1)
    if apply_outdoor_fraction:
        population *= outdoor_population_fraction(hour)
    return population.astype(np.float64)

