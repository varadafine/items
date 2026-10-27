"""Optional OpenStreetMap/Overpass provider with an offline JSON cache.

The ground-risk calculation does not depend on a live network.  These queries
are retained for later school, hospital, fuel-station and industrial-area
consequence layers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import requests

from .population import DATA_DIR, HOHHOT_BOUNDS


OVERPASS_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)

OSM_FEATURE_TAGS: Dict[str, Tuple[str, ...]] = {
    "residential": ("landuse=residential",),
    "commercial": ("landuse=commercial", "landuse=retail"),
    "industrial": ("landuse=industrial",),
    "school": ("amenity=school", "amenity=college", "amenity=university"),
    "hospital": ("amenity=hospital", "amenity=clinic"),
    "fuel_station": ("amenity=fuel",),
}


class OSMProvider:
    def __init__(self, cache_dir: Path | None = None, timeout_seconds: int = 45):
        self.cache_dir = Path(cache_dir or (DATA_DIR / "osm_json_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = int(timeout_seconds)

    @staticmethod
    def _cache_name(tag: str, bounds: Tuple[float, float, float, float]) -> str:
        safe_tag = re.sub(r"[^a-zA-Z0-9_-]+", "_", tag)
        bound_key = "_".join(f"{value:.4f}" for value in bounds)
        return f"{safe_tag}_{bound_key}.json"

    @staticmethod
    def _query(tag: str, bounds: Tuple[float, float, float, float]) -> str:
        min_lon, min_lat, max_lon, max_lat = bounds
        key, value = tag.split("=", 1)
        return f"""
[out:json][timeout:40][bbox:{min_lat},{min_lon},{max_lat},{max_lon}];
(
  node[\"{key}\"=\"{value}\"];
  way[\"{key}\"=\"{value}\"];
  relation[\"{key}\"=\"{value}\"];
);
out center geom;
""".strip()

    def query_tag(
        self,
        tag: str,
        bounds: Tuple[float, float, float, float] = HOHHOT_BOUNDS,
        force_refresh: bool = False,
    ) -> dict:
        cache_path = self.cache_dir / self._cache_name(tag, bounds)
        if cache_path.exists() and not force_refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        query = self._query(tag, bounds)
        errors = []
        for url in OVERPASS_URLS:
            try:
                response = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": "INTENTRA-X-ground-risk/1.0"},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                )
                return payload
            except (requests.RequestException, ValueError) as exc:
                errors.append(f"{url}: {exc}")
        raise RuntimeError("all Overpass endpoints failed: " + " | ".join(errors))

    def query_category(
        self,
        category: str,
        bounds: Tuple[float, float, float, float] = HOHHOT_BOUNDS,
        force_refresh: bool = False,
    ) -> dict:
        if category not in OSM_FEATURE_TAGS:
            raise KeyError(f"unknown OSM category: {category}")
        elements = []
        for tag in OSM_FEATURE_TAGS[category]:
            payload = self.query_tag(tag, bounds=bounds, force_refresh=force_refresh)
            elements.extend(payload.get("elements", []))
        return {"category": category, "bounds": bounds, "elements": elements}

    def available_offline_shapefiles(self) -> Iterable[Path]:
        """Return the original Hohhot OSM shapefile caches migrated for offline use."""

        return sorted((DATA_DIR / "osm_cache").glob("*.shp"))

