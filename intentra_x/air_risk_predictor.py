"""Inference wrapper for the legacy 9-feature and experimental 12-feature models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Dict, Iterable, Sequence

import numpy as np
import torch

try:
    from .attention_risk_lstm import FeatureAttentionRiskLSTM
    from .air_risk_features import (
        FEATURE_NAMES as FEATURE_NAMES_12,
        LEGACY_FEATURE_NAMES,
        row_to_features as extract_features,
    )
except ImportError:  # Allow: python intentra_x/run_joint_demo.py
    from attention_risk_lstm import FeatureAttentionRiskLSTM
    from air_risk_features import (
        FEATURE_NAMES as FEATURE_NAMES_12,
        LEGACY_FEATURE_NAMES,
        row_to_features as extract_features,
    )


# The default joint-risk demo still uses its original 9-feature checkpoint.
# The 12-feature model is available explicitly as schema="manual12".
FEATURE_NAMES = LEGACY_FEATURE_NAMES
SEQUENCE_LENGTH = 20
RISK_THRESHOLD = 0.31

# Exact z-score parameters from the training-flight split used by
# train_airsim_attention.py (seed=42, first three shuffled flights).
NORMALIZATION_MEAN = np.asarray(
    [
        5.0961503982543945,
        7.325197696685791,
        -1.776472568511963,
        -10.065469741821289,
        8.854437828063965,
        30.006227493286133,
        1.0,
        -0.04404900595545769,
        0.05241537094116211,
    ],
    dtype=np.float32,
)
NORMALIZATION_STD = np.asarray(
    [
        3.2530648708343506,
        4.551985263824463,
        16.367094039916992,
        12.335988998413086,
        8.049261093139648,
        32.117733001708984,
        1.0,
        0.7254416346549988,
        0.6848823428153992,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class AirRiskResult:
    risk_type: str
    risk_probability: float
    risk_detected: bool
    threshold: float
    risk_factor_weights: Dict[str, float]

    def to_dict(self):
        return asdict(self)


def row_to_features(row: dict) -> np.ndarray:
    """Keep the old demo's public 9-feature function for compatibility."""
    position = row.get("position") or [0.0, 0.0, 0.0]
    velocity = row.get("velocity") or [0.0, 0.0, 0.0]
    attitude = row.get("attitude") or {}
    speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float32)))
    altitude = max(0.0, -float(position[2]))
    sensor_distance = row.get("sensor_obstacle_distance", 100.0) or 100.0
    return np.asarray(
        [
            row.get("wind_speed", 0.0) or 0.0,
            speed,
            attitude.get("roll", 0.0) or 0.0,
            attitude.get("pitch", 0.0) or 0.0,
            altitude,
            sensor_distance,
            1.0 if sensor_distance < 100.0 else 0.0,
            row.get("wind_direction_sin", 0.0) or 0.0,
            row.get("wind_direction_cos", 1.0) or 1.0,
        ],
        dtype=np.float32,
    )


class AirRiskPredictor:
    def __init__(self, model_path=None, device=None, threshold=None,
                 schema="legacy9", metadata_path=None):
        base = Path(__file__).resolve().parent
        if schema not in {"legacy9", "manual12"}:
            raise ValueError("schema must be 'legacy9' or 'manual12'")
        self.schema = schema
        self.feature_names = FEATURE_NAMES if schema == "legacy9" else FEATURE_NAMES_12
        if schema == "manual12":
            experiment_dir = base / "models" / "manual_12_experiment"
            if model_path is not None and metadata_path is None:
                raise ValueError("custom 12-feature checkpoint requires matching metadata_path")
            metadata_file = Path(metadata_path or (experiment_dir / "comparison.json"))
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            candidate = next(item for item in metadata["results"] if item["schema"] == "x12")
            if tuple(candidate["features"]) != FEATURE_NAMES_12:
                raise ValueError("12-feature model metadata has a different feature order")
            self.normalization_mean = np.asarray(candidate["normalization_mean"], dtype=np.float32)
            self.normalization_std = np.asarray(candidate["normalization_std"], dtype=np.float32)
            default_path = experiment_dir / "attention_lstm_x12.pt"
            default_threshold = float(candidate["val"]["threshold"])
        else:
            self.normalization_mean = NORMALIZATION_MEAN
            self.normalization_std = NORMALIZATION_STD
            default_path = base / "models" / "airsim_attention_risk_model.pt"
            default_threshold = RISK_THRESHOLD
        self.model_path = Path(
            model_path or default_path
        )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(default_threshold if threshold is None else threshold)
        self.model = FeatureAttentionRiskLSTM(n_features=len(self.feature_names)).to(self.device)
        try:
            state_dict = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
        except TypeError:  # Compatibility with older PyTorch releases.
            state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def prepare_sequence(self, sequence) -> np.ndarray:
        values = np.asarray(sequence, dtype=np.float32)
        expected = (SEQUENCE_LENGTH, len(self.feature_names))
        if values.shape != expected:
            raise ValueError(f"air-risk sequence must have shape {expected}, got {values.shape}")
        return ((values - self.normalization_mean) / self.normalization_std).astype(np.float32)

    def predict(self, sequence) -> AirRiskResult:
        normalized = self.prepare_sequence(sequence)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        with torch.no_grad():
            risk_tensor, importance_tensor = self.model(tensor)
        probability = float(risk_tensor.squeeze().cpu().item())
        importance = importance_tensor.squeeze(0).cpu().numpy()
        weights = {
            name: float(value) for name, value in zip(self.feature_names, importance)
        }
        detected = probability >= self.threshold
        return AirRiskResult(
            risk_type=(
                "OBSTACLE_COLLISION_RISK" if self.schema == "manual12"
                else "COLLISION_RISK"
            ) if detected else "NORMAL",
            risk_probability=probability,
            risk_detected=detected,
            threshold=self.threshold,
            risk_factor_weights=weights,
        )

    def predict_rows(self, rows: Sequence[dict]) -> AirRiskResult:
        if len(rows) != SEQUENCE_LENGTH:
            raise ValueError(f"exactly {SEQUENCE_LENGTH} telemetry rows are required")
        if self.schema == "legacy9":
            features = [row_to_features(row) for row in rows]
        else:
            features = [extract_features(row, self.feature_names) for row in rows]
        return self.predict(np.stack(features))
