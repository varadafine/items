"""Regression tests for the opt-in 12-feature obstacle-risk model."""

import json
import unittest
from pathlib import Path

import numpy as np

from intentra_x.air_risk_features import FEATURE_NAMES, row_to_features
from intentra_x.air_risk_predictor import AirRiskPredictor


class TwelveFeatureTests(unittest.TestCase):
    def test_feature_order_and_no_collision_label_leakage(self):
        row = {
            "position": [20.0, 30.0, -10.0],
            "velocity": [3.0, 4.0, 0.0],
            "attitude": {"roll": 1.0, "pitch": 2.0},
            "wind_speed": 6.0,
            "wind_direction_sin": 0.0,
            "wind_direction_cos": 0.0,
            "sensor_obstacle_distance": 8.0,
            "obstacle_collision": False,
            "ground_collision": False,
        }
        expected = [10, 3, 4, 0, 5, 1, 2, 6, 0, 0, 8, 1]
        self.assertEqual(len(FEATURE_NAMES), 12)
        np.testing.assert_allclose(row_to_features(row), expected)
        row["obstacle_collision"] = True
        row["ground_collision"] = True
        np.testing.assert_allclose(row_to_features(row), expected)

    def test_manual12_checkpoint_runs_with_twenty_rows(self):
        source = Path(__file__).resolve().parent / "data/UAV_gust_manual_sessions.jsonl"
        rows = []
        with source.open(encoding="utf-8") as stream:
            for line in stream:
                rows.append(json.loads(line))
                if len(rows) == 20:
                    break
        result = AirRiskPredictor(device="cpu", schema="manual12").predict_rows(rows)
        self.assertTrue(0.0 <= result.risk_probability <= 1.0)
        self.assertEqual(tuple(result.risk_factor_weights), FEATURE_NAMES)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in result.risk_factor_weights.values()))


if __name__ == "__main__":
    unittest.main()
