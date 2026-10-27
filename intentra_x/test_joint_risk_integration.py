"""Focused tests for the migrated ground-risk and joint-inference chain."""

import unittest

import numpy as np

from intentra_x.air_risk_predictor import AirRiskPredictor
from intentra_x.ground_risk import GroundRiskConfig, GroundRiskEngine
from intentra_x.ground_risk.harm_models.fatality_model import probability_of_fatality
from intentra_x.ground_risk.population import load_hohhot_population
from intentra_x.run_joint_demo import load_collision_demo_window


class GroundRiskMigrationTests(unittest.TestCase):
    def test_hohhot_population_loads(self):
        population = load_hohhot_population(hour=9, resolution_m=50)
        self.assertEqual(population.shape, (133, 268))
        self.assertGreater(float(population.max()), float(population.min()))

    def test_fatality_probability_increases_with_energy(self):
        low = probability_of_fatality(100, 1e6, 34, 0.3)
        high = probability_of_fatality(10_000, 1e6, 34, 0.3)
        self.assertGreater(high, low)

    def test_existing_air_checkpoint_runs(self):
        _, _, sequence = load_collision_demo_window()
        result = AirRiskPredictor(device="cpu").predict(sequence)
        self.assertGreaterEqual(result.risk_probability, 0.0)
        self.assertLessEqual(result.risk_probability, 1.0)
        self.assertEqual(len(result.risk_factor_weights), 9)

    def test_ground_engine_runs_on_small_grid(self):
        engine = GroundRiskEngine(
            config=GroundRiskConfig(
                resolution_m=50,
                monte_carlo_samples=200,
                random_seed=7,
            )
        )
        population = np.full((32, 48), 1_000.0)
        result = engine.evaluate(
            population_grid=population,
            accident_probability=0.2,
            altitude_m=120,
            airspeed_mps=20,
            wind_speed_mps=5,
            wind_direction_deg=180,
        )
        self.assertEqual(result.fatality_risk.shape, population.shape)
        self.assertTrue(np.isfinite(result.fatality_risk).all())
        self.assertAlmostEqual(float(result.impact_probability.sum()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
