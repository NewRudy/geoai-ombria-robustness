from __future__ import annotations

import unittest

from geoai_ombria_robustness.quality_uncertainty_full import (
    build_full_shard_plan,
    ombria_training_route_args,
)


class QualityUncertaintyFullTests(unittest.TestCase):
    def test_seed_shard_keeps_the_complete_frozen_matrix(self) -> None:
        shard = build_full_shard_plan(7)

        self.assertEqual(shard.planned_seeds, (7, 13, 21, 29, 37))
        self.assertEqual(shard.active_seed, 7)
        self.assertEqual(shard.epochs, 25)
        self.assertEqual(shard.error_rates, (0.0, 0.05, 0.1, 0.2, 0.4))
        self.assertEqual(shard.perturbation_repetitions, 3)
        self.assertEqual(shard.external_seed_condition_rows, 550)
        self.assertEqual(shard.external_raw_summary_rows, 1650)
        self.assertEqual(shard.ombria_evaluation_cells, 101)
        self.assertEqual(shard.ombria_raw_summary_rows, 301)
        self.assertFalse(shard.scientific_interpretation_allowed)

    def test_ombria_error_aware_routes_are_explicitly_distinct(self) -> None:
        rates = (0.0, 0.05, 0.1, 0.2, 0.4)
        oracle = ombria_training_route_args("hard_oracle", rates)
        error_aware = ombria_training_route_args("hard_error_aware", rates)

        self.assertNotIn("--train-quality-error-rates", oracle)
        self.assertIn("--train-quality-error-rates", error_aware)
        self.assertIn("quality_gated_fusion", oracle)
        self.assertIn("quality_gated_fusion", error_aware)
        with self.assertRaisesRegex(ValueError, "Unknown OMBRIA route"):
            ombria_training_route_args("invented", rates)


if __name__ == "__main__":
    unittest.main()
