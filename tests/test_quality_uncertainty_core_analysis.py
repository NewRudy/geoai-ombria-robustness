from __future__ import annotations

import unittest

import numpy as np

from geoai_ombria_robustness.quality_uncertainty_core_analysis import (
    evaluate_method_gate,
    paired_t_summary,
)
from geoai_ombria_robustness.quality_uncertainty_full_audit import FULL_SEEDS
from geoai_ombria_robustness.quality_uncertainty_hierarchical_bootstrap import (
    Estimand,
    StateKey,
    UnitTable,
    hierarchical_contrast_summary,
)


class QualityUncertaintyCoreAnalysisTests(unittest.TestCase):
    def test_paired_t_summary_uses_five_frozen_seeds_and_df4(self) -> None:
        summary = paired_t_summary(
            {7: 0.10, 13: 0.12, 21: 0.08, 29: 0.11, 37: 0.09}
        )

        self.assertAlmostEqual(summary["mean"], 0.10)
        self.assertAlmostEqual(summary["sample_standard_deviation"], 0.0158113883)
        self.assertAlmostEqual(summary["ci95_lower"], 0.0803675688)
        self.assertAlmostEqual(summary["ci95_upper"], 0.1196324312)

    def test_paired_t_summary_rejects_partial_seed_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected frozen seeds"):
            paired_t_summary({7: 0.1, 13: 0.2})

    def test_method_claim_fails_when_one_dataset_exceeds_clean_loss(self) -> None:
        surfaces = {
            "ombria": {
                "hard_error_aware": {
                    "maximum_mean_regret": 0.0,
                    "lower_nonnegative_cells": 25,
                },
                "hard_oracle": {
                    "maximum_mean_regret": 0.0,
                    "lower_nonnegative_cells": 24,
                },
            },
            "sen1floods11": {
                "test": {
                    "hard_quality_gate_error_aware": {
                        "maximum_mean_regret": 0.0,
                        "lower_nonnegative_cells": 25,
                    },
                    "hard_quality_gate": {
                        "maximum_mean_regret": 0.12,
                        "lower_nonnegative_cells": 8,
                    },
                }
            },
        }
        rows = []
        for dataset, intervention, baseline, clean in (
            ("ombria", "hard_error_aware", "hard_oracle", -0.015),
            (
                "sen1floods11",
                "hard_quality_gate_error_aware",
                "hard_quality_gate",
                0.02,
            ),
        ):
            for fa in (0.0, 0.05, 0.1, 0.2, 0.4):
                for fu in (0.0, 0.05, 0.1, 0.2, 0.4):
                    mean = clean if (fa, fu) == (0.0, 0.0) else 0.03
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": "test",
                            "intervention": intervention,
                            "baseline": baseline,
                            "false_available_rate": fa,
                            "false_unavailable_rate": fu,
                            "iou_difference_mean": mean,
                            "iou_difference_ci95_lower": mean - 0.01,
                        }
                    )

        gate = evaluate_method_gate(surfaces, rows)

        self.assertFalse(gate["statistical_gate"])
        self.assertFalse(gate["datasets"]["ombria"]["clean_gate"])
        self.assertTrue(gate["datasets"]["sen1floods11"]["clean_gate"])
        self.assertFalse(gate["method_claim_authorized"])

    def test_hierarchical_bootstrap_is_paired_and_deterministic(self) -> None:
        keys = (("event_a", "chip_1"), ("event_b", "chip_2"))
        states = {}
        for seed in FULL_SEEDS:
            states[StateKey("test", "test", "a", "condition", seed)] = UnitTable(
                keys=keys,
                repetitions=(0,),
                counts=np.asarray([[[2, 0, 0]], [[2, 0, 0]]], dtype=np.int64),
            )
            states[StateKey("test", "test", "b", "condition", seed)] = UnitTable(
                keys=keys,
                repetitions=(0,),
                counts=np.asarray([[[1, 0, 1]], [[1, 0, 1]]], dtype=np.int64),
            )
        estimand = Estimand(
            estimand_id="constant_paired_difference",
            family="test",
            dataset="test",
            split="test",
            route="a_minus_b",
            condition_a="condition",
            condition_b="condition",
            state_a_route="a",
            state_a_condition="condition",
            state_b_route="b",
            state_b_condition="condition",
        )

        first = hierarchical_contrast_summary(
            states, estimand, replicates=100, random_seed=123
        )
        second = hierarchical_contrast_summary(
            states, estimand, replicates=100, random_seed=123
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_mean"], 0.5)
        self.assertAlmostEqual(first["ci95_lower"], 0.5)
        self.assertAlmostEqual(first["ci95_upper"], 0.5)


if __name__ == "__main__":
    unittest.main()
