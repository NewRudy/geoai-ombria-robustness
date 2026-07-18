from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_sen1floods11_quality_uncertainty import (  # noqa: E402
    Counts,
    event_equal_iou,
    metrics,
    normalize_evaluation_conditions,
    stable_sample_seed,
)


class Sen1Floods11EvaluationTests(unittest.TestCase):
    def test_counts_metrics_are_pooled(self) -> None:
        result = metrics(Counts(tp=3, fp=1, fn=2, tn=4))
        self.assertAlmostEqual(result["iou"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.75)

    def test_event_equal_iou_weights_events_equally(self) -> None:
        result = event_equal_iou(
            {
                "large": Counts(tp=90, fp=10, fn=0, tn=0),
                "small": Counts(tp=1, fp=0, fn=1, tn=0),
            }
        )
        self.assertAlmostEqual(result, 0.7)

    def test_sample_seed_is_call_order_independent(self) -> None:
        first = stable_sample_seed(9, 1, "chip", "quality")
        self.assertEqual(first, stable_sample_seed(9, 1, "chip", "quality"))
        self.assertNotEqual(first, stable_sample_seed(9, 2, "chip", "quality"))

    def test_condition_matrix_is_normalized_and_identifiers_are_unique(self) -> None:
        conditions = normalize_evaluation_conditions(
            [
                {
                    "condition_id": "fa0_fu0",
                    "quality_mode": "independent",
                },
                {
                    "condition_id": "shift",
                    "quality_mode": "translate",
                    "shift_x": 26,
                },
            ]
        )
        self.assertEqual(conditions[0]["false_available_rate"], 0.0)
        self.assertEqual(conditions[1]["shift_x"], 26)
        self.assertEqual(conditions[1]["radius"], 0)
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_evaluation_conditions(
                [
                    {"condition_id": "same", "quality_mode": "reference"},
                    {"condition_id": "same", "quality_mode": "reference"},
                ]
            )


if __name__ == "__main__":
    unittest.main()
