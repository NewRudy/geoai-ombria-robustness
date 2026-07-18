from __future__ import annotations

import argparse
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_sen1floods11_quality_uncertainty as evaluation  # noqa: E402

from evaluate_sen1floods11_quality_uncertainty import (  # noqa: E402
    Counts,
    event_equal_iou,
    metrics,
    normalize_evaluation_conditions,
    stable_sample_seed,
)
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    Sen1Floods11Chip,
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

    def test_all_invalid_target_chip_does_not_crash_quality_accounting(self) -> None:
        shape = (4, 4)
        chip = Sen1Floods11Chip(
            image=np.zeros((7, *shape), dtype=np.float32),
            target=np.zeros(shape, dtype=np.float32),
            valid_target=np.zeros(shape, dtype=bool),
            reference_quality=np.ones(shape, dtype=bool),
            optical_valid=np.ones(shape, dtype=bool),
            scl=np.full(shape, 4, dtype=np.uint8),
        )
        args = argparse.Namespace(
            quality_mode="reference",
            perturb_seed=20260716,
            false_available_rate=0.0,
            false_unavailable_rate=0.0,
            shift_y=0,
            shift_x=0,
            radius=0,
            matched_source_mode="translate",
        )
        record = {"chip_id": "all-invalid", "event": "fixture"}
        fake_torch = types.ModuleType("torch")
        fake_torch.from_numpy = np.asarray
        fake_utils = types.ModuleType("torch.utils")
        fake_data = types.ModuleType("torch.utils.data")
        fake_data.Dataset = object
        with (
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "torch.utils": fake_utils,
                    "torch.utils.data": fake_data,
                },
            ),
            patch.object(evaluation, "load_hand_labeled_chip", return_value=chip),
        ):
            dataset = evaluation.EvaluationDataset(
                [record], Path("."), "s1_reference", args, repetition=0
            ).dataset
            _, _, valid, index, quality = dataset[0]

        self.assertFalse(valid.any().item())
        self.assertEqual(index, 0)
        np.testing.assert_array_equal(np.asarray(quality)[7:11], np.zeros(4))
        self.assertEqual(float(np.asarray(quality)[13]), 1.0)
        self.assertIsNone(
            evaluation.masked_mean_probability(
                np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
            )
        )


if __name__ == "__main__":
    unittest.main()
