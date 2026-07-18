from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from train_sen1floods11_unet import (  # noqa: E402
    masked_bce_with_logits,
    metrics_from_counts,
    stable_stream_seed,
)


class Sen1Floods11TrainingTests(unittest.TestCase):
    def test_stable_stream_seed_separates_epoch_and_stream(self) -> None:
        first = stable_stream_seed(7, 1, "chip", "quality")
        self.assertEqual(first, stable_stream_seed(7, 1, "chip", "quality"))
        self.assertNotEqual(first, stable_stream_seed(7, 2, "chip", "quality"))
        self.assertNotEqual(first, stable_stream_seed(7, 1, "chip", "augment"))

    def test_masked_loss_ignores_invalid_pixels(self) -> None:
        logits = torch.tensor([[[[0.0, 100.0]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        valid = torch.tensor([[[[True, False]]]])
        loss = masked_bce_with_logits(logits, target, valid)
        self.assertAlmostEqual(float(loss), 0.693147, places=5)

    def test_metrics_are_computed_from_pooled_counts(self) -> None:
        metrics = metrics_from_counts(tp=3, fp=1, fn=2, tn=4)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.75)
        self.assertAlmostEqual(metrics["recall"], 0.6)


if __name__ == "__main__":
    unittest.main()
