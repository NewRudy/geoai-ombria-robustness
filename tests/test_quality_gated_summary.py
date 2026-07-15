from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_quality_gated_v3 import (  # noqa: E402
    classify_positive_contrast,
    paired_summary,
)


class QualityGatedSummaryTests(unittest.TestCase):
    def test_interval_is_computed_on_paired_differences(self) -> None:
        left = {7: 0.60, 13: 0.58, 21: 0.62, 29: 0.59, 37: 0.61}
        right = {7: 0.55, 13: 0.54, 21: 0.57, 29: 0.55, 37: 0.56}
        summary = paired_summary(left, right)
        self.assertAlmostEqual(summary["paired_difference_mean"], 0.046)
        self.assertEqual(summary["positive_seed_differences"], 5)
        self.assertGreater(summary["paired_ci95_lower"], 0.0)
        self.assertEqual(
            classify_positive_contrast(summary), "superiority_supported"
        )

    def test_one_seed_smoke_is_not_evaluable(self) -> None:
        summary = paired_summary({7: 0.5}, {7: 0.4})
        self.assertIsNone(summary["paired_ci95_half_width"])
        self.assertEqual(classify_positive_contrast(summary), "not_evaluable")


if __name__ == "__main__":
    unittest.main()
