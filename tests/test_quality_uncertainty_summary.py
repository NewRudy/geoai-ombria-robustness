from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_quality_uncertainty import summarize  # noqa: E402


class QualityUncertaintySummaryTests(unittest.TestCase):
    def test_delta_is_computed_against_s1_reference(self) -> None:
        rows = [
            {
                "route": "s1_reference",
                "content_degradation": "none",
                "requested_false_available_rate": "0",
                "requested_false_unavailable_rate": "0",
                "realized_false_available_rate": "0",
                "realized_false_unavailable_rate": "0",
                "iou": "0.50",
            },
            {
                "route": "hard_oracle",
                "content_degradation": "cloud_after_50",
                "requested_false_available_rate": "0.2",
                "requested_false_unavailable_rate": "0.1",
                "realized_false_available_rate": "0.2",
                "realized_false_unavailable_rate": "0.1",
                "iou": "0.55",
            },
        ]
        result = summarize(rows)
        hard = next(row for row in result if row["route"] == "hard_oracle")
        self.assertAlmostEqual(float(hard["s1_reference_iou"]), 0.50)
        self.assertAlmostEqual(float(hard["delta_s1_iou"]), 0.05)

    def test_missing_s1_reference_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "s1_reference"):
            summarize(
                [
                    {
                        "route": "hard",
                        "content_degradation": "cloud_after_50",
                        "requested_false_available_rate": "0",
                        "requested_false_unavailable_rate": "0",
                        "realized_false_available_rate": "0",
                        "realized_false_unavailable_rate": "0",
                        "iou": "0.5",
                    }
                ]
            )


if __name__ == "__main__":
    unittest.main()
