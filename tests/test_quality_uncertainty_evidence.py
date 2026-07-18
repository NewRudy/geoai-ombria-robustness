from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_evidence import (  # noqa: E402
    summarize_seed_conditions,
)


def row(
    route: str,
    seed: int,
    split: str,
    condition: str,
    repetition: int,
    iou: float,
) -> dict[str, str]:
    return {
        "route": route,
        "model_seed": str(seed),
        "split": split,
        "condition_id": condition,
        "quality_mode": "reference",
        "false_available_rate": "0",
        "false_unavailable_rate": "0",
        "shift_y": "0",
        "shift_x": "0",
        "radius": "0",
        "matched_source_mode": "translate",
        "repetition": str(repetition),
        "iou": str(iou),
        "f1": str(iou + 0.1),
        "precision": "0.8",
        "recall": "0.7",
        "event_equal_iou": str(iou - 0.05),
        "realized_false_available_rate": "0",
        "realized_false_unavailable_rate": "0",
        "valid_realized_false_available_rate": "0",
        "valid_realized_false_unavailable_rate": "0",
    }


class QualityUncertaintyEvidenceTests(unittest.TestCase):
    def test_seed_summary_averages_repetitions_and_attaches_s1_delta(self) -> None:
        rows = [
            row("s1_reference", 7, "test", "reference", 0, 0.4),
            row("quality_concat", 7, "test", "fa0_fu0", 0, 0.5),
            row("quality_concat", 7, "test", "fa0_fu0", 1, 0.7),
        ]
        summary = summarize_seed_conditions(rows)
        quality = next(value for value in summary if value["route"] == "quality_concat")
        self.assertAlmostEqual(float(quality["iou"]), 0.6)
        self.assertAlmostEqual(float(quality["s1_reference_iou"]), 0.4)
        self.assertAlmostEqual(float(quality["delta_s1_iou"]), 0.2)
        self.assertEqual(quality["repetitions"], 2)

    def test_seed_summary_requires_a_matching_s1_reference(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "S1 reference"):
            summarize_seed_conditions(
                [row("quality_concat", 7, "test", "fa0_fu0", 0, 0.5)]
            )


if __name__ == "__main__":
    unittest.main()
