from __future__ import annotations

import unittest

import numpy as np

from geoai_ombria_robustness.sen1floods11_audit import (
    percentile_stretch,
    select_alignment_audit_records,
)


def record(
    chip_id: str,
    event: str,
    split: str,
    provider: str,
    asset_count: int = 1,
) -> dict:
    return {
        "chip_id": chip_id,
        "event": event,
        "split": split,
        "scl_assets": [
            {"provider": provider, "item_id": f"{chip_id}-{index}"}
            for index in range(asset_count)
        ],
    }


class Sen1Floods11AuditTests(unittest.TestCase):
    def test_selection_is_deterministic_and_covers_strata(self) -> None:
        records = [
            record("A1", "A", "train", "earth-search"),
            record("A2", "A", "validation", "earth-search"),
            record("B1", "B", "test", "planetary-computer"),
            record("B2", "B", "bolivia", "earth-search", asset_count=2),
        ]
        first = select_alignment_audit_records(records)
        second = select_alignment_audit_records(list(reversed(records)))
        self.assertEqual(
            [value["chip_id"] for value in first],
            [value["chip_id"] for value in second],
        )
        self.assertEqual({value["event"] for value in first}, {"A", "B"})
        self.assertEqual(
            {value["split"] for value in first},
            {"train", "validation", "test", "bolivia"},
        )
        providers = {
            asset["provider"] for value in first for asset in value["scl_assets"]
        }
        self.assertEqual(
            providers,
            {"earth-search", "planetary-computer"},
        )
        self.assertTrue(any(len(value["scl_assets"]) > 1 for value in first))

    def test_selection_rejects_empty_and_invalid_count(self) -> None:
        with self.assertRaises(ValueError):
            select_alignment_audit_records([])
        with self.assertRaises(ValueError):
            select_alignment_audit_records(
                [record("A1", "A", "train", "earth-search")],
                per_event=0,
            )

    def test_percentile_stretch_is_finite_and_bounded(self) -> None:
        source = np.array(
            [
                [[0.0, 1.0], [2.0, 100.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
            dtype=np.float32,
        )
        stretched = percentile_stretch(source)
        self.assertTrue(np.isfinite(stretched).all())
        self.assertGreaterEqual(float(stretched.min()), 0.0)
        self.assertLessEqual(float(stretched.max()), 1.0)
        np.testing.assert_array_equal(stretched[1], 0.0)


if __name__ == "__main__":
    unittest.main()
