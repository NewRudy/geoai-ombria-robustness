from __future__ import annotations

import copy
import unittest

from geoai_ombria_robustness.quality_uncertainty_shard_set_audit import (
    ombria_semantic_split_signature,
)


class QualityUncertaintyShardSetAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.split = {
            "train": [
                {
                    "split": "train",
                    "chip_id": "0001",
                    "s1_before": "/kaggle/working/run-a/s1/train/0001.png",
                    "s2_before": "/kaggle/working/run-a/s2/train/0001.png",
                }
            ],
            "val": [
                {
                    "split": "train",
                    "chip_id": "0002",
                    "s1_before": "/kaggle/working/run-a/s1/train/0002.png",
                    "s2_before": "/kaggle/working/run-a/s2/train/0002.png",
                }
            ],
            "test": [
                {
                    "split": "test",
                    "chip_id": "0003",
                    "s1_before": "/kaggle/working/run-a/s1/test/0003.png",
                    "s2_before": "/kaggle/working/run-a/s2/test/0003.png",
                }
            ],
        }

    def test_absolute_path_only_change_preserves_semantic_signature(self) -> None:
        other = copy.deepcopy(self.split)
        for records in other.values():
            for record in records:
                record["s1_before"] = record["s1_before"].replace("run-a", "run-b")
                record["s2_before"] = record["s2_before"].replace("run-a", "run-b")

        self.assertEqual(
            ombria_semantic_split_signature(self.split),
            ombria_semantic_split_signature(other),
        )

    def test_chip_partition_change_changes_semantic_signature(self) -> None:
        other = copy.deepcopy(self.split)
        moved = other["train"].pop()
        other["val"].append(moved)

        self.assertNotEqual(
            ombria_semantic_split_signature(self.split),
            ombria_semantic_split_signature(other),
        )

    def test_duplicate_source_chip_assignment_is_rejected(self) -> None:
        other = copy.deepcopy(self.split)
        other["val"].append(copy.deepcopy(other["train"][0]))

        with self.assertRaisesRegex(ValueError, "duplicate chip assignments"):
            ombria_semantic_split_signature(other)


if __name__ == "__main__":
    unittest.main()
