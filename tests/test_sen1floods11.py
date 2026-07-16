from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from geoai_ombria_robustness.sen1floods11 import (
    local_paths,
    manifest_records,
    normalize_sentinel1,
    normalize_sentinel2,
    resolve_scl_href,
)


class Sen1Floods11Tests(unittest.TestCase):
    def test_manifest_filter_respects_split_and_scl_requirement(self) -> None:
        document = {
            "records": [
                {"chip_id": "A", "split": "train", "scl_assets": [{"href": "x"}]},
                {"chip_id": "B", "split": "test", "scl_assets": []},
            ]
        }
        self.assertEqual(
            [record["chip_id"] for record in manifest_records(document, ["train"])],
            ["A"],
        )
        self.assertEqual(
            [
                record["chip_id"]
                for record in manifest_records(
                    document,
                    ["test"],
                    require_scl=False,
                )
            ],
            ["B"],
        )

    def test_local_paths_are_split_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = local_paths(
                Path(temporary),
                {"chip_id": "Spain_1", "split": "validation"},
            )
            self.assertEqual(paths.s1.name, "Spain_1_S1Hand.tif")
            self.assertEqual(paths.s2.parent.name, "S2")
            self.assertEqual(paths.quality.suffix, ".npz")
            self.assertIn("validation", paths.quality.parts)

    def test_sentinel1_normalization_is_fixed_and_bounded(self) -> None:
        source = np.array([[-60.0, -50.0, -24.5, 1.0, 5.0, np.nan]])
        normalized = normalize_sentinel1(source)
        np.testing.assert_allclose(
            normalized[0, :5],
            [0.0, 0.0, 0.5, 1.0, 1.0],
        )
        self.assertEqual(normalized[0, 5], 0.0)

    def test_sentinel2_normalization_is_fixed_and_bounded(self) -> None:
        source = np.array([[-1.0, 0.0, 5000.0, 10000.0, 12000.0, np.nan]])
        normalized = normalize_sentinel2(source)
        np.testing.assert_allclose(
            normalized,
            [[0.0, 0.0, 0.5, 1.0, 1.0, 0.0]],
        )

    def test_earth_search_asset_is_not_signed(self) -> None:
        href = "https://example.test/SCL.tif"
        self.assertEqual(
            resolve_scl_href({"href": href, "provider": "earth-search"}),
            href,
        )


if __name__ == "__main__":
    unittest.main()
