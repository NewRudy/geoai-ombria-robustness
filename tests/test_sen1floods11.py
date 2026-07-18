from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from geoai_ombria_robustness.sen1floods11 import (
    _download,
    local_paths,
    manifest_records,
    normalize_sentinel1,
    normalize_sentinel2,
    resolve_scl_href,
    scl_reference_quality,
)


class Sen1Floods11Tests(unittest.TestCase):
    def test_download_retries_transient_disconnect_and_writes_atomically(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.side_effect = [b"payload", b""]
        response.__exit__.return_value = False
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "asset.tif"
            with (
                patch(
                    "geoai_ombria_robustness.sen1floods11.urllib.request.urlopen",
                    side_effect=[ConnectionResetError("transient"), response],
                ) as urlopen,
                patch("geoai_ombria_robustness.sen1floods11.time.sleep") as sleep,
            ):
                _download(
                    "https://example.test/asset.tif",
                    destination,
                    attempts=2,
                    backoff_seconds=0.0,
                )
            self.assertEqual(destination.read_bytes(), b"payload")
            self.assertFalse(destination.with_suffix(".tif.part").exists())
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(0.0)

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

    def test_reference_quality_excludes_official_optical_nodata(self) -> None:
        scl = np.array([[4, 4], [9, 6]], dtype=np.uint8)
        optical_valid = np.array([[True, False], [True, True]])
        quality = scl_reference_quality(scl, optical_valid)
        np.testing.assert_array_equal(
            quality,
            [[True, False], [False, True]],
        )

    def test_earth_search_asset_is_not_signed(self) -> None:
        href = "https://example.test/SCL.tif"
        self.assertEqual(
            resolve_scl_href({"href": href, "provider": "earth-search"}),
            href,
        )


if __name__ == "__main__":
    unittest.main()
