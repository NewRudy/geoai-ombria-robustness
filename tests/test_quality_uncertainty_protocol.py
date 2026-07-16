from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geoai_ombria_robustness.ombria import (
    OmbriaSample,
    load_multimodal_quality_uncertainty_sample,
)


class QualityUncertaintyProtocolTests(unittest.TestCase):
    def _sample(self, root: Path) -> OmbriaSample:
        height = width = 96
        paths = {
            "s1_before": root / "s1_before.png",
            "s1_after": root / "s1_after.png",
            "s1_mask": root / "s1_mask.png",
            "s2_before": root / "s2_before.png",
            "s2_after": root / "s2_after.png",
            "s2_mask": root / "s2_mask.png",
        }
        Image.fromarray(
            np.full((height, width), 80, dtype=np.uint8)
        ).save(paths["s1_before"])
        Image.fromarray(
            np.full((height, width), 120, dtype=np.uint8)
        ).save(paths["s1_after"])
        Image.fromarray(
            np.zeros((height, width), dtype=np.uint8)
        ).save(paths["s1_mask"])
        Image.fromarray(
            np.full((height, width, 3), 160, dtype=np.uint8)
        ).save(paths["s2_before"])
        Image.fromarray(
            np.full((height, width, 3), 200, dtype=np.uint8)
        ).save(paths["s2_after"])
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[20:50, 30:70] = 255
        Image.fromarray(mask).save(paths["s2_mask"])
        return OmbriaSample(
            split="train",
            chip_id="test",
            **paths,
        )

    def test_content_and_quality_errors_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = load_multimodal_quality_uncertainty_sample(
                self._sample(Path(temporary)),
                degrade_s2="cloud_after_25",
                degradation_rng=np.random.default_rng(41),
                quality_rng=np.random.default_rng(43),
                false_available_rate=0.20,
                false_unavailable_rate=0.10,
            )
        self.assertEqual(result.image.shape, (96, 96, 10))
        np.testing.assert_array_equal(
            result.image[:, :, 8:] >= 0.5,
            result.observed_quality,
        )
        self.assertGreater(result.quality_confusion.false_available, 0)
        self.assertGreater(result.quality_confusion.false_unavailable, 0)
        self.assertTrue(np.all(result.reference_quality[:, :, 0]))
        self.assertTrue(
            np.any(~result.reference_quality[:, :, 1])
        )

    def test_zero_quality_error_returns_reference_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = load_multimodal_quality_uncertainty_sample(
                self._sample(Path(temporary)),
                degrade_s2="cloud_after_25",
                degradation_rng=np.random.default_rng(47),
                quality_rng=np.random.default_rng(53),
                false_available_rate=0.0,
                false_unavailable_rate=0.0,
            )
        np.testing.assert_array_equal(
            result.reference_quality,
            result.observed_quality,
        )
        self.assertEqual(result.quality_confusion.false_available, 0)
        self.assertEqual(result.quality_confusion.false_unavailable, 0)


if __name__ == "__main__":
    unittest.main()
