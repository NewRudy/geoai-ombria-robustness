from __future__ import annotations

import unittest

import numpy as np

from geoai_ombria_robustness.quality_maps import (
    dilate_unavailable,
    erode_unavailable,
    perturb_quality_map,
    quality_map_confusion,
    random_error_control,
    translate_quality_map,
)


class QualityMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.ones((10, 10), dtype=bool)
        self.reference[2:6, 3:8] = False

    def test_independent_rates_have_exact_realized_counts(self) -> None:
        result = perturb_quality_map(
            self.reference,
            false_available_rate=0.25,
            false_unavailable_rate=0.10,
            rng=np.random.default_rng(17),
        )
        confusion = result.confusion
        self.assertEqual(confusion.unavailable_pixels, 20)
        self.assertEqual(confusion.available_pixels, 80)
        self.assertEqual(confusion.false_available, 5)
        self.assertEqual(confusion.false_unavailable, 8)
        self.assertAlmostEqual(confusion.false_available_rate, 0.25)
        self.assertAlmostEqual(confusion.false_unavailable_rate, 0.10)

    def test_independent_perturbation_is_seed_deterministic(self) -> None:
        first = perturb_quality_map(
            self.reference,
            0.30,
            0.20,
            np.random.default_rng(23),
        )
        second = perturb_quality_map(
            self.reference,
            0.30,
            0.20,
            np.random.default_rng(23),
        )
        np.testing.assert_array_equal(first.observed, second.observed)

    def test_empty_denominator_has_zero_realized_rate(self) -> None:
        all_available = np.ones((4, 4), dtype=bool)
        result = perturb_quality_map(
            all_available,
            false_available_rate=1.0,
            false_unavailable_rate=0.25,
            rng=np.random.default_rng(3),
        )
        self.assertEqual(result.confusion.false_available, 0)
        self.assertEqual(result.confusion.false_available_rate, 0.0)
        self.assertEqual(result.confusion.false_unavailable, 4)

    def test_translation_does_not_wrap(self) -> None:
        reference = np.ones((5, 5), dtype=bool)
        reference[0, 0] = False
        shifted = translate_quality_map(
            reference,
            shift_y=1,
            shift_x=2,
            fill_available=True,
        )
        self.assertFalse(shifted.observed[1, 2])
        self.assertTrue(shifted.observed[0, 0])
        self.assertEqual(int((~shifted.observed).sum()), 1)

    def test_dilation_and_erosion_change_unavailable_region(self) -> None:
        reference = np.ones((7, 7), dtype=bool)
        reference[3, 3] = False
        dilated = dilate_unavailable(reference, radius=1)
        self.assertEqual(int((~dilated.observed).sum()), 9)
        self.assertEqual(dilated.confusion.false_unavailable, 8)

        block = np.ones((7, 7), dtype=bool)
        block[2:5, 2:5] = False
        eroded = erode_unavailable(block, radius=1)
        self.assertEqual(int((~eroded.observed).sum()), 1)
        self.assertEqual(eroded.confusion.false_available, 8)

    def test_random_control_matches_both_error_counts(self) -> None:
        structured = translate_quality_map(
            self.reference,
            shift_y=1,
            shift_x=-2,
            fill_available=True,
        )
        control = random_error_control(
            self.reference,
            structured.observed,
            np.random.default_rng(101),
        )
        self.assertEqual(
            control.confusion.false_available,
            structured.confusion.false_available,
        )
        self.assertEqual(
            control.confusion.false_unavailable,
            structured.confusion.false_unavailable,
        )

    def test_random_control_also_matches_counts_inside_comparison_mask(self) -> None:
        reference = np.ones((8, 8), dtype=bool)
        reference[1:7, 2:6] = False
        valid = np.zeros((8, 8), dtype=bool)
        valid[:, :5] = True
        structured = translate_quality_map(
            reference,
            shift_y=1,
            shift_x=2,
            fill_available=True,
        )
        control = random_error_control(
            reference,
            structured.observed,
            np.random.default_rng(103),
            comparison_mask=valid,
        )
        structured_valid = quality_map_confusion(
            reference[valid][None, :],
            structured.observed[valid][None, :],
        )
        control_valid = quality_map_confusion(
            reference[valid][None, :],
            control.observed[valid][None, :],
        )
        self.assertEqual(
            control.confusion.false_available,
            structured.confusion.false_available,
        )
        self.assertEqual(
            control.confusion.false_unavailable,
            structured.confusion.false_unavailable,
        )
        self.assertEqual(
            control_valid.false_available,
            structured_valid.false_available,
        )
        self.assertEqual(
            control_valid.false_unavailable,
            structured_valid.false_unavailable,
        )

    def test_quality_iou_for_empty_available_union_is_one(self) -> None:
        unavailable = np.zeros((3, 3), dtype=bool)
        confusion = quality_map_confusion(unavailable, unavailable)
        self.assertEqual(confusion.quality_iou, 1.0)

    def test_validation_rejects_nonbinary_and_invalid_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "binary"):
            quality_map_confusion(np.array([[0, 2]]), np.array([[0, 1]]))
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            perturb_quality_map(
                self.reference,
                false_available_rate=-0.01,
                false_unavailable_rate=0.0,
                rng=np.random.default_rng(1),
            )


if __name__ == "__main__":
    unittest.main()
