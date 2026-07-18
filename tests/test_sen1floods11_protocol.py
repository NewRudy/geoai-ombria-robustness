from __future__ import annotations

import unittest

import numpy as np

from geoai_ombria_robustness.sen1floods11 import Sen1Floods11Chip
from geoai_ombria_robustness.sen1floods11_protocol import (
    augment_spatially,
    build_observed_quality,
    build_quality_condition,
    build_route_input,
    route_config,
)
from geoai_ombria_robustness.quality_maps import quality_map_confusion


def chip() -> Sen1Floods11Chip:
    image = np.zeros((7, 4, 4), dtype=np.float32)
    image[:4] = 0.75
    image[4:6] = 0.25
    quality = np.ones((4, 4), dtype=bool)
    quality[0, 0] = False
    image[6] = quality
    return Sen1Floods11Chip(
        image=image,
        target=np.eye(4, dtype=np.float32),
        valid_target=np.ones((4, 4), dtype=bool),
        reference_quality=quality,
        optical_valid=np.ones((4, 4), dtype=bool),
        scl=np.zeros((4, 4), dtype=np.uint8),
    )


class Sen1Floods11ProtocolTests(unittest.TestCase):
    def test_route_inputs_have_expected_channels(self) -> None:
        source = chip()
        self.assertEqual(build_route_input(source, "s1_reference").shape[0], 2)
        self.assertEqual(build_route_input(source, "early_fusion").shape[0], 6)
        self.assertEqual(build_route_input(source, "quality_concat").shape[0], 7)

    def test_complete_absence_zeros_only_optical_and_quality(self) -> None:
        source = chip()
        image = build_route_input(
            source,
            "hard_quality_gate",
            complete_optical_absence=True,
        )
        np.testing.assert_array_equal(image[:4], 0.0)
        np.testing.assert_array_equal(image[4:6], 0.25)
        np.testing.assert_array_equal(image[6], 0.0)

    def test_observed_quality_errors_do_not_modify_reference(self) -> None:
        source = chip()
        reference = source.reference_quality.copy()
        perturbed = build_observed_quality(
            source.reference_quality,
            false_available_rate=1.0,
            false_unavailable_rate=0.0,
            rng=np.random.default_rng(7),
        )
        self.assertTrue(perturbed.observed[0, 0])
        np.testing.assert_array_equal(source.reference_quality, reference)

    def test_augmentation_keeps_arrays_aligned(self) -> None:
        source = chip()
        image = build_route_input(source, "quality_concat")
        image[6] = source.target
        augmented, target, valid = augment_spatially(
            image,
            source.target,
            source.valid_target,
            np.random.default_rng(9),
        )
        np.testing.assert_array_equal(augmented[6] > 0.5, target > 0.5)
        self.assertTrue(valid.all())

    def test_unknown_route_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            route_config("unknown")

    def test_matched_random_preserves_structured_error_counts(self) -> None:
        reference = np.ones((7, 7), dtype=bool)
        reference[2:5, 2:5] = False
        comparison_mask = np.zeros((7, 7), dtype=bool)
        comparison_mask[:, :4] = True
        structured = build_quality_condition(
            reference,
            mode="translate",
            rng=np.random.default_rng(1),
            shift_y=1,
            shift_x=0,
            comparison_mask=comparison_mask,
        )
        random_control = build_quality_condition(
            reference,
            mode="matched-random",
            rng=np.random.default_rng(2),
            shift_y=1,
            shift_x=0,
            matched_source_mode="translate",
            comparison_mask=comparison_mask,
        )
        self.assertEqual(
            structured.perturbation.confusion.false_available,
            random_control.perturbation.confusion.false_available,
        )
        self.assertEqual(
            structured.perturbation.confusion.false_unavailable,
            random_control.perturbation.confusion.false_unavailable,
        )
        structured_valid = quality_map_confusion(
            reference[comparison_mask][None, :],
            structured.observed[comparison_mask][None, :],
        )
        random_valid = quality_map_confusion(
            reference[comparison_mask][None, :],
            random_control.observed[comparison_mask][None, :],
        )
        self.assertEqual(
            structured_valid.false_available,
            random_valid.false_available,
        )
        self.assertEqual(
            structured_valid.false_unavailable,
            random_valid.false_unavailable,
        )

    def test_complete_absence_sets_explicit_scene_flag(self) -> None:
        condition = build_quality_condition(
            np.ones((4, 4), dtype=bool),
            mode="complete-absence",
            rng=np.random.default_rng(3),
        )
        self.assertTrue(condition.complete_optical_absence)
        self.assertFalse(condition.observed.any())


if __name__ == "__main__":
    unittest.main()
