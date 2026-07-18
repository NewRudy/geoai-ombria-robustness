from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .quality_maps import (
    PerturbedQualityMap,
    dilate_unavailable,
    erode_unavailable,
    perturb_quality_map,
    quality_map_confusion,
    random_error_control,
    translate_quality_map,
)
from .sen1floods11 import Sen1Floods11Chip


@dataclass(frozen=True)
class Sen1Floods11Route:
    architecture: str
    uses_optical: bool
    uses_quality: bool
    error_aware_training: bool = False
    modality_dropout_training: bool = False


@dataclass(frozen=True)
class QualityCondition:
    observed: np.ndarray
    perturbation: PerturbedQualityMap
    complete_optical_absence: bool = False


SEN1FLOODS11_ROUTES: dict[str, Sen1Floods11Route] = {
    "s1_reference": Sen1Floods11Route("s1_only_unet", False, False),
    "early_fusion": Sen1Floods11Route("early_fusion_unet", True, False),
    "early_fusion_dropout": Sen1Floods11Route(
        "early_fusion_unet",
        True,
        False,
        modality_dropout_training=True,
    ),
    "quality_concat": Sen1Floods11Route("quality_concat_unet", True, True),
    "quality_concat_error_aware": Sen1Floods11Route(
        "quality_concat_unet",
        True,
        True,
        error_aware_training=True,
    ),
    "hard_quality_gate": Sen1Floods11Route("hard_quality_gate", True, True),
    "hard_quality_gate_error_aware": Sen1Floods11Route(
        "hard_quality_gate",
        True,
        True,
        error_aware_training=True,
    ),
    "soft_quality_prior_error_aware": Sen1Floods11Route(
        "soft_quality_prior",
        True,
        True,
        error_aware_training=True,
    ),
}


def route_config(route: str) -> Sen1Floods11Route:
    try:
        return SEN1FLOODS11_ROUTES[route]
    except KeyError as exc:
        choices = ", ".join(SEN1FLOODS11_ROUTES)
        raise ValueError(
            f"Unknown Sen1Floods11 route {route!r}; choose from {choices}"
        ) from exc


def build_observed_quality(
    reference_quality: np.ndarray,
    false_available_rate: float,
    false_unavailable_rate: float,
    rng: np.random.Generator,
) -> PerturbedQualityMap:
    return perturb_quality_map(
        reference_quality,
        false_available_rate=false_available_rate,
        false_unavailable_rate=false_unavailable_rate,
        rng=rng,
    )


def build_quality_condition(
    reference_quality: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    false_available_rate: float = 0.0,
    false_unavailable_rate: float = 0.0,
    shift_y: int = 0,
    shift_x: int = 0,
    radius: int = 0,
    matched_source_mode: str = "translate",
    comparison_mask: np.ndarray | None = None,
) -> QualityCondition:
    """Apply one frozen quality-map error condition."""

    reference = np.asarray(reference_quality, dtype=bool)
    if mode == "reference":
        perturbation = PerturbedQualityMap(
            observed=reference.copy(),
            confusion=quality_map_confusion(reference, reference),
        )
    elif mode == "independent":
        perturbation = build_observed_quality(
            reference,
            false_available_rate,
            false_unavailable_rate,
            rng,
        )
    elif mode == "translate":
        perturbation = translate_quality_map(reference, shift_y, shift_x)
    elif mode == "dilate":
        perturbation = dilate_unavailable(reference, radius)
    elif mode == "erode":
        perturbation = erode_unavailable(reference, radius)
    elif mode == "matched-random":
        if matched_source_mode == "translate":
            target = translate_quality_map(reference, shift_y, shift_x)
        elif matched_source_mode == "dilate":
            target = dilate_unavailable(reference, radius)
        elif matched_source_mode == "erode":
            target = erode_unavailable(reference, radius)
        else:
            raise ValueError("matched_source_mode must be translate, dilate, or erode")
        perturbation = random_error_control(
            reference,
            target.observed,
            rng,
            comparison_mask=comparison_mask,
        )
    elif mode == "complete-absence":
        observed = np.zeros_like(reference, dtype=bool)
        perturbation = PerturbedQualityMap(
            observed=observed,
            confusion=quality_map_confusion(reference, observed),
        )
        return QualityCondition(
            observed=observed,
            perturbation=perturbation,
            complete_optical_absence=True,
        )
    else:
        raise ValueError(f"Unknown quality condition mode: {mode}")
    return QualityCondition(
        observed=perturbation.observed,
        perturbation=perturbation,
    )


def build_route_input(
    chip: Sen1Floods11Chip,
    route: str,
    observed_quality: np.ndarray | None = None,
    complete_optical_absence: bool = False,
) -> np.ndarray:
    """Build one route input without changing the original chip arrays."""

    config = route_config(route)
    s2 = chip.image[:4].copy()
    s1 = chip.image[4:6].copy()
    if complete_optical_absence:
        s2.fill(0.0)
    if not config.uses_optical:
        return s1.astype(np.float32, copy=False)
    if not config.uses_quality:
        return np.concatenate([s2, s1], axis=0).astype(np.float32, copy=False)

    quality = (
        chip.reference_quality
        if observed_quality is None
        else np.asarray(observed_quality)
    )
    if quality.shape != chip.reference_quality.shape:
        raise ValueError("observed_quality must match the chip quality shape")
    if not np.all(np.isin(quality, (0, 1))):
        raise ValueError("observed_quality must be binary")
    quality = quality.astype(np.float32, copy=True)
    if complete_optical_absence:
        quality.fill(0.0)
    return np.concatenate([s2, s1, quality[None, :, :]], axis=0).astype(
        np.float32,
        copy=False,
    )


def augment_spatially(
    image: np.ndarray,
    target: np.ndarray,
    valid_target: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply deterministic right-angle augmentation to every aligned array."""

    image = np.asarray(image)
    target = np.asarray(target)
    valid_target = np.asarray(valid_target)
    if image.ndim != 3 or target.ndim != 2 or valid_target.ndim != 2:
        raise ValueError("Expected [C,H,W] image and 2D target masks")
    if image.shape[1:] != target.shape or target.shape != valid_target.shape:
        raise ValueError("Image, target, and valid_target shapes must align")

    rotation = int(rng.integers(0, 4))
    image = np.rot90(image, rotation, axes=(1, 2))
    target = np.rot90(target, rotation, axes=(0, 1))
    valid_target = np.rot90(valid_target, rotation, axes=(0, 1))
    if bool(rng.integers(0, 2)):
        image = np.flip(image, axis=2)
        target = np.flip(target, axis=1)
        valid_target = np.flip(valid_target, axis=1)
    if bool(rng.integers(0, 2)):
        image = np.flip(image, axis=1)
        target = np.flip(target, axis=0)
        valid_target = np.flip(valid_target, axis=0)
    return (
        np.ascontiguousarray(image),
        np.ascontiguousarray(target),
        np.ascontiguousarray(valid_target),
    )


def route_manifest() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "architecture": value.architecture,
            "uses_optical": value.uses_optical,
            "uses_quality": value.uses_quality,
            "error_aware_training": value.error_aware_training,
            "modality_dropout_training": value.modality_dropout_training,
        }
        for name, value in SEN1FLOODS11_ROUTES.items()
    }
