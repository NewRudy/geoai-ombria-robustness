from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class QualityMapConfusion:
    """Pixel counts for an observed availability map against a reference map.

    True means optically available. A false-available pixel is therefore
    unusable in the reference map but reported as usable in the observed map.
    """

    available_pixels: int
    unavailable_pixels: int
    true_available: int
    true_unavailable: int
    false_available: int
    false_unavailable: int
    false_available_rate: float
    false_unavailable_rate: float
    quality_iou: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbedQualityMap:
    observed: np.ndarray
    confusion: QualityMapConfusion
    requested_false_available_rate: float | None = None
    requested_false_unavailable_rate: float | None = None


def _as_bool_map(quality: np.ndarray, name: str) -> np.ndarray:
    quality = np.asarray(quality)
    if quality.ndim not in {2, 3}:
        raise ValueError(f"{name} must be a 2D or 3D quality map")
    if quality.size == 0:
        raise ValueError(f"{name} must not be empty")
    if quality.dtype == np.bool_:
        return quality.copy()
    if not np.all(np.isin(quality, (0, 1))):
        raise ValueError(f"{name} must contain only binary values")
    return quality.astype(bool, copy=True)


def _validate_rate(rate: float, name: str) -> float:
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return rate


def quality_map_confusion(
    reference: np.ndarray,
    observed: np.ndarray,
) -> QualityMapConfusion:
    reference = _as_bool_map(reference, "reference")
    observed = _as_bool_map(observed, "observed")
    if reference.shape != observed.shape:
        raise ValueError("reference and observed quality maps must have equal shapes")

    available = reference
    unavailable = ~reference
    false_available = unavailable & observed
    false_unavailable = available & ~observed
    true_available = available & observed
    true_unavailable = unavailable & ~observed

    available_count = int(available.sum())
    unavailable_count = int(unavailable.sum())
    intersection = int(true_available.sum())
    union = int((reference | observed).sum())

    return QualityMapConfusion(
        available_pixels=available_count,
        unavailable_pixels=unavailable_count,
        true_available=intersection,
        true_unavailable=int(true_unavailable.sum()),
        false_available=int(false_available.sum()),
        false_unavailable=int(false_unavailable.sum()),
        false_available_rate=(
            float(false_available.sum() / unavailable_count)
            if unavailable_count
            else 0.0
        ),
        false_unavailable_rate=(
            float(false_unavailable.sum() / available_count)
            if available_count
            else 0.0
        ),
        quality_iou=float(intersection / union) if union else 1.0,
    )


def perturb_quality_map(
    reference: np.ndarray,
    false_available_rate: float,
    false_unavailable_rate: float,
    rng: np.random.Generator,
) -> PerturbedQualityMap:
    """Independently flip unavailable and available reference pixels.

    Requested rates are converted to exact counts with round and sampled
    without replacement. Returned confusion values always describe realized
    rates, including maps that have no pixels in one denominator class.
    """

    reference = _as_bool_map(reference, "reference")
    false_available_rate = _validate_rate(
        false_available_rate, "false_available_rate"
    )
    false_unavailable_rate = _validate_rate(
        false_unavailable_rate, "false_unavailable_rate"
    )

    observed = reference.copy().reshape(-1)
    reference_flat = reference.reshape(-1)
    unavailable_indices = np.flatnonzero(~reference_flat)
    available_indices = np.flatnonzero(reference_flat)
    false_available_count = int(
        round(false_available_rate * unavailable_indices.size)
    )
    false_unavailable_count = int(
        round(false_unavailable_rate * available_indices.size)
    )

    if false_available_count:
        chosen = rng.choice(
            unavailable_indices,
            size=false_available_count,
            replace=False,
        )
        observed[chosen] = True
    if false_unavailable_count:
        chosen = rng.choice(
            available_indices,
            size=false_unavailable_count,
            replace=False,
        )
        observed[chosen] = False

    observed = observed.reshape(reference.shape)
    return PerturbedQualityMap(
        observed=observed,
        confusion=quality_map_confusion(reference, observed),
        requested_false_available_rate=false_available_rate,
        requested_false_unavailable_rate=false_unavailable_rate,
    )


def random_error_control(
    reference: np.ndarray,
    target_observed: np.ndarray,
    rng: np.random.Generator,
) -> PerturbedQualityMap:
    """Create a random map with the target map's exact two error counts."""

    reference = _as_bool_map(reference, "reference")
    target_observed = _as_bool_map(target_observed, "target_observed")
    target = quality_map_confusion(reference, target_observed)

    observed = reference.copy().reshape(-1)
    reference_flat = reference.reshape(-1)
    unavailable_indices = np.flatnonzero(~reference_flat)
    available_indices = np.flatnonzero(reference_flat)

    if target.false_available:
        chosen = rng.choice(
            unavailable_indices,
            size=target.false_available,
            replace=False,
        )
        observed[chosen] = True
    if target.false_unavailable:
        chosen = rng.choice(
            available_indices,
            size=target.false_unavailable,
            replace=False,
        )
        observed[chosen] = False

    observed = observed.reshape(reference.shape)
    return PerturbedQualityMap(
        observed=observed,
        confusion=quality_map_confusion(reference, observed),
    )


def translate_quality_map(
    reference: np.ndarray,
    shift_y: int,
    shift_x: int,
    fill_available: bool = True,
) -> PerturbedQualityMap:
    """Translate a quality map without circular wraparound."""

    reference = _as_bool_map(reference, "reference")
    if reference.ndim == 2:
        translated = _translate_2d(
            reference,
            int(shift_y),
            int(shift_x),
            fill_available,
        )
    else:
        translated = np.stack(
            [
                _translate_2d(
                    reference[:, :, channel],
                    int(shift_y),
                    int(shift_x),
                    fill_available,
                )
                for channel in range(reference.shape[2])
            ],
            axis=2,
        )
    return PerturbedQualityMap(
        observed=translated,
        confusion=quality_map_confusion(reference, translated),
    )


def _translate_2d(
    quality: np.ndarray,
    shift_y: int,
    shift_x: int,
    fill_available: bool,
) -> np.ndarray:
    height, width = quality.shape
    translated = np.full_like(quality, bool(fill_available))
    source_y0 = max(0, -shift_y)
    source_y1 = min(height, height - shift_y)
    source_x0 = max(0, -shift_x)
    source_x1 = min(width, width - shift_x)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return translated
    target_y0 = source_y0 + shift_y
    target_y1 = source_y1 + shift_y
    target_x0 = source_x0 + shift_x
    target_x1 = source_x1 + shift_x
    translated[target_y0:target_y1, target_x0:target_x1] = quality[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return translated


def dilate_unavailable(
    reference: np.ndarray,
    radius: int,
) -> PerturbedQualityMap:
    """Expand unavailable regions with a square structuring element."""

    return _morph_unavailable(reference, radius, operation="dilate")


def erode_unavailable(
    reference: np.ndarray,
    radius: int,
) -> PerturbedQualityMap:
    """Shrink unavailable regions with a square structuring element."""

    return _morph_unavailable(reference, radius, operation="erode")


def _morph_unavailable(
    reference: np.ndarray,
    radius: int,
    operation: str,
) -> PerturbedQualityMap:
    reference = _as_bool_map(reference, "reference")
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if operation not in {"dilate", "erode"}:
        raise ValueError("operation must be 'dilate' or 'erode'")
    if radius == 0:
        observed = reference.copy()
    elif reference.ndim == 2:
        observed = ~_morph_binary(~reference, radius, operation)
    else:
        observed = np.stack(
            [
                ~_morph_binary(~reference[:, :, channel], radius, operation)
                for channel in range(reference.shape[2])
            ],
            axis=2,
        )
    return PerturbedQualityMap(
        observed=observed,
        confusion=quality_map_confusion(reference, observed),
    )


def _morph_binary(mask: np.ndarray, radius: int, operation: str) -> np.ndarray:
    padding_value = operation == "erode"
    padded = np.pad(
        mask,
        radius,
        mode="constant",
        constant_values=padding_value,
    )
    result = np.zeros_like(mask) if operation == "dilate" else np.ones_like(mask)
    height, width = mask.shape
    for offset_y in range(2 * radius + 1):
        for offset_x in range(2 * radius + 1):
            view = padded[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
            if operation == "dilate":
                result |= view
            else:
                result &= view
    return result
