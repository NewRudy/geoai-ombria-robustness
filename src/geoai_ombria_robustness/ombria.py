from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


VARIANTS = (
    "s1_after",
    "s1_bitemporal",
    "s2_after",
    "s2_bitemporal",
    "multimodal",
)


@dataclass(frozen=True)
class OmbriaSample:
    split: str
    chip_id: str
    s1_before: Path
    s1_after: Path
    s1_mask: Path
    s2_before: Path
    s2_after: Path
    s2_mask: Path


def _chip_id(path: Path) -> str:
    return path.stem.split("_")[-1]


def _index_pngs(folder: Path) -> dict[str, Path]:
    return {_chip_id(path): path for path in folder.glob("*.png")}


def collect_ombria_samples(root: Path, split: str) -> list[OmbriaSample]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unknown split: {split}")

    s1_base = root / "OmbriaS1" / split
    s2_base = root / "OmbriaS2" / split

    s1_before = _index_pngs(s1_base / "BEFORE")
    s1_after = _index_pngs(s1_base / "AFTER")
    s1_mask = _index_pngs(s1_base / "MASK")
    s2_before = _index_pngs(s2_base / "BEFORE")
    s2_after = _index_pngs(s2_base / "AFTER")
    s2_mask = _index_pngs(s2_base / "MASK")

    chip_ids = set.intersection(
        set(s1_before),
        set(s1_after),
        set(s1_mask),
        set(s2_before),
        set(s2_after),
        set(s2_mask),
    )

    return [
        OmbriaSample(
            split=split,
            chip_id=chip_id,
            s1_before=s1_before[chip_id],
            s1_after=s1_after[chip_id],
            s1_mask=s1_mask[chip_id],
            s2_before=s2_before[chip_id],
            s2_after=s2_after[chip_id],
            s2_mask=s2_mask[chip_id],
        )
        for chip_id in sorted(chip_ids)
    ]


def variant_channels(variant: str, s2_quality: str = "none") -> int:
    channels = {
        "s1_after": 1,
        "s1_bitemporal": 2,
        "s2_after": 3,
        "s2_bitemporal": 6,
        "multimodal": 8,
    }
    if s2_quality not in {"none", "binary"}:
        raise ValueError("s2_quality must be 'none' or 'binary'")
    try:
        base_channels = channels[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown variant {variant!r}; choose from {VARIANTS}"
        ) from exc
    if variant == "multimodal" and s2_quality == "binary":
        return base_channels + 2
    return base_channels


def read_image(path: Path, mode: str) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return arr


def read_mask(path: Path) -> np.ndarray:
    return (read_image(path, "L")[:, :, 0] > 0.5).astype(np.float32)


def load_sample(
    sample: OmbriaSample,
    variant: str,
    degrade_s2: str = "none",
    rng: Optional[np.random.Generator] = None,
    s2_quality: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    if s2_quality not in {"none", "binary"}:
        raise ValueError("s2_quality must be 'none' or 'binary'")

    if rng is None:
        rng = np.random.default_rng()

    s1_before = read_image(sample.s1_before, "L")
    s1_after = read_image(sample.s1_after, "L")
    s2_before = read_image(sample.s2_before, "RGB")
    s2_after = read_image(sample.s2_after, "RGB")

    s2_before, s2_after = degrade_s2_pair(s2_before, s2_after, degrade_s2, rng)

    if variant == "s1_after":
        image = s1_after
    elif variant == "s1_bitemporal":
        image = np.concatenate([s1_before, s1_after], axis=2)
    elif variant == "s2_after":
        image = s2_after
    elif variant == "s2_bitemporal":
        image = np.concatenate([s2_before, s2_after], axis=2)
    else:
        image = np.concatenate([s2_before, s2_after, s1_before, s1_after], axis=2)
        if s2_quality == "binary":
            image = np.concatenate(
                [image, s2_quality_channels(s2_before, s2_after, degrade_s2)],
                axis=2,
            )

    return image.astype(np.float32), read_mask(sample.s2_mask)


def s2_quality_channels(
    before: np.ndarray,
    after: np.ndarray,
    mode: str,
) -> np.ndarray:
    h, w, _ = before.shape
    before_quality = np.ones((h, w, 1), dtype=np.float32)
    after_quality = np.ones((h, w, 1), dtype=np.float32)
    if mode == "zero_all":
        before_quality.fill(0.0)
        after_quality.fill(0.0)
    elif mode in {"zero_after", "noise_after"}:
        after_quality.fill(0.0)
    elif mode == "patch_after" or mode.startswith("cloud_after_"):
        after_quality = (after.sum(axis=2, keepdims=True) > 0.0).astype(np.float32)
    return np.concatenate([before_quality, after_quality], axis=2)


def degrade_s2_pair(
    before: np.ndarray,
    after: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none":
        return before, after
    if mode == "zero_all":
        return np.zeros_like(before), np.zeros_like(after)
    if mode == "zero_after":
        return before, np.zeros_like(after)
    if mode == "noise_after":
        return before, rng.random(after.shape, dtype=np.float32)
    if mode == "patch_after":
        degraded = after.copy()
        h, w, _ = degraded.shape
        patch = max(16, min(h, w) // 4)
        for _ in range(8):
            y = int(rng.integers(0, h - patch + 1))
            x = int(rng.integers(0, w - patch + 1))
            degraded[y : y + patch, x : x + patch, :] = 0.0
        return before, degraded
    if mode.startswith("cloud_after_"):
        fraction = _parse_cloud_fraction(mode)
        return before, apply_cloud_like_mask(after, fraction, rng)
    raise ValueError(
        "Unknown S2 degradation mode "
        f"{mode!r}; choose none, zero_all, zero_after, noise_after, patch_after, "
        "or cloud_after_<percent>"
    )


def _parse_cloud_fraction(mode: str) -> float:
    try:
        percent = int(mode.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Invalid cloud-like degradation {mode!r}; use cloud_after_<percent>"
        ) from exc
    if percent <= 0 or percent >= 100:
        raise ValueError("cloud_after_<percent> must use 1..99")
    return percent / 100.0


def apply_cloud_like_mask(
    image: np.ndarray,
    target_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    degraded = image.copy()
    h, w, _ = degraded.shape
    mask = np.zeros((h, w), dtype=bool)
    target_pixels = int(round(h * w * target_fraction))
    attempts = 0
    while int(mask.sum()) < target_pixels and attempts < 200:
        attempts += 1
        cy = int(rng.integers(0, h))
        cx = int(rng.integers(0, w))
        radius_y = int(rng.integers(max(8, h // 18), max(12, h // 6)))
        radius_x = int(rng.integers(max(8, w // 18), max(12, w // 6)))
        y0 = max(0, cy - radius_y)
        y1 = min(h, cy + radius_y + 1)
        x0 = max(0, cx - radius_x)
        x1 = min(w, cx + radius_x + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        blob = (
            ((yy - cy) / max(radius_y, 1)) ** 2
            + ((xx - cx) / max(radius_x, 1)) ** 2
            <= 1.0
        )
        mask[y0:y1, x0:x1] |= blob
    degraded[mask, :] = 0.0
    return degraded


def summarize_samples(samples: Iterable[OmbriaSample]) -> dict[str, float]:
    samples = list(samples)
    if not samples:
        return {"count": 0, "mean_flood_fraction": 0.0}
    flood_fractions = [float(read_mask(sample.s2_mask).mean()) for sample in samples]
    return {
        "count": len(samples),
        "mean_flood_fraction": float(np.mean(flood_fractions)),
    }
