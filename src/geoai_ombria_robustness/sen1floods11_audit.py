from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .sen1floods11 import Sen1Floods11Chip, Sen1Floods11LocalPaths


AUDIT_SELECTION_SALT = "sen1floods11-alignment-audit-v1"


def _record_score(record: dict[str, Any], stratum: str) -> str:
    token = (f"{AUDIT_SELECTION_SALT}:{stratum}:{record['chip_id']}").encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def _providers(record: dict[str, Any]) -> set[str]:
    return {str(asset["provider"]) for asset in record.get("scl_assets", [])}


def select_alignment_audit_records(
    records: Iterable[dict[str, Any]],
    per_event: int = 1,
) -> list[dict[str, Any]]:
    """Select an outcome-independent audit panel from manifest metadata.

    The frozen rule selects a stable hash sample within every event and then
    adds the minimum number of records needed to cover every split, SCL
    provider, and multi-tile mosaicking when those strata exist.
    """

    records = list(records)
    if not records:
        raise ValueError("At least one manifest record is required")
    if per_event < 1:
        raise ValueError("per_event must be positive")

    selected: dict[str, dict[str, Any]] = {}
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_event[str(record["event"])].append(record)
    for event, values in sorted(by_event.items()):
        ranked = sorted(values, key=lambda row: _record_score(row, f"event:{event}"))
        for record in ranked[:per_event]:
            selected[str(record["chip_id"])] = record

    def add_for_stratum(
        name: str,
        values: list[dict[str, Any]],
        covered: bool,
    ) -> None:
        if covered or not values:
            return
        record = min(values, key=lambda row: _record_score(row, name))
        selected[str(record["chip_id"])] = record

    for split in sorted({str(record["split"]) for record in records}):
        values = [record for record in records if str(record["split"]) == split]
        add_for_stratum(
            f"split:{split}",
            values,
            any(str(record["split"]) == split for record in selected.values()),
        )

    providers = sorted(
        {provider for record in records for provider in _providers(record)}
    )
    for provider in providers:
        values = [record for record in records if provider in _providers(record)]
        add_for_stratum(
            f"provider:{provider}",
            values,
            any(provider in _providers(record) for record in selected.values()),
        )

    multi_asset = [
        record for record in records if len(record.get("scl_assets", [])) > 1
    ]
    add_for_stratum(
        "multi-asset",
        multi_asset,
        any(len(record.get("scl_assets", [])) > 1 for record in selected.values()),
    )
    return sorted(
        selected.values(),
        key=lambda record: (str(record["event"]), str(record["chip_id"])),
    )


def percentile_stretch(image: np.ndarray) -> np.ndarray:
    """Return a finite [0, 1] display stretch without changing source data."""

    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in {2, 3}:
        raise ValueError("image must be 2D or 3D")
    channels = image[None, :, :] if image.ndim == 2 else image
    stretched = np.zeros_like(channels, dtype=np.float32)
    for index, channel in enumerate(channels):
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            continue
        low, high = np.percentile(finite, (2.0, 98.0))
        if high <= low:
            continue
        stretched[index] = np.clip((channel - low) / (high - low), 0.0, 1.0)
    return stretched[0] if image.ndim == 2 else stretched


def audit_display_arrays(chip: Sen1Floods11Chip) -> dict[str, np.ndarray]:
    """Build explicitly labeled display arrays for human alignment review."""

    s2 = chip.image[:4]
    s1 = chip.image[4:6]
    s2_rgb = np.moveaxis(percentile_stretch(s2[[2, 1, 0]]), 0, 2)
    s1_composite = np.moveaxis(
        percentile_stretch(np.stack([s1[0], s1[1], s1.mean(axis=0)])),
        0,
        2,
    )
    unavailable_overlay = s2_rgb.copy()
    unavailable = ~chip.reference_quality
    unavailable_overlay[unavailable] = 0.35 * unavailable_overlay[
        unavailable
    ] + 0.65 * np.array([1.0, 0.35, 0.0], dtype=np.float32)
    return {
        "s1_composite": s1_composite,
        "s2_rgb": s2_rgb,
        "s2_unavailable_overlay": unavailable_overlay,
        "scl": chip.scl,
        "label": np.where(chip.valid_target, chip.target, np.nan),
    }


def inspect_raster_grids(paths: Sen1Floods11LocalPaths) -> dict[str, Any]:
    """Check that official S1, S2, and label rasters share one chip grid."""

    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Raster grid inspection requires rasterio") from exc

    metadata: dict[str, dict[str, Any]] = {}
    for name, path in (("s1", paths.s1), ("s2", paths.s2), ("label", paths.label)):
        with rasterio.open(path) as source:
            metadata[name] = {
                "shape": [source.height, source.width],
                "count": source.count,
                "crs": str(source.crs),
                "transform": [float(value) for value in tuple(source.transform)],
                "bounds": [float(value) for value in source.bounds],
                "dtypes": list(source.dtypes),
            }
    reference = metadata["s2"]
    same_shape = all(
        value["shape"] == reference["shape"] for value in metadata.values()
    )
    same_crs = all(value["crs"] == reference["crs"] for value in metadata.values())
    same_transform = all(
        np.allclose(value["transform"], reference["transform"], atol=1e-9)
        for value in metadata.values()
    )
    same_bounds = all(
        np.allclose(value["bounds"], reference["bounds"], atol=1e-6)
        for value in metadata.values()
    )
    return {
        "rasters": metadata,
        "same_shape": same_shape,
        "same_crs": same_crs,
        "same_transform": same_transform,
        "same_bounds": same_bounds,
        "pass": same_shape and same_crs and same_transform and same_bounds,
    }


def render_alignment_panel(
    rows: list[tuple[dict[str, Any], Sen1Floods11Chip]],
    output: Path,
) -> None:
    if not rows:
        raise ValueError("At least one successful audit row is required")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    scl_colors = [
        "#000000",
        "#d73027",
        "#166534",
        "#8c510a",
        "#33a02c",
        "#f1c40f",
        "#2166ac",
        "#969696",
        "#d9d9d9",
        "#ffffff",
        "#67c5e8",
        "#b2182b",
    ]
    scl_cmap = ListedColormap(scl_colors)
    figure, axes = plt.subplots(
        len(rows),
        5,
        figsize=(10.5, 2.05 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    headings = (
        "S1 VV/VH composite",
        "S2 RGB",
        "S2 + unavailable",
        "SCL class",
        "Flood label",
    )
    for column, heading in enumerate(headings):
        axes[0, column].set_title(heading, fontsize=9, fontweight="bold")
    for row_index, (record, chip) in enumerate(rows):
        arrays = audit_display_arrays(chip)
        axes[row_index, 0].imshow(arrays["s1_composite"])
        axes[row_index, 1].imshow(arrays["s2_rgb"])
        axes[row_index, 2].imshow(arrays["s2_unavailable_overlay"])
        axes[row_index, 3].imshow(
            arrays["scl"],
            cmap=scl_cmap,
            vmin=0,
            vmax=11,
            interpolation="nearest",
        )
        axes[row_index, 4].imshow(
            arrays["label"],
            cmap="Blues",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        unavailable_fraction = float((~chip.reference_quality).mean())
        axes[row_index, 0].set_ylabel(
            f"{record['event']}\n{record['chip_id']}\n"
            f"{record['split']}; unavailable={unavailable_fraction:.1%}",
            fontsize=7,
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Sen1Floods11 / Sentinel-2 L2A SCL alignment audit",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compress_level": 9},
    )
    plt.close(figure)
