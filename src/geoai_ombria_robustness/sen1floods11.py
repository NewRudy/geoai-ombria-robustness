from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .sen1floods11_manifest import SCL_UNAVAILABLE_CLASSES


@dataclass(frozen=True)
class Sen1Floods11LocalPaths:
    s1: Path
    s2: Path
    label: Path
    quality: Path


@dataclass(frozen=True)
class Sen1Floods11Chip:
    image: np.ndarray
    target: np.ndarray
    valid_target: np.ndarray
    reference_quality: np.ndarray
    optical_valid: np.ndarray
    scl: np.ndarray


_SAS_CACHE: dict[tuple[str, str], tuple[str, datetime]] = {}
QUALITY_CACHE_SCHEMA = "sen1floods11-scl-s2-valid-v2"


def load_sen1floods11_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "geoai-sen1floods11-scl-manifest-v1":
        raise ValueError(f"Unsupported Sen1Floods11 manifest schema in {path}")
    return document


def manifest_records(
    document: dict[str, Any],
    splits: Iterable[str] | None = None,
    require_scl: bool = True,
) -> list[dict[str, Any]]:
    requested = set(splits) if splits is not None else None
    records = []
    for record in document["records"]:
        if requested is not None and record["split"] not in requested:
            continue
        if require_scl and not record["scl_assets"]:
            continue
        records.append(record)
    return records


def local_paths(root: Path, record: dict[str, Any]) -> Sen1Floods11LocalPaths:
    chip_id = str(record["chip_id"])
    split = str(record["split"])
    base = root / split
    return Sen1Floods11LocalPaths(
        s1=base / "S1" / f"{chip_id}_S1Hand.tif",
        s2=base / "S2" / f"{chip_id}_S2Hand.tif",
        label=base / "label" / f"{chip_id}_LabelHand.tif",
        quality=base / "quality" / f"{chip_id}_SCL_quality.npz",
    )


def _download(
    url: str,
    destination: Path,
    attempts: int = 5,
    backoff_seconds: float = 1.0,
) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    if attempts < 1:
        raise ValueError("attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "geoai-ombria-robustness/0.4"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180.0) as response:
                with temporary.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            os.replace(temporary, destination)
            return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(backoff_seconds * (2**attempt))


def download_hand_labeled_assets(
    records: Iterable[dict[str, Any]],
    root: Path,
    workers: int = 8,
) -> list[Sen1Floods11LocalPaths]:
    records = list(records)
    paths = [local_paths(root, record) for record in records]
    jobs: list[tuple[str, Path]] = []
    for record, record_paths in zip(records, paths, strict=True):
        jobs.extend(
            [
                (str(record["s1_url"]), record_paths.s1),
                (str(record["s2_url"]), record_paths.s2),
                (str(record["label_url"]), record_paths.label),
            ]
        )
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        list(executor.map(lambda job: _download(*job), jobs))
    return paths


def _planetary_computer_token(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    account = parsed.netloc.split(".", 1)[0]
    container = parsed.path.lstrip("/").split("/", 1)[0]
    key = (account, container)
    now = datetime.now(UTC)
    cached = _SAS_CACHE.get(key)
    if cached is not None and cached[1] - now > timedelta(minutes=5):
        return cached[0]

    token_url = (
        "https://planetarycomputer.microsoft.com/api/sas/v1/token/"
        f"{account}/{container}"
    )
    request = urllib.request.Request(
        token_url,
        headers={"User-Agent": "geoai-ombria-robustness/0.4"},
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        document = json.load(response)
    token = str(document["token"])
    expiry = datetime.fromisoformat(str(document["msft:expiry"]).replace("Z", "+00:00"))
    _SAS_CACHE[key] = (token, expiry)
    return token


def resolve_scl_href(asset: dict[str, Any]) -> str:
    href = str(asset["href"])
    if asset.get("provider") != "planetary-computer":
        return href
    separator = "&" if "?" in href else "?"
    return f"{href}{separator}{_planetary_computer_token(href)}"


def _write_quality_cache(
    output_path: Path,
    scl: np.ndarray,
    quality: np.ndarray,
    optical_valid: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            schema=np.array(QUALITY_CACHE_SCHEMA),
            scl=scl,
            quality=quality.astype(np.uint8),
            optical_valid=optical_valid.astype(np.uint8),
        )
    os.replace(temporary, output_path)


def build_scl_reference_quality(
    record: dict[str, Any],
    s2_path: Path,
    output_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject pinned L2A SCL assets onto one official S2Hand chip grid."""

    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as exc:
        raise RuntimeError(
            "Sen1Floods11 SCL preparation requires rasterio. "
            "Install the package with the sen1floods11 optional dependency."
        ) from exc

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(s2_path) as target:
            target_shape = (target.height, target.width)
            target_crs = target.crs
            target_transform = target.transform
            optical_valid = target.dataset_mask() > 0

        scl = np.zeros(target_shape, dtype=np.uint8)
        for asset in record["scl_assets"]:
            href = resolve_scl_href(asset)
            with rasterio.open(href) as source:
                warped = np.zeros(target_shape, dtype=np.uint8)
                reproject(
                    source=rasterio.band(source, 1),
                    destination=warped,
                    src_transform=source.transform,
                    src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    dst_nodata=0,
                    resampling=Resampling.nearest,
                )
            scl[(scl == 0) & (warped != 0)] = warped[(scl == 0) & (warped != 0)]

    quality = scl_reference_quality(scl, optical_valid)
    if output_path is not None:
        _write_quality_cache(
            output_path,
            scl,
            quality,
            optical_valid,
        )
    return scl, quality


def scl_reference_quality(
    scl: np.ndarray,
    optical_valid: np.ndarray,
) -> np.ndarray:
    """Combine SCL semantics with the official chip's own nodata mask."""

    scl = np.asarray(scl)
    optical_valid = np.asarray(optical_valid, dtype=bool)
    if scl.shape != optical_valid.shape:
        raise ValueError("SCL and optical-valid masks must have equal shapes")
    return (~np.isin(scl, SCL_UNAVAILABLE_CLASSES)) & optical_valid


def normalize_sentinel1(s1: np.ndarray) -> np.ndarray:
    s1 = np.asarray(s1, dtype=np.float32)
    s1 = np.nan_to_num(s1, nan=-50.0, posinf=1.0, neginf=-50.0)
    return ((np.clip(s1, -50.0, 1.0) + 50.0) / 51.0).astype(np.float32)


def normalize_sentinel2(s2: np.ndarray) -> np.ndarray:
    s2 = np.asarray(s2, dtype=np.float32)
    s2 = np.nan_to_num(s2, nan=0.0, posinf=10000.0, neginf=0.0)
    return np.clip(s2 / 10000.0, 0.0, 1.0).astype(np.float32)


def load_hand_labeled_chip(
    record: dict[str, Any],
    root: Path,
) -> Sen1Floods11Chip:
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError(
            "Sen1Floods11 loading requires rasterio. Install the package "
            "with the sen1floods11 optional dependency."
        ) from exc

    paths = local_paths(root, record)
    with rasterio.open(paths.s1) as source:
        s1 = normalize_sentinel1(source.read([1, 2]))
    with rasterio.open(paths.s2) as source:
        # B2, B3, B4, and B8 in the official 13-band ordering.
        s2 = normalize_sentinel2(source.read([2, 3, 4, 8]))
        optical_valid = source.dataset_mask() > 0
    with rasterio.open(paths.label) as source:
        label = source.read(1)

    cache_valid = False
    upgrade_cache = False
    if paths.quality.exists():
        with np.load(paths.quality) as quality_file:
            schema = (
                str(quality_file["schema"].item()) if "schema" in quality_file else ""
            )
            if schema == QUALITY_CACHE_SCHEMA:
                scl = quality_file["scl"].astype(np.uint8)
                quality = quality_file["quality"].astype(bool)
                cached_optical_valid = quality_file["optical_valid"].astype(bool)
                cache_valid = np.array_equal(cached_optical_valid, optical_valid)
            elif "scl" in quality_file:
                scl = quality_file["scl"].astype(np.uint8)
                if scl.shape == optical_valid.shape:
                    quality = scl_reference_quality(scl, optical_valid)
                    cache_valid = True
                    upgrade_cache = True
    if upgrade_cache:
        _write_quality_cache(
            paths.quality,
            scl,
            quality,
            optical_valid,
        )
    if not cache_valid:
        scl, quality = build_scl_reference_quality(
            record,
            paths.s2,
            output_path=paths.quality,
        )

    if s1.shape[1:] != s2.shape[1:] or s2.shape[1:] != label.shape:
        raise RuntimeError(f"Shape mismatch for {record['chip_id']}")
    if quality.shape != label.shape:
        raise RuntimeError(f"SCL quality shape mismatch for {record['chip_id']}")

    image = np.concatenate(
        [s2, s1, quality[None, :, :].astype(np.float32)],
        axis=0,
    )
    return Sen1Floods11Chip(
        image=image,
        target=(label == 1).astype(np.float32),
        valid_target=(label >= 0),
        reference_quality=quality,
        optical_valid=optical_valid,
        scl=scl,
    )
