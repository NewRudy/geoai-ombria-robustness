from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_experiment import (  # noqa: E402
    build_experiment_plan,
    build_selected_manifest,
)
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    download_hand_labeled_assets,
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    local_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and prepare the Sen1Floods11 records used by one "
            "quality-map uncertainty experiment."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare_record(
    record: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    try:
        chip = load_hand_labeled_chip(record, data_root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to prepare Sen1Floods11 chip {record['chip_id']}"
        ) from exc
    if chip.image.shape != (7, 512, 512):
        raise RuntimeError(
            f"Unexpected image shape for {record['chip_id']}: {chip.image.shape}"
        )
    if chip.target.shape != (512, 512):
        raise RuntimeError(f"Unexpected target shape for {record['chip_id']}")
    if not np.isfinite(chip.image).all():
        raise RuntimeError(f"Non-finite input values for {record['chip_id']}")
    paths = local_paths(data_root, record)
    return {
        "chip_id": str(record["chip_id"]),
        "event": str(record["event"]),
        "split": str(record["split"]),
        "providers": sorted({str(asset["provider"]) for asset in record["scl_assets"]}),
        "scl_asset_count": len(record["scl_assets"]),
        "valid_target_pixels": int(chip.valid_target.sum()),
        "flood_pixels": int((chip.target.astype(bool) & chip.valid_target).sum()),
        "available_quality_pixels": int(chip.reference_quality.sum()),
        "unavailable_quality_pixels": int((~chip.reference_quality).sum()),
        "optical_valid_pixels": int(chip.optical_valid.sum()),
        "files": {
            "s1": {"bytes": paths.s1.stat().st_size, "sha256": file_sha256(paths.s1)},
            "s2": {"bytes": paths.s2.stat().st_size, "sha256": file_sha256(paths.s2)},
            "label": {
                "bytes": paths.label.stat().st_size,
                "sha256": file_sha256(paths.label),
            },
            "quality": {
                "bytes": paths.quality.stat().st_size,
                "sha256": file_sha256(paths.quality),
            },
        },
    }


def aggregate_preparation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "chips": 0,
            "valid_target_pixels": 0,
            "flood_pixels": 0,
            "available_quality_pixels": 0,
            "unavailable_quality_pixels": 0,
        }
    )
    for row in rows:
        summary = by_split[str(row["split"])]
        summary["chips"] += 1
        for name in (
            "valid_target_pixels",
            "flood_pixels",
            "available_quality_pixels",
            "unavailable_quality_pixels",
        ):
            summary[name] += int(row[name])
    for summary in by_split.values():
        denominator = (
            summary["available_quality_pixels"] + summary["unavailable_quality_pixels"]
        )
        summary["unavailable_quality_fraction"] = (
            summary["unavailable_quality_pixels"] / denominator if denominator else 0.0
        )
    return dict(sorted(by_split.items()))


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    plan = build_experiment_plan(args.mode)
    source = load_sen1floods11_manifest(args.source_manifest)
    selected = build_selected_manifest(
        source,
        plan,
        source_manifest_sha256=file_sha256(args.source_manifest),
    )
    write_json_atomic(args.out_manifest, selected)

    records = list(selected["records"])
    print(f"Downloading {len(records)} selected Sen1Floods11 chips", flush=True)
    download_hand_labeled_assets(records, args.data_root, workers=args.workers)
    print("Official S1/S2/label downloads complete", flush=True)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(prepare_record, record, args.data_root): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"Prepared {completed}/{len(records)}: {row['chip_id']}",
                flush=True,
            )
    rows.sort(key=lambda row: (row["split"], row["event"], row["chip_id"]))

    report = {
        "schema": "geoai-sen1floods11-preparation-report-v1",
        "status": "pass",
        "mode": plan.mode,
        "pipeline_only": plan.pipeline_only,
        "selected_manifest": str(args.out_manifest),
        "selected_manifest_sha256": file_sha256(args.out_manifest),
        "reference_quality": (
            "available Sentinel-2 L2A SCL class intersected with the official "
            "S2Hand chip valid-data mask"
        ),
        "record_count": len(rows),
        "split_summary": aggregate_preparation(rows),
        "records": rows,
    }
    write_json_atomic(args.out_report, report)
    print(json.dumps({key: report[key] for key in ("status", "mode", "record_count")}))


if __name__ == "__main__":
    main()
