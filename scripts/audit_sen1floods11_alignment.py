from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    download_hand_labeled_assets,
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    local_paths,
    manifest_records,
)
from geoai_ombria_robustness.sen1floods11_audit import (  # noqa: E402
    AUDIT_SELECTION_SALT,
    inspect_raster_grids,
    render_alignment_panel,
    select_alignment_audit_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an outcome-independent visual and automated alignment audit "
            "for the pinned Sen1Floods11/SCL workflow."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-event", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_sen1floods11_manifest(args.manifest)
    selected = select_alignment_audit_records(
        manifest_records(document),
        per_event=args.per_event,
    )
    download_hand_labeled_assets(selected, args.work_dir, workers=args.workers)

    audit_rows: list[dict[str, object]] = []
    panel_rows = []
    for record in selected:
        try:
            paths = local_paths(args.work_dir, record)
            grid = inspect_raster_grids(paths)
            chip = load_hand_labeled_chip(record, args.work_dir)
            classes, counts = np.unique(chip.scl, return_counts=True)
            label_values, label_counts = np.unique(
                chip.target[chip.valid_target],
                return_counts=True,
            )
            row_pass = bool(
                grid["pass"]
                and chip.scl.shape == chip.target.shape == (512, 512)
                and np.isfinite(chip.image).all()
                and chip.valid_target.any()
            )
            audit_rows.append(
                {
                    "chip_id": record["chip_id"],
                    "event": record["event"],
                    "split": record["split"],
                    "s1_date": record["s1_date"],
                    "s2_date": record["s2_date"],
                    "providers": sorted(
                        {asset["provider"] for asset in record["scl_assets"]}
                    ),
                    "scl_item_ids": [
                        asset["item_id"] for asset in record["scl_assets"]
                    ],
                    "scl_asset_count": len(record["scl_assets"]),
                    "unavailable_fraction": float((~chip.reference_quality).mean()),
                    "optical_valid_fraction": float(chip.optical_valid.mean()),
                    "scl_classes": {
                        str(int(value)): int(count)
                        for value, count in zip(classes, counts, strict=True)
                    },
                    "label_counts": {
                        str(int(value)): int(count)
                        for value, count in zip(
                            label_values,
                            label_counts,
                            strict=True,
                        )
                    },
                    "grid_checks": grid,
                    "automated_status": "pass" if row_pass else "fail",
                }
            )
            panel_rows.append((record, chip))
        except Exception as exc:  # noqa: BLE001
            audit_rows.append(
                {
                    "chip_id": record["chip_id"],
                    "event": record["event"],
                    "split": record["split"],
                    "automated_status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_dir / "sen1floods11_alignment_audit.png"
    if panel_rows:
        render_alignment_panel(panel_rows, panel_path)
    all_pass = len(panel_rows) == len(selected) and all(
        row["automated_status"] == "pass" for row in audit_rows
    )
    report = {
        "schema": "geoai-sen1floods11-alignment-audit-v1",
        "selection_salt": AUDIT_SELECTION_SALT,
        "selection_rule": (
            "Stable hash sample within every event, followed by minimum additions "
            "needed to cover all splits, SCL providers, and multi-asset mosaics."
        ),
        "selected_count": len(selected),
        "selected_chip_ids": [record["chip_id"] for record in selected],
        "automated_status": "pass" if all_pass else "fail",
        "visual_status": "requires_human_review",
        "panel": str(panel_path),
        "rows": audit_rows,
        "boundary": (
            "Automated grid equality and a visual panel can detect gross alignment "
            "failures. They do not establish SCL as human cloud truth."
        ),
    }
    report_path = args.out_dir / "sen1floods11_alignment_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not all_pass:
        raise SystemExit("Sen1Floods11 alignment audit failed automated checks")


if __name__ == "__main__":
    main()
