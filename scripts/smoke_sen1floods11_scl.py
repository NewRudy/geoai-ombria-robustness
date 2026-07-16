from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    build_scl_reference_quality,
    download_hand_labeled_assets,
    load_sen1floods11_manifest,
    local_paths,
)


AUDIT_CHIPS = ("Spain_7370579", "India_1017769")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = load_sen1floods11_manifest(args.manifest)
    records_by_id = {
        record["chip_id"]: record for record in document["records"]
    }
    records = [records_by_id[chip_id] for chip_id in AUDIT_CHIPS]
    download_hand_labeled_assets(records, args.work_dir, workers=4)

    audit_rows = []
    for record in records:
        paths = local_paths(args.work_dir, record)
        scl, quality = build_scl_reference_quality(
            record,
            paths.s2,
            output_path=paths.quality,
        )
        values, counts = np.unique(scl, return_counts=True)
        audit_rows.append(
            {
                "chip_id": record["chip_id"],
                "event": record["event"],
                "split": record["split"],
                "providers": sorted(
                    {asset["provider"] for asset in record["scl_assets"]}
                ),
                "scl_shape": list(scl.shape),
                "scl_classes": {
                    str(int(value)): int(count)
                    for value, count in zip(values, counts, strict=True)
                },
                "unavailable_fraction": float((~quality).mean()),
                "quality_npz": str(paths.quality),
            }
        )

    summary = document["summary"]
    output = {
        "schema": "geoai-sen1floods11-scl-smoke-v1",
        "manifest_record_count": summary["record_count"],
        "manifest_matched_count": summary["matched_count"],
        "manifest_match_fraction": summary["match_fraction"],
        "unmatched_chip_ids": summary["unmatched_chip_ids"],
        "audit_rows": audit_rows,
        "status": (
            "pass"
            if summary["match_fraction"] >= 0.95
            and all(row["scl_shape"] == [512, 512] for row in audit_rows)
            else "fail"
        ),
        "boundary": (
            "SCL is an operational quality proxy, not human cloud truth."
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))
    if output["status"] != "pass":
        raise SystemExit("Sen1Floods11 SCL smoke gate failed")


if __name__ == "__main__":
    main()
