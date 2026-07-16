#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from geoai_ombria_robustness.sen1floods11_manifest import (
    build_manifest_document,
    build_manifest_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pinned Sen1Floods11-to-Sentinel-2-SCL manifest from "
            "official STAC metadata."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    document = build_manifest_document(build_manifest_records())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = document["summary"]
    print(
        f"Wrote {args.out}: {summary['matched_count']}/"
        f"{summary['record_count']} matched "
        f"({summary['match_fraction']:.1%})."
    )


if __name__ == "__main__":
    main()
