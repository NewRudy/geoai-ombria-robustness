from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_core_analysis import (  # noqa: E402
    analyze_quality_uncertainty_core,
    render_core_analysis_markdown,
)


TABLE_FILENAMES = {
    "external_seed_rows": "external_seed_rows.csv",
    "external_paired_summary": "external_paired_summary.csv",
    "ombria_seed_rows": "ombria_seed_rows.csv",
    "ombria_paired_summary": "ombria_paired_summary.csv",
    "endpoint_asymmetry": "endpoint_asymmetry.csv",
    "route_pair_contrasts": "route_pair_contrasts.csv",
    "structured_contrasts": "structured_vs_matched_random.csv",
    "hierarchical_bootstrap": "selected_hierarchical_bootstrap.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge and analyze the five audited quality-uncertainty shards."
    )
    parser.add_argument(
        "--artifact", action="append", required=True, metavar="SEED=ZIP"
    )
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-random-seed", type=int, default=20260720)
    args = parser.parse_args()
    artifacts: list[tuple[int, Path]] = []
    for value in args.artifact:
        try:
            seed_text, path_text = value.split("=", 1)
            artifacts.append((int(seed_text), Path(path_text)))
        except (TypeError, ValueError):
            parser.error(f"invalid --artifact {value!r}; expected SEED=ZIP")
    args.artifact = artifacts
    return args


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    result = analyze_quality_uncertainty_core(
        args.artifact,
        code_root=args.code_root,
        hierarchical_bootstrap_replicates=args.bootstrap_replicates,
        hierarchical_bootstrap_seed=args.bootstrap_random_seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for key, filename in TABLE_FILENAMES.items():
        if key not in result["tables"]:
            continue
        path = args.out_dir / filename
        _write_csv(path, result["tables"][key])
        outputs[filename] = {
            "rows": len(result["tables"][key]),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    report = result["report"]
    report["outputs"] = outputs
    report_path = args.out_dir / "core_analysis.json"
    markdown_path = args.out_dir / "core_analysis.md"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_core_analysis_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
