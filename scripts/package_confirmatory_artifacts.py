from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PATTERNS = (
    "tables/*",
    "figures/*",
    "evaluations/**/summary_metrics.csv",
    "evaluations/**/per_chip_metrics.csv",
    "evaluations/**/evaluation_config.json",
    "runs/*/config.json",
    "runs/*/splits.json",
    "runs/*/metrics.csv",
    "runtime_manifest.json",
    "experiment_manifest.json",
    "environment_freeze.txt",
    "run.log",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    experiment = json.loads((root / "experiment_manifest.json").read_text())
    seeds = experiment["model_seeds"]
    routes = experiment["routes"]
    modes = experiment["evaluation_modes"]
    expected_runs = len(seeds) * len(routes)
    expected_evaluations = expected_runs * len(modes)

    checks = {
        "training_configs": (len(list(root.glob("runs/*/config.json"))), expected_runs),
        "training_splits": (len(list(root.glob("runs/*/splits.json"))), expected_runs),
        "training_metrics": (len(list(root.glob("runs/*/metrics.csv"))), expected_runs),
        "evaluation_configs": (
            len(list(root.glob("evaluations/**/evaluation_config.json"))),
            expected_evaluations,
        ),
        "summary_metrics": (
            len(list(root.glob("evaluations/**/summary_metrics.csv"))),
            expected_evaluations,
        ),
        "per_chip_metrics": (
            len(list(root.glob("evaluations/**/per_chip_metrics.csv"))),
            expected_evaluations,
        ),
    }
    failed = {name: counts for name, counts in checks.items() if counts[0] != counts[1]}
    if failed:
        raise RuntimeError(f"Artifact completeness gate failed: {failed}")

    files: list[Path] = []
    for pattern in PATTERNS:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    files = sorted(set(files))
    project_root = root.parents[1]
    manifest = {
        "schema": "geoai-ombria-confirmatory-artifact-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completeness_checks": checks,
        "file_count_excluding_manifest": len(files),
        "files": [
            {
                "path": str(path.relative_to(project_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    manifest_path = root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, arcname=path.relative_to(project_root))
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "artifact": str(args.out),
                "bytes": args.out.stat().st_size,
                "files": len(files) + 1,
                "completeness": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
