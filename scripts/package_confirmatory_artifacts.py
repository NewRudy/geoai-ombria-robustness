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
    "runs/*/checkpoint_selection.json",
    "runtime_manifest.json",
    "experiment_manifest.json",
    "environment_freeze.txt",
    "run.log",
    "checkpoint_manifest.json",
    "split_near_duplicate_audit.json",
)

RUN_DIR_TEMPLATES = {
    "clean": "multimodal_none_seed{seed}",
    "light": "multimodal_none_train-modality_dropout_light_seed{seed}",
    "matched_control": "multimodal_none_train-quality_matched_light_seed{seed}",
    "matched_quality": "multimodal_quality-binary_none_train-quality_matched_light_seed{seed}",
    "s1_reference": "s1_bitemporal_none_seed{seed}",
    "s2_reference": "s2_bitemporal_none_seed{seed}",
    "mislocalized_quality": "multimodal_quality-mislocalized_none_train-quality_matched_light_seed{seed}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--include-checkpoints",
        action="store_true",
        help="Include the exact clean- and robust-selected weights in the archive.",
    )
    return parser.parse_args()


def checkpoint_path(run_dir: Path, policy: str) -> Path:
    candidate = run_dir / f"best_{policy}.pt"
    if policy == "clean" and not candidate.exists():
        return run_dir / "best_model.pt"
    return candidate


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
    policies = experiment.get("checkpoint_policies", ["clean"])
    modes = experiment["evaluation_modes"]
    expected_runs = len(seeds) * len(routes)
    expected_evaluations = expected_runs * len(policies) * len(modes)
    if not set(routes).issubset(RUN_DIR_TEMPLATES):
        raise RuntimeError(f"Unknown confirmatory route set: {routes}")
    run_dir_by_route_seed = {
        (route, int(seed)): root / "runs" / RUN_DIR_TEMPLATES[route].format(seed=seed)
        for route in routes
        for seed in seeds
    }
    checkpoint_by_route_seed_policy = {
        (route, int(seed), policy): checkpoint_path(
            run_dir_by_route_seed[(route, int(seed))], policy
        )
        for route in routes
        for seed in seeds
        for policy in policies
    }
    expected_checkpoints = set(checkpoint_by_route_seed_policy.values())
    actual_checkpoints = set()
    for policy in policies:
        actual_checkpoints.update(root.glob(f"runs/*/best_{policy}.pt"))
    if policies == ["clean"] and not actual_checkpoints:
        actual_checkpoints = set(root.glob("runs/*/best_model.pt"))
    checkpoints = sorted(expected_checkpoints)

    checks = {
        "training_configs": (len(list(root.glob("runs/*/config.json"))), expected_runs),
        "training_splits": (len(list(root.glob("runs/*/splits.json"))), expected_runs),
        "training_metrics": (len(list(root.glob("runs/*/metrics.csv"))), expected_runs),
        "validation_selected_checkpoints": (
            len(checkpoints),
            expected_runs * len(policies),
        ),
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
    if actual_checkpoints != expected_checkpoints:
        missing = sorted(
            str(path.relative_to(root))
            for path in expected_checkpoints - actual_checkpoints
        )
        unexpected = sorted(
            str(path.relative_to(root))
            for path in actual_checkpoints - expected_checkpoints
        )
        raise RuntimeError(
            f"Checkpoint path coverage mismatch; missing={missing}, unexpected={unexpected}"
        )

    checkpoint_hashes = {path: sha256(path) for path in checkpoints}
    evaluation_keys: list[tuple[str, int, str, str]] = []
    evaluation_counts = {path: 0 for path in checkpoints}
    evaluation_errors: list[str] = []
    for config_path in sorted(root.glob("evaluations/**/evaluation_config.json")):
        config = json.loads(config_path.read_text())
        key = (
            str(config["route"]),
            int(config["model_seed"]),
            str(config.get("checkpoint_policy", "clean")),
        )
        mode = str(config["degrade_s2"])
        evaluation_keys.append((key[0], key[1], key[2], mode))
        expected_checkpoint = checkpoint_by_route_seed_policy.get(key)
        if expected_checkpoint is None:
            evaluation_errors.append(
                f"unknown route/seed in {config_path.relative_to(root)}"
            )
            continue
        evaluation_counts[expected_checkpoint] += 1
        configured_checkpoint = Path(config["checkpoint"])
        if not configured_checkpoint.is_absolute():
            configured_checkpoint = (Path.cwd() / configured_checkpoint).resolve()
        if configured_checkpoint != expected_checkpoint.resolve():
            evaluation_errors.append(
                f"checkpoint path mismatch in {config_path.relative_to(root)}"
            )
        if config.get("checkpoint_sha256") != checkpoint_hashes[expected_checkpoint]:
            evaluation_errors.append(
                f"checkpoint hash mismatch in {config_path.relative_to(root)}"
            )
        if (
            int(config.get("checkpoint_bytes", -1))
            != expected_checkpoint.stat().st_size
        ):
            evaluation_errors.append(
                f"checkpoint size mismatch in {config_path.relative_to(root)}"
            )

    expected_evaluation_keys = {
        (route, int(seed), policy, mode)
        for route in routes
        for seed in seeds
        for policy in policies
        for mode in modes
    }
    if set(evaluation_keys) != expected_evaluation_keys or len(evaluation_keys) != len(
        expected_evaluation_keys
    ):
        evaluation_errors.append(
            "route/seed/mode evaluation-config coverage is incomplete or duplicated"
        )
    for checkpoint, count in evaluation_counts.items():
        if count != len(modes):
            evaluation_errors.append(
                f"{checkpoint.relative_to(root)} is referenced by {count} evaluations; expected {len(modes)}"
            )
    if evaluation_errors:
        raise RuntimeError(
            f"Checkpoint-to-evaluation traceability gate failed: {evaluation_errors}"
        )

    checkpoint_manifest_path = root / "checkpoint_manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(
            {
                "schema": "geoai-ombria-confirmatory-checkpoints-v2",
                "weights_included": args.include_checkpoints,
                "note": "Hashes identify every validation-selected checkpoint used for evaluation. The v0.2 Full artifact includes the exact weights when include-checkpoints is enabled.",
                "checkpoints": [
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "sha256": checkpoint_hashes[path],
                        "evaluation_config_count": evaluation_counts[path],
                    }
                    for path in checkpoints
                ],
            },
            indent=2,
        )
        + "\n"
    )

    files: list[Path] = []
    for pattern in PATTERNS:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    if args.include_checkpoints:
        files.extend(checkpoints)
    files = sorted(set(files))
    project_root = root.parents[1]
    manifest = {
        "schema": "geoai-ombria-confirmatory-artifact-v2",
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
