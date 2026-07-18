from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_evidence import (  # noqa: E402
    summarize_seed_conditions,
)
from geoai_ombria_robustness.quality_uncertainty_experiment import (  # noqa: E402
    build_experiment_plan,
    evaluation_conditions_for_route,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Sen1Floods11 quality-map uncertainty experiment "
            "with resumable training and matrix evaluation."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
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


def run(command: list[str], dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def checkpoint_is_complete(
    run_dir: Path,
    expected_config: dict[str, object] | None = None,
) -> bool:
    checkpoint = run_dir / "best_clean.pt"
    manifest_path = run_dir / "checkpoint_manifest.json"
    config_path = run_dir / "config.json"
    if (
        not checkpoint.is_file()
        or not manifest_path.is_file()
        or not config_path.is_file()
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("best_clean_sha256") != file_sha256(checkpoint):
        return False
    if expected_config is not None:
        for name, expected in expected_config.items():
            if configuration.get(name) != expected:
                return False
    return True


def evaluation_is_complete(
    out_dir: Path,
    expected_conditions: list[dict[str, object]],
    repetitions: int,
    route: str,
    seed: int,
    split: str,
    checkpoint_sha256: str,
    manifest_sha256: str,
    perturb_seed: int,
) -> bool:
    required = (
        out_dir / "evaluation_config.json",
        out_dir / "summary_metrics.csv",
        out_dir / "per_chip_metrics.csv",
        out_dir / "per_event_metrics.csv",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        configuration = json.loads(required[0].read_text(encoding="utf-8"))
        with required[1].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, json.JSONDecodeError):
        return False
    if (
        configuration.get("conditions") != expected_conditions
        or configuration.get("checkpoint_sha256") != checkpoint_sha256
        or configuration.get("manifest_sha256") != manifest_sha256
        or configuration.get("repetitions") != repetitions
        or configuration.get("perturb_seed") != perturb_seed
    ):
        return False
    try:
        wrong_identity = any(
            row.get("route") != route
            or int(row.get("model_seed", -1)) != seed
            or row.get("split") != split
            for row in rows
        )
    except (TypeError, ValueError):
        return False
    if wrong_identity:
        return False
    counts = Counter(row.get("condition_id", "") for row in rows)
    return counts == Counter(
        {
            str(condition["condition_id"]): repetitions
            for condition in expected_conditions
        }
    )


def read_summary_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("summary_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No evaluation summaries found under {root}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def current_source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    plan = build_experiment_plan(args.mode)
    result_root = args.result_root.resolve()
    data_root = args.data_root.resolve()
    source_manifest = args.source_manifest.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    runs_dir = result_root / "runs"
    evaluations_dir = result_root / "evaluations"
    conditions_dir = result_root / "conditions"
    tables_dir = result_root / "tables"
    for directory in (runs_dir, evaluations_dir, conditions_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    plan_document = {
        **plan.to_dict(),
        "source_commit": current_source_commit(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": file_sha256(source_manifest),
        "official_smagnet": {
            "repository": "https://github.com/ASUcicilab/SMAGNet",
            "commit": "4371df08e6ca3b9d71c0385ad57b589830469a0c",
            "license": "MIT",
            "status": "required_full_feasibility_gate_not_executed_by_smoke",
        },
    }
    write_json_atomic(result_root / "experiment_plan.json", plan_document)
    condition_paths: dict[str, Path] = {}
    for route in plan.routes:
        conditions_path = conditions_dir / f"{route}.json"
        write_json_atomic(
            conditions_path,
            {
                "schema": "geoai-quality-uncertainty-evaluation-conditions-v1",
                "mode": plan.mode,
                "route": route,
                "pipeline_only": plan.pipeline_only,
                "conditions": evaluation_conditions_for_route(plan, route),
            },
        )
        condition_paths[route] = conditions_path

    selected_manifest = result_root / "sen1floods11_selected_manifest.json"
    preparation_report = result_root / "sen1floods11_preparation_report.json"
    run(
        [
            sys.executable,
            "scripts/prepare_sen1floods11_experiment.py",
            "--mode",
            plan.mode,
            "--source-manifest",
            str(source_manifest),
            "--data-root",
            str(data_root),
            "--out-manifest",
            str(selected_manifest),
            "--out-report",
            str(preparation_report),
            "--workers",
            str(args.workers),
        ],
        args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(plan_document, indent=2))
        return

    loader_workers = min(2, args.workers)
    for route in plan.routes:
        conditions = evaluation_conditions_for_route(plan, route)
        conditions_path = condition_paths[route]
        for seed in plan.seeds:
            run_name = f"{route}_seed{seed}"
            run_dir = runs_dir / run_name
            expected_config = {
                "route": route,
                "seed": seed,
                "epochs": plan.epochs,
                "base_channels": plan.base_channels,
                "manifest_sha256": file_sha256(selected_manifest),
                "train_quality_error_rates": list(plan.error_rates),
            }
            if not checkpoint_is_complete(run_dir, expected_config):
                command = [
                    sys.executable,
                    "scripts/train_sen1floods11_unet.py",
                    "--manifest",
                    str(selected_manifest),
                    "--data-root",
                    str(data_root),
                    "--route",
                    route,
                    "--out-dir",
                    str(runs_dir),
                    "--run-name",
                    run_name,
                    "--epochs",
                    str(plan.epochs),
                    "--batch-size",
                    str(plan.batch_size),
                    "--base-channels",
                    str(plan.base_channels),
                    "--seed",
                    str(seed),
                    "--loader-seed",
                    str(seed + 200_000),
                    "--augmentation-seed",
                    str(seed + 300_000),
                    "--quality-error-seed",
                    str(seed + 400_000),
                    "--train-quality-error-rates",
                    *[str(rate) for rate in plan.error_rates],
                    "--num-workers",
                    str(loader_workers),
                ]
                run(command, False)
            for split in plan.evaluation_splits:
                out_dir = evaluations_dir / route / f"seed{seed}" / split
                if evaluation_is_complete(
                    out_dir,
                    expected_conditions=conditions,
                    repetitions=plan.perturbation_repetitions,
                    route=route,
                    seed=seed,
                    split=split,
                    checkpoint_sha256=file_sha256(run_dir / "best_clean.pt"),
                    manifest_sha256=file_sha256(selected_manifest),
                    perturb_seed=plan.perturb_seed,
                ):
                    continue
                run(
                    [
                        sys.executable,
                        "scripts/evaluate_sen1floods11_quality_uncertainty.py",
                        "--manifest",
                        str(selected_manifest),
                        "--data-root",
                        str(data_root),
                        "--checkpoint",
                        str(run_dir / "best_clean.pt"),
                        "--route",
                        route,
                        "--split",
                        split,
                        "--conditions-json",
                        str(conditions_path),
                        "--perturb-seed",
                        str(plan.perturb_seed),
                        "--repetitions",
                        str(plan.perturbation_repetitions),
                        "--batch-size",
                        str(plan.batch_size),
                        "--num-workers",
                        str(loader_workers),
                        "--out-dir",
                        str(out_dir),
                    ],
                    False,
                )

    raw_rows = read_summary_rows(evaluations_dir)
    seed_rows = summarize_seed_conditions(raw_rows)
    write_csv(tables_dir / "sen1floods11_seed_condition_summary.csv", seed_rows)
    expected_runs = len(plan.routes) * len(plan.seeds)
    expected_seed_conditions = (
        sum(len(evaluation_conditions_for_route(plan, route)) for route in plan.routes)
        * len(plan.seeds)
        * len(plan.evaluation_splits)
    )
    complete_training_runs = sum(
        checkpoint_is_complete(
            runs_dir / f"{route}_seed{seed}",
            {
                "route": route,
                "seed": seed,
                "epochs": plan.epochs,
                "base_channels": plan.base_channels,
                "manifest_sha256": file_sha256(selected_manifest),
                "train_quality_error_rates": list(plan.error_rates),
            },
        )
        for route in plan.routes
        for seed in plan.seeds
    )
    preparation = json.loads(preparation_report.read_text(encoding="utf-8"))
    selected = json.loads(selected_manifest.read_text(encoding="utf-8"))
    preparation_pass = (
        preparation.get("status") == "pass"
        and preparation.get("mode") == plan.mode
        and preparation.get("record_count")
        == selected.get("summary", {}).get("record_count")
    )
    finite = all(
        math.isfinite(float(row[metric]))
        for row in seed_rows
        for metric in ("iou", "event_equal_iou", "delta_s1_iou")
    )
    gate = {
        "schema": "geoai-sen1floods11-quality-uncertainty-gate-v1",
        "status": (
            "pass"
            if len(seed_rows) == expected_seed_conditions
            and finite
            and complete_training_runs == expected_runs
            and preparation_pass
            else "fail"
        ),
        "mode": plan.mode,
        "pipeline_only": plan.pipeline_only,
        "scientific_interpretation_allowed": False,
        "post_run_scientific_audit_required": True,
        "expected_training_runs": expected_runs,
        "complete_training_runs": complete_training_runs,
        "preparation_pass": preparation_pass,
        "expected_seed_condition_rows": expected_seed_conditions,
        "seed_condition_rows": len(seed_rows),
        "finite_primary_metrics": finite,
        "claim_boundary": (
            "A passing Smoke gate validates execution and packaging only; "
            "its scores are prohibited from manuscript claims."
            if plan.pipeline_only
            else "Full completeness does not itself establish a scientific claim."
        ),
    }
    write_json_atomic(result_root / "sen1floods11_decision_gate.json", gate)
    if gate["status"] != "pass":
        raise RuntimeError(f"Sen1Floods11 experiment gate failed: {gate}")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
