from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_full import (  # noqa: E402
    build_full_shard_plan,
    ombria_training_route_args,
)


ROOT = Path(__file__).resolve().parents[1]
OMBRIA_COMMIT = "38a490355f76da8ce27ed051138f03f3492a6e46"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one resumable OMBRIA quality-uncertainty Full seed shard."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=0)
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


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def current_source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def ombria_source_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def checkpoint_is_complete(
    run_dir: Path,
    expected_config: dict[str, object],
) -> bool:
    checkpoint = run_dir / "best_clean.pt"
    selection_path = run_dir / "checkpoint_selection.json"
    config_path = run_dir / "config.json"
    if not all(path.is_file() for path in (checkpoint, selection_path, config_path)):
        return False
    try:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        selection.get("best_clean_sha256") == file_sha256(checkpoint)
        and all(config.get(name) == value for name, value in expected_config.items())
    )


def evaluation_is_complete(
    out_dir: Path,
    *,
    route: str,
    false_available: float,
    false_unavailable: float,
    repetitions: int,
    checkpoint_sha256: str,
) -> bool:
    config_path = out_dir / "evaluation_config.json"
    summary_path = out_dir / "summary_metrics.csv"
    chip_path = out_dir / "per_chip_metrics.csv"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (config_path, summary_path, chip_path)):
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with summary_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, json.JSONDecodeError):
        return False
    try:
        return (
            config.get("route") == route
            and float(config.get("false_available_rate", -1)) == false_available
            and float(config.get("false_unavailable_rate", -1)) == false_unavailable
            and config.get("perturb_seed") == 20260716
            and config.get("repetitions") == repetitions
            and config.get("checkpoint_sha256") == checkpoint_sha256
            and len(rows) == repetitions
            and {int(row["repetition"]) for row in rows} == set(range(repetitions))
            and all(row["route"] == route for row in rows)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _rate_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def read_summary_rows(evaluations_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(evaluations_dir.rglob("summary_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    shard = build_full_shard_plan(args.seed)
    result_root = args.result_root.resolve()
    runs_dir = result_root / "runs"
    evaluations_dir = result_root / "evaluations"
    tables_dir = result_root / "tables"
    for directory in (result_root, runs_dir, evaluations_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)
    plan_document = {
        **shard.to_dict(),
        "source_commit": current_source_commit(),
        "ombria_commit": OMBRIA_COMMIT,
        "root": str(args.root.resolve()),
        "batch_size": 8,
        "base_channels": 16,
        "split_seed": 20260710,
        "perturb_seed": 20260716,
        "content_degradation": "cloud_after_50",
    }
    write_json_atomic(result_root / "experiment_plan.json", plan_document)
    if args.dry_run:
        print(json.dumps(plan_document, indent=2))
        return

    required_paths = (
        "OmbriaS1/train",
        "OmbriaS1/test",
        "OmbriaS2/train",
        "OmbriaS2/test",
    )
    missing = [name for name in required_paths if not (args.root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing OMBRIA directories: {missing}")
    actual_ombria_commit = ombria_source_commit(args.root)
    if actual_ombria_commit != OMBRIA_COMMIT:
        raise RuntimeError(
            f"OMBRIA source commit {actual_ombria_commit} != {OMBRIA_COMMIT}"
        )

    for route in shard.ombria_routes:
        run_name = f"{route}_seed{args.seed}"
        run_dir = runs_dir / run_name
        route_args = ombria_training_route_args(route, shard.error_rates)
        expected_config: dict[str, object] = {
            "run_name": run_name,
            "epochs": shard.epochs,
            "batch_size": 8,
            "base_channels": 16,
            "seed": args.seed,
            "split_seed": 20260710,
            "eval_perturb_seed": 20260716,
        }
        if "--train-quality-error-rates" in route_args:
            expected_config["train_quality_error_rates"] = list(shard.error_rates)
        else:
            expected_config["train_quality_error_rates"] = []
        if not checkpoint_is_complete(run_dir, expected_config):
            run(
                [
                    sys.executable,
                    "scripts/train_ombria_unet.py",
                    "--root",
                    str(args.root.resolve()),
                    "--out-dir",
                    str(runs_dir),
                    "--run-name",
                    run_name,
                    "--epochs",
                    str(shard.epochs),
                    "--batch-size",
                    "8",
                    "--base-channels",
                    "16",
                    "--num-workers",
                    str(args.workers),
                    "--seed",
                    str(args.seed),
                    "--split-seed",
                    "20260710",
                    "--eval-perturb-seed",
                    "20260716",
                    "--loader-seed",
                    str(args.seed + 200_000),
                    "--corruption-seed",
                    str(args.seed + 300_000),
                    "--quality-error-seed",
                    str(args.seed + 400_000),
                    "--robust-val-modes",
                    "none",
                    "cloud_after_50",
                    *route_args,
                ]
            )

    cells = {
        "s1_reference": ((0.0, 0.0),),
        **{
            route: tuple(
                (false_available, false_unavailable)
                for false_available in shard.error_rates
                for false_unavailable in shard.error_rates
            )
            for route in shard.ombria_routes
            if route != "s1_reference"
        },
    }
    for route, route_cells in cells.items():
        checkpoint = runs_dir / f"{route}_seed{args.seed}" / "best_clean.pt"
        checkpoint_hash = file_sha256(checkpoint)
        repetitions = 1 if route == "s1_reference" else shard.perturbation_repetitions
        for false_available, false_unavailable in route_cells:
            out_dir = (
                evaluations_dir
                / route
                / f"fa{_rate_key(false_available)}_fu{_rate_key(false_unavailable)}"
            )
            if evaluation_is_complete(
                out_dir,
                route=route,
                false_available=false_available,
                false_unavailable=false_unavailable,
                repetitions=repetitions,
                checkpoint_sha256=checkpoint_hash,
            ):
                continue
            run(
                [
                    sys.executable,
                    "scripts/evaluate_ombria_quality_uncertainty.py",
                    "--root",
                    str(args.root.resolve()),
                    "--checkpoint",
                    str(checkpoint),
                    "--route",
                    route,
                    "--content-degradation",
                    "cloud_after_50",
                    "--false-available-rate",
                    str(false_available),
                    "--false-unavailable-rate",
                    str(false_unavailable),
                    "--perturb-seed",
                    "20260716",
                    "--repetitions",
                    str(repetitions),
                    "--batch-size",
                    "8",
                    "--out-dir",
                    str(out_dir),
                ]
            )

    response_csv = tables_dir / "response_surface.csv"
    response_md = tables_dir / "response_surface.md"
    run(
        [
            sys.executable,
            "scripts/summarize_quality_uncertainty.py",
            "--evaluations-dir",
            str(evaluations_dir),
            "--out-csv",
            str(response_csv),
            "--out-md",
            str(response_md),
        ]
    )
    raw_rows = read_summary_rows(evaluations_dir)
    with response_csv.open(newline="", encoding="utf-8") as handle:
        response_rows = list(csv.DictReader(handle))
    complete_runs = sum(
        checkpoint_is_complete(
            runs_dir / f"{route}_seed{args.seed}",
            {
                "run_name": f"{route}_seed{args.seed}",
                "epochs": shard.epochs,
                "batch_size": 8,
                "base_channels": 16,
                "seed": args.seed,
                "split_seed": 20260710,
                "eval_perturb_seed": 20260716,
                "train_quality_error_rates": (
                    list(shard.error_rates)
                    if "--train-quality-error-rates"
                    in ombria_training_route_args(route, shard.error_rates)
                    else []
                ),
            },
        )
        for route in shard.ombria_routes
    )
    finite = all(
        math.isfinite(float(row[name]))
        for row in raw_rows
        for name in ("iou", "f1", "precision", "recall")
    )
    passed = (
        complete_runs == len(shard.ombria_routes)
        and len(raw_rows) == shard.ombria_raw_summary_rows
        and len(response_rows) == shard.ombria_evaluation_cells
        and finite
    )
    gate = {
        "schema": "geoai-ombria-quality-uncertainty-full-shard-gate-v1",
        "status": "pass" if passed else "fail",
        "active_seed": args.seed,
        "planned_seeds": list(shard.planned_seeds),
        "complete_training_runs": complete_runs,
        "expected_training_runs": len(shard.ombria_routes),
        "raw_summary_rows": len(raw_rows),
        "expected_raw_summary_rows": shard.ombria_raw_summary_rows,
        "response_surface_rows": len(response_rows),
        "expected_response_surface_rows": shard.ombria_evaluation_cells,
        "finite_primary_metrics": finite,
        "scientific_interpretation_allowed": False,
        "post_run_scientific_audit_required": True,
        "claim_boundary": (
            "One OMBRIA Full seed shard is incomplete evidence until all five "
            "frozen seeds and the cross-dataset audit are complete."
        ),
    }
    write_json_atomic(result_root / "ombria_decision_gate.json", gate)
    print(json.dumps(gate, indent=2))
    if not passed:
        raise RuntimeError(f"OMBRIA Full shard gate failed: {gate}")


if __name__ == "__main__":
    main()
