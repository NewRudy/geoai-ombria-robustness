from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_experiment import (  # noqa: E402
    build_experiment_plan,
    evaluation_conditions_for_route,
)
from geoai_ombria_robustness.quality_uncertainty_full_audit import (  # noqa: E402
    FULL_SEEDS,
)
from geoai_ombria_robustness.smagnet_adapter import (  # noqa: E402
    OFFICIAL_SMAGNET_COMMIT,
    OFFICIAL_SMAGNET_MODEL,
    OFFICIAL_SMAGNET_PAPER_DOI,
    OFFICIAL_SMAGNET_REPOSITORY,
    file_sha256,
    validate_official_smagnet_checkout,
)


ROOT = Path(__file__).resolve().parents[1]
MODE_SETTINGS = {
    "smoke": {
        "epochs": 2,
        "repetitions": 1,
        "pipeline_only": True,
    },
    "full": {
        "epochs": 200,
        "repetitions": 3,
        "pipeline_only": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SMAGNet Sen1Floods11 adaptation shard."
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smagnet-checkout", type=Path, required=True)
    parser.add_argument("--official-source-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], *, dry_run: bool) -> None:
    print("$", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def current_source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def checkpoint_complete(run_dir: Path, expected: dict[str, Any]) -> bool:
    required = (
        run_dir / "config.json",
        run_dir / "best_validation_loss.pt",
        run_dir / "checkpoint_manifest.json",
        run_dir / "threshold_selection.json",
        run_dir / "normalization.json",
        run_dir / "fallback_boundary.json",
        run_dir / "metrics.csv",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        config = json.loads(required[0].read_text(encoding="utf-8"))
        manifest = json.loads(required[2].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("best_checkpoint_sha256") != file_sha256(required[1]):
        return False
    return all(config.get(key) == value for key, value in expected.items())


def evaluation_complete(
    out_dir: Path,
    *,
    conditions: list[dict[str, Any]],
    repetitions: int,
    split: str,
    seed: int,
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
        config = json.loads(required[0].read_text(encoding="utf-8"))
        with required[1].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, json.JSONDecodeError):
        return False
    expected_counts = Counter(
        {
            str(condition["condition_id"]): repetitions
            for condition in conditions
        }
    )
    observed_counts = Counter(row.get("condition_id", "") for row in rows)
    identities_match = all(
        row.get("route") == "smagnet_official"
        and row.get("split") == split
        and int(row.get("model_seed", -1)) == seed
        for row in rows
    )
    return bool(
        config.get("conditions") == conditions
        and observed_counts == expected_counts
        and identities_match
    )


def build_decision_gate(
    args: argparse.Namespace,
    *,
    source: dict[str, Any],
    settings: dict[str, Any],
    conditions: list[dict[str, Any]],
    selected_manifest: Path,
    run_dir: Path,
    evaluation_dirs: dict[str, Path],
) -> dict[str, Any]:
    configuration = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    fallback = json.loads(
        (run_dir / "fallback_boundary.json").read_text(encoding="utf-8")
    )
    split_rows: dict[str, int] = {}
    split_chip_rows: dict[str, int] = {}
    finite = True
    for split, directory in evaluation_dirs.items():
        with (directory / "summary_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            summaries = list(csv.DictReader(handle))
        with (directory / "per_chip_metrics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            chips = list(csv.DictReader(handle))
        split_rows[split] = len(summaries)
        split_chip_rows[split] = len(chips)
        for row in summaries:
            for metric in ("iou", "f1", "precision", "recall", "accuracy"):
                try:
                    finite = finite and float("-inf") < float(row[metric]) < float(
                        "inf"
                    )
                except (KeyError, TypeError, ValueError):
                    finite = False
    selected = json.loads(selected_manifest.read_text(encoding="utf-8"))
    expected_split_samples = Counter(
        str(record["split"]) for record in selected["records"]
    )
    expected_summary_rows = len(conditions) * int(settings["repetitions"])
    complete = bool(
        source["commit"] == OFFICIAL_SMAGNET_COMMIT
        and source["source_sha256"] == file_sha256(
            args.smagnet_checkout / "src" / "smagnet.py"
        )
        and fallback.get("status") == "pass"
        and finite
        and all(rows == expected_summary_rows for rows in split_rows.values())
        and all(
            split_chip_rows[split]
            == expected_summary_rows * expected_split_samples[split]
            for split in evaluation_dirs
        )
    )
    return {
        "schema": "geoai-quality-map-uncertainty-smagnet-gate-v1",
        "status": "pass" if complete else "fail",
        "mode": args.mode,
        "pipeline_only": settings["pipeline_only"],
        "model_seed": args.seed,
        "source_commit": current_source_commit(),
        "official_source": source,
        "paper_doi": OFFICIAL_SMAGNET_PAPER_DOI,
        "architecture": OFFICIAL_SMAGNET_MODEL,
        "training": {
            "epochs": configuration["epochs"],
            "optimizer": configuration["optimizer"],
            "loss": configuration["loss"],
            "checkpoint_rule": configuration["checkpoint_rule"],
            "threshold_rule": configuration["threshold_rule"],
            "effective_batch_size": configuration["effective_batch_size"],
            "parameter_count": configuration["model_parameters"],
            "paper_default_deviations": configuration[
                "paper_default_deviations"
            ],
        },
        "fallback_boundary": fallback,
        "condition_count": len(conditions),
        "repetitions": settings["repetitions"],
        "summary_rows": split_rows,
        "per_chip_rows": split_chip_rows,
        "finite_metrics": finite,
        "scientific_interpretation_allowed": False,
        "claim_boundary": (
            "Smoke validates execution only. A Full shard remains incomplete "
            "until all five official-architecture seeds are audited and merged "
            "with the frozen S1 reference."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    settings = MODE_SETTINGS[args.mode]
    plan = build_experiment_plan(args.mode)
    if args.mode == "full" and args.seed not in plan.seeds:
        raise ValueError(f"Full SMAGNet seed must be one of {plan.seeds}")
    if args.mode == "smoke" and args.seed != 7:
        raise ValueError("SMAGNet Smoke is frozen to seed 7")
    source = validate_official_smagnet_checkout(args.smagnet_checkout)
    official_manifest = json.loads(
        args.official_source_manifest.read_text(encoding="utf-8")
    )
    if official_manifest.get("source_sha256") != source["source_sha256"]:
        raise ValueError("official source manifest is stale")
    conditions = evaluation_conditions_for_route(plan, "hard_quality_gate")
    args.result_root.mkdir(parents=True, exist_ok=True)
    selected_manifest = args.result_root / "sen1floods11_selected_manifest.json"
    preparation_report = args.result_root / "sen1floods11_preparation_report.json"
    conditions_path = args.result_root / "smagnet_conditions.json"
    write_json(
        conditions_path,
        {
            "schema": "geoai-quality-uncertainty-evaluation-conditions-v1",
            "mode": args.mode,
            "route": "smagnet_official",
            "pipeline_only": settings["pipeline_only"],
            "conditions": conditions,
        },
    )
    plan_document = {
        "schema": "geoai-quality-map-uncertainty-smagnet-plan-v1",
        "mode": args.mode,
        "pipeline_only": settings["pipeline_only"],
        "seed": args.seed,
        "planned_full_seeds": list(FULL_SEEDS),
        "epochs": settings["epochs"],
        "repetitions": settings["repetitions"],
        "official_repository": OFFICIAL_SMAGNET_REPOSITORY,
        "official_commit": OFFICIAL_SMAGNET_COMMIT,
        "paper_doi": OFFICIAL_SMAGNET_PAPER_DOI,
        "official_model": OFFICIAL_SMAGNET_MODEL,
        "condition_count": len(conditions),
        "evaluation_splits": list(plan.evaluation_splits),
        "source_commit": current_source_commit(),
    }
    write_json(args.result_root / "experiment_plan.json", plan_document)
    source_copy = args.result_root / "official_source"
    source_copy.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.smagnet_checkout / "src" / "smagnet.py", source_copy)
    shutil.copy2(args.smagnet_checkout / "LICENSE", source_copy)
    shutil.copy2(args.official_source_manifest, source_copy)

    run(
        [
            sys.executable,
            "scripts/prepare_sen1floods11_experiment.py",
            "--mode",
            args.mode,
            "--source-manifest",
            str(args.source_manifest),
            "--data-root",
            str(args.data_root),
            "--out-manifest",
            str(selected_manifest),
            "--out-report",
            str(preparation_report),
            "--workers",
            str(args.workers),
        ],
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps(plan_document, indent=2, sort_keys=True))
        return

    runs_dir = args.result_root / "runs"
    run_name = f"smagnet_official_seed{args.seed}"
    run_dir = runs_dir / run_name
    expected_config = {
        "architecture": "official_smagnet",
        "seed": args.seed,
        "epochs": settings["epochs"],
        "manifest_sha256": file_sha256(selected_manifest),
    }
    if not checkpoint_complete(run_dir, expected_config):
        command = [
            sys.executable,
            "scripts/train_sen1floods11_smagnet.py",
            "--manifest",
            str(selected_manifest),
            "--data-root",
            str(args.data_root),
            "--smagnet-source",
            str(args.smagnet_checkout / "src" / "smagnet.py"),
            "--official-source-manifest",
            str(args.official_source_manifest),
            "--out-dir",
            str(runs_dir),
            "--run-name",
            run_name,
            "--epochs",
            str(settings["epochs"]),
            "--micro-batch-size",
            str(args.micro_batch_size),
            "--gradient-accumulation",
            str(args.gradient_accumulation),
            "--seed",
            str(args.seed),
            "--loader-seed",
            str(args.seed + 200_000),
            "--augmentation-seed",
            str(args.seed + 300_000),
            "--num-workers",
            str(min(2, args.workers)),
            "--encoder-weights-msi",
            "imagenet",
        ]
        if args.amp:
            command.append("--amp")
        run(command, dry_run=False)

    checkpoint = run_dir / "best_validation_loss.pt"
    evaluation_dirs: dict[str, Path] = {}
    for split in plan.evaluation_splits:
        out_dir = args.result_root / "evaluations" / f"seed{args.seed}" / split
        evaluation_dirs[split] = out_dir
        if evaluation_complete(
            out_dir,
            conditions=conditions,
            repetitions=int(settings["repetitions"]),
            split=split,
            seed=args.seed,
        ):
            continue
        command = [
            sys.executable,
            "scripts/evaluate_sen1floods11_smagnet_quality_uncertainty.py",
            "--manifest",
            str(selected_manifest),
            "--data-root",
            str(args.data_root),
            "--checkpoint",
            str(checkpoint),
            "--smagnet-source",
            str(args.smagnet_checkout / "src" / "smagnet.py"),
            "--split",
            split,
            "--conditions-json",
            str(conditions_path),
            "--perturb-seed",
            str(plan.perturb_seed),
            "--repetitions",
            str(settings["repetitions"]),
            "--batch-size",
            str(args.micro_batch_size),
            "--out-dir",
            str(out_dir),
        ]
        if args.amp:
            command.append("--amp")
        run(command, dry_run=False)

    gate = build_decision_gate(
        args,
        source=source,
        settings=settings,
        conditions=conditions,
        selected_manifest=selected_manifest,
        run_dir=run_dir,
        evaluation_dirs=evaluation_dirs,
    )
    write_json(args.result_root / "published_architecture_gate.json", gate)
    if gate["status"] != "pass":
        raise RuntimeError("SMAGNet published-architecture gate failed")
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
