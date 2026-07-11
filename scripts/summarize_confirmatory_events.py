from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob("**/summary_metrics.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No summary_metrics.csv files below {root}")
    return rows


T_CRITICAL_975 = {
    2: 4.302652729911275,
    4: 2.7764451051977987,
}


def ci95(values: list[float]) -> float | None:
    if len(values) == 1:
        return None
    degrees_freedom = len(values) - 1
    if degrees_freedom not in T_CRITICAL_975:
        raise ValueError(
            "Run-level intervals support exactly three or five model seeds"
        )
    return T_CRITICAL_975[degrees_freedom] * stdev(values) / math.sqrt(len(values))


def format_mean_interval(center: float, half_width: float | None) -> str:
    if half_width is None:
        return f"{center:.4f} [not estimable from one seed]"
    return f"{center:.4f} ± {half_width:.4f}"


def main() -> None:
    args = parse_args()
    rows = read_rows(args.evaluations_dir)
    pooled_repetitions: dict[tuple[str, str, str, int], list[dict[str, str]]] = (
        defaultdict(list)
    )
    for row in rows:
        if row["event"] == "ALL":
            pooled_repetitions[
                (
                    row["route"],
                    row.get("checkpoint_policy", "clean"),
                    row["degrade_s2"],
                    int(row["model_seed"]),
                )
            ].append(row)

    seed_rows: list[dict[str, object]] = []
    for (route, checkpoint_policy, mode, model_seed), group in sorted(
        pooled_repetitions.items()
    ):
        seed_rows.append(
            {
                "route": route,
                "checkpoint_policy": checkpoint_policy,
                "degrade_s2": mode,
                "model_seed": model_seed,
                "perturb_repetitions": len(group),
                "iou": mean(float(row["iou"]) for row in group),
                "f1": mean(float(row["f1"]) for row in group),
                "precision": mean(float(row["precision"]) for row in group),
                "recall": mean(float(row["recall"]) for row in group),
                "accuracy": mean(float(row["accuracy"]) for row in group),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[
            (
                str(row["route"]),
                str(row["checkpoint_policy"]),
                str(row["degrade_s2"]),
            )
        ].append(row)

    summary: list[dict[str, object]] = []
    for (route, checkpoint_policy, mode), group in sorted(grouped.items()):
        ious = [float(row["iou"]) for row in group]
        f1s = [float(row["f1"]) for row in group]
        repetition_counts = {int(row["perturb_repetitions"]) for row in group}
        if len(repetition_counts) != 1:
            raise ValueError(
                f"Perturbation repetition coverage differs across model seeds for {(route, mode)}: "
                f"{sorted(repetition_counts)}"
            )
        summary.append(
            {
                "route": route,
                "checkpoint_policy": checkpoint_policy,
                "degrade_s2": mode,
                "model_seeds": len(group),
                "perturb_repetitions_per_seed": next(iter(repetition_counts)),
                "iou_mean": mean(ious),
                "iou_ci95_run_level": ci95(ious),
                "f1_mean": mean(f1s),
                "f1_ci95_run_level": ci95(f1s),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    seed_counts = {int(row["model_seeds"]) for row in summary}
    if seed_counts == {3}:
        interval_note = (
            "The displayed interval is a two-sided 95% Student-t interval (df = 2) "
            "across three model seeds after averaging perturbation repetitions within each seed."
        )
    elif seed_counts == {5}:
        interval_note = (
            "The displayed interval is a two-sided 95% Student-t interval (df = 4) "
            "across five model seeds after averaging perturbation repetitions within each seed."
        )
    elif seed_counts == {1}:
        interval_note = (
            "This Smoke run has one model seed, so run-level uncertainty is not estimable "
            "and interval cells are reported as such."
        )
    else:
        raise ValueError(
            f"Expected one, three, or five model seeds per route/state, found {seed_counts}"
        )

    lines = [
        "# OMBRIA 2021 Event-Held-Out Confirmatory Summary",
        "",
        "Metrics pool confusion counts across the four released 2021 event folders within each model-seed/perturbation repetition. "
        + interval_note,
        "",
        "| Route | Checkpoint policy | S2 state | Seeds | Perturbation repetitions/seed | IoU mean [run-level interval] | F1 mean [run-level interval] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['route']} | {row['checkpoint_policy']} | {row['degrade_s2']} | {row['model_seeds']} | "
            f"{row['perturb_repetitions_per_seed']} | "
            f"{format_mean_interval(float(row['iou_mean']), row['iou_ci95_run_level'])} | "
            f"{format_mean_interval(float(row['f1_mean']), row['f1_ci95_run_level'])} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(args.out_md)


if __name__ == "__main__":
    main()
