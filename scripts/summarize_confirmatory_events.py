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


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.evaluations_dir)
    pooled_repetitions: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["event"] == "ALL":
            pooled_repetitions[(row["route"], row["degrade_s2"], int(row["model_seed"]))].append(row)

    seed_rows: list[dict[str, object]] = []
    for (route, mode, model_seed), group in sorted(pooled_repetitions.items()):
        seed_rows.append(
            {
                "route": route,
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

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["route"]), str(row["degrade_s2"]))].append(row)

    summary: list[dict[str, object]] = []
    for (route, mode), group in sorted(grouped.items()):
        ious = [float(row["iou"]) for row in group]
        f1s = [float(row["f1"]) for row in group]
        summary.append(
            {
                "route": route,
                "degrade_s2": mode,
                "model_seeds": len(group),
                "perturb_repetitions_per_seed": min(
                    int(row["perturb_repetitions"]) for row in group
                ),
                "iou_mean": mean(ious),
                "iou_ci95_run_level": ci95(ious),
                "f1_mean": mean(f1s),
                "f1_ci95_run_level": ci95(f1s),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# OMBRIA 2021 Event-Held-Out Confirmatory Summary",
        "",
        "Metrics pool confusion counts across the four released 2021 event folders within each model-seed/perturbation repetition. The displayed interval is descriptive run-level variation across model seeds after averaging perturbation repetitions within each seed.",
        "",
        "| Route | S2 state | Seeds | Perturbation repetitions/seed | IoU mean ± 95% CI | F1 mean ± 95% CI |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {route} | {degrade_s2} | {model_seeds} | {perturb_repetitions_per_seed} | "
            "{iou_mean:.4f} ± {iou_ci95_run_level:.4f} | {f1_mean:.4f} ± {f1_ci95_run_level:.4f} |".format(**row)
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(args.out_md)


if __name__ == "__main__":
    main()
