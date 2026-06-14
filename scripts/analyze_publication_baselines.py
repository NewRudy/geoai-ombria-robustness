from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("test_iou", "test_f1", "test_precision", "test_recall")
VARIANT_ORDER = ("s1_bitemporal", "s2_bitemporal", "multimodal")
DEGRADATION_ORDER = (
    "none",
    "patch_after",
    "cloud_after_30",
    "cloud_after_50",
    "cloud_after_70",
    "noise_after",
    "zero_after",
    "zero_all",
)


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mu = mean(values)
    assert mu is not None
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def route_label(row: dict[str, str]) -> str:
    variant = row.get("variant", "")
    train_mode = row.get("train_degrade_s2", "none") or "none"
    s2_quality = row.get("s2_quality", "none") or "none"
    if variant == "s1_bitemporal":
        return "S1 bitemporal"
    if variant == "s2_bitemporal":
        return "S2 bitemporal"
    if variant == "multimodal" and train_mode == "none":
        return "Clean multimodal"
    if variant == "multimodal" and train_mode == "modality_dropout_light":
        return "Light degradation training"
    if variant == "multimodal" and train_mode == "quality_dropout_light":
        if s2_quality == "binary":
            return "Quality-aware degradation training"
        return "Quality dropout training"
    return f"{variant}:{train_mode}:{s2_quality}"


def include_row(row: dict[str, str]) -> bool:
    if row.get("record_type") != "eval":
        return False
    variant = row.get("variant", "")
    train_mode = row.get("train_degrade_s2", "none") or "none"
    if variant in {"s1_bitemporal", "s2_bitemporal"}:
        return True
    if variant == "multimodal" and train_mode in {
        "none",
        "modality_dropout_light",
        "quality_dropout_light",
    }:
        return True
    return False


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if include_row(row):
            groups[(row.get("degrade_s2", "none") or "none", route_label(row))].append(row)

    route_order = (
        "S1 bitemporal",
        "S2 bitemporal",
        "Clean multimodal",
        "Light degradation training",
        "Quality-aware degradation training",
    )
    summary: list[dict[str, str]] = []
    for degradation in DEGRADATION_ORDER:
        for route in route_order:
            values = groups.get((degradation, route), [])
            if not values:
                continue
            record: dict[str, str] = {
                "degrade_s2": degradation,
                "route": route,
                "n": str(len(values)),
                "seeds": ",".join(sorted({str(row.get("seed", "")) for row in values})),
            }
            for metric in METRICS:
                metric_values = [
                    value
                    for value in (as_float(row.get(metric)) for row in values)
                    if value is not None
                ]
                record[f"{metric}_mean"] = fmt(mean(metric_values))
                record[f"{metric}_std"] = fmt(sample_std(metric_values))
            summary.append(record)
    return summary


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Test S2 degradation | Route | n | IoU mean | IoU std | F1 mean | F1 std | Precision mean | Recall mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("degrade_s2", ""),
                    row.get("route", ""),
                    row.get("n", ""),
                    row.get("test_iou_mean", ""),
                    row.get("test_iou_std", ""),
                    row.get("test_f1_mean", ""),
                    row.get("test_f1_std", ""),
                    row.get("test_precision_mean", ""),
                    row.get("test_recall_mean", ""),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/publication_upgrade_run_summary.csv"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("results/tables/publication_upgrade_baseline_summary.csv"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/tables/publication_upgrade_baseline_summary.md"),
    )
    args = parser.parse_args()

    rows = summarize(read_rows(args.summary))
    write_csv(rows, args.out_csv)
    write_markdown(rows, args.out_md)
    print(f"wrote {len(rows)} rows to {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
