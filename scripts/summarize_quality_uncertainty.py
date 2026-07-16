from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def load_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("summary_metrics.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No summary_metrics.csv files found under {root}")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row["route"],
            row["content_degradation"],
            float(row["requested_false_available_rate"]),
            float(row["requested_false_unavailable_rate"]),
        )
        grouped[key].append(row)

    s1_rows = [
        row
        for key, values in grouped.items()
        if key[0] == "s1_reference"
        for row in values
    ]
    if not s1_rows:
        raise RuntimeError("Missing s1_reference evaluation")
    s1_iou = mean([float(row["iou"]) for row in s1_rows])

    result: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        route, content, false_available, false_unavailable = key
        iou = mean([float(row["iou"]) for row in values])
        result.append(
            {
                "route": route,
                "content_degradation": content,
                "requested_false_available_rate": false_available,
                "requested_false_unavailable_rate": false_unavailable,
                "realized_false_available_rate": mean(
                    [
                        float(row["realized_false_available_rate"])
                        for row in values
                    ]
                ),
                "realized_false_unavailable_rate": mean(
                    [
                        float(row["realized_false_unavailable_rate"])
                        for row in values
                    ]
                ),
                "iou": iou,
                "s1_reference_iou": s1_iou,
                "delta_s1_iou": iou - s1_iou,
                "repetitions": len(values),
            }
        )
    return result


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Quality-map uncertainty smoke summary",
        "",
        "Smoke values validate the pipeline only and are not scientific evidence.",
        "",
        "| Route | Content state | False available | False unavailable | IoU | Δ vs S1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {route} | {content_degradation} | {fa:.2f} | {fu:.2f} | "
            "{iou:.4f} | {delta:+.4f} |".format(
                route=row["route"],
                content_degradation=row["content_degradation"],
                fa=float(row["requested_false_available_rate"]),
                fu=float(row["requested_false_unavailable_rate"]),
                iou=float(row["iou"]),
                delta=float(row["delta_s1_iou"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = summarize(load_rows(args.evaluations_dir))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, args.out_md)
    print(args.out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
