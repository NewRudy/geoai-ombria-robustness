from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.append(str(Path(__file__).resolve().parent))

from summarize_confirmatory_events import ci95  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--decision-json", type=Path, required=True)
    return parser.parse_args()


def read_seed_scores(root: Path) -> dict[tuple[str, str, str, int], float]:
    repetitions: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for path in sorted(root.glob("**/summary_metrics.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["event"] != "ALL":
                    continue
                key = (
                    row["route"],
                    row.get("checkpoint_policy", "clean"),
                    row["degrade_s2"],
                    int(row["model_seed"]),
                )
                repetitions[key].append(float(row["iou"]))
    if not repetitions:
        raise FileNotFoundError(f"No pooled event rows below {root}")
    return {key: mean(values) for key, values in repetitions.items()}


def route_composite(
    scores: dict[tuple[str, str, str, int], float],
    route: str,
    policy: str,
    modes: tuple[str, ...],
    seeds: list[int],
) -> dict[int, float]:
    composite: dict[int, float] = {}
    for seed in seeds:
        values = [scores.get((route, policy, mode, seed)) for mode in modes]
        if all(value is not None for value in values):
            composite[seed] = mean(float(value) for value in values)
    return composite


def paired_summary(
    left: dict[int, float], right: dict[int, float]
) -> dict[str, object]:
    paired_seeds = sorted(set(left) & set(right))
    if not paired_seeds:
        return {
            "model_seeds": 0,
            "left_mean": None,
            "right_mean": None,
            "paired_difference_mean": None,
            "paired_ci95_half_width": None,
            "paired_ci95_lower": None,
            "paired_ci95_upper": None,
            "positive_seed_differences": 0,
            "seed_differences": {},
        }
    differences = [left[seed] - right[seed] for seed in paired_seeds]
    center = mean(differences)
    half_width = ci95(differences)
    return {
        "model_seeds": len(paired_seeds),
        "left_mean": mean(left[seed] for seed in paired_seeds),
        "right_mean": mean(right[seed] for seed in paired_seeds),
        "paired_difference_mean": center,
        "paired_ci95_half_width": half_width,
        "paired_ci95_lower": None if half_width is None else center - half_width,
        "paired_ci95_upper": None if half_width is None else center + half_width,
        "positive_seed_differences": sum(value > 0 for value in differences),
        "seed_differences": {
            str(seed): left[seed] - right[seed] for seed in paired_seeds
        },
    }


def classify_positive_contrast(summary: dict[str, object]) -> str:
    count = int(summary["model_seeds"])
    center = summary["paired_difference_mean"]
    lower = summary["paired_ci95_lower"]
    positives = int(summary["positive_seed_differences"])
    if count < 5 or center is None:
        return "not_evaluable"
    if float(center) > 0 and lower is not None and float(lower) > 0:
        return "superiority_supported"
    if float(center) > 0 and positives >= math.ceil(0.8 * count):
        return "descriptive_consistency_only"
    return "not_supported"


def build_contrast(
    name: str,
    left_route: str,
    right_route: str,
    modes: tuple[str, ...],
    policy: str,
    scores: dict[tuple[str, str, str, int], float],
    seeds: list[int],
) -> dict[str, object]:
    summary = paired_summary(
        route_composite(scores, left_route, policy, modes, seeds),
        route_composite(scores, right_route, policy, modes, seeds),
    )
    return {
        "contrast": name,
        "left_route": left_route,
        "right_route": right_route,
        "checkpoint_policy": policy,
        "modes": " ".join(modes),
        **summary,
    }


def format_number(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    scores = read_seed_scores(args.evaluations_dir)
    seeds = [int(seed) for seed in manifest["model_seeds"]]
    policies = [str(policy) for policy in manifest["checkpoint_policies"]]
    primary_modes = tuple(
        manifest.get("primary_modes")
        or ("patch_after", "cloud_after_30", "cloud_after_50", "cloud_after_70")
    )
    definitions = (
        (
            "architecture_partial",
            "quality_gated",
            "quality_concat",
            primary_modes,
        ),
        (
            "localization_partial",
            "quality_gated",
            "gated_misaligned",
            primary_modes,
        ),
        (
            "information_partial",
            "quality_gated",
            "matched_control",
            primary_modes,
        ),
        ("clean_preservation", "quality_gated", "quality_concat", ("none",)),
        ("fallback_consistency", "quality_gated", "s1_reference", ("zero_all",)),
    )
    rows = [
        build_contrast(name, left, right, modes, policy, scores, seeds)
        for policy in policies
        for name, left, right, modes in definitions
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [
        {key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value for key, value in row.items()}
        for row in rows
    ]
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)

    clean_rows = {
        str(row["contrast"]): row
        for row in rows
        if row["checkpoint_policy"] == "clean"
    }
    mode = str(manifest["mode"])
    if mode == "smoke":
        decision = {
            "status": "pipeline_only",
            "scientific_interpretation": "prohibited",
            "reason": "Smoke has one model seed and exists only to validate execution and packaging.",
        }
    else:
        architecture = classify_positive_contrast(clean_rows["architecture_partial"])
        localization = classify_positive_contrast(clean_rows["localization_partial"])
        information = classify_positive_contrast(clean_rows["information_partial"])
        clean_difference = clean_rows["clean_preservation"][
            "paired_difference_mean"
        ]
        fallback_difference = clean_rows["fallback_consistency"][
            "paired_difference_mean"
        ]
        clean_preserved = (
            clean_difference is not None and float(clean_difference) >= -0.02
        )
        fallback_consistent = (
            fallback_difference is not None and abs(float(fallback_difference)) <= 0.02
        )
        strong = (
            architecture == "superiority_supported"
            and localization == "superiority_supported"
            and information == "superiority_supported"
            and clean_preserved
            and fallback_consistent
        )
        weak = (
            architecture
            in {"superiority_supported", "descriptive_consistency_only"}
            and localization
            in {"superiority_supported", "descriptive_consistency_only"}
            and information
            in {"superiority_supported", "descriptive_consistency_only"}
            and clean_preserved
            and fallback_consistent
        )
        decision = {
            "status": (
                "strong_method_claim_supported"
                if strong
                else "descriptive_method_consistency_only"
                if weak
                else "method_claim_not_supported"
            ),
            "architecture_partial": architecture,
            "localization_partial": localization,
            "information_partial": information,
            "clean_preserved_margin_minus_0_02": clean_preserved,
            "fallback_within_absolute_0_02": fallback_consistent,
        }
    decision_payload = {
        "schema": "geoai-ombria-quality-gated-decision-v1",
        "protocol": manifest.get("protocol"),
        "checkpoint_policy_for_decision": "clean",
        "decision": decision,
        "contrasts": clean_rows,
    }
    args.decision_json.write_text(json.dumps(decision_payload, indent=2) + "\n")

    lines = [
        "# Prespecified Quality-Gated Fusion Contrasts",
        "",
        "Differences are paired by model seed. The partial-state composite averages patch and cloud-like 30/50/70 IoU within each seed before differencing. Smoke scores are pipeline checks only.",
        "",
        "| Contrast | Checkpoint | Modes | Seeds | Left mean | Right mean | Paired difference | 95% run-level interval | Positive seeds |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        center = format_number(row["paired_difference_mean"])
        lower = format_number(row["paired_ci95_lower"])
        upper = format_number(row["paired_ci95_upper"])
        interval = "not estimable" if lower == "NA" else f"[{lower}, {upper}]"
        lines.append(
            f"| {row['contrast']} | {row['checkpoint_policy']} | {row['modes']} | "
            f"{row['model_seeds']} | {format_number(row['left_mean'])} | "
            f"{format_number(row['right_mean'])} | {center} | {interval} | "
            f"{row['positive_seed_differences']}/{row['model_seeds']} |"
        )
    lines.extend(
        [
            "",
            f"Decision status: **{decision['status']}**.",
            "",
            "The decision applies only to the prespecified OMBRIA follow-up and does not establish observed-cloud or cross-dataset robustness.",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(decision, sort_keys=True))


if __name__ == "__main__":
    main()
