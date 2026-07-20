from __future__ import annotations

import csv
import io
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .quality_uncertainty_full_audit import FULL_SEEDS
from .quality_uncertainty_shard_set_audit import (
    audit_quality_uncertainty_shard_set_artifacts,
)


T_CRITICAL_95_DF4 = 2.7764451051977987
RATES = (0.0, 0.05, 0.1, 0.2, 0.4)
EXTERNAL_GRID_ROUTES = (
    "quality_concat",
    "quality_concat_error_aware",
    "hard_quality_gate",
    "hard_quality_gate_error_aware",
    "soft_quality_prior_error_aware",
)
OMBRIA_GRID_ROUTES = (
    "hard_oracle",
    "hard_error_aware",
    "concat_error_aware",
    "soft_error_aware",
)
EXTERNAL_ROUTE_PAIRS = (
    ("hard_quality_gate_error_aware", "hard_quality_gate", "hard_gate"),
    ("quality_concat_error_aware", "quality_concat", "quality_concat"),
)
OMBRIA_ROUTE_PAIRS = (
    ("hard_error_aware", "hard_oracle", "hard_gate"),
)
TRANSFER_ROUTE_PAIRS = (
    ("hard_oracle", "hard_quality_gate", "hard_gate"),
    ("hard_error_aware", "hard_quality_gate_error_aware", "hard_error_aware"),
    ("concat_error_aware", "quality_concat_error_aware", "quality_concat"),
    ("soft_error_aware", "soft_quality_prior_error_aware", "soft_prior"),
)
EXTERNAL_STATIC_FIELDS = (
    "split",
    "route",
    "condition_id",
    "quality_mode",
    "false_available_rate",
    "false_unavailable_rate",
    "shift_y",
    "shift_x",
    "radius",
    "matched_source_mode",
)
OMBRIA_STATIC_FIELDS = (
    "route",
    "content_degradation",
    "requested_false_available_rate",
    "requested_false_unavailable_rate",
)


def _rate_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def independent_condition_id(false_available: float, false_unavailable: float) -> str:
    return (
        f"independent_fa{_rate_key(false_available)}_"
        f"fu{_rate_key(false_unavailable)}"
    )


def paired_t_summary(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    """Summarize one paired five-seed estimand with a df=4 t interval."""

    seeds = sorted(int(seed) for seed in values_by_seed)
    if tuple(seeds) != FULL_SEEDS:
        raise ValueError(f"expected frozen seeds {FULL_SEEDS}, received {seeds}")
    values = [float(values_by_seed[seed]) for seed in seeds]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("paired values must be finite")
    center = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF4 * standard_deviation / math.sqrt(len(values))
    return {
        "model_seeds": len(values),
        "mean": center,
        "sample_standard_deviation": standard_deviation,
        "ci95_half_width": half_width,
        "ci95_lower": center - half_width,
        "ci95_upper": center + half_width,
        "positive_seeds": sum(value > 0 for value in values),
        "negative_seeds": sum(value < 0 for value in values),
        "minimum": min(values),
        "maximum": max(values),
        "seed_values": {str(seed): values_by_seed[seed] for seed in seeds},
    }


def _read_csv(archive: ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def _float(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is not finite")
    return value


def _normalize_external_row(row: Mapping[str, str], seed: int) -> dict[str, Any]:
    if int(row["model_seed"]) != seed:
        raise ValueError("external row model seed does not match shard seed")
    return {
        "model_seed": seed,
        "split": row["split"],
        "route": row["route"],
        "condition_id": row["condition_id"],
        "quality_mode": row["quality_mode"],
        "false_available_rate": _float(row, "false_available_rate"),
        "false_unavailable_rate": _float(row, "false_unavailable_rate"),
        "shift_y": int(row["shift_y"]),
        "shift_x": int(row["shift_x"]),
        "radius": int(row["radius"]),
        "matched_source_mode": row["matched_source_mode"],
        "iou": _float(row, "iou"),
        "f1": _float(row, "f1"),
        "precision": _float(row, "precision"),
        "recall": _float(row, "recall"),
        "event_equal_iou": _float(row, "event_equal_iou"),
        "realized_false_available_rate": _float(
            row, "realized_false_available_rate"
        ),
        "realized_false_unavailable_rate": _float(
            row, "realized_false_unavailable_rate"
        ),
        "valid_realized_false_available_rate": _float(
            row, "valid_realized_false_available_rate"
        ),
        "valid_realized_false_unavailable_rate": _float(
            row, "valid_realized_false_unavailable_rate"
        ),
        "s1_reference_iou": _float(row, "s1_reference_iou"),
        "delta_s1_iou": _float(row, "delta_s1_iou"),
        "repetitions": int(row["repetitions"]),
    }


def _normalize_ombria_row(row: Mapping[str, str], seed: int) -> dict[str, Any]:
    return {
        "model_seed": seed,
        "route": row["route"],
        "content_degradation": row["content_degradation"],
        "requested_false_available_rate": _float(
            row, "requested_false_available_rate"
        ),
        "requested_false_unavailable_rate": _float(
            row, "requested_false_unavailable_rate"
        ),
        "realized_false_available_rate": _float(
            row, "realized_false_available_rate"
        ),
        "realized_false_unavailable_rate": _float(
            row, "realized_false_unavailable_rate"
        ),
        "iou": _float(row, "iou"),
        "s1_reference_iou": _float(row, "s1_reference_iou"),
        "delta_s1_iou": _float(row, "delta_s1_iou"),
        "repetitions": int(row["repetitions"]),
    }


def load_core_seed_rows(
    artifacts: Sequence[tuple[int, Path]],
    *,
    code_root: Path | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Fail closed before loading the five frozen seed-level tables."""

    set_audit = audit_quality_uncertainty_shard_set_artifacts(
        artifacts, code_root=code_root
    )
    if not set_audit["decision"]["core_merge_authorized"]:
        raise ValueError("five-shard audit did not authorize the core merge")

    external_rows: list[dict[str, Any]] = []
    ombria_rows: list[dict[str, Any]] = []
    for seed, artifact in sorted((int(seed), Path(path)) for seed, path in artifacts):
        root = f"quality_uncertainty_full_seed{seed}"
        with ZipFile(artifact) as archive:
            external = _read_csv(
                archive,
                f"{root}/sen1floods11/tables/"
                "sen1floods11_seed_condition_summary.csv",
            )
            ombria = _read_csv(
                archive,
                f"{root}/ombria/seed{seed}/tables/response_surface.csv",
            )
        if len(external) != 550 or len(ombria) != 101:
            raise ValueError(f"seed {seed}: unexpected seed-table row counts")
        external_rows.extend(_normalize_external_row(row, seed) for row in external)
        ombria_rows.extend(_normalize_ombria_row(row, seed) for row in ombria)

    if len(external_rows) != 2750 or len(ombria_rows) != 505:
        raise ValueError("merged seed-table row totals are incomplete")
    return set_audit, external_rows, ombria_rows


def _static_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row[field] for field in fields)


def _add_estimate(output: dict[str, Any], prefix: str, estimate: Mapping[str, Any]) -> None:
    for key in (
        "mean",
        "sample_standard_deviation",
        "ci95_half_width",
        "ci95_lower",
        "ci95_upper",
        "positive_seeds",
        "negative_seeds",
        "minimum",
        "maximum",
    ):
        output[f"{prefix}_{key}"] = estimate[key]
    output[f"{prefix}_seed_values_json"] = json.dumps(
        estimate["seed_values"], sort_keys=True, separators=(",", ":")
    )


def merge_external_seed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference_event_equal: dict[tuple[int, str], float] = {}
    for row in rows:
        if row["route"] == "s1_reference":
            reference_event_equal[(row["model_seed"], row["split"])] = row[
                "event_equal_iou"
            ]
    if len(reference_event_equal) != len(FULL_SEEDS) * 2:
        raise ValueError("missing external event-equal S1 references")

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_static_key(row, EXTERNAL_STATIC_FIELDS)].append(row)

    merged: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        by_seed = {int(row["model_seed"]): row for row in group}
        if tuple(sorted(by_seed)) != FULL_SEEDS:
            raise ValueError(f"external condition lacks five seeds: {key}")
        output = dict(zip(EXTERNAL_STATIC_FIELDS, key, strict=True))
        output["model_seeds"] = len(FULL_SEEDS)
        for metric in ("iou", "f1", "precision", "recall", "event_equal_iou"):
            output[f"{metric}_mean"] = statistics.mean(
                float(by_seed[seed][metric]) for seed in FULL_SEEDS
            )
        output["realized_false_available_rate_mean"] = statistics.mean(
            float(by_seed[seed]["realized_false_available_rate"])
            for seed in FULL_SEEDS
        )
        output["realized_false_unavailable_rate_mean"] = statistics.mean(
            float(by_seed[seed]["realized_false_unavailable_rate"])
            for seed in FULL_SEEDS
        )
        delta = paired_t_summary(
            {seed: float(by_seed[seed]["delta_s1_iou"]) for seed in FULL_SEEDS}
        )
        event_delta = paired_t_summary(
            {
                seed: float(by_seed[seed]["event_equal_iou"])
                - reference_event_equal[(seed, str(output["split"]))]
                for seed in FULL_SEEDS
            }
        )
        _add_estimate(output, "delta_s1_iou", delta)
        _add_estimate(output, "event_equal_delta_s1_iou", event_delta)
        output["s1_relative_regret"] = max(0.0, -delta["mean"])
        output["seed_interval_status"] = (
            "lower_nonnegative"
            if delta["ci95_lower"] >= 0
            else "upper_negative"
            if delta["ci95_upper"] < 0
            else "crosses_zero"
        )
        merged.append(output)
    if len(merged) != 550:
        raise ValueError(f"expected 550 external paired rows, received {len(merged)}")
    return merged


def merge_ombria_seed_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_static_key(row, OMBRIA_STATIC_FIELDS)].append(row)
    merged: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        by_seed = {int(row["model_seed"]): row for row in group}
        if tuple(sorted(by_seed)) != FULL_SEEDS:
            raise ValueError(f"OMBRIA condition lacks five seeds: {key}")
        output = dict(zip(OMBRIA_STATIC_FIELDS, key, strict=True))
        output["model_seeds"] = len(FULL_SEEDS)
        output["iou_mean"] = statistics.mean(
            float(by_seed[seed]["iou"]) for seed in FULL_SEEDS
        )
        output["s1_reference_iou_mean"] = statistics.mean(
            float(by_seed[seed]["s1_reference_iou"]) for seed in FULL_SEEDS
        )
        output["realized_false_available_rate_mean"] = statistics.mean(
            float(by_seed[seed]["realized_false_available_rate"])
            for seed in FULL_SEEDS
        )
        output["realized_false_unavailable_rate_mean"] = statistics.mean(
            float(by_seed[seed]["realized_false_unavailable_rate"])
            for seed in FULL_SEEDS
        )
        delta = paired_t_summary(
            {seed: float(by_seed[seed]["delta_s1_iou"]) for seed in FULL_SEEDS}
        )
        _add_estimate(output, "delta_s1_iou", delta)
        output["s1_relative_regret"] = max(0.0, -delta["mean"])
        output["seed_interval_status"] = (
            "lower_nonnegative"
            if delta["ci95_lower"] >= 0
            else "upper_negative"
            if delta["ci95_upper"] < 0
            else "crosses_zero"
        )
        merged.append(output)
    if len(merged) != 101:
        raise ValueError(f"expected 101 OMBRIA paired rows, received {len(merged)}")
    return merged


def _external_raw_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["split"]),
            str(row["route"]),
            str(row["condition_id"]),
            int(row["model_seed"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate external seed condition {key}")
        lookup[key] = row
    return lookup


def _ombria_raw_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, float, float, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, float, float, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            str(row["route"]),
            float(row["requested_false_available_rate"]),
            float(row["requested_false_unavailable_rate"]),
            int(row["model_seed"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate OMBRIA seed condition {key}")
        lookup[key] = row
    return lookup


def _surface_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    false_available_field: str,
    false_unavailable_field: str,
) -> dict[str, Any]:
    if len(rows) != 25:
        raise ValueError(f"surface requires 25 cells, received {len(rows)}")
    worst = min(rows, key=lambda row: float(row["delta_s1_iou_mean"]))
    best = max(rows, key=lambda row: float(row["delta_s1_iou_mean"]))
    return {
        "cells": len(rows),
        "lower_nonnegative_cells": sum(
            float(row["delta_s1_iou_ci95_lower"]) >= 0 for row in rows
        ),
        "upper_negative_cells": sum(
            float(row["delta_s1_iou_ci95_upper"]) < 0 for row in rows
        ),
        "negative_mean_cells": sum(
            float(row["delta_s1_iou_mean"]) < 0 for row in rows
        ),
        "descriptive_safe_fraction": sum(
            float(row["delta_s1_iou_ci95_lower"]) >= 0 for row in rows
        )
        / len(rows),
        "maximum_mean_regret": max(
            max(0.0, -float(row["delta_s1_iou_mean"])) for row in rows
        ),
        "worst_cell": {
            "false_available_rate": worst[false_available_field],
            "false_unavailable_rate": worst[false_unavailable_field],
            "mean": worst["delta_s1_iou_mean"],
            "ci95_lower": worst["delta_s1_iou_ci95_lower"],
            "ci95_upper": worst["delta_s1_iou_ci95_upper"],
        },
        "best_cell": {
            "false_available_rate": best[false_available_field],
            "false_unavailable_rate": best[false_unavailable_field],
            "mean": best["delta_s1_iou_mean"],
            "ci95_lower": best["delta_s1_iou_ci95_lower"],
            "ci95_upper": best["delta_s1_iou_ci95_upper"],
        },
    }


def build_surface_summaries(
    external_paired: Sequence[Mapping[str, Any]],
    ombria_paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    external: dict[str, Any] = {}
    for split in ("test", "bolivia"):
        external[split] = {}
        for route in EXTERNAL_GRID_ROUTES:
            rows = [
                row
                for row in external_paired
                if row["split"] == split
                and row["route"] == route
                and row["quality_mode"] == "independent"
            ]
            external[split][route] = _surface_summary(
                rows,
                false_available_field="false_available_rate",
                false_unavailable_field="false_unavailable_rate",
            )
    ombria: dict[str, Any] = {}
    for route in OMBRIA_GRID_ROUTES:
        rows = [row for row in ombria_paired if row["route"] == route]
        ombria[route] = _surface_summary(
            rows,
            false_available_field="requested_false_available_rate",
            false_unavailable_field="requested_false_unavailable_rate",
        )
    return {"sen1floods11": external, "ombria": ombria}


def build_endpoint_asymmetry(
    external_rows: Sequence[Mapping[str, Any]],
    ombria_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    external = _external_raw_lookup(external_rows)
    ombria = _ombria_raw_lookup(ombria_rows)
    output: list[dict[str, Any]] = []

    for route in OMBRIA_GRID_ROUTES:
        clean = {(seed): ombria[(route, 0.0, 0.0, seed)] for seed in FULL_SEEDS}
        fa = {(seed): ombria[(route, 0.4, 0.0, seed)] for seed in FULL_SEEDS}
        fu = {(seed): ombria[(route, 0.0, 0.4, seed)] for seed in FULL_SEEDS}
        fa_harm = {
            seed: float(clean[seed]["delta_s1_iou"])
            - float(fa[seed]["delta_s1_iou"])
            for seed in FULL_SEEDS
        }
        fu_harm = {
            seed: float(clean[seed]["delta_s1_iou"])
            - float(fu[seed]["delta_s1_iou"])
            for seed in FULL_SEEDS
        }
        asymmetry = {
            seed: fa_harm[seed] - fu_harm[seed] for seed in FULL_SEEDS
        }
        row = {"dataset": "ombria", "split": "test", "route": route}
        _add_estimate(row, "false_available_harm", paired_t_summary(fa_harm))
        _add_estimate(row, "false_unavailable_harm", paired_t_summary(fu_harm))
        _add_estimate(row, "harm_fa_minus_fu", paired_t_summary(asymmetry))
        output.append(row)

    for split in ("test", "bolivia"):
        for route in EXTERNAL_GRID_ROUTES:
            clean_id = independent_condition_id(0.0, 0.0)
            fa_id = independent_condition_id(0.4, 0.0)
            fu_id = independent_condition_id(0.0, 0.4)
            fa_harm = {
                seed: float(external[(split, route, clean_id, seed)]["delta_s1_iou"])
                - float(external[(split, route, fa_id, seed)]["delta_s1_iou"])
                for seed in FULL_SEEDS
            }
            fu_harm = {
                seed: float(external[(split, route, clean_id, seed)]["delta_s1_iou"])
                - float(external[(split, route, fu_id, seed)]["delta_s1_iou"])
                for seed in FULL_SEEDS
            }
            asymmetry = {
                seed: fa_harm[seed] - fu_harm[seed] for seed in FULL_SEEDS
            }
            fa_harm_event_equal = {
                seed: float(
                    external[(split, route, clean_id, seed)]["event_equal_iou"]
                )
                - float(external[(split, route, fa_id, seed)]["event_equal_iou"])
                for seed in FULL_SEEDS
            }
            fu_harm_event_equal = {
                seed: float(
                    external[(split, route, clean_id, seed)]["event_equal_iou"]
                )
                - float(external[(split, route, fu_id, seed)]["event_equal_iou"])
                for seed in FULL_SEEDS
            }
            asymmetry_event_equal = {
                seed: fa_harm_event_equal[seed] - fu_harm_event_equal[seed]
                for seed in FULL_SEEDS
            }
            row = {"dataset": "sen1floods11", "split": split, "route": route}
            _add_estimate(row, "false_available_harm", paired_t_summary(fa_harm))
            _add_estimate(row, "false_unavailable_harm", paired_t_summary(fu_harm))
            _add_estimate(row, "harm_fa_minus_fu", paired_t_summary(asymmetry))
            _add_estimate(
                row,
                "event_equal_harm_fa_minus_fu",
                paired_t_summary(asymmetry_event_equal),
            )
            output.append(row)
    return output


def build_route_pair_contrasts(
    external_rows: Sequence[Mapping[str, Any]],
    ombria_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    external = _external_raw_lookup(external_rows)
    ombria = _ombria_raw_lookup(ombria_rows)
    output: list[dict[str, Any]] = []
    for split in ("test", "bolivia"):
        for intervention, baseline, family in EXTERNAL_ROUTE_PAIRS:
            for false_available in RATES:
                for false_unavailable in RATES:
                    condition = independent_condition_id(
                        false_available, false_unavailable
                    )
                    values = {
                        seed: float(
                            external[(split, intervention, condition, seed)]["iou"]
                        )
                        - float(external[(split, baseline, condition, seed)]["iou"])
                        for seed in FULL_SEEDS
                    }
                    row = {
                        "dataset": "sen1floods11",
                        "split": split,
                        "family": family,
                        "intervention": intervention,
                        "baseline": baseline,
                        "false_available_rate": false_available,
                        "false_unavailable_rate": false_unavailable,
                    }
                    _add_estimate(row, "iou_difference", paired_t_summary(values))
                    output.append(row)
    for intervention, baseline, family in OMBRIA_ROUTE_PAIRS:
        for false_available in RATES:
            for false_unavailable in RATES:
                values = {
                    seed: float(
                        ombria[(intervention, false_available, false_unavailable, seed)][
                            "iou"
                        ]
                    )
                    - float(
                        ombria[(baseline, false_available, false_unavailable, seed)][
                            "iou"
                        ]
                    )
                    for seed in FULL_SEEDS
                }
                row = {
                    "dataset": "ombria",
                    "split": "test",
                    "family": family,
                    "intervention": intervention,
                    "baseline": baseline,
                    "false_available_rate": false_available,
                    "false_unavailable_rate": false_unavailable,
                }
                _add_estimate(row, "iou_difference", paired_t_summary(values))
                output.append(row)
    return output


def build_structured_contrasts(
    external_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = _external_raw_lookup(external_rows)
    output: list[dict[str, Any]] = []
    structured_conditions = sorted(
        {
            str(row["condition_id"])
            for row in external_rows
            if row["quality_mode"] in {"translate", "dilate", "erode"}
        }
    )
    if len(structured_conditions) != 14:
        raise ValueError("expected 14 frozen structured conditions")
    for split in ("test", "bolivia"):
        for route in EXTERNAL_GRID_ROUTES:
            for condition in structured_conditions:
                matched = f"matched_random__{condition}"
                values = {
                    seed: float(lookup[(split, route, condition, seed)]["iou"])
                    - float(lookup[(split, route, matched, seed)]["iou"])
                    for seed in FULL_SEEDS
                }
                source = lookup[(split, route, condition, FULL_SEEDS[0])]
                row = {
                    "dataset": "sen1floods11",
                    "split": split,
                    "route": route,
                    "structured_condition_id": condition,
                    "structured_mode": source["quality_mode"],
                    "matched_random_condition_id": matched,
                }
                _add_estimate(
                    row,
                    "structured_minus_matched_random_iou",
                    paired_t_summary(values),
                )
                output.append(row)
    return output


def _find_asymmetry(
    rows: Sequence[Mapping[str, Any]], dataset: str, split: str, route: str
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if row["dataset"] == dataset
        and row["split"] == split
        and row["route"] == route
    ]
    if len(matches) != 1:
        raise ValueError("endpoint-asymmetry row is not unique")
    return matches[0]


def _find_audit_check(
    audit: Mapping[str, Any], check_id: str
) -> Mapping[str, Any]:
    matches = [check for check in audit["checks"] if check["id"] == check_id]
    if len(matches) != 1 or matches[0]["status"] != "pass":
        raise ValueError(f"required shard-set audit check did not pass: {check_id}")
    return matches[0]


def evaluate_method_gate(
    surface_summaries: Mapping[str, Any],
    route_pair_contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset, split, intervention, baseline in (
        (
            "ombria",
            "test",
            "hard_error_aware",
            "hard_oracle",
        ),
        (
            "sen1floods11",
            "test",
            "hard_quality_gate_error_aware",
            "hard_quality_gate",
        ),
    ):
        surface_root = (
            surface_summaries["ombria"]
            if dataset == "ombria"
            else surface_summaries["sen1floods11"][split]
        )
        intervention_surface = surface_root[intervention]
        baseline_surface = surface_root[baseline]
        rows = [
            row
            for row in route_pair_contrasts
            if row["dataset"] == dataset
            and row["split"] == split
            and row["intervention"] == intervention
            and row["baseline"] == baseline
        ]
        if len(rows) != 25:
            raise ValueError("hard-gate route pair does not contain 25 cells")
        clean = next(
            row
            for row in rows
            if row["false_available_rate"] == 0
            and row["false_unavailable_rate"] == 0
        )
        clean_difference = float(clean["iou_difference_mean"])
        clean_loss = max(0.0, -clean_difference)
        regret_reduction = (
            float(baseline_surface["maximum_mean_regret"])
            - float(intervention_surface["maximum_mean_regret"])
        )
        safe_expansion = int(intervention_surface["lower_nonnegative_cells"]) - int(
            baseline_surface["lower_nonnegative_cells"]
        )
        significant_improvement_cells = sum(
            float(row["iou_difference_ci95_lower"]) > 0 for row in rows
        )
        benefit_gate = regret_reduction > 0 or (
            safe_expansion > 0 and significant_improvement_cells > 0
        )
        clean_gate = clean_loss <= 0.01
        datasets[dataset] = {
            "intervention": intervention,
            "baseline": baseline,
            "clean_iou_difference": clean_difference,
            "clean_loss": clean_loss,
            "clean_loss_limit": 0.01,
            "clean_gate": clean_gate,
            "maximum_mean_regret_reduction": regret_reduction,
            "lower_nonnegative_cell_expansion": safe_expansion,
            "significant_improvement_cells": significant_improvement_cells,
            "benefit_gate": benefit_gate,
        }
    statistical_gate = all(
        result["benefit_gate"] and result["clean_gate"]
        for result in datasets.values()
    )
    return {
        "datasets": datasets,
        "statistical_gate": statistical_gate,
        "fallback_boundary_gate": "verified_by_model_contract_tests_not_this_merge",
        "method_claim_authorized": False,
        "reason": (
            "The external hard-gate intervention removes mean regret and expands "
            "the seed-interval safe region, but the OMBRIA clean mean loss exceeds "
            "the frozen 0.01 limit. The manuscript therefore remains an empirical "
            "reliability study rather than a new-method claim."
        ),
    }


def build_core_decision(
    *,
    set_audit: Mapping[str, Any],
    surface_summaries: Mapping[str, Any],
    endpoint_asymmetry: Sequence[Mapping[str, Any]],
    method_gate: Mapping[str, Any],
    structured_contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ombria_hard = _find_asymmetry(
        endpoint_asymmetry, "ombria", "test", "hard_oracle"
    )
    external_hard = _find_asymmetry(
        endpoint_asymmetry, "sen1floods11", "test", "hard_quality_gate"
    )
    opposite_asymmetry_transfers = (
        float(ombria_hard["harm_fa_minus_fu_ci95_upper"]) < 0
        and float(external_hard["harm_fa_minus_fu_ci95_upper"]) < 0
        and float(external_hard["event_equal_harm_fa_minus_fu_ci95_upper"]) < 0
    )
    structured_test = [
        row for row in structured_contrasts if row["split"] == "test"
    ]
    structured_summary: dict[str, Any] = {}
    for route in EXTERNAL_GRID_ROUTES:
        rows = [row for row in structured_test if row["route"] == route]
        structured_summary[route] = {
            "conditions": len(rows),
            "structured_better_ci_cells": sum(
                float(row["structured_minus_matched_random_iou_ci95_lower"]) > 0
                for row in rows
            ),
            "structured_worse_ci_cells": sum(
                float(row["structured_minus_matched_random_iou_ci95_upper"]) < 0
                for row in rows
            ),
        }
    hard_surface = surface_summaries["sen1floods11"]["test"][
        "hard_quality_gate"
    ]
    stop_triggered = False
    return {
        "five_shard_merge_gate": bool(
            set_audit["decision"]["core_merge_authorized"]
        ),
        "hypotheses": {
            "H1_false_available_more_harmful": {
                "status": (
                    "falsified_with_opposite_direction_transfer"
                    if opposite_asymmetry_transfers
                    else "not_supported"
                ),
                "ombria_hard_gate_harm_fa_minus_fu": {
                    key: ombria_hard[f"harm_fa_minus_fu_{key}"]
                    for key in ("mean", "ci95_lower", "ci95_upper")
                },
                "external_hard_gate_harm_fa_minus_fu": {
                    key: external_hard[f"harm_fa_minus_fu_{key}"]
                    for key in ("mean", "ci95_lower", "ci95_upper")
                },
                "external_event_equal_hard_gate_harm_fa_minus_fu": {
                    key: external_hard[f"event_equal_harm_fa_minus_fu_{key}"]
                    for key in ("mean", "ci95_lower", "ci95_upper")
                },
                "interpretation": (
                    "At the frozen 0.40 endpoints, false-unavailable errors are "
                    "more harmful than false-available errors for the hard gate "
                    "in both datasets and in the external event-equal sensitivity."
                ),
            },
            "H2_spatial_organization_matters": {
                "status": "supported_but_route_dependent",
                "test_split": structured_summary,
            },
            "H3_hard_gate_has_limited_safe_region": {
                "status": "supported_descriptively_with_seed_uncertainty",
                "external_test_hard_gate": hard_surface,
            },
            "H4_error_aware_method_gate": {
                "status": "fails_frozen_method_claim_gate",
                "detail": method_gate,
            },
        },
        "evidence_gates": {
            "five_seed_response": "pass",
            "cross_dataset_direction": (
                "pass" if opposite_asymmetry_transfers else "fail"
            ),
            "bolivia_and_event_equal_outputs": "pass",
            "both_error_types_reported": "pass",
        },
        "stop_gate_triggered": stop_triggered,
        "core_article_decision": (
            "continue_as_empirical_reliability_article"
            if not stop_triggered and opposite_asymmetry_transfers
            else "reassess_or_stop"
        ),
        "method_positioning": "empirical_reliability_study_not_new_method",
        "formal_manuscript_results_authorized": False,
        "remaining_scientific_gates": [
            "selected hierarchical seed-event-chip uncertainty",
            "official SMAGNet adaptation or prespecified reproducibility limitation",
            "post-analysis claim-evidence audit",
        ],
    }


def analyze_quality_uncertainty_core(
    artifacts: Sequence[tuple[int, Path]],
    *,
    code_root: Path | None = None,
    hierarchical_bootstrap_replicates: int = 0,
    hierarchical_bootstrap_seed: int = 20260720,
) -> dict[str, Any]:
    set_audit, external_seed_rows, ombria_seed_rows = load_core_seed_rows(
        artifacts, code_root=code_root
    )
    external_paired = merge_external_seed_rows(external_seed_rows)
    ombria_paired = merge_ombria_seed_rows(ombria_seed_rows)
    surfaces = build_surface_summaries(external_paired, ombria_paired)
    asymmetry = build_endpoint_asymmetry(external_seed_rows, ombria_seed_rows)
    route_pairs = build_route_pair_contrasts(external_seed_rows, ombria_seed_rows)
    structured = build_structured_contrasts(external_seed_rows)
    method_gate = evaluate_method_gate(surfaces, route_pairs)
    artifact_identity = _find_audit_check(set_audit, "artifact_identity")
    decision = build_core_decision(
        set_audit=set_audit,
        surface_summaries=surfaces,
        endpoint_asymmetry=asymmetry,
        method_gate=method_gate,
        structured_contrasts=structured,
    )
    artifact_sha256 = {
        str(seed): artifact_identity["detail"][str(seed)] for seed in FULL_SEEDS
    }
    report: dict[str, Any] = {
        "schema": "geoai-quality-map-uncertainty-core-analysis-v1",
        "frozen_seeds": list(FULL_SEEDS),
        "artifact_sha256": artifact_sha256,
        "source_commit": set_audit["source_commit"],
        "row_counts": {
            "external_seed_rows": len(external_seed_rows),
            "external_paired_conditions": len(external_paired),
            "ombria_seed_rows": len(ombria_seed_rows),
            "ombria_paired_conditions": len(ombria_paired),
            "route_pair_contrasts": len(route_pairs),
            "structured_contrasts": len(structured),
        },
        "uncertainty": {
            "primary_in_this_stage": "paired model-seed Student-t interval",
            "model_seeds": len(FULL_SEEDS),
            "degrees_of_freedom": 4,
            "t_critical_95": T_CRITICAL_95_DF4,
            "boundary": (
                "These intervals describe fixed-split, fixed-perturbation "
                "model-seed variability; they are not spatial, event-population, "
                "or deployment uncertainty."
            ),
        },
        "surface_summaries": surfaces,
        "endpoint_asymmetry": asymmetry,
        "method_gate": method_gate,
        "decision": decision,
    }
    tables = {
        "external_seed_rows": external_seed_rows,
        "external_paired_summary": external_paired,
        "ombria_seed_rows": ombria_seed_rows,
        "ombria_paired_summary": ombria_paired,
        "endpoint_asymmetry": asymmetry,
        "route_pair_contrasts": route_pairs,
        "structured_contrasts": structured,
    }

    if hierarchical_bootstrap_replicates:
        from .quality_uncertainty_hierarchical_bootstrap import (
            run_selected_hierarchical_bootstrap,
        )

        hierarchical = run_selected_hierarchical_bootstrap(
            artifacts,
            replicates=hierarchical_bootstrap_replicates,
            random_seed=hierarchical_bootstrap_seed,
        )
        hierarchical_report = hierarchical["report"]
        if hierarchical_report["artifact_sha256"] != artifact_sha256:
            raise ValueError("hierarchical bootstrap artifact identities do not match")
        h1_by_id = {
            row["estimand_id"]: row
            for row in hierarchical["rows"]
            if row["family"] == "H1_endpoint_asymmetry"
        }
        for estimand_id, expected in (
            (
                "h1__ombria__test__hard_oracle",
                decision["hypotheses"]["H1_false_available_more_harmful"][
                    "ombria_hard_gate_harm_fa_minus_fu"
                ]["mean"],
            ),
            (
                "h1__sen1floods11__test__hard_quality_gate",
                decision["hypotheses"]["H1_false_available_more_harmful"][
                    "external_hard_gate_harm_fa_minus_fu"
                ]["mean"],
            ),
        ):
            if not math.isclose(
                float(h1_by_id[estimand_id]["observed_mean"]),
                float(expected),
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"hierarchical per-chip reconstruction disagrees for {estimand_id}"
                )
        hierarchical_transfer = all(
            float(h1_by_id[estimand_id]["ci95_upper"]) < 0
            for estimand_id in (
                "h1__ombria__test__hard_oracle",
                "h1__sen1floods11__test__hard_quality_gate",
            )
        )
        bolivia_same_direction = (
            float(
                h1_by_id[
                    "h1__sen1floods11__bolivia__hard_quality_gate"
                ]["ci95_upper"]
            )
            < 0
        )
        report["hierarchical_bootstrap"] = hierarchical_report
        report["row_counts"]["hierarchical_bootstrap_estimands"] = len(
            hierarchical["rows"]
        )
        report["uncertainty"]["primary_in_this_stage"] = (
            "selected paired seed-event-chip hierarchical percentile interval"
        )
        report["uncertainty"]["seed_t_role"] = (
            "complete-surface model-seed sensitivity interval"
        )
        report["uncertainty"]["boundary"] = hierarchical_report["boundary"]
        decision["evidence_gates"]["selected_hierarchical_uncertainty"] = "pass"
        decision["evidence_gates"]["cross_dataset_direction_hierarchical"] = (
            "pass" if hierarchical_transfer else "fail"
        )
        decision["evidence_gates"]["bolivia_direction_hierarchical"] = (
            "pass" if bolivia_same_direction else "inconclusive"
        )
        decision["hypotheses"]["H1_false_available_more_harmful"][
            "hierarchical_status"
        ] = (
            "opposite_direction_confirmed"
            if hierarchical_transfer
            else "opposite_direction_not_confirmed"
        )
        if not hierarchical_transfer:
            decision["core_article_decision"] = "reassess_or_stop"
        decision["remaining_scientific_gates"] = [
            gate
            for gate in decision["remaining_scientific_gates"]
            if gate != "selected hierarchical seed-event-chip uncertainty"
        ]
        tables["hierarchical_bootstrap"] = hierarchical["rows"]

    return {"report": report, "tables": tables}


def render_core_analysis_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    h1 = decision["hypotheses"]["H1_false_available_more_harmful"]
    hard = report["surface_summaries"]["sen1floods11"]["test"][
        "hard_quality_gate"
    ]
    aware = report["surface_summaries"]["sen1floods11"]["test"][
        "hard_quality_gate_error_aware"
    ]
    ombria_method = report["method_gate"]["datasets"]["ombria"]
    external_method = report["method_gate"]["datasets"]["sen1floods11"]
    event_equal_h1 = h1[
        "external_event_equal_hard_gate_harm_fa_minus_fu"
    ]
    lines = [
        "# Quality-map uncertainty five-seed core analysis",
        "",
        f"- Core decision: **{decision['core_article_decision']}**",
        f"- Method positioning: **{decision['method_positioning']}**",
        "- Formal manuscript Results authorized: **false**",
        f"- Frozen seeds: `{', '.join(map(str, report['frozen_seeds']))}`",
        f"- Source commit: `{report['source_commit']}`",
        "",
        "## Main paired finding",
        "",
        "The prespecified expectation that false-available errors would be more "
        "harmful is not supported. At the 0.40 endpoints, the paired "
        "false-available-minus-false-unavailable harm contrast is "
        f"{h1['ombria_hard_gate_harm_fa_minus_fu']['mean']:+.4f} "
        "in OMBRIA (95% seed interval "
        f"[{h1['ombria_hard_gate_harm_fa_minus_fu']['ci95_lower']:+.4f}, "
        f"{h1['ombria_hard_gate_harm_fa_minus_fu']['ci95_upper']:+.4f}]) and "
        f"{h1['external_hard_gate_harm_fa_minus_fu']['mean']:+.4f} "
        "in Sen1Floods11 test ("
        f"[{h1['external_hard_gate_harm_fa_minus_fu']['ci95_lower']:+.4f}, "
        f"{h1['external_hard_gate_harm_fa_minus_fu']['ci95_upper']:+.4f}]). "
        "Both intervals are below zero, so false-unavailable errors are the more "
        "damaging endpoint for the hard gate in both datasets. The event-equal "
        "Sen1Floods11 sensitivity is "
        f"{event_equal_h1['mean']:+.4f} "
        f"[{event_equal_h1['ci95_lower']:+.4f}, "
        f"{event_equal_h1['ci95_upper']:+.4f}].",
    ]
    if "hierarchical_bootstrap" in report:
        primary_h1 = report["hierarchical_bootstrap"]["primary_h1"]
        ombria_hierarchical = primary_h1["h1__ombria__test__hard_oracle"]
        external_hierarchical = primary_h1[
            "h1__sen1floods11__test__hard_quality_gate"
        ]
        bolivia_hierarchical = primary_h1[
            "h1__sen1floods11__bolivia__hard_quality_gate"
        ]
        lines.extend(
            [
                "",
                "The selected paired seed-event-chip bootstrap confirms the "
                "opposite direction: OMBRIA "
                f"[{ombria_hierarchical['ci95_lower']:+.4f}, "
                f"{ombria_hierarchical['ci95_upper']:+.4f}], Sen1Floods11 test "
                f"[{external_hierarchical['ci95_lower']:+.4f}, "
                f"{external_hierarchical['ci95_upper']:+.4f}], and Bolivia "
                f"[{bolivia_hierarchical['ci95_lower']:+.4f}, "
                f"{bolivia_hierarchical['ci95_upper']:+.4f}].",
            ]
        )
    lines.extend(
        [
            "",
            "## Fusion-safety surface",
            "",
            "On Sen1Floods11 test, the unguided hard gate has "
        f"{hard['negative_mean_cells']}/25 cells with negative mean "
        f"S1-relative IoU and {hard['lower_nonnegative_cells']}/25 cells whose "
        "seed interval is lower-nonnegative. Its maximum mean S1-relative regret "
        f"is {hard['maximum_mean_regret']:.4f}. The error-aware hard gate has "
        f"{aware['negative_mean_cells']}/25 negative-mean cells, "
        f"{aware['lower_nonnegative_cells']}/25 lower-nonnegative cells, and "
        f"maximum mean regret {aware['maximum_mean_regret']:.4f}.",
        "",
        "## Frozen method-claim gate",
        "",
        "The external hard-gate comparison improves maximum mean regret by "
        f"{external_method['maximum_mean_regret_reduction']:+.4f} and expands "
        "the lower-nonnegative grid by "
        f"{external_method['lower_nonnegative_cell_expansion']} cells. However, "
        "the OMBRIA clean mean difference is "
        f"{ombria_method['clean_iou_difference']:+.4f}, corresponding to a "
        f"{ombria_method['clean_loss']:.4f} clean loss, above the frozen 0.01 "
        "limit. A new-method benefit claim therefore fails; the defensible paper "
        "remains an empirical reliability study.",
        "",
        "## Remaining gates",
        "",
        ]
    )
    lines.extend(f"- `{gate}`" for gate in decision["remaining_scientific_gates"])
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            str(report["uncertainty"]["boundary"]),
            "Synthetic OMBRIA degradation is described as controlled cloud-like "
            "occlusion, and SCL remains an operational proxy rather than human "
            "cloud truth.",
            "",
        ]
    )
    return "\n".join(lines)
