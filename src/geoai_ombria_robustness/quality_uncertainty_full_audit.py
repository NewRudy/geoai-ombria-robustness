from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from .quality_uncertainty_artifact_audit import (
    ARTIFACT_SCHEMA,
    EXPECTED_EXTERNAL_ROUTES,
    EXPECTED_OMBRIA_COMMIT,
    EXPECTED_OMBRIA_ROUTES,
    EXPECTED_QUALITY_ROUTES,
    METRIC_NAMES,
    _all_finite,
    _check,
    _close,
    _file_sha256,
    _read_csv,
    _read_json,
    _safe_member,
    _sha256,
    _validate_aggregate_metrics,
)


FULL_SOURCE_COMMIT = "abf6a792ba158ca2302850f3234097e06f9a1d8e"
FULL_SMOKE_SHA256 = "32ebcd1d8bfa5cadcf9b007985548ae7d03b9ecb1015ea41b92e93b12b47e67e"
FULL_SMAGNET_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
FULL_SEEDS = (7, 13, 21, 29, 37)
FULL_RATES = (0.0, 0.05, 0.1, 0.2, 0.4)
FULL_SPLIT_COUNTS = {
    "train": 252,
    "validation": 89,
    "test": 90,
    "bolivia": 15,
}
FULL_CONDITION_COUNTS = {
    "s1_reference": 1,
    "early_fusion": 2,
    "early_fusion_dropout": 2,
    **{route: 54 for route in EXPECTED_QUALITY_ROUTES},
}


def _git_blob_sha256(code_root: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(code_root), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    )
    return _sha256(completed.stdout)


def _rate_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _audit_protocol_and_boundary(
    archive: ZipFile,
    root: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    plan = _read_json(archive, f"{root}/full_shard_plan.json")
    gate = _read_json(archive, f"{root}/full_shard_decision_gate.json")
    ombria_gate = _read_json(
        archive, f"{root}/ombria/seed{seed}/ombria_decision_gate.json"
    )
    external_gate = _read_json(
        archive,
        f"{root}/sen1floods11/sen1floods11_decision_gate.json",
    )
    expected_remaining = [value for value in FULL_SEEDS if value != seed]
    passed = (
        plan.get("schema") == "geoai-quality-map-uncertainty-full-shard-plan-v1"
        and plan.get("source_commit") == FULL_SOURCE_COMMIT
        and plan.get("active_seed") == seed
        and tuple(plan.get("planned_seeds", ())) == FULL_SEEDS
        and plan.get("epochs") == 25
        and tuple(plan.get("error_rates", ())) == FULL_RATES
        and plan.get("perturbation_repetitions") == 3
        and tuple(plan.get("external_routes", ())) == EXPECTED_EXTERNAL_ROUTES
        and tuple(plan.get("ombria_routes", ())) == EXPECTED_OMBRIA_ROUTES
        and plan.get("external_seed_condition_rows") == 550
        and plan.get("external_raw_summary_rows") == 1650
        and plan.get("ombria_evaluation_cells") == 101
        and plan.get("ombria_raw_summary_rows") == 301
        and plan.get("published_architecture_gate", {}).get("official_commit")
        == FULL_SMAGNET_COMMIT
        and plan.get("scientific_interpretation_allowed") is False
        and gate.get("status") == "pass"
        and gate.get("active_seed") == seed
        and gate.get("remaining_core_seeds") == expected_remaining
        and gate.get("cuda_conv2d_gate") == "pass"
        and gate.get("ombria_gate") == "pass"
        and gate.get("sen1floods11_gate") == "pass"
        and gate.get("authorized_evaluation_hotfix") == "pass"
        and gate.get("published_architecture_gate") == "pending_separate_shard"
        and gate.get("scientific_interpretation_allowed") is False
        and ombria_gate.get("status") == "pass"
        and ombria_gate.get("active_seed") == seed
        and ombria_gate.get("raw_summary_rows") == 301
        and ombria_gate.get("response_surface_rows") == 101
        and ombria_gate.get("scientific_interpretation_allowed") is False
        and external_gate.get("status") == "pass"
        and external_gate.get("mode") == "full"
        and external_gate.get("pipeline_only") is False
        and external_gate.get("active_seeds") == [seed]
        and external_gate.get("seed_condition_rows") == 550
        and external_gate.get("complete_training_runs") == 8
        and external_gate.get("shard_complete") is True
        and external_gate.get("all_full_seeds_present") is False
        and external_gate.get("scientific_interpretation_allowed") is False
    )
    return passed, {
        "active_seed": seed,
        "planned_seeds": list(FULL_SEEDS),
        "remaining_core_seeds": expected_remaining,
        "source_commit": plan.get("source_commit"),
        "top_gate": gate.get("status"),
        "ombria_gate": ombria_gate.get("status"),
        "external_gate": external_gate.get("status"),
        "published_architecture_gate": gate.get("published_architecture_gate"),
        "scientific_interpretation_allowed": gate.get(
            "scientific_interpretation_allowed"
        ),
    }


def _audit_runtime_and_source(
    archive: ZipFile,
    root: str,
    code_root: Path | None,
) -> tuple[bool, dict[str, Any]]:
    runtime = _read_json(archive, f"{root}/runtime_manifest.json")
    smoke = _read_json(
        archive, f"{root}/quality_uncertainty_smoke_authorization.json"
    )
    equivalence = _read_json(
        archive, f"{root}/quality_uncertainty_core_equivalence.json"
    )
    source_issues: list[str] = []
    if code_root is not None:
        try:
            for relative, expected in equivalence.get(
                "byte_identical_files", {}
            ).items():
                if (
                    _git_blob_sha256(code_root, FULL_SOURCE_COMMIT, relative)
                    != expected
                ):
                    source_issues.append(relative)
            for relative, exception in equivalence.get(
                "authorized_exceptions", {}
            ).items():
                if (
                    _git_blob_sha256(code_root, FULL_SOURCE_COMMIT, relative)
                    != exception.get("full_sha256")
                ):
                    source_issues.append(relative)
        except (OSError, subprocess.CalledProcessError) as exc:
            source_issues.append(str(exc))
    passed = (
        runtime.get("repository_commit") == FULL_SOURCE_COMMIT
        and runtime.get("repository_dirty_tracked") is False
        and runtime.get("cuda_conv2d_gate") == "pass"
        and runtime.get("device") not in {None, "", "cpu"}
        and runtime.get("required_arch") in runtime.get("compiled_arches", [])
        and smoke.get("status") == "pass"
        and smoke.get("artifact", {}).get("sha256") == FULL_SMOKE_SHA256
        and smoke.get("audit", {}).get("full_authorized") is True
        and smoke.get("scientific_interpretation_allowed") is False
        and equivalence.get("status")
        == "pass-with-authorized-evaluation-hotfix"
        and equivalence.get("failed_full_source_commit")
        == "d25bc67fa42c71e1a0d565b8f024a352b312d0bf"
        and equivalence.get("authorized_exceptions", {})
        .get("scripts/evaluate_sen1floods11_quality_uncertainty.py", {})
        .get("regression_status")
        == "pass"
        and equivalence.get("scientific_interpretation_allowed") is False
        and not source_issues
    )
    return passed, {
        "repository_commit": runtime.get("repository_commit"),
        "repository_dirty_tracked": runtime.get("repository_dirty_tracked"),
        "cuda_conv2d_gate": runtime.get("cuda_conv2d_gate"),
        "device": runtime.get("device"),
        "torch": runtime.get("torch"),
        "torch_cuda": runtime.get("torch_cuda"),
        "required_arch": runtime.get("required_arch"),
        "smoke_sha256": smoke.get("artifact", {}).get("sha256"),
        "core_equivalence": equivalence.get("status"),
        "source_blob_issues": source_issues,
    }


def _audit_external_preparation(
    archive: ZipFile,
    root: str,
    seed: int,
) -> tuple[bool, dict[str, Any], set[str]]:
    prefix = f"{root}/sen1floods11"
    plan = _read_json(archive, f"{prefix}/experiment_plan.json")
    preparation = _read_json(
        archive, f"{prefix}/sen1floods11_preparation_report.json"
    )
    selected_name = f"{prefix}/sen1floods11_selected_manifest.json"
    source_name = f"{root}/sen1floods11_scl_manifest.json"
    selected = _read_json(archive, selected_name)
    source = _read_json(archive, source_name)
    selected_hash = _file_sha256(archive, selected_name)
    source_hash = _file_sha256(archive, source_name)
    selected_records = selected.get("records", [])
    prepared_records = preparation.get("records", [])
    record_ids = [str(record.get("chip_id")) for record in selected_records]
    prepared_ids = [str(record.get("chip_id")) for record in prepared_records]
    split_counts = Counter(str(record.get("split")) for record in selected_records)
    provider_counts = Counter(
        str(asset.get("provider"))
        for record in selected_records
        for asset in record.get("scl_assets", [])
    )
    zero_valid = {
        str(record["chip_id"])
        for record in prepared_records
        if int(record.get("valid_target_pixels", -1)) == 0
    }
    zero_valid_test = {
        str(record["chip_id"])
        for record in prepared_records
        if int(record.get("valid_target_pixels", -1)) == 0
        and record.get("split") == "test"
    }
    passed = (
        plan.get("mode") == "full"
        and plan.get("pipeline_only") is False
        and plan.get("source_commit") == FULL_SOURCE_COMMIT
        and plan.get("active_seeds") == [seed]
        and tuple(plan.get("planned_seeds", ())) == FULL_SEEDS
        and plan.get("epochs") == 25
        and tuple(plan.get("error_rates", ())) == FULL_RATES
        and tuple(plan.get("routes", ())) == EXPECTED_EXTERNAL_ROUTES
        and plan.get("sample_limits") == {}
        and plan.get("perturbation_repetitions") == 3
        and plan.get("source_manifest_sha256") == source_hash
        and plan.get("official_smagnet", {}).get("commit") == FULL_SMAGNET_COMMIT
        and preparation.get("status") == "pass"
        and preparation.get("mode") == "full"
        and preparation.get("pipeline_only") is False
        and preparation.get("record_count") == 446
        and preparation.get("selected_manifest_sha256") == selected_hash
        and selected.get("summary", {}).get("record_count") == 446
        and selected.get("summary", {}).get("split_counts") == FULL_SPLIT_COUNTS
        and dict(split_counts) == FULL_SPLIT_COUNTS
        and len(record_ids) == len(set(record_ids)) == 446
        and set(record_ids) == set(prepared_ids)
        and source.get("summary", {}).get("record_count") == 446
        and len(zero_valid) == 5
        and len(zero_valid_test) == 1
        and provider_counts["earth-search"] > 0
        and provider_counts["planetary-computer"] > 0
    )
    return passed, {
        "records": len(record_ids),
        "split_counts": dict(split_counts),
        "events": len({record.get("event") for record in selected_records}),
        "provider_counts": dict(provider_counts),
        "source_manifest_sha256": source_hash,
        "selected_manifest_sha256": selected_hash,
        "zero_valid_target_chips": sorted(zero_valid),
        "zero_valid_test_chips": sorted(zero_valid_test),
    }, zero_valid_test


def _audit_external_training(
    archive: ZipFile,
    root: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/sen1floods11"
    selected_hash = _file_sha256(
        archive, f"{prefix}/sen1floods11_selected_manifest.json"
    )
    names = set(archive.namelist())
    issues: list[str] = []
    best_epochs: dict[str, int] = {}
    final_elapsed: dict[str, float] = {}
    split_hashes: set[str] = set()
    for route in EXPECTED_EXTERNAL_ROUTES:
        run_prefix = f"{prefix}/runs/{route}_seed{seed}"
        required = (
            "best_clean.pt",
            "last.pt",
            "checkpoint_manifest.json",
            "config.json",
            "metrics.csv",
            "splits.json",
        )
        missing = [name for name in required if f"{run_prefix}/{name}" not in names]
        if missing:
            issues.append(f"{route}: missing {','.join(missing)}")
            continue
        config = _read_json(archive, f"{run_prefix}/config.json")
        checkpoint = _read_json(
            archive, f"{run_prefix}/checkpoint_manifest.json"
        )
        rows = _read_csv(archive, f"{run_prefix}/metrics.csv")
        best_row = max(rows, key=lambda row: float(row["val_iou"]))
        split_hashes.add(_file_sha256(archive, f"{run_prefix}/splits.json"))
        route_ok = (
            config.get("route") == route
            and config.get("seed") == seed
            and config.get("epochs") == 25
            and config.get("base_channels") == 16
            and config.get("device") == "cuda"
            and config.get("manifest_sha256") == selected_hash
            and tuple(config.get("train_quality_error_rates", ())) == FULL_RATES
            and config.get("train_count") == FULL_SPLIT_COUNTS["train"]
            and config.get("validation_count") == FULL_SPLIT_COUNTS["validation"]
            and checkpoint.get("best_clean_sha256")
            == _file_sha256(archive, f"{run_prefix}/best_clean.pt")
            and checkpoint.get("last_sha256")
            == _file_sha256(archive, f"{run_prefix}/last.pt")
            and checkpoint.get("best_clean_epoch") == int(best_row["epoch"])
            and _close(
                float(checkpoint.get("best_clean_iou", -1)),
                float(best_row["val_iou"]),
            )
            and len(rows) == 25
            and [int(row["epoch"]) for row in rows] == list(range(1, 26))
            and _all_finite(
                rows,
                (
                    "train_loss",
                    "val_loss",
                    "val_iou",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "val_accuracy",
                    "elapsed_seconds",
                ),
            )
            and all(0.0 <= float(row["val_iou"]) <= 1.0 for row in rows)
        )
        if not route_ok:
            issues.append(f"{route}: configuration, hashes, or trajectory disagree")
        else:
            best_epochs[route] = int(best_row["epoch"])
            final_elapsed[route] = float(rows[-1]["elapsed_seconds"])
    if len(split_hashes) != 1:
        issues.append("external routes do not share one frozen split file")
    return not issues, {
        "expected_runs": len(EXPECTED_EXTERNAL_ROUTES),
        "complete_runs": len(EXPECTED_EXTERNAL_ROUTES) - len(
            [
                issue
                for issue in issues
                if issue.split(":", 1)[0] in EXPECTED_EXTERNAL_ROUTES
            ]
        ),
        "best_clean_epochs": best_epochs,
        "elapsed_seconds": final_elapsed,
        "shared_split_hashes": len(split_hashes),
        "issues": issues,
    }


def _audit_external_evaluations(
    archive: ZipFile,
    root: str,
    seed: int,
    zero_valid_test_ids: set[str],
) -> tuple[bool, dict[str, Any], list[dict[str, str]]]:
    prefix = f"{root}/sen1floods11"
    selected_hash = _file_sha256(
        archive, f"{prefix}/sen1floods11_selected_manifest.json"
    )
    names = set(archive.namelist())
    issues: list[str] = []
    all_summaries: list[dict[str, str]] = []
    expected_summary_rows = 0
    expected_chip_rows = 0
    actual_chip_rows = 0
    zero_valid_rows = 0
    matched_max_difference = 0.0
    for route in EXPECTED_EXTERNAL_ROUTES:
        conditions_name = f"{prefix}/conditions/{route}.json"
        if conditions_name not in names:
            issues.append(f"{route}: missing condition manifest")
            continue
        conditions_document = _read_json(archive, conditions_name)
        conditions = conditions_document.get("conditions", [])
        condition_ids = [str(condition.get("condition_id")) for condition in conditions]
        if (
            conditions_document.get("mode") != "full"
            or conditions_document.get("pipeline_only") is not False
            or conditions_document.get("route") != route
            or len(conditions) != FULL_CONDITION_COUNTS[route]
            or len(condition_ids) != len(set(condition_ids))
        ):
            issues.append(f"{route}: invalid Full condition manifest")
            continue
        for split, samples in (("test", 90), ("bolivia", 15)):
            eval_prefix = f"{prefix}/evaluations/{route}/seed{seed}/{split}"
            required = (
                "evaluation_config.json",
                "summary_metrics.csv",
                "per_chip_metrics.csv",
                "per_event_metrics.csv",
            )
            missing = [
                name for name in required if f"{eval_prefix}/{name}" not in names
            ]
            if missing:
                issues.append(f"{route}/{split}: missing {','.join(missing)}")
                continue
            config = _read_json(archive, f"{eval_prefix}/evaluation_config.json")
            summaries = _read_csv(archive, f"{eval_prefix}/summary_metrics.csv")
            chips = _read_csv(archive, f"{eval_prefix}/per_chip_metrics.csv")
            events = _read_csv(archive, f"{eval_prefix}/per_event_metrics.csv")
            checkpoint_name = f"{prefix}/runs/{route}_seed{seed}/best_clean.pt"
            if not (
                config.get("route") == route
                and config.get("split") == split
                and config.get("sample_count") == samples
                and config.get("manifest_sha256") == selected_hash
                and config.get("checkpoint_sha256")
                == _file_sha256(archive, checkpoint_name)
                and config.get("perturb_seed") == 20260716
                and config.get("repetitions") == 3
                and config.get("conditions") == conditions
            ):
                issues.append(f"{route}/{split}: evaluation config mismatch")
            expected_keys = {
                (identifier, repetition)
                for identifier in condition_ids
                for repetition in range(3)
            }
            summary_by_key = {
                (row["condition_id"], int(row["repetition"])): row
                for row in summaries
            }
            chip_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
            event_by_key: dict[
                tuple[str, int], list[dict[str, str]]
            ] = defaultdict(list)
            for row in chips:
                chip_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
            for row in events:
                event_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
            if (
                set(summary_by_key) != expected_keys
                or set(chip_by_key) != expected_keys
                or set(event_by_key) != expected_keys
            ):
                issues.append(
                    f"{route}/{split}: condition/repetition coverage mismatch"
                )
            for key in expected_keys & set(summary_by_key) & set(chip_by_key):
                summary = summary_by_key[key]
                key_chips = chip_by_key[key]
                key_events = event_by_key.get(key, [])
                aggregate_ok, reason = _validate_aggregate_metrics(
                    summary, key_chips, key_events
                )
                if not (
                    summary.get("route") == route
                    and summary.get("split") == split
                    and int(summary.get("model_seed", -1)) == seed
                    and len(key_chips) == samples
                    and len({row.get("chip_id") for row in key_chips}) == samples
                    and aggregate_ok
                ):
                    issues.append(f"{route}/{split}/{key}: {reason}")
                if summary.get("quality_mode") == "independent":
                    rate_fields = (
                        ("false_available_rate", "realized_false_available_rate"),
                        ("false_unavailable_rate", "realized_false_unavailable_rate"),
                        (
                            "false_available_rate",
                            "valid_realized_false_available_rate",
                        ),
                        (
                            "false_unavailable_rate",
                            "valid_realized_false_unavailable_rate",
                        ),
                    )
                    if any(
                        abs(float(summary[requested]) - float(summary[realized]))
                        > 0.005
                        for requested, realized in rate_fields
                    ):
                        issues.append(f"{route}/{split}/{key}: realized rate drift")
                for row in key_chips:
                    valid_pixels = int(row.get("valid_target_pixels", -1))
                    has_valid = row.get("has_valid_target") == "True"
                    mean_probability = row.get("mean_probability", "")
                    if valid_pixels == 0:
                        zero_valid_rows += 1
                        if (
                            split != "test"
                            or row.get("chip_id") not in zero_valid_test_ids
                            or has_valid
                            or mean_probability != ""
                            or float(row["valid_quality_false_available_rate"]) != 0.0
                            or float(row["valid_quality_false_unavailable_rate"]) != 0.0
                        ):
                            issues.append(
                                f"{route}/{split}/{key}: invalid zero-domain encoding"
                            )
                    elif (
                        not has_valid
                        or not mean_probability
                        or not math.isfinite(float(mean_probability))
                    ):
                        issues.append(
                            f"{route}/{split}/{key}: invalid mean probability"
                        )
            if route in EXPECTED_QUALITY_ROUTES:
                rows_by_key = {
                    (row["condition_id"], row["chip_id"], int(row["repetition"])): row
                    for row in chips
                }
                for condition in conditions:
                    identifier = str(condition["condition_id"])
                    if condition.get("quality_mode") not in {
                        "translate",
                        "dilate",
                        "erode",
                    }:
                        continue
                    matched_identifier = f"matched_random__{identifier}"
                    for repetition in range(3):
                        for chip_id in {
                            row["chip_id"]
                            for row in chips
                            if row["condition_id"] == identifier
                            and int(row["repetition"]) == repetition
                        }:
                            structured = rows_by_key[(identifier, chip_id, repetition)]
                            matched = rows_by_key[
                                (matched_identifier, chip_id, repetition)
                            ]
                            for field in (
                                "quality_false_available_rate",
                                "quality_false_unavailable_rate",
                                "valid_quality_false_available_rate",
                                "valid_quality_false_unavailable_rate",
                            ):
                                difference = abs(
                                    float(structured[field]) - float(matched[field])
                                )
                                matched_max_difference = max(
                                    matched_max_difference, difference
                                )
            all_summaries.extend(summaries)
            expected_summary_rows += len(condition_ids) * 3
            expected_chip_rows += len(condition_ids) * 3 * samples
            actual_chip_rows += len(chips)
    if len(all_summaries) != expected_summary_rows or expected_summary_rows != 1650:
        issues.append("external raw summary total is not the frozen 1650")
    if actual_chip_rows != expected_chip_rows or expected_chip_rows != 86625:
        issues.append("external per-chip total is not the frozen 86625")
    if zero_valid_rows != 825:
        issues.append(f"zero-valid-domain row total is {zero_valid_rows}, expected 825")
    if matched_max_difference > 1e-12:
        issues.append("structured and matched-random error rates differ")
    finite = _all_finite(
        all_summaries,
        (
            *METRIC_NAMES,
            "event_equal_iou",
            "realized_false_available_rate",
            "realized_false_unavailable_rate",
            "valid_realized_false_available_rate",
            "valid_realized_false_unavailable_rate",
        ),
    )
    if not finite:
        issues.append("external summaries contain non-finite metrics")
    return not issues, {
        "raw_summary_rows": len(all_summaries),
        "per_chip_rows": actual_chip_rows,
        "zero_valid_domain_rows": zero_valid_rows,
        "matched_control_max_rate_difference": matched_max_difference,
        "finite_metrics": finite,
        "issues": issues,
    }, all_summaries


def _audit_external_seed_summary(
    archive: ZipFile,
    root: str,
    seed: int,
    raw_rows: list[dict[str, str]],
) -> tuple[bool, dict[str, Any]]:
    rows = _read_csv(
        archive,
        f"{root}/sen1floods11/tables/sen1floods11_seed_condition_summary.csv",
    )
    grouped: dict[tuple[int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_rows:
        grouped[
            (
                int(row["model_seed"]),
                row["split"],
                row["route"],
                row["condition_id"],
            )
        ].append(row)
    table = {
        (
            int(row["model_seed"]),
            row["split"],
            row["route"],
            row["condition_id"],
        ): row
        for row in rows
    }
    s1_by_split = {
        split: sum(float(row["iou"]) for row in values) / len(values)
        for (model_seed, split, route, _), values in grouped.items()
        if model_seed == seed and route == "s1_reference"
    }
    issues: list[str] = []
    if set(table) != set(grouped) or len(rows) != 550:
        issues.append("seed-condition table coverage does not match 550 raw groups")
    for key in set(table) & set(grouped):
        row = table[key]
        values = grouped[key]
        iou = sum(float(value["iou"]) for value in values) / len(values)
        event_iou = sum(float(value["event_equal_iou"]) for value in values) / len(
            values
        )
        reference = s1_by_split[key[1]]
        if not (
            len(values) == 3
            and int(row["repetitions"]) == 3
            and _close(float(row["iou"]), iou)
            and _close(float(row["event_equal_iou"]), event_iou)
            and _close(float(row["s1_reference_iou"]), reference)
            and _close(float(row["delta_s1_iou"]), iou - reference)
        ):
            issues.append(f"seed-condition recomputation failed for {key}")
            break
    finite = _all_finite(
        rows,
        (
            "iou",
            "f1",
            "precision",
            "recall",
            "event_equal_iou",
            "s1_reference_iou",
            "delta_s1_iou",
        ),
    )
    if not finite:
        issues.append("seed-condition table contains non-finite metrics")
    preview: dict[str, float] = {}
    for row in rows:
        if row["split"] != "test":
            continue
        reference_condition = (
            row["condition_id"] == "reference"
            if row["route"] in {
                "s1_reference",
                "early_fusion",
                "early_fusion_dropout",
            }
            else row["condition_id"] == "independent_fa0_fu0"
        )
        if reference_condition:
            preview[row["route"]] = float(row["iou"])
    return not issues, {
        "rows": len(rows),
        "paired_s1_splits": sorted(s1_by_split),
        "finite_metrics": finite,
        "test_reference_iou_qc_only": preview,
        "issues": issues,
    }


def _audit_ombria(
    archive: ZipFile,
    root: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/ombria/seed{seed}"
    names = set(archive.namelist())
    issues: list[str] = []
    split_hashes: set[str] = set()
    checkpoint_hashes: dict[str, str] = {}
    for route in EXPECTED_OMBRIA_ROUTES:
        run_prefix = f"{prefix}/runs/{route}_seed{seed}"
        required = (
            "best_clean.pt",
            "best_model.pt",
            "best_robust.pt",
            "checkpoint_selection.json",
            "config.json",
            "metrics.csv",
            "splits.json",
        )
        missing = [name for name in required if f"{run_prefix}/{name}" not in names]
        if missing:
            issues.append(f"{route}: missing {','.join(missing)}")
            continue
        config = _read_json(archive, f"{run_prefix}/config.json")
        selection = _read_json(
            archive, f"{run_prefix}/checkpoint_selection.json"
        )
        rows = _read_csv(archive, f"{run_prefix}/metrics.csv")
        split_hashes.add(_file_sha256(archive, f"{run_prefix}/splits.json"))
        checkpoint_hashes[route] = _file_sha256(
            archive, f"{run_prefix}/best_clean.pt"
        )
        best_clean = max(rows, key=lambda row: float(row["val_iou"]))
        best_robust = max(rows, key=lambda row: float(row["robust_val_iou_mean"]))
        if not (
            config.get("run_name") == f"{route}_seed{seed}"
            and config.get("seed") == seed
            and config.get("epochs") == 25
            and config.get("base_channels") == 16
            and config.get("split_seed") == 20260710
            and config.get("eval_perturb_seed") == 20260716
            and selection.get("best_clean_sha256") == checkpoint_hashes[route]
            and selection.get("best_robust_sha256")
            == _file_sha256(archive, f"{run_prefix}/best_robust.pt")
            and selection.get("best_clean_epoch") == int(best_clean["epoch"])
            and selection.get("best_robust_epoch") == int(best_robust["epoch"])
            and len(rows) == 25
            and [int(row["epoch"]) for row in rows] == list(range(1, 26))
            and _all_finite(
                rows,
                (
                    "train_loss",
                    "val_loss",
                    "val_iou",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "val_accuracy",
                    "robust_val_iou_mean",
                    "val_cloud_after_50_iou",
                ),
            )
        ):
            issues.append(f"{route}: configuration, hashes, or trajectory disagree")
    if len(split_hashes) != 1:
        issues.append("OMBRIA routes do not share one frozen split file")

    expected_cells = {
        "s1_reference": ((0.0, 0.0),),
        **{
            route: tuple((fa, fu) for fa in FULL_RATES for fu in FULL_RATES)
            for route in EXPECTED_OMBRIA_ROUTES
            if route != "s1_reference"
        },
    }
    raw: dict[tuple[str, float, float], list[dict[str, str]]] = {}
    raw_row_count = 0
    for route, cells in expected_cells.items():
        for fa, fu in cells:
            eval_prefix = (
                f"{prefix}/evaluations/{route}/fa{_rate_key(fa)}_fu{_rate_key(fu)}"
            )
            required = (
                "evaluation_config.json",
                "summary_metrics.csv",
                "per_chip_metrics.csv",
            )
            missing = [
                name for name in required if f"{eval_prefix}/{name}" not in names
            ]
            if missing:
                issues.append(f"{route}/{fa}/{fu}: missing files")
                continue
            config = _read_json(archive, f"{eval_prefix}/evaluation_config.json")
            summaries = _read_csv(archive, f"{eval_prefix}/summary_metrics.csv")
            chips = _read_csv(archive, f"{eval_prefix}/per_chip_metrics.csv")
            repetitions = 1 if route == "s1_reference" else 3
            summary_by_rep = {int(row["repetition"]): row for row in summaries}
            chip_by_rep: dict[int, list[dict[str, str]]] = defaultdict(list)
            for row in chips:
                chip_by_rep[int(row["repetition"])].append(row)
            if not (
                config.get("route") == route
                and config.get("content_degradation") == "cloud_after_50"
                and float(config.get("false_available_rate", -1)) == fa
                and float(config.get("false_unavailable_rate", -1)) == fu
                and config.get("perturb_seed") == 20260716
                and config.get("repetitions") == repetitions
                and config.get("checkpoint_sha256") == checkpoint_hashes.get(route)
                and set(summary_by_rep) == set(range(repetitions))
                and set(chip_by_rep) == set(range(repetitions))
            ):
                issues.append(f"{route}/{fa}/{fu}: config or repetition mismatch")
            for repetition in set(summary_by_rep) & set(chip_by_rep):
                aggregate_ok, reason = _validate_aggregate_metrics(
                    summary_by_rep[repetition], chip_by_rep[repetition]
                )
                if len(chip_by_rep[repetition]) != 70 or not aggregate_ok:
                    issues.append(f"{route}/{fa}/{fu}/{repetition}: {reason}")
            if route != "s1_reference" and any(
                abs(float(row["realized_false_available_rate"]) - fa) > 0.001
                or abs(float(row["realized_false_unavailable_rate"]) - fu) > 0.001
                for row in summaries
            ):
                issues.append(f"{route}/{fa}/{fu}: realized rate drift")
            raw[(route, fa, fu)] = summaries
            raw_row_count += len(summaries)

    response = _read_csv(archive, f"{prefix}/tables/response_surface.csv")
    response_by_key = {
        (
            row["route"],
            float(row["requested_false_available_rate"]),
            float(row["requested_false_unavailable_rate"]),
        ): row
        for row in response
    }
    expected_keys = {
        (route, fa, fu)
        for route, cells in expected_cells.items()
        for fa, fu in cells
    }
    s1_iou = sum(float(row["iou"]) for row in raw[("s1_reference", 0.0, 0.0)])
    if set(response_by_key) != expected_keys or len(response) != 101:
        issues.append("OMBRIA response surface is not the frozen 101 cells")
    for key in set(response_by_key) & set(raw):
        row = response_by_key[key]
        values = raw[key]
        iou = sum(float(value["iou"]) for value in values) / len(values)
        if not (
            int(row["repetitions"]) == len(values)
            and _close(float(row["iou"]), iou)
            and _close(float(row["s1_reference_iou"]), s1_iou)
            and _close(float(row["delta_s1_iou"]), iou - s1_iou)
        ):
            issues.append(f"OMBRIA response recomputation failed for {key}")
            break
    finite = _all_finite(
        [row for rows in raw.values() for row in rows],
        (
            *METRIC_NAMES,
            "realized_false_available_rate",
            "realized_false_unavailable_rate",
        ),
    ) and _all_finite(response, ("iou", "s1_reference_iou", "delta_s1_iou"))
    if raw_row_count != 301:
        issues.append(f"OMBRIA raw summary total is {raw_row_count}, expected 301")
    if not finite:
        issues.append("OMBRIA outputs contain non-finite metrics")
    deltas = [
        float(row["delta_s1_iou"])
        for row in response
        if row["route"] != "s1_reference"
    ]
    return not issues, {
        "training_runs": len(EXPECTED_OMBRIA_ROUTES),
        "evaluation_cells": len(raw),
        "raw_summary_rows": raw_row_count,
        "response_surface_rows": len(response),
        "shared_split_hashes": len(split_hashes),
        "finite_metrics": finite,
        "delta_s1_range_qc_only": [min(deltas), max(deltas)] if deltas else [],
        "issues": issues,
    }


def _audit_logs_and_payload(
    archive: ZipFile,
    root: str,
    seed: int,
) -> tuple[bool, dict[str, Any]]:
    names = archive.namelist()
    raw_raster_pattern = re.compile(r"_(?:S1|S2|Label)Hand\.tiff?$", re.IGNORECASE)
    raw_rasters = [name for name in names if raw_raster_pattern.search(name)]
    current = archive.read(f"{root}/run.log").decode("utf-8", errors="replace")
    anomaly_patterns = {
        "traceback": r"\bTraceback \(most recent call last\)",
        "cuda_error": r"\bCUDA error\b",
        "out_of_memory": r"\b(?:out of memory|OOM)\b",
        "runtime_error": r"\bRuntimeError:\s",
        "non_finite": r"(?i)(?:^|[\s,:=])(nan|inf)(?:$|[\s,}])",
    }
    current_anomalies = {
        name: len(re.findall(pattern, current, flags=re.MULTILINE))
        for name, pattern in anomaly_patterns.items()
    }
    prior_names = sorted(
        name
        for name in names
        if name.startswith(f"{root}/prior_attempts/") and name.endswith(".log")
    )
    prior_text = "\n".join(
        archive.read(name).decode("utf-8", errors="replace") for name in prior_names
    )
    prior_expected = True
    if seed == 7:
        prior_expected = (
            len(prior_names) == 1
            and "ValueError: reference must not be empty" in prior_text
            and "out of memory" not in prior_text.lower()
            and "CUDA error" not in prior_text
        )
    passed = (
        not raw_rasters
        and not any(current_anomalies.values())
        and '"status": "pass"' in current
        and prior_expected
    )
    return passed, {
        "raw_rasters": raw_rasters,
        "current_anomalies": current_anomalies,
        "current_final_pass_marker": '"status": "pass"' in current,
        "prior_attempt_logs": prior_names,
        "prior_failure_signature_retained": (
            "ValueError: reference must not be empty" in prior_text
        ),
    }


def audit_quality_uncertainty_full_shard_artifact(
    archive_path: Path,
    *,
    seed: int,
    code_root: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed audit for one returned Full seed shard."""

    archive_path = Path(archive_path)
    code_root = Path(code_root) if code_root is not None else None
    checks: list[dict[str, Any]] = []
    artifact_sha256 = ""
    artifact_bytes = 0
    member_count = 0
    root = f"quality_uncertainty_full_seed{seed}"
    try:
        payload = archive_path.read_bytes()
        artifact_sha256 = _sha256(payload)
        artifact_bytes = len(payload)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            member_count = len(names)
            safe = (
                len(names) == len(set(names))
                and all(_safe_member(name) for name in names)
                and all(name.startswith(f"{root}/") for name in names)
            )
            checks.append(
                _check(
                    "archive_path_safety",
                    safe,
                    {"members": len(names), "unique_members": len(set(names))},
                )
            )
            bad_member = archive.testzip()
            checks.append(
                _check("archive_crc", bad_member is None, {"bad_member": bad_member})
            )

            manifest_name = f"{root}/artifact_manifest.json"
            manifest_ok = manifest_name in names
            manifest_detail: dict[str, Any] = {}
            if manifest_ok:
                manifest = _read_json(archive, manifest_name)
                records = manifest.get("files", [])
                expected_names = {f"{root}/{record['path']}" for record in records}
                actual_names = set(names) - {manifest_name}
                mismatched: list[str] = []
                for record in records:
                    member = archive.read(f"{root}/{record['path']}")
                    if (
                        len(member) != int(record["bytes"])
                        or _sha256(member) != record["sha256"]
                    ):
                        mismatched.append(str(record["path"]))
                manifest_ok = (
                    manifest.get("schema") == ARTIFACT_SCHEMA
                    and manifest.get("root") == root
                    and len(records) == len({record["path"] for record in records})
                    and expected_names == actual_names
                    and not mismatched
                )
                manifest_detail = {
                    "schema": manifest.get("schema"),
                    "root": manifest.get("root"),
                    "records": len(records),
                    "coverage_match": expected_names == actual_names,
                    "mismatched": mismatched,
                }
            checks.append(
                _check("artifact_manifest_integrity", manifest_ok, manifest_detail)
            )

            if manifest_ok:
                audit_steps = [
                    (
                        "protocol_and_claim_boundary",
                        lambda: _audit_protocol_and_boundary(archive, root, seed),
                    ),
                    (
                        "runtime_source_and_hotfix",
                        lambda: _audit_runtime_and_source(archive, root, code_root),
                    ),
                ]
                for check_id, operation in audit_steps:
                    try:
                        passed, detail = operation()
                        checks.append(_check(check_id, passed, detail))
                    except Exception as exc:
                        checks.append(_check(check_id, False, {"error": str(exc)}))

                zero_valid_test: set[str] = set()
                try:
                    passed, detail, zero_valid_test = _audit_external_preparation(
                        archive, root, seed
                    )
                    checks.append(_check("external_data_preparation", passed, detail))
                except Exception as exc:
                    checks.append(
                        _check("external_data_preparation", False, {"error": str(exc)})
                    )
                try:
                    passed, detail = _audit_external_training(archive, root, seed)
                    checks.append(_check("external_training_runs", passed, detail))
                except Exception as exc:
                    checks.append(
                        _check("external_training_runs", False, {"error": str(exc)})
                    )
                external_rows: list[dict[str, str]] = []
                try:
                    passed, detail, external_rows = _audit_external_evaluations(
                        archive, root, seed, zero_valid_test
                    )
                    checks.append(
                        _check("external_evaluation_integrity", passed, detail)
                    )
                except Exception as exc:
                    checks.append(
                        _check(
                            "external_evaluation_integrity", False, {"error": str(exc)}
                        )
                    )
                try:
                    passed, detail = _audit_external_seed_summary(
                        archive, root, seed, external_rows
                    )
                    checks.append(_check("external_seed_summary", passed, detail))
                except Exception as exc:
                    checks.append(
                        _check("external_seed_summary", False, {"error": str(exc)})
                    )
                try:
                    passed, detail = _audit_ombria(archive, root, seed)
                    checks.append(_check("ombria_integrity", passed, detail))
                except Exception as exc:
                    checks.append(
                        _check("ombria_integrity", False, {"error": str(exc)})
                    )
                try:
                    passed, detail = _audit_logs_and_payload(archive, root, seed)
                    checks.append(_check("logs_resume_and_payload", passed, detail))
                except Exception as exc:
                    checks.append(
                        _check("logs_resume_and_payload", False, {"error": str(exc)})
                    )
    except (OSError, BadZipFile, KeyError, ValueError, json.JSONDecodeError) as exc:
        checks.append(_check("archive_open", False, {"error": str(exc)}))

    passed = bool(checks) and all(check["status"] == "pass" for check in checks)
    return {
        "schema": "geoai-quality-map-uncertainty-full-shard-audit-v1",
        "artifact": {
            "path": str(archive_path.resolve()),
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "members": member_count,
        },
        "seed": seed,
        "checks": checks,
        "decision": {
            "status": "pass" if passed else "fail",
            "remaining_core_seeds_authorized": passed,
            "manuscript_results_authorized": False,
            "scientific_interpretation_allowed": False,
            "claim_boundary": (
                "A passing seed shard authorizes the remaining frozen core seeds "
                "only. Manuscript interpretation remains prohibited until all five "
                "core seeds, the SMAGNet gate, merge audit, and post-run scientific "
                "audit pass."
            ),
        },
    }


def render_quality_uncertainty_full_shard_audit_markdown(
    report: dict[str, Any],
) -> str:
    artifact = report["artifact"]
    decision = report["decision"]
    lines = [
        f"# Quality-map uncertainty Full seed {report['seed']} artifact audit",
        "",
        f"- Decision: **{decision['status'].upper()}**",
        "- Remaining frozen core seeds authorized: "
        f"**{str(decision['remaining_core_seeds_authorized']).lower()}**",
        "- Manuscript results authorized: **false**",
        f"- Artifact: `{artifact['path']}`",
        f"- SHA-256: `{artifact['sha256']}`",
        f"- Bytes / members: {artifact['bytes']} / {artifact['members']}",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True, ensure_ascii=False)
        if len(detail) > 520:
            detail = detail[:517] + "..."
        detail = detail.replace("|", "\\|")
        lines.append(f"| `{check['id']}` | {check['status']} | `{detail}` |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            decision["claim_boundary"],
            "",
            f"Seed-{report['seed']} score previews in the JSON detail are "
            "execution-health checks only; "
            "they are not manuscript evidence and must not be quoted as results.",
            "",
        ]
    )
    return "\n".join(lines)
