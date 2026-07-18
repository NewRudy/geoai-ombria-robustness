from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile


ARTIFACT_SCHEMA = "geoai-quality-map-uncertainty-artifacts-v1"
EXPECTED_SOURCE_COMMIT = "3e97f93bdea999e1781869b74842a88050f9bdb1"
EXPECTED_OMBRIA_COMMIT = "38a490355f76da8ce27ed051138f03f3492a6e46"
EXPECTED_RATES = (0.0, 0.2, 0.4)
EXPECTED_OMBRIA_ROUTES = (
    "hard_oracle",
    "hard_error_aware",
    "concat_error_aware",
    "soft_error_aware",
    "s1_reference",
)
EXPECTED_EXTERNAL_ROUTES = (
    "s1_reference",
    "early_fusion",
    "early_fusion_dropout",
    "quality_concat",
    "quality_concat_error_aware",
    "hard_quality_gate",
    "hard_quality_gate_error_aware",
    "soft_quality_prior_error_aware",
)
EXPECTED_QUALITY_ROUTES = frozenset(EXPECTED_EXTERNAL_ROUTES[3:])
EXPECTED_SPLIT_COUNTS = {
    "train": 24,
    "validation": 12,
    "test": 12,
    "bolivia": 4,
}
METRIC_NAMES = ("iou", "f1", "precision", "recall", "accuracy")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _check(check_id: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith(("/", "\\"))
        and "\\" not in name
        and ".." not in path.parts
    )


def _read_json(archive: ZipFile, name: str) -> dict[str, Any]:
    document = json.loads(archive.read(name))
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {name}")
    return document


def _read_csv(archive: ZipFile, name: str) -> list[dict[str, str]]:
    text = archive.read(name).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"Expected non-empty CSV rows in {name}")
    return rows


def _file_sha256(archive: ZipFile, name: str) -> str:
    return _sha256(archive.read(name))


def _metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def _close(left: float, right: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _all_finite(rows: list[dict[str, str]], names: tuple[str, ...]) -> bool:
    try:
        return all(
            math.isfinite(float(row[name]))
            for row in rows
            for name in names
            if row.get(name, "") != ""
        )
    except (KeyError, TypeError, ValueError):
        return False


def _audit_runtime(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    runtime = _read_json(archive, f"{root}/runtime_manifest.json")
    experiment = _read_json(archive, f"{root}/experiment_manifest.json")
    passed = (
        runtime.get("repository_commit") == EXPECTED_SOURCE_COMMIT
        and runtime.get("repository_dirty_tracked") is False
        and runtime.get("cuda_conv2d_gate") == "pass"
        and runtime.get("device") not in {None, "", "cpu"}
        and experiment.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and experiment.get("mode") == "smoke"
        and experiment.get("pipeline_only") is True
        and experiment.get("epochs") == 2
        and experiment.get("seed") == 7
        and tuple(experiment.get("quality_error_rates", ())) == EXPECTED_RATES
        and tuple(experiment.get("routes", ())) == EXPECTED_OMBRIA_ROUTES
        and experiment.get("ombria_commit") == EXPECTED_OMBRIA_COMMIT
        and experiment.get("content_degradation") == "cloud_after_50"
    )
    return passed, {
        "repository_commit": runtime.get("repository_commit"),
        "repository_dirty_tracked": runtime.get("repository_dirty_tracked"),
        "cuda_conv2d_gate": runtime.get("cuda_conv2d_gate"),
        "device": runtime.get("device"),
        "torch": runtime.get("torch"),
        "torch_cuda": runtime.get("torch_cuda"),
        "ombria_commit": experiment.get("ombria_commit"),
    }


def _audit_external_preparation(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/sen1floods11"
    plan = _read_json(archive, f"{prefix}/experiment_plan.json")
    gate = _read_json(archive, f"{prefix}/sen1floods11_decision_gate.json")
    preparation = _read_json(
        archive,
        f"{prefix}/sen1floods11_preparation_report.json",
    )
    selected_name = f"{prefix}/sen1floods11_selected_manifest.json"
    selected = _read_json(archive, selected_name)
    source_name = f"{root}/sen1floods11_scl_manifest.json"
    scl_smoke = _read_json(archive, f"{root}/sen1floods11_scl_smoke.json")
    selected_hash = _file_sha256(archive, selected_name)
    source_hash = _file_sha256(archive, source_name)
    selected_records = selected.get("records", [])
    record_ids = [record.get("chip_id") for record in selected_records]
    prepared_ids = [
        record.get("chip_id") for record in preparation.get("records", [])
    ]
    split_counts = Counter(record.get("split") for record in selected_records)
    provider_counts = Counter(
        asset.get("provider")
        for record in selected_records
        for asset in record.get("scl_assets", [])
    )
    passed = (
        plan.get("mode") == "smoke"
        and plan.get("pipeline_only") is True
        and plan.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and tuple(plan.get("seeds", ())) == (7,)
        and plan.get("epochs") == 2
        and tuple(plan.get("error_rates", ())) == EXPECTED_RATES
        and tuple(plan.get("routes", ())) == EXPECTED_EXTERNAL_ROUTES
        and plan.get("sample_limits") == EXPECTED_SPLIT_COUNTS
        and plan.get("perturbation_repetitions") == 1
        and plan.get("source_manifest_sha256") == source_hash
        and gate.get("status") == "pass"
        and gate.get("mode") == "smoke"
        and gate.get("pipeline_only") is True
        and gate.get("scientific_interpretation_allowed") is False
        and gate.get("expected_training_runs") == 8
        and gate.get("complete_training_runs") == 8
        and gate.get("expected_seed_condition_rows") == 170
        and gate.get("seed_condition_rows") == 170
        and gate.get("finite_primary_metrics") is True
        and preparation.get("status") == "pass"
        and preparation.get("record_count") == 52
        and preparation.get("selected_manifest_sha256") == selected_hash
        and selected.get("summary", {}).get("split_counts")
        == EXPECTED_SPLIT_COUNTS
        and dict(split_counts) == EXPECTED_SPLIT_COUNTS
        and len(record_ids) == len(set(record_ids)) == 52
        and set(record_ids) == set(prepared_ids)
        and scl_smoke.get("status") == "pass"
        and scl_smoke.get("manifest_match_fraction") == 1.0
        and scl_smoke.get("manifest_record_count") == 446
        and scl_smoke.get("unmatched_chip_ids") == []
        and provider_counts["earth-search"] > 0
        and provider_counts["planetary-computer"] > 0
    )
    return passed, {
        "selected_manifest_sha256": selected_hash,
        "source_manifest_sha256": source_hash,
        "records": len(record_ids),
        "split_counts": dict(split_counts),
        "events": len({record.get("event") for record in selected_records}),
        "provider_counts": dict(provider_counts),
        "gate_status": gate.get("status"),
        "seed_condition_rows": gate.get("seed_condition_rows"),
    }


def _audit_alignment(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    alignment = _read_json(
        archive,
        f"{root}/sen1floods11_alignment_audit/sen1floods11_alignment_audit.json",
    )
    rows = alignment.get("rows", [])
    events = {row.get("event") for row in rows}
    grid_pass = all(
        row.get("automated_status") == "pass"
        and row.get("grid_checks", {}).get("pass") is True
        for row in rows
    )
    passed = (
        alignment.get("automated_status") == "pass"
        and alignment.get("selected_count") == 11
        and len(rows) == 11
        and len(events) == 11
        and grid_pass
    )
    return passed, {
        "automated_status": alignment.get("automated_status"),
        "reported_visual_status": alignment.get("visual_status"),
        "selected_count": alignment.get("selected_count"),
        "events": sorted(str(event) for event in events),
        "grid_pass": grid_pass,
    }


def _audit_external_training_runs(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/sen1floods11"
    selected_hash = _file_sha256(
        archive,
        f"{prefix}/sen1floods11_selected_manifest.json",
    )
    names = set(archive.namelist())
    issues: list[str] = []
    elapsed_by_route: dict[str, float] = {}
    for route in EXPECTED_EXTERNAL_ROUTES:
        run_prefix = f"{prefix}/runs/{route}_seed7"
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
            archive,
            f"{run_prefix}/checkpoint_manifest.json",
        )
        metrics_rows = _read_csv(archive, f"{run_prefix}/metrics.csv")
        best_hash = _file_sha256(archive, f"{run_prefix}/best_clean.pt")
        last_hash = _file_sha256(archive, f"{run_prefix}/last.pt")
        route_ok = (
            config.get("route") == route
            and config.get("seed") == 7
            and config.get("epochs") == 2
            and config.get("base_channels") == 16
            and config.get("device") == "cuda"
            and config.get("manifest_sha256") == selected_hash
            and tuple(config.get("train_quality_error_rates", ()))
            == EXPECTED_RATES
            and config.get("train_count") == 24
            and config.get("validation_count") == 12
            and checkpoint.get("best_clean_sha256") == best_hash
            and checkpoint.get("last_sha256") == last_hash
            and len(metrics_rows) == 2
            and [int(row["epoch"]) for row in metrics_rows] == [1, 2]
            and _all_finite(
                metrics_rows,
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
        )
        if not route_ok:
            issues.append(f"{route}: configuration, hashes, or metrics disagree")
        else:
            elapsed_by_route[route] = float(metrics_rows[-1]["elapsed_seconds"])
    return not issues, {
        "expected_runs": len(EXPECTED_EXTERNAL_ROUTES),
        "complete_runs": len(EXPECTED_EXTERNAL_ROUTES) - len(issues),
        "issues": issues,
        "elapsed_seconds": elapsed_by_route,
    }


def _validate_aggregate_metrics(
    summary: dict[str, str],
    chip_rows: list[dict[str, str]],
    event_rows: list[dict[str, str]] | None = None,
) -> tuple[bool, str]:
    try:
        counts = {
            name: sum(int(row[name]) for row in chip_rows)
            for name in ("tp", "fp", "fn", "tn")
        }
        if any(int(summary[name]) != value for name, value in counts.items()):
            return False, "summary confusion counts disagree with per-chip rows"
        expected_metrics = _metrics(**counts)
        if any(
            not _close(float(summary[name]), expected_metrics[name])
            for name in METRIC_NAMES
        ):
            return False, "summary metrics disagree with confusion counts"
        if int(summary["samples"]) != len(chip_rows):
            return False, "summary sample count disagrees with per-chip rows"
        if event_rows is None:
            return True, "pass"
        chip_by_event: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        for row in chip_rows:
            event_counts = chip_by_event[row["event"]]
            for name in event_counts:
                event_counts[name] += int(row[name])
        event_by_name = {row["event"]: row for row in event_rows}
        if set(event_by_name) != set(chip_by_event):
            return False, "per-event coverage disagrees with per-chip events"
        event_ious: list[float] = []
        for event, counts_for_event in chip_by_event.items():
            event_row = event_by_name[event]
            if any(
                int(event_row[name]) != value
                for name, value in counts_for_event.items()
            ):
                return False, f"per-event counts disagree for {event}"
            event_metrics = _metrics(**counts_for_event)
            if any(
                not _close(float(event_row[name]), event_metrics[name])
                for name in METRIC_NAMES
            ):
                return False, f"per-event metrics disagree for {event}"
            event_ious.append(event_metrics["iou"])
        expected_event_equal = sum(event_ious) / len(event_ious)
        if not _close(float(summary["event_equal_iou"]), expected_event_equal):
            return False, "event-equal IoU disagrees with per-event rows"
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return False, str(exc)
    return True, "pass"


def _audit_external_evaluations(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any], list[dict[str, str]]]:
    prefix = f"{root}/sen1floods11"
    selected_hash = _file_sha256(
        archive,
        f"{prefix}/sen1floods11_selected_manifest.json",
    )
    names = set(archive.namelist())
    issues: list[str] = []
    summary_rows_all: list[dict[str, str]] = []
    matched_max_difference = 0.0
    expected_condition_counts = {
        "s1_reference": 1,
        "early_fusion": 2,
        "early_fusion_dropout": 2,
        **{route: 16 for route in EXPECTED_QUALITY_ROUTES},
    }
    expected_summary_rows = 0
    expected_chip_rows = 0
    actual_chip_rows = 0
    for route in EXPECTED_EXTERNAL_ROUTES:
        conditions_name = f"{prefix}/conditions/{route}.json"
        if conditions_name not in names:
            issues.append(f"{route}: missing condition manifest")
            continue
        conditions_document = _read_json(archive, conditions_name)
        conditions = conditions_document.get("conditions", [])
        condition_ids = [str(condition.get("condition_id")) for condition in conditions]
        if (
            conditions_document.get("mode") != "smoke"
            or conditions_document.get("pipeline_only") is not True
            or conditions_document.get("route") != route
            or len(conditions) != expected_condition_counts[route]
            or len(condition_ids) != len(set(condition_ids))
        ):
            issues.append(f"{route}: invalid condition manifest")
            continue
        for split, samples in (("test", 12), ("bolivia", 4)):
            eval_prefix = f"{prefix}/evaluations/{route}/seed7/{split}"
            required = (
                "evaluation_config.json",
                "summary_metrics.csv",
                "per_chip_metrics.csv",
                "per_event_metrics.csv",
            )
            missing = [name for name in required if f"{eval_prefix}/{name}" not in names]
            if missing:
                issues.append(f"{route}/{split}: missing {','.join(missing)}")
                continue
            config = _read_json(archive, f"{eval_prefix}/evaluation_config.json")
            summary_rows = _read_csv(archive, f"{eval_prefix}/summary_metrics.csv")
            chip_rows = _read_csv(archive, f"{eval_prefix}/per_chip_metrics.csv")
            event_rows = _read_csv(archive, f"{eval_prefix}/per_event_metrics.csv")
            checkpoint_name = f"{prefix}/runs/{route}_seed7/best_clean.pt"
            config_ok = (
                config.get("route") == route
                and config.get("split") == split
                and config.get("sample_count") == samples
                and config.get("manifest_sha256") == selected_hash
                and config.get("checkpoint_sha256")
                == _file_sha256(archive, checkpoint_name)
                and config.get("perturb_seed") == 20260716
                and config.get("repetitions") == 1
                and config.get("conditions") == conditions
            )
            if not config_ok:
                issues.append(f"{route}/{split}: evaluation config mismatch")
            expected_keys = {(identifier, 0) for identifier in condition_ids}
            summary_by_key = {
                (row["condition_id"], int(row["repetition"])): row
                for row in summary_rows
            }
            chip_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
            event_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
            for row in chip_rows:
                chip_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
            for row in event_rows:
                event_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
            if (
                set(summary_by_key) != expected_keys
                or set(chip_by_key) != expected_keys
                or set(event_by_key) != expected_keys
            ):
                issues.append(f"{route}/{split}: condition coverage mismatch")
            for key in expected_keys & set(summary_by_key) & set(chip_by_key):
                summary = summary_by_key[key]
                chips = chip_by_key[key]
                events = event_by_key.get(key, [])
                identities_ok = (
                    summary.get("route") == route
                    and summary.get("split") == split
                    and int(summary.get("model_seed", -1)) == 7
                    and len(chips) == samples
                    and len({row.get("chip_id") for row in chips}) == samples
                    and all(
                        row.get("route") == route
                        and row.get("split") == split
                        and int(row.get("model_seed", -1)) == 7
                        for row in chips
                    )
                )
                aggregate_ok, reason = _validate_aggregate_metrics(
                    summary,
                    chips,
                    events,
                )
                if not identities_ok or not aggregate_ok:
                    issues.append(f"{route}/{split}/{key[0]}: {reason}")
                if summary.get("quality_mode") == "independent":
                    rate_fields = (
                        ("false_available_rate", "realized_false_available_rate"),
                        (
                            "false_unavailable_rate",
                            "realized_false_unavailable_rate",
                        ),
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
                        issues.append(f"{route}/{split}/{key[0]}: realized rate drift")
            if route in EXPECTED_QUALITY_ROUTES:
                rows_by_condition_chip = {
                    (row["condition_id"], row["chip_id"], int(row["repetition"])): row
                    for row in chip_rows
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
                    for chip_id in {
                        row["chip_id"]
                        for row in chip_rows
                        if row["condition_id"] == identifier
                    }:
                        structured = rows_by_condition_chip[(identifier, chip_id, 0)]
                        matched = rows_by_condition_chip[
                            (matched_identifier, chip_id, 0)
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
                                matched_max_difference,
                                difference,
                            )
            summary_rows_all.extend(summary_rows)
            expected_summary_rows += len(condition_ids)
            expected_chip_rows += len(condition_ids) * samples
            actual_chip_rows += len(chip_rows)
    if matched_max_difference > 1e-12:
        issues.append(
            "structured and matched-random controls differ inside the evaluation domain"
        )
    if len(summary_rows_all) != expected_summary_rows or expected_summary_rows != 170:
        issues.append("external summary row total is not the frozen 170")
    if actual_chip_rows != expected_chip_rows:
        issues.append("external per-chip row total is incomplete")
    finite = _all_finite(
        summary_rows_all,
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
        issues.append("external summaries contain missing or non-finite metrics")
    return not issues, {
        "summary_rows": len(summary_rows_all),
        "per_chip_rows": actual_chip_rows,
        "matched_control_max_rate_difference": matched_max_difference,
        "finite_metrics": finite,
        "issues": issues,
    }, summary_rows_all


def _audit_external_seed_summary(
    archive: ZipFile,
    root: str,
    raw_summary_rows: list[dict[str, str]],
) -> tuple[bool, dict[str, Any]]:
    table_rows = _read_csv(
        archive,
        f"{root}/sen1floods11/tables/sen1floods11_seed_condition_summary.csv",
    )
    raw_by_key = {
        (
            int(row["model_seed"]),
            row["split"],
            row["route"],
            row["condition_id"],
        ): row
        for row in raw_summary_rows
    }
    table_by_key = {
        (
            int(row["model_seed"]),
            row["split"],
            row["route"],
            row["condition_id"],
        ): row
        for row in table_rows
    }
    s1_by_split = {
        row["split"]: float(row["iou"])
        for row in raw_summary_rows
        if row["route"] == "s1_reference"
    }
    issues: list[str] = []
    if set(table_by_key) != set(raw_by_key) or len(table_rows) != 170:
        issues.append("seed-condition table coverage does not match raw summaries")
    for key in set(table_by_key) & set(raw_by_key):
        table = table_by_key[key]
        raw = raw_by_key[key]
        reference = s1_by_split[key[1]]
        if (
            int(table["repetitions"]) != 1
            or not _close(float(table["iou"]), float(raw["iou"]))
            or not _close(float(table["s1_reference_iou"]), reference)
            or not _close(
                float(table["delta_s1_iou"]),
                float(table["iou"]) - reference,
            )
        ):
            issues.append("seed-condition values disagree with raw paired summaries")
            break
    finite = _all_finite(
        table_rows,
        ("iou", "event_equal_iou", "s1_reference_iou", "delta_s1_iou"),
    )
    if not finite:
        issues.append("seed-condition table contains non-finite primary metrics")
    return not issues, {
        "rows": len(table_rows),
        "paired_s1_splits": sorted(s1_by_split),
        "finite_metrics": finite,
        "issues": issues,
    }


def _audit_ombria_runs_and_evaluations(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    names = set(archive.namelist())
    issues: list[str] = []
    for route in EXPECTED_OMBRIA_ROUTES:
        run_prefix = f"{root}/runs/{route}_seed7"
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
            archive,
            f"{run_prefix}/checkpoint_selection.json",
        )
        metric_rows = _read_csv(archive, f"{run_prefix}/metrics.csv")
        run_ok = (
            config.get("run_name") == f"{route}_seed7"
            and config.get("seed") == 7
            and config.get("epochs") == 2
            and config.get("base_channels") == 16
            and config.get("split_seed") == 20260710
            and config.get("eval_perturb_seed") == 20260716
            and selection.get("best_clean_sha256")
            == _file_sha256(archive, f"{run_prefix}/best_clean.pt")
            and selection.get("best_robust_sha256")
            == _file_sha256(archive, f"{run_prefix}/best_robust.pt")
            and len(metric_rows) == 2
            and _all_finite(
                metric_rows,
                (
                    "train_loss",
                    "val_loss",
                    "val_iou",
                    "val_f1",
                    "val_precision",
                    "val_recall",
                    "val_accuracy",
                ),
            )
        )
        if not run_ok:
            issues.append(f"{route}: configuration, hashes, or metrics disagree")

    expected_cells = {
        "s1_reference": ((0.0, 0.0),),
        **{
            route: tuple((fa, fu) for fa in EXPECTED_RATES for fu in EXPECTED_RATES)
            for route in EXPECTED_OMBRIA_ROUTES
            if route != "s1_reference"
        },
    }
    raw_summaries: list[dict[str, str]] = []
    for route, cells in expected_cells.items():
        for false_available, false_unavailable in cells:
            fa_key = str(false_available).replace(".", "p")
            fu_key = str(false_unavailable).replace(".", "p")
            if fa_key.endswith("p0"):
                fa_key = fa_key[:-2]
            if fu_key.endswith("p0"):
                fu_key = fu_key[:-2]
            eval_prefix = f"{root}/evaluations/{route}/fa{fa_key}_fu{fu_key}"
            required = (
                "evaluation_config.json",
                "summary_metrics.csv",
                "per_chip_metrics.csv",
            )
            missing = [name for name in required if f"{eval_prefix}/{name}" not in names]
            if missing:
                issues.append(f"{route}/fa{fa_key}_fu{fu_key}: missing files")
                continue
            config = _read_json(archive, f"{eval_prefix}/evaluation_config.json")
            summaries = _read_csv(archive, f"{eval_prefix}/summary_metrics.csv")
            chips = _read_csv(archive, f"{eval_prefix}/per_chip_metrics.csv")
            if len(summaries) != 1:
                issues.append(f"{route}/fa{fa_key}_fu{fu_key}: expected one summary")
                continue
            summary = summaries[0]
            aggregate_ok, reason = _validate_aggregate_metrics(summary, chips)
            cell_ok = (
                config.get("route") == route
                and config.get("content_degradation") == "cloud_after_50"
                and float(config.get("false_available_rate", -1))
                == false_available
                and float(config.get("false_unavailable_rate", -1))
                == false_unavailable
                and config.get("perturb_seed") == 20260716
                and config.get("repetitions") == 1
                and summary.get("route") == route
                and len(chips) == 70
                and aggregate_ok
            )
            if route != "s1_reference":
                cell_ok = cell_ok and (
                    abs(
                        float(summary["realized_false_available_rate"])
                        - false_available
                    )
                    <= 0.001
                    and abs(
                        float(summary["realized_false_unavailable_rate"])
                        - false_unavailable
                    )
                    <= 0.001
                )
            if not cell_ok:
                issues.append(f"{route}/fa{fa_key}_fu{fu_key}: {reason}")
            raw_summaries.append(summary)

    response_rows = _read_csv(archive, f"{root}/tables/response_surface.csv")
    response_keys = {
        (
            row["route"],
            float(row["requested_false_available_rate"]),
            float(row["requested_false_unavailable_rate"]),
        )
        for row in response_rows
    }
    expected_response_keys = {
        (route, fa, fu)
        for route, cells in expected_cells.items()
        for fa, fu in cells
    }
    s1_rows = [row for row in response_rows if row["route"] == "s1_reference"]
    if len(s1_rows) != 1:
        issues.append("OMBRIA response surface lacks one S1 reference")
    else:
        reference = float(s1_rows[0]["iou"])
        for row in response_rows:
            if (
                not _close(float(row["s1_reference_iou"]), reference)
                or not _close(
                    float(row["delta_s1_iou"]),
                    float(row["iou"]) - reference,
                )
            ):
                issues.append("OMBRIA response surface has invalid paired S1 deltas")
                break
    if response_keys != expected_response_keys or len(response_rows) != 37:
        issues.append("OMBRIA response surface coverage is not the frozen 37 cells")
    finite = _all_finite(
        raw_summaries,
        (
            *METRIC_NAMES,
            "realized_false_available_rate",
            "realized_false_unavailable_rate",
        ),
    ) and _all_finite(
        response_rows,
        ("iou", "s1_reference_iou", "delta_s1_iou"),
    )
    if not finite:
        issues.append("OMBRIA outputs contain non-finite primary metrics")
    return not issues, {
        "training_runs": len(EXPECTED_OMBRIA_ROUTES),
        "evaluation_cells": len(raw_summaries),
        "response_surface_rows": len(response_rows),
        "finite_metrics": finite,
        "issues": issues,
    }


def _audit_logs_and_payload(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    names = archive.namelist()
    raw_raster_pattern = re.compile(r"_(?:S1|S2|Label)Hand\.tiff?$", re.IGNORECASE)
    raw_rasters = [name for name in names if raw_raster_pattern.search(name)]
    log = archive.read(f"{root}/run.log").decode("utf-8", errors="replace")
    anomaly_patterns = {
        "traceback": r"\bTraceback \(most recent call last\)",
        "cuda_error": r"\bCUDA error\b",
        "out_of_memory": r"\b(?:out of memory|OOM)\b",
        "runtime_error": r"\bRuntimeError:\s",
        "non_finite": r"(?i)(?:^|[\s,:=])(nan|inf)(?:$|[\s,}])",
    }
    anomalies = {
        name: len(re.findall(pattern, log, flags=re.MULTILINE))
        for name, pattern in anomaly_patterns.items()
    }
    command_counts = {
        "external_training": log.count("scripts/train_sen1floods11_unet.py"),
        "external_evaluation": log.count(
            "scripts/evaluate_sen1floods11_quality_uncertainty.py"
        ),
        "ombria_training": log.count("scripts/train_ombria_unet.py"),
        "ombria_evaluation": log.count(
            "scripts/evaluate_ombria_quality_uncertainty.py"
        ),
    }
    passed = (
        not raw_rasters
        and not any(anomalies.values())
        and command_counts["external_training"] == 8
        and command_counts["external_evaluation"] == 16
        and '"status": "pass"' in log
    )
    return passed, {
        "raw_rasters": raw_rasters,
        "anomalies": anomalies,
        "command_counts": command_counts,
        "final_pass_marker": '"status": "pass"' in log,
    }


def audit_quality_uncertainty_smoke_artifact(
    archive_path: Path,
    *,
    alignment_visual_status: str = "not-reviewed",
) -> dict[str, Any]:
    """Audit one returned Smoke ZIP and decide whether Full may be released.

    The report is deliberately fail-closed: malformed or incomplete archives
    return failed checks instead of being treated as partial success.
    """

    archive_path = Path(archive_path)
    if alignment_visual_status not in {"not-reviewed", "pass", "fail"}:
        raise ValueError("alignment_visual_status must be not-reviewed, pass, or fail")

    checks: list[dict[str, Any]] = []
    artifact_sha256 = ""
    artifact_bytes = 0
    member_count = 0
    try:
        payload = archive_path.read_bytes()
        artifact_sha256 = _sha256(payload)
        artifact_bytes = len(payload)
        with ZipFile(archive_path) as archive:
            names = archive.namelist()
            member_count = len(names)
            safe = len(names) == len(set(names)) and all(
                _safe_member(name) for name in names
            )
            checks.append(
                _check(
                    "archive_path_safety",
                    safe,
                    {
                        "members": len(names),
                        "unique_members": len(set(names)),
                    },
                )
            )
            bad_member = archive.testzip()
            checks.append(
                _check("archive_crc", bad_member is None, {"bad_member": bad_member})
            )

            manifest_names = [
                name for name in names if name.endswith("/artifact_manifest.json")
            ]
            manifest_pass = len(manifest_names) == 1
            root = ""
            manifest_detail: dict[str, Any] = {
                "manifest_count": len(manifest_names),
            }
            if manifest_pass:
                manifest_name = manifest_names[0]
                try:
                    manifest = json.loads(archive.read(manifest_name))
                    root = str(manifest.get("root", ""))
                    records = manifest.get("files", [])
                    expected_names = {
                        f"{root}/{record['path']}" for record in records
                    }
                    actual_names = set(names) - {manifest_name}
                    record_names = [str(record.get("path", "")) for record in records]
                    manifest_pass = (
                        manifest.get("schema") == ARTIFACT_SCHEMA
                        and manifest_name == f"{root}/artifact_manifest.json"
                        and len(record_names) == len(set(record_names))
                        and expected_names == actual_names
                    )
                    mismatched: list[str] = []
                    if manifest_pass:
                        for record in records:
                            member_name = f"{root}/{record['path']}"
                            member = archive.read(member_name)
                            if (
                                len(member) != int(record["bytes"])
                                or _sha256(member) != record["sha256"]
                            ):
                                mismatched.append(str(record["path"]))
                    manifest_pass = manifest_pass and not mismatched
                    manifest_detail.update(
                        {
                            "schema": manifest.get("schema"),
                            "root": root,
                            "records": len(records),
                            "mismatched": mismatched,
                            "coverage_match": expected_names == actual_names,
                        }
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    manifest_pass = False
                    manifest_detail["error"] = str(exc)
            checks.append(
                _check(
                    "artifact_manifest_integrity",
                    manifest_pass,
                    manifest_detail,
                )
            )

            required_suffixes = (
                "/runtime_manifest.json",
                "/experiment_manifest.json",
                "/sen1floods11/experiment_plan.json",
                "/sen1floods11/sen1floods11_decision_gate.json",
            )
            missing = [
                suffix
                for suffix in required_suffixes
                if not any(name.endswith(suffix) for name in names)
            ]
            checks.append(
                _check(
                    "protocol_completeness",
                    not missing,
                    {"missing": missing},
                )
            )
            if manifest_pass and not missing:
                try:
                    runtime_ok, runtime_detail = _audit_runtime(archive, root)
                    checks.append(
                        _check("runtime_and_source_pin", runtime_ok, runtime_detail)
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check("runtime_and_source_pin", False, {"error": str(exc)})
                    )
                try:
                    preparation_ok, preparation_detail = _audit_external_preparation(
                        archive,
                        root,
                    )
                    checks.append(
                        _check(
                            "external_data_preparation",
                            preparation_ok,
                            preparation_detail,
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check(
                            "external_data_preparation",
                            False,
                            {"error": str(exc)},
                        )
                    )
                try:
                    alignment_ok, alignment_detail = _audit_alignment(archive, root)
                    checks.append(
                        _check(
                            "alignment_automated",
                            alignment_ok,
                            alignment_detail,
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check("alignment_automated", False, {"error": str(exc)})
                    )
                try:
                    training_ok, training_detail = _audit_external_training_runs(
                        archive,
                        root,
                    )
                    checks.append(
                        _check(
                            "external_training_runs",
                            training_ok,
                            training_detail,
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check(
                            "external_training_runs",
                            False,
                            {"error": str(exc)},
                        )
                    )
                external_rows: list[dict[str, str]] = []
                try:
                    evaluation_ok, evaluation_detail, external_rows = (
                        _audit_external_evaluations(archive, root)
                    )
                    checks.append(
                        _check(
                            "external_evaluation_integrity",
                            evaluation_ok,
                            evaluation_detail,
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check(
                            "external_evaluation_integrity",
                            False,
                            {"error": str(exc)},
                        )
                    )
                try:
                    summary_ok, summary_detail = _audit_external_seed_summary(
                        archive,
                        root,
                        external_rows,
                    )
                    checks.append(
                        _check(
                            "external_seed_summary",
                            summary_ok,
                            summary_detail,
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check(
                            "external_seed_summary",
                            False,
                            {"error": str(exc)},
                        )
                    )
                try:
                    ombria_ok, ombria_detail = _audit_ombria_runs_and_evaluations(
                        archive,
                        root,
                    )
                    checks.append(
                        _check("ombria_integrity", ombria_ok, ombria_detail)
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check("ombria_integrity", False, {"error": str(exc)})
                    )
                try:
                    log_ok, log_detail = _audit_logs_and_payload(archive, root)
                    checks.append(
                        _check("logs_and_payload", log_ok, log_detail)
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check("logs_and_payload", False, {"error": str(exc)})
                    )
                try:
                    external_gate = _read_json(
                        archive,
                        f"{root}/sen1floods11/"
                        "sen1floods11_decision_gate.json",
                    )
                    experiment = _read_json(
                        archive,
                        f"{root}/experiment_manifest.json",
                    )
                    boundary_ok = (
                        experiment.get("pipeline_only") is True
                        and external_gate.get("pipeline_only") is True
                        and external_gate.get("scientific_interpretation_allowed")
                        is False
                        and "prohibited" in external_gate.get("claim_boundary", "")
                    )
                    checks.append(
                        _check(
                            "smoke_claim_boundary",
                            boundary_ok,
                            {
                                "pipeline_only": external_gate.get("pipeline_only"),
                                "scientific_interpretation_allowed": (
                                    external_gate.get(
                                        "scientific_interpretation_allowed"
                                    )
                                ),
                                "claim_boundary": external_gate.get(
                                    "claim_boundary"
                                ),
                            },
                        )
                    )
                except Exception as exc:  # fail-closed audit boundary
                    checks.append(
                        _check("smoke_claim_boundary", False, {"error": str(exc)})
                    )
            else:
                for check_id in (
                    "runtime_and_source_pin",
                    "external_data_preparation",
                    "alignment_automated",
                    "external_training_runs",
                    "external_evaluation_integrity",
                    "external_seed_summary",
                    "ombria_integrity",
                    "logs_and_payload",
                    "smoke_claim_boundary",
                ):
                    checks.append(
                        _check(
                            check_id,
                            False,
                            {"error": "archive manifest or protocol is incomplete"},
                        )
                    )
    except (OSError, BadZipFile) as exc:
        checks.extend(
            [
                _check("archive_path_safety", False, {"error": str(exc)}),
                _check("archive_crc", False, {"error": str(exc)}),
                _check(
                    "artifact_manifest_integrity",
                    False,
                    {"error": str(exc)},
                ),
                _check("protocol_completeness", False, {"error": str(exc)}),
            ]
        )

    checks.append(
        _check(
            "alignment_visual_review",
            alignment_visual_status == "pass",
            {"review_status": alignment_visual_status},
        )
    )
    full_authorized = all(check["status"] == "pass" for check in checks)
    return {
        "schema": "geoai-quality-map-uncertainty-smoke-audit-v1",
        "artifact": {
            "path": str(archive_path.resolve()),
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "members": member_count,
        },
        "checks": checks,
        "decision": {
            "status": "pass" if full_authorized else "fail",
            "full_authorized": full_authorized,
            "smoke_scores_publishable": False,
            "claim_boundary": (
                "Smoke validates execution and packaging only; its scores are "
                "prohibited from manuscript claims."
            ),
        },
    }


def render_quality_uncertainty_audit_markdown(report: dict[str, Any]) -> str:
    """Render a compact reviewer-facing audit receipt from an audit report."""

    artifact = report["artifact"]
    decision = report["decision"]
    lines = [
        "# Quality-map uncertainty Smoke artifact audit",
        "",
        f"- Decision: **{decision['status'].upper()}**",
        f"- Full authorized: **{str(decision['full_authorized']).lower()}**",
        f"- Artifact: `{artifact['path']}`",
        f"- SHA-256: `{artifact['sha256']}`",
        f"- Bytes / members: {artifact['bytes']} / {artifact['members']}",
        "- Smoke scores publishable: **false**",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True, ensure_ascii=False)
        if len(detail) > 360:
            detail = detail[:357] + "..."
        detail = detail.replace("|", "\\|")
        lines.append(f"| `{check['id']}` | {check['status']} | `{detail}` |")
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            decision["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)
