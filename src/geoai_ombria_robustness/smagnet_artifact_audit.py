from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from .quality_uncertainty_artifact_audit import (
    ARTIFACT_SCHEMA,
    METRIC_NAMES,
    _all_finite,
    _check,
    _close,
    _metrics,
    _read_csv,
    _read_json,
    _safe_member,
    _validate_aggregate_metrics,
)


EXPECTED_ROOT = "quality_uncertainty_smagnet_smoke"
EXPECTED_SOURCE_COMMIT = "8b5a4f9ed7d0393a3b9259451f7e7dd3089f5d64"
EXPECTED_OFFICIAL_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
EXPECTED_OFFICIAL_SOURCE_SHA256 = (
    "daf00d0533ca7865b4bd7b47404f1c0fa42e4a0bdc70706dee45bedcc1420f25"
)
EXPECTED_OFFICIAL_LICENSE_SHA256 = (
    "4261bd84b3a36788cb1bb4e25d3f59a2cf2ac79abb93cb45cf09fc043b39265c"
)
EXPECTED_PAPER_DOI = "10.1016/j.isprsjprs.2025.12.023"
EXPECTED_FULL_SEEDS = (7, 13, 21, 29, 37)
EXPECTED_SPLIT_COUNTS = {
    "train": 24,
    "validation": 12,
    "test": 12,
    "bolivia": 4,
}
EXPECTED_CONDITION_IDS = (
    "independent_fa0_fu0",
    "independent_fa0_fu0p2",
    "independent_fa0_fu0p4",
    "independent_fa0p2_fu0",
    "independent_fa0p2_fu0p2",
    "independent_fa0p2_fu0p4",
    "independent_fa0p4_fu0",
    "independent_fa0p4_fu0p2",
    "independent_fa0p4_fu0p4",
    "translate_east_5pct",
    "matched_random__translate_east_5pct",
    "dilate_unavailable_r8",
    "matched_random__dilate_unavailable_r8",
    "erode_unavailable_r8",
    "matched_random__erode_unavailable_r8",
    "complete_absence",
)
EXPECTED_ARCHITECTURE = {
    "activation": None,
    "classes": 1,
    "decoder_channels": [256, 128, 64, 32, 16],
    "decoder_use_batchnorm": False,
    "enable_spatial_mask": True,
    "encoder_depth": 5,
    "encoder_name": "resnet50",
    "encoder_weights_msi": "imagenet",
    "encoder_weights_sar": None,
    "sarmsiff_method": "sar_msi_gated",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member_sha256(archive: ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _audit_manifest(
    archive: ZipFile,
) -> tuple[bool, dict[str, Any], str, dict[str, str]]:
    names = archive.namelist()
    manifest_names = [
        name for name in names if name.endswith("/artifact_manifest.json")
    ]
    if len(manifest_names) != 1:
        return False, {"manifest_count": len(manifest_names)}, "", {}
    manifest_name = manifest_names[0]
    manifest = _read_json(archive, manifest_name)
    root = str(manifest.get("root", ""))
    records = manifest.get("files", [])
    if not isinstance(records, list):
        return False, {"error": "manifest files is not a list"}, root, {}
    record_names = [str(record.get("path", "")) for record in records]
    expected_names = {f"{root}/{name}" for name in record_names}
    actual_names = set(names) - {manifest_name}
    mismatched: list[str] = []
    hashes: dict[str, str] = {}
    structurally_valid = (
        manifest.get("schema") == ARTIFACT_SCHEMA
        and root == EXPECTED_ROOT
        and manifest_name == f"{root}/artifact_manifest.json"
        and len(record_names) == len(set(record_names))
        and expected_names == actual_names
    )
    if structurally_valid:
        for record in records:
            member_name = f"{root}/{record['path']}"
            digest = _member_sha256(archive, member_name)
            hashes[member_name] = digest
            if (
                archive.getinfo(member_name).file_size != int(record["bytes"])
                or digest != record["sha256"]
            ):
                mismatched.append(str(record["path"]))
    passed = structurally_valid and not mismatched
    return passed, {
        "schema": manifest.get("schema"),
        "root": root,
        "records": len(records),
        "coverage_match": expected_names == actual_names,
        "mismatched": mismatched,
    }, root, hashes


def _audit_runtime_and_official_source(
    archive: ZipFile,
    root: str,
    hashes: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    runtime = _read_json(archive, f"{root}/runtime_manifest.json")
    source = _read_json(archive, f"{root}/official_source_manifest.json")
    copied_source = _read_json(
        archive,
        f"{root}/official_source/official_source_manifest.json",
    )
    source_name = f"{root}/official_source/smagnet.py"
    license_name = f"{root}/official_source/LICENSE"
    required_arch = runtime.get("required_arch")
    passed = (
        runtime.get("repository_commit") == EXPECTED_SOURCE_COMMIT
        and runtime.get("repository_dirty_tracked") is False
        and runtime.get("cuda_conv2d_gate") == "pass"
        and runtime.get("device") not in {None, "", "cpu"}
        and required_arch in runtime.get("compiled_arches", [])
        and source == copied_source
        and source.get("commit") == EXPECTED_OFFICIAL_COMMIT
        and source.get("checkout_commit") == EXPECTED_OFFICIAL_COMMIT
        and source.get("source_sha256") == EXPECTED_OFFICIAL_SOURCE_SHA256
        and source.get("license_sha256") == EXPECTED_OFFICIAL_LICENSE_SHA256
        and source.get("paper_doi") == EXPECTED_PAPER_DOI
        and source.get("model_configuration") == EXPECTED_ARCHITECTURE
        and hashes.get(source_name) == EXPECTED_OFFICIAL_SOURCE_SHA256
        and hashes.get(license_name) == EXPECTED_OFFICIAL_LICENSE_SHA256
    )
    return passed, {
        "repository_commit": runtime.get("repository_commit"),
        "repository_dirty_tracked": runtime.get("repository_dirty_tracked"),
        "device": runtime.get("device"),
        "capability": runtime.get("capability"),
        "required_arch": required_arch,
        "cuda_conv2d_gate": runtime.get("cuda_conv2d_gate"),
        "torch": runtime.get("torch"),
        "torch_cuda": runtime.get("torch_cuda"),
        "official_commit": source.get("commit"),
        "official_source_sha256": source.get("source_sha256"),
        "official_license_sha256": source.get("license_sha256"),
    }


def _audit_protocol(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    gate = _read_json(archive, f"{root}/published_architecture_gate.json")
    plan = _read_json(archive, f"{root}/experiment_plan.json")
    condition_document = _read_json(archive, f"{root}/smagnet_conditions.json")
    conditions = condition_document.get("conditions", [])
    condition_ids = tuple(str(item.get("condition_id")) for item in conditions)
    fallback = gate.get("fallback_boundary", {})
    training = gate.get("training", {})
    passed = (
        gate.get("schema") == "geoai-quality-map-uncertainty-smagnet-gate-v1"
        and gate.get("status") == "pass"
        and gate.get("mode") == "smoke"
        and gate.get("pipeline_only") is True
        and gate.get("scientific_interpretation_allowed") is False
        and gate.get("model_seed") == 7
        and gate.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and gate.get("architecture") == EXPECTED_ARCHITECTURE
        and gate.get("paper_doi") == EXPECTED_PAPER_DOI
        and gate.get("condition_count") == len(EXPECTED_CONDITION_IDS)
        and gate.get("repetitions") == 1
        and gate.get("summary_rows") == {"bolivia": 16, "test": 16}
        and gate.get("per_chip_rows") == {"bolivia": 64, "test": 192}
        and gate.get("finite_metrics") is True
        and training.get("epochs") == 2
        and training.get("effective_batch_size") == 16
        and training.get("parameter_count") == 56_035_958
        and training.get("optimizer") == "Adam"
        and fallback.get("status") == "pass"
        and fallback.get("maximum_fused_sar_logit_difference") == 0.0
        and fallback.get("maximum_masked_gate") == 0.0
        and float(fallback.get("tolerance", math.inf)) == 1e-6
        and plan.get("schema") == "geoai-quality-map-uncertainty-smagnet-plan-v1"
        and plan.get("mode") == "smoke"
        and plan.get("pipeline_only") is True
        and plan.get("seed") == 7
        and tuple(plan.get("planned_full_seeds", ())) == EXPECTED_FULL_SEEDS
        and plan.get("epochs") == 2
        and plan.get("repetitions") == 1
        and plan.get("official_commit") == EXPECTED_OFFICIAL_COMMIT
        and plan.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and condition_document.get("mode") == "smoke"
        and condition_document.get("pipeline_only") is True
        and condition_document.get("route") == "smagnet_official"
        and condition_ids == EXPECTED_CONDITION_IDS
    )
    return passed, {
        "gate_status": gate.get("status"),
        "mode": gate.get("mode"),
        "pipeline_only": gate.get("pipeline_only"),
        "scientific_interpretation_allowed": gate.get(
            "scientific_interpretation_allowed"
        ),
        "planned_full_seeds": plan.get("planned_full_seeds"),
        "condition_ids": list(condition_ids),
        "epochs": training.get("epochs"),
        "effective_batch_size": training.get("effective_batch_size"),
        "parameter_count": training.get("parameter_count"),
        "fallback": fallback,
    }, conditions


def _audit_data_preparation(
    archive: ZipFile,
    root: str,
    hashes: dict[str, str],
) -> tuple[bool, dict[str, Any], dict[str, list[str]]]:
    selected_name = f"{root}/sen1floods11_selected_manifest.json"
    selected = _read_json(archive, selected_name)
    preparation = _read_json(
        archive,
        f"{root}/sen1floods11_preparation_report.json",
    )
    records = selected.get("records", [])
    prepared = preparation.get("records", [])
    split_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        split_ids[str(record.get("split"))].append(str(record.get("chip_id")))
    split_counts = {key: len(values) for key, values in split_ids.items()}
    record_ids = [str(record.get("chip_id")) for record in records]
    prepared_ids = [str(record.get("chip_id")) for record in prepared]
    provider_counts = Counter(
        str(asset.get("provider"))
        for record in records
        for asset in record.get("scl_assets", [])
    )
    passed = (
        selected.get("schema") == "geoai-sen1floods11-scl-manifest-v1"
        and selected.get("experiment_mode") == "smoke"
        and selected.get("pipeline_only") is True
        and selected.get("summary", {}).get("record_count") == 52
        and selected.get("summary", {}).get("split_counts")
        == EXPECTED_SPLIT_COUNTS
        and split_counts == EXPECTED_SPLIT_COUNTS
        and len(record_ids) == len(set(record_ids)) == 52
        and preparation.get("schema") == "geoai-sen1floods11-preparation-report-v1"
        and preparation.get("status") == "pass"
        and preparation.get("mode") == "smoke"
        and preparation.get("pipeline_only") is True
        and preparation.get("record_count") == 52
        and set(record_ids) == set(prepared_ids)
        and preparation.get("selected_manifest_sha256") == hashes[selected_name]
        and "official S2Hand chip valid-data mask"
        in preparation.get("reference_quality", "")
        and provider_counts["earth-search"] > 0
        and provider_counts["planetary-computer"] > 0
    )
    return passed, {
        "records": len(record_ids),
        "unique_records": len(set(record_ids)),
        "split_counts": split_counts,
        "events": len({str(record.get("event")) for record in records}),
        "provider_counts": dict(provider_counts),
        "selected_manifest_sha256": hashes.get(selected_name),
        "preparation_status": preparation.get("status"),
        "reference_quality": preparation.get("reference_quality"),
    }, split_ids


def _audit_training(
    archive: ZipFile,
    root: str,
    hashes: dict[str, str],
    split_ids: dict[str, list[str]],
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/runs/smagnet_official_seed7"
    config = _read_json(archive, f"{prefix}/config.json")
    checkpoint = _read_json(archive, f"{prefix}/checkpoint_manifest.json")
    threshold = _read_json(archive, f"{prefix}/threshold_selection.json")
    fallback = _read_json(archive, f"{prefix}/fallback_boundary.json")
    normalization = _read_json(archive, f"{prefix}/normalization.json")
    splits = _read_json(archive, f"{prefix}/splits.json")
    rows = _read_csv(archive, f"{prefix}/metrics.csv")
    best_checkpoint_name = f"{prefix}/best_validation_loss.pt"
    threshold_name = f"{prefix}/threshold_selection.json"
    fallback_name = f"{prefix}/fallback_boundary.json"
    val_losses = [float(row["val_loss"]) for row in rows]
    min_index = val_losses.index(min(val_losses))
    passed = (
        config.get("architecture") == "official_smagnet"
        and config.get("route") == "smagnet_official"
        and config.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and config.get("seed") == 7
        and config.get("epochs") == 2
        and config.get("train_count") == 24
        and config.get("validation_count") == 12
        and config.get("validation_patches") == 48
        and config.get("micro_batch_size") == 4
        and config.get("gradient_accumulation") == 4
        and config.get("effective_batch_size") == 16
        and config.get("model_parameters") == 56_035_958
        and config.get("device") == "cuda"
        and config.get("amp") is True
        and config.get("amp_effective") is True
        and config.get("segmentation_models_pytorch") == "0.5.0"
        and config.get("official_model_configuration") == EXPECTED_ARCHITECTURE
        and config.get("normalization") == normalization
        and normalization.get("source") == "frozen training records only"
        and normalization.get("optical_order")
        == ["B4_red", "B3_green", "B2_blue", "B8_nir"]
        and normalization.get("radar_order") == ["VV", "VH"]
        and set(splits.get("train", [])) == set(split_ids["train"])
        and set(splits.get("validation", [])) == set(split_ids["validation"])
        and len(rows) == 2
        and [int(row["epoch"]) for row in rows] == [1, 2]
        and _all_finite(
            rows,
            (
                "train_loss",
                "val_loss",
                "val_iou_at_0p5",
                "val_f1_at_0p5",
                "val_precision_at_0p5",
                "val_recall_at_0p5",
                "val_accuracy_at_0p5",
                "elapsed_seconds",
            ),
        )
        and checkpoint.get("best_checkpoint_sha256")
        == hashes.get(best_checkpoint_name)
        and checkpoint.get("best_validation_loss_epoch")
        == int(rows[min_index]["epoch"])
        and _close(checkpoint.get("best_validation_loss"), min(val_losses))
        and checkpoint.get("threshold_selection_sha256")
        == hashes.get(threshold_name)
        and checkpoint.get("fallback_boundary_sha256")
        == hashes.get(fallback_name)
        and _close(checkpoint.get("threshold"), threshold.get("threshold"))
        and threshold.get("selection_split") == "validation"
        and threshold.get("selection_rule")
        == "precision_recall_threshold_maximizing_pixel_iou"
        and math.isfinite(float(threshold.get("threshold")))
        and 0.0 <= float(threshold.get("threshold")) <= 1.0
        and fallback.get("status") == "pass"
        and fallback.get("maximum_fused_sar_logit_difference") == 0.0
        and fallback.get("maximum_masked_gate") == 0.0
    )
    return passed, {
        "epochs": len(rows),
        "best_epoch": checkpoint.get("best_validation_loss_epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "threshold": threshold.get("threshold"),
        "checkpoint_sha256": hashes.get(best_checkpoint_name),
        "parameter_count": config.get("model_parameters"),
        "effective_batch_size": config.get("effective_batch_size"),
        "amp_effective": config.get("amp_effective"),
        "fallback": fallback,
    }


def _row_metrics_valid(row: dict[str, str]) -> bool:
    try:
        counts = {name: int(row[name]) for name in ("tp", "fp", "fn", "tn")}
        expected = _metrics(**counts)
        return (
            sum(counts.values()) == int(row.get("valid_target_pixels", sum(counts.values())))
            and all(_close(float(row[name]), expected[name]) for name in METRIC_NAMES)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _audit_evaluation_split(
    archive: ZipFile,
    root: str,
    split: str,
    split_ids: dict[str, list[str]],
    conditions: list[dict[str, Any]],
    hashes: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/evaluations/seed7/{split}"
    config = _read_json(archive, f"{prefix}/evaluation_config.json")
    summaries = _read_csv(archive, f"{prefix}/summary_metrics.csv")
    chips = _read_csv(archive, f"{prefix}/per_chip_metrics.csv")
    events = _read_csv(archive, f"{prefix}/per_event_metrics.csv")
    expected_ids = set(EXPECTED_CONDITION_IDS)
    summary_keys = {(row["condition_id"], int(row["repetition"])) for row in summaries}
    chips_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    events_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in chips:
        chips_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
    for row in events:
        events_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
    issues: list[str] = []
    for summary in summaries:
        key = (summary["condition_id"], int(summary["repetition"]))
        aggregate_ok, reason = _validate_aggregate_metrics(
            summary,
            chips_by_key[key],
            events_by_key[key],
        )
        if not aggregate_ok:
            issues.append(f"{key}: {reason}")
    if not all(_row_metrics_valid(row) for row in chips):
        issues.append("per-chip metrics do not reconstruct from confusion counts")
    if not all(_row_metrics_valid(row) for row in events):
        issues.append("per-event metrics do not reconstruct from confusion counts")
    rows_by_condition_chip = {
        (row["condition_id"], row["chip_id"]): row for row in chips
    }
    max_matched_rate_difference = 0.0
    for identifier in (
        "translate_east_5pct",
        "dilate_unavailable_r8",
        "erode_unavailable_r8",
    ):
        matched_identifier = f"matched_random__{identifier}"
        for chip_id in split_ids[split]:
            structured = rows_by_condition_chip[(identifier, chip_id)]
            matched = rows_by_condition_chip[(matched_identifier, chip_id)]
            for field in (
                "quality_false_available_rate",
                "quality_false_unavailable_rate",
                "valid_quality_false_available_rate",
                "valid_quality_false_unavailable_rate",
            ):
                max_matched_rate_difference = max(
                    max_matched_rate_difference,
                    abs(float(structured[field]) - float(matched[field])),
                )
    complete_absence = next(
        row for row in summaries if row["condition_id"] == "complete_absence"
    )
    checkpoint_config = config.get("checkpoint_config", {})
    fallback = config.get("fallback_boundary", {})
    expected_summary_keys = {(identifier, 0) for identifier in expected_ids}
    expected_chip_rows = len(expected_ids) * len(split_ids[split])
    expected_events = len(
        {
            row["event"]
            for row in chips
            if row["condition_id"] == "independent_fa0_fu0"
        }
    )
    checkpoint_name = f"{root}/runs/smagnet_official_seed7/best_validation_loss.pt"
    passed = (
        config.get("split") == split
        and config.get("model_seed") == 7
        and checkpoint_config.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and checkpoint_config.get("seed") == 7
        and checkpoint_config.get("official_model_configuration")
        == EXPECTED_ARCHITECTURE
        and config.get("repetitions") == 1
        and config.get("perturb_seed") == 20260716
        and config.get("patch_size") == 256
        and config.get("amp") is True
        and config.get("amp_effective") is True
        and config.get("conditions") == conditions
        and config.get("checkpoint_sha256") == hashes.get(checkpoint_name)
        and config.get("manifest_sha256")
        == hashes.get(f"{root}/sen1floods11_selected_manifest.json")
        and fallback.get("status") == "pass"
        and fallback.get("maximum_fused_sar_logit_difference") == 0.0
        and fallback.get("maximum_masked_gate") == 0.0
        and summary_keys == expected_summary_keys
        and set(chips_by_key) == expected_summary_keys
        and set(events_by_key) == expected_summary_keys
        and len(summaries) == len(expected_ids)
        and len(chips) == expected_chip_rows
        and all(len(chips_by_key[key]) == len(split_ids[split]) for key in summary_keys)
        and all(
            len(events_by_key[key]) == expected_events for key in summary_keys
        )
        and {row["chip_id"] for row in chips} == set(split_ids[split])
        and _all_finite(
            summaries,
            (
                *METRIC_NAMES,
                "event_equal_iou",
                "realized_false_available_rate",
                "realized_false_unavailable_rate",
                "valid_realized_false_available_rate",
                "valid_realized_false_unavailable_rate",
            ),
        )
        and max_matched_rate_difference <= 1e-12
        and _close(float(complete_absence["realized_false_unavailable_rate"]), 1.0)
        and _close(
            float(complete_absence["valid_realized_false_unavailable_rate"]),
            1.0,
        )
        and not issues
    )
    return passed, {
        "summary_rows": len(summaries),
        "per_chip_rows": len(chips),
        "per_event_rows": len(events),
        "samples_per_condition": len(split_ids[split]),
        "events_per_condition": expected_events,
        "matched_control_max_rate_difference": max_matched_rate_difference,
        "finite_metrics": _all_finite(summaries, METRIC_NAMES),
        "issues": issues,
    }


def _audit_logs_and_payload(
    archive: ZipFile,
    root: str,
) -> tuple[bool, dict[str, Any]]:
    names = archive.namelist()
    raw_pattern = re.compile(r"_(?:S1|S2|Label)Hand\.tiff?$", re.IGNORECASE)
    raw_rasters = [name for name in names if raw_pattern.search(name)]
    log = archive.read(f"{root}/run.log").decode("utf-8", errors="replace")
    anomaly_patterns = {
        "traceback": r"\bTraceback \(most recent call last\)",
        "called_process_error": r"\bCalledProcessError\b",
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
        "training": log.count("scripts/train_sen1floods11_smagnet.py"),
        "evaluation": log.count(
            "scripts/evaluate_sen1floods11_smagnet_quality_uncertainty.py"
        ),
    }
    passed = (
        not raw_rasters
        and not any(anomalies.values())
        and command_counts == {"training": 1, "evaluation": 2}
        and '"status": "pass"' in log
        and "official SMAGNet quality-uncertainty smoke" in log
    )
    return passed, {
        "raw_rasters": raw_rasters,
        "anomalies": anomalies,
        "command_counts": command_counts,
        "final_pass_marker": '"status": "pass"' in log,
    }


def audit_smagnet_smoke_artifact(archive_path: Path) -> dict[str, Any]:
    """Fail-closed audit of one returned official-SMAGNet Smoke archive."""

    archive_path = Path(archive_path)
    checks: list[dict[str, Any]] = []
    artifact_bytes = 0
    artifact_sha256 = ""
    member_count = 0
    try:
        artifact_bytes = archive_path.stat().st_size
        artifact_sha256 = _file_sha256(archive_path)
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
                    {"members": len(names), "unique_members": len(set(names))},
                )
            )
            bad_member = archive.testzip()
            checks.append(
                _check("archive_crc", bad_member is None, {"bad_member": bad_member})
            )
            manifest_ok, manifest_detail, root, hashes = _audit_manifest(archive)
            checks.append(
                _check("artifact_manifest_integrity", manifest_ok, manifest_detail)
            )
            if manifest_ok:
                audits = (
                    (
                        "runtime_and_official_source",
                        lambda: _audit_runtime_and_official_source(
                            archive, root, hashes
                        ),
                    ),
                    (
                        "frozen_protocol_and_boundary",
                        lambda: _audit_protocol(archive, root)[:2],
                    ),
                    (
                        "data_preparation",
                        lambda: _audit_data_preparation(archive, root, hashes)[:2],
                    ),
                    (
                        "training_and_checkpoint",
                        lambda: _audit_training(
                            archive,
                            root,
                            hashes,
                            _audit_data_preparation(archive, root, hashes)[2],
                        ),
                    ),
                )
                for check_id, audit in audits:
                    try:
                        passed, detail = audit()
                    except Exception as exc:  # fail-closed audit boundary
                        passed, detail = False, {"error": str(exc)}
                    checks.append(_check(check_id, passed, detail))
                try:
                    protocol_ok, _, conditions = _audit_protocol(archive, root)
                    data_ok, _, split_ids = _audit_data_preparation(
                        archive, root, hashes
                    )
                except Exception as exc:  # fail-closed audit boundary
                    protocol_ok = data_ok = False
                    conditions = []
                    split_ids = defaultdict(list)
                    checks.append(
                        _check("evaluation_prerequisites", False, {"error": str(exc)})
                    )
                if protocol_ok and data_ok:
                    for split in ("test", "bolivia"):
                        try:
                            passed, detail = _audit_evaluation_split(
                                archive,
                                root,
                                split,
                                split_ids,
                                conditions,
                                hashes,
                            )
                        except Exception as exc:  # fail-closed audit boundary
                            passed, detail = False, {"error": str(exc)}
                        checks.append(
                            _check(f"evaluation_{split}_integrity", passed, detail)
                        )
                else:
                    for split in ("test", "bolivia"):
                        checks.append(
                            _check(
                                f"evaluation_{split}_integrity",
                                False,
                                {"error": "protocol or data audit failed"},
                            )
                        )
                try:
                    log_ok, log_detail = _audit_logs_and_payload(archive, root)
                except Exception as exc:  # fail-closed audit boundary
                    log_ok, log_detail = False, {"error": str(exc)}
                checks.append(_check("logs_and_payload", log_ok, log_detail))
            else:
                for check_id in (
                    "runtime_and_official_source",
                    "frozen_protocol_and_boundary",
                    "data_preparation",
                    "training_and_checkpoint",
                    "evaluation_test_integrity",
                    "evaluation_bolivia_integrity",
                    "logs_and_payload",
                ):
                    checks.append(
                        _check(
                            check_id,
                            False,
                            {"error": "artifact manifest is incomplete"},
                        )
                    )
    except (OSError, BadZipFile) as exc:
        checks.extend(
            [
                _check("archive_path_safety", False, {"error": str(exc)}),
                _check("archive_crc", False, {"error": str(exc)}),
                _check(
                    "artifact_manifest_integrity", False, {"error": str(exc)}
                ),
            ]
        )
    full_authorized = all(check["status"] == "pass" for check in checks)
    return {
        "schema": "geoai-quality-map-uncertainty-smagnet-smoke-audit-v1",
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
            "scientific_interpretation_allowed": False,
            "claim_boundary": (
                "Smoke validates the official-source execution, fallback identity, "
                "evaluation, and packaging pipeline only. Its scores are prohibited "
                "from manuscript claims."
            ),
        },
    }


def render_smagnet_smoke_audit_markdown(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    decision = report["decision"]
    lines = [
        "# Official SMAGNet Smoke artifact audit",
        "",
        f"- Decision: **{decision['status'].upper()}**",
        f"- Full authorized: **{str(decision['full_authorized']).lower()}**",
        f"- Artifact: `{artifact['path']}`",
        f"- SHA-256: `{artifact['sha256']}`",
        f"- Bytes / members: {artifact['bytes']} / {artifact['members']}",
        "- Smoke scores publishable: **false**",
        "- Scientific interpretation allowed: **false**",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True, ensure_ascii=False)
        if len(detail) > 420:
            detail = detail[:417] + "..."
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
