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
from .smagnet_artifact_audit import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_FULL_SEEDS,
    EXPECTED_OFFICIAL_COMMIT,
    EXPECTED_OFFICIAL_LICENSE_SHA256,
    EXPECTED_OFFICIAL_SOURCE_SHA256,
    EXPECTED_PAPER_DOI,
    EXPECTED_SOURCE_COMMIT,
)


EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "738037cdd69b300037552793fc81d4e454a2bc14f16561c4a71d9f86bcf509c9"
)
EXPECTED_SELECTED_MANIFEST_SHA256 = (
    "3e75c42d31ead77bced230b287622faa7620f8278eca66b318424041f4a8ed27"
)
EXPECTED_SPLIT_COUNTS = {
    "train": 252,
    "validation": 89,
    "test": 90,
    "bolivia": 15,
}
EXPECTED_RATES = (0.0, 0.05, 0.1, 0.2, 0.4)
EXPECTED_REPETITIONS = 3
EXPECTED_EPOCHS = 200
EXPECTED_PERTURB_SEED = 20260716
EXPECTED_OFFICIAL_MANIFEST_SHA256 = (
    "506212dbbd79a5c04d530525a218feab3cbe8000f2d1368e4b5a8a313a8ad9e3"
)


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


def _rate_key(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def expected_full_conditions() -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for false_available in EXPECTED_RATES:
        for false_unavailable in EXPECTED_RATES:
            conditions.append(
                {
                    "condition_id": (
                        f"independent_fa{_rate_key(false_available)}"
                        f"_fu{_rate_key(false_unavailable)}"
                    ),
                    "false_available_rate": false_available,
                    "false_unavailable_rate": false_unavailable,
                    "matched_source_mode": "translate",
                    "quality_mode": "independent",
                    "radius": 0,
                    "shift_x": 0,
                    "shift_y": 0,
                }
            )
    translations = (
        ("north", 5, 0, -26),
        ("south", 5, 0, 26),
        ("west", 5, -26, 0),
        ("east", 5, 26, 0),
        ("north", 10, 0, -51),
        ("south", 10, 0, 51),
        ("west", 10, -51, 0),
        ("east", 10, 51, 0),
    )
    for direction, percentage, shift_x, shift_y in translations:
        identifier = f"translate_{direction}_{percentage}pct"
        base = {
            "false_available_rate": 0.0,
            "false_unavailable_rate": 0.0,
            "matched_source_mode": "translate",
            "radius": 0,
            "shift_x": shift_x,
            "shift_y": shift_y,
        }
        conditions.append(
            {"condition_id": identifier, "quality_mode": "translate", **base}
        )
        conditions.append(
            {
                "condition_id": f"matched_random__{identifier}",
                "quality_mode": "matched-random",
                **base,
            }
        )
    for radius in (4, 8, 16):
        for mode in ("dilate", "erode"):
            identifier = f"{mode}_unavailable_r{radius}"
            base = {
                "false_available_rate": 0.0,
                "false_unavailable_rate": 0.0,
                "radius": radius,
                "shift_x": 0,
                "shift_y": 0,
            }
            conditions.append(
                {
                    "condition_id": identifier,
                    "matched_source_mode": "translate",
                    "quality_mode": mode,
                    **base,
                }
            )
            conditions.append(
                {
                    "condition_id": f"matched_random__{identifier}",
                    "matched_source_mode": mode,
                    "quality_mode": "matched-random",
                    **base,
                }
            )
    conditions.append(
        {
            "condition_id": "complete_absence",
            "false_available_rate": 0.0,
            "false_unavailable_rate": 0.0,
            "matched_source_mode": "translate",
            "quality_mode": "complete-absence",
            "radius": 0,
            "shift_x": 0,
            "shift_y": 0,
        }
    )
    return conditions


def _audit_manifest(
    archive: ZipFile,
    expected_root: str,
) -> tuple[bool, dict[str, Any], dict[str, str]]:
    names = archive.namelist()
    manifest_names = [
        name for name in names if name.endswith("/artifact_manifest.json")
    ]
    if len(manifest_names) != 1:
        return False, {"manifest_count": len(manifest_names)}, {}
    manifest_name = manifest_names[0]
    manifest = _read_json(archive, manifest_name)
    records = manifest.get("files", [])
    if not isinstance(records, list):
        return False, {"error": "manifest files is not a list"}, {}
    root = str(manifest.get("root", ""))
    record_names = [str(record.get("path", "")) for record in records]
    expected_names = {f"{root}/{name}" for name in record_names}
    actual_names = set(names) - {manifest_name}
    structurally_valid = (
        manifest.get("schema") == ARTIFACT_SCHEMA
        and root == expected_root
        and manifest_name == f"{root}/artifact_manifest.json"
        and len(names) == 29
        and len(records) == 28
        and len(record_names) == len(set(record_names))
        and all(_safe_member(name) for name in record_names)
        and expected_names == actual_names
    )
    mismatched: list[str] = []
    hashes: dict[str, str] = {}
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
    return structurally_valid and not mismatched, {
        "schema": manifest.get("schema"),
        "root": root,
        "members": len(names),
        "records": len(records),
        "coverage_match": expected_names == actual_names,
        "mismatched": mismatched,
    }, hashes


def _audit_runtime_and_source(
    archive: ZipFile,
    root: str,
    hashes: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    runtime = _read_json(archive, f"{root}/runtime_manifest.json")
    source = _read_json(archive, f"{root}/official_source_manifest.json")
    copied_source = _read_json(
        archive, f"{root}/official_source/official_source_manifest.json"
    )
    environment = archive.read(f"{root}/environment_freeze.txt").decode(
        "utf-8", errors="replace"
    )
    required_arch = runtime.get("required_arch")
    passed = (
        runtime.get("repository_commit") == EXPECTED_SOURCE_COMMIT
        and runtime.get("repository_dirty_tracked") is False
        and runtime.get("cuda_conv2d_gate") == "pass"
        and runtime.get("device") not in {None, "", "cpu"}
        and runtime.get("capability") == [6, 0]
        and required_arch == "sm_60"
        and required_arch in runtime.get("compiled_arches", [])
        and runtime.get("torch") == "2.7.1+cu126"
        and runtime.get("torch_cuda") == "12.6"
        and source == copied_source
        and source.get("commit") == EXPECTED_OFFICIAL_COMMIT
        and source.get("checkout_commit") == EXPECTED_OFFICIAL_COMMIT
        and source.get("source_sha256") == EXPECTED_OFFICIAL_SOURCE_SHA256
        and source.get("license_sha256") == EXPECTED_OFFICIAL_LICENSE_SHA256
        and source.get("paper_doi") == EXPECTED_PAPER_DOI
        and source.get("model_configuration") == EXPECTED_ARCHITECTURE
        and hashes.get(f"{root}/official_source/smagnet.py")
        == EXPECTED_OFFICIAL_SOURCE_SHA256
        and hashes.get(f"{root}/official_source/LICENSE")
        == EXPECTED_OFFICIAL_LICENSE_SHA256
        and hashes.get(f"{root}/official_source_manifest.json")
        == EXPECTED_OFFICIAL_MANIFEST_SHA256
        and "torch==2.7.1+cu126" in environment
        and "torchvision==0.22.1+cu126" in environment
        and "segmentation_models_pytorch==0.5.0" in environment
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
    seed: int,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    gate = _read_json(archive, f"{root}/published_architecture_gate.json")
    plan = _read_json(archive, f"{root}/experiment_plan.json")
    condition_document = _read_json(archive, f"{root}/smagnet_conditions.json")
    conditions = condition_document.get("conditions", [])
    expected_conditions = expected_full_conditions()
    training = gate.get("training", {})
    fallback = gate.get("fallback_boundary", {})
    expected_rows = 54 * EXPECTED_REPETITIONS
    passed = (
        plan.get("schema") == "geoai-quality-map-uncertainty-smagnet-plan-v1"
        and plan.get("mode") == "full"
        and plan.get("pipeline_only") is False
        and plan.get("seed") == seed
        and tuple(plan.get("planned_full_seeds", ())) == EXPECTED_FULL_SEEDS
        and plan.get("epochs") == EXPECTED_EPOCHS
        and plan.get("repetitions") == EXPECTED_REPETITIONS
        and plan.get("condition_count") == len(expected_conditions) == 54
        and plan.get("evaluation_splits") == ["test", "bolivia"]
        and plan.get("official_commit") == EXPECTED_OFFICIAL_COMMIT
        and plan.get("official_model") == EXPECTED_ARCHITECTURE
        and plan.get("paper_doi") == EXPECTED_PAPER_DOI
        and plan.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and condition_document.get("schema")
        == "geoai-quality-uncertainty-evaluation-conditions-v1"
        and condition_document.get("mode") == "full"
        and condition_document.get("pipeline_only") is False
        and condition_document.get("route") == "smagnet_official"
        and conditions == expected_conditions
        and gate.get("schema") == "geoai-quality-map-uncertainty-smagnet-gate-v1"
        and gate.get("status") == "pass"
        and gate.get("mode") == "full"
        and gate.get("pipeline_only") is False
        and gate.get("model_seed") == seed
        and gate.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and gate.get("architecture") == EXPECTED_ARCHITECTURE
        and gate.get("paper_doi") == EXPECTED_PAPER_DOI
        and gate.get("condition_count") == 54
        and gate.get("repetitions") == EXPECTED_REPETITIONS
        and gate.get("summary_rows")
        == {"bolivia": expected_rows, "test": expected_rows}
        and gate.get("per_chip_rows")
        == {
            "bolivia": expected_rows * EXPECTED_SPLIT_COUNTS["bolivia"],
            "test": expected_rows * EXPECTED_SPLIT_COUNTS["test"],
        }
        and gate.get("finite_metrics") is True
        and gate.get("scientific_interpretation_allowed") is False
        and training.get("epochs") == EXPECTED_EPOCHS
        and training.get("effective_batch_size") == 16
        and training.get("parameter_count") == 56_035_958
        and training.get("optimizer") == "Adam"
        and fallback.get("status") == "pass"
        and fallback.get("maximum_fused_sar_logit_difference") == 0.0
        and fallback.get("maximum_masked_gate") == 0.0
        and float(fallback.get("tolerance", math.inf)) == 1e-6
    )
    return passed, {
        "mode": gate.get("mode"),
        "pipeline_only": gate.get("pipeline_only"),
        "model_seed": gate.get("model_seed"),
        "planned_full_seeds": plan.get("planned_full_seeds"),
        "condition_count": len(conditions),
        "repetitions": gate.get("repetitions"),
        "epochs": training.get("epochs"),
        "summary_rows": gate.get("summary_rows"),
        "per_chip_rows": gate.get("per_chip_rows"),
        "scientific_interpretation_allowed": gate.get(
            "scientific_interpretation_allowed"
        ),
        "fallback": fallback,
    }, conditions


def _audit_data_preparation(
    archive: ZipFile,
    root: str,
    hashes: dict[str, str],
) -> tuple[
    bool,
    dict[str, Any],
    dict[str, list[str]],
    dict[str, dict[str, str]],
]:
    selected_name = f"{root}/sen1floods11_selected_manifest.json"
    selected = _read_json(archive, selected_name)
    preparation = _read_json(
        archive, f"{root}/sen1floods11_preparation_report.json"
    )
    records = selected.get("records", [])
    prepared = preparation.get("records", [])
    selected_by_id = {
        str(record.get("chip_id")): record for record in records
    }
    prepared_by_id = {
        str(record.get("chip_id")): record for record in prepared
    }
    split_ids: dict[str, list[str]] = defaultdict(list)
    chip_metadata: dict[str, dict[str, str]] = {}
    for record in records:
        chip_id = str(record.get("chip_id"))
        split = str(record.get("split"))
        event = str(record.get("event"))
        split_ids[split].append(chip_id)
        chip_metadata[chip_id] = {"split": split, "event": event}
    split_counts = {key: len(values) for key, values in split_ids.items()}
    event_counts = Counter(str(record.get("event")) for record in records)
    provider_counts = Counter(
        str(asset.get("provider"))
        for record in records
        for asset in record.get("scl_assets", [])
    )
    hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    preparation_records_valid = True
    for chip_id, item in prepared_by_id.items():
        selected_item = selected_by_id.get(chip_id, {})
        files = item.get("files", {})
        available = int(item.get("available_quality_pixels", -1))
        unavailable = int(item.get("unavailable_quality_pixels", -1))
        valid = int(item.get("valid_target_pixels", -1))
        flood = int(item.get("flood_pixels", -1))
        optical_valid = int(item.get("optical_valid_pixels", -1))
        preparation_records_valid = preparation_records_valid and (
            item.get("split") == selected_item.get("split")
            and item.get("event") == selected_item.get("event")
            and set(files) == {"label", "quality", "s1", "s2"}
            and all(
                int(value.get("bytes", 0)) > 0
                and bool(hash_pattern.fullmatch(str(value.get("sha256", ""))))
                for value in files.values()
            )
            and available >= 0
            and unavailable >= 0
            and available + unavailable == 512 * 512
            and 0 <= valid <= 512 * 512
            and 0 <= flood <= valid
            and 0 <= available <= optical_valid <= 512 * 512
            and int(item.get("scl_asset_count", -1))
            == len(selected_item.get("scl_assets", []))
            and set(item.get("providers", []))
            == {
                str(asset.get("provider"))
                for asset in selected_item.get("scl_assets", [])
            }
        )
    aggregate_valid = True
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        values = [item for item in prepared if item.get("split") == split]
        reported = preparation.get("split_summary", {}).get(split, {})
        available = sum(int(item["available_quality_pixels"]) for item in values)
        unavailable = sum(
            int(item["unavailable_quality_pixels"]) for item in values
        )
        aggregate_valid = aggregate_valid and (
            len(values) == expected_count
            and reported.get("chips") == expected_count
            and reported.get("available_quality_pixels") == available
            and reported.get("unavailable_quality_pixels") == unavailable
            and reported.get("flood_pixels")
            == sum(int(item["flood_pixels"]) for item in values)
            and reported.get("valid_target_pixels")
            == sum(int(item["valid_target_pixels"]) for item in values)
            and _close(
                float(reported.get("unavailable_quality_fraction", math.nan)),
                unavailable / (available + unavailable),
            )
        )
    zero_valid = sorted(
        chip_id
        for chip_id, item in prepared_by_id.items()
        if int(item.get("valid_target_pixels", -1)) == 0
    )
    zero_valid_test = sorted(
        chip_id
        for chip_id in zero_valid
        if chip_metadata.get(chip_id, {}).get("split") == "test"
    )
    passed = (
        selected.get("schema") == "geoai-sen1floods11-scl-manifest-v1"
        and selected.get("selection_schema")
        == "event-stratified-outcome-independent-v1"
        and selected.get("experiment_mode") == "full"
        and selected.get("pipeline_only") is False
        and selected.get("source_manifest_sha256")
        == EXPECTED_SOURCE_MANIFEST_SHA256
        and selected.get("summary", {}).get("record_count") == 446
        and selected.get("summary", {}).get("split_counts")
        == EXPECTED_SPLIT_COUNTS
        and selected.get("summary", {}).get("event_counts")
        == dict(event_counts)
        and hashes.get(selected_name) == EXPECTED_SELECTED_MANIFEST_SHA256
        and split_counts == EXPECTED_SPLIT_COUNTS
        and len(records) == len(selected_by_id) == 446
        and preparation.get("schema")
        == "geoai-sen1floods11-preparation-report-v1"
        and preparation.get("status") == "pass"
        and preparation.get("mode") == "full"
        and preparation.get("pipeline_only") is False
        and preparation.get("record_count") == 446
        and preparation.get("selected_manifest_sha256")
        == EXPECTED_SELECTED_MANIFEST_SHA256
        and len(prepared) == len(prepared_by_id) == 446
        and set(selected_by_id) == set(prepared_by_id)
        and preparation_records_valid
        and aggregate_valid
        and len(zero_valid) == 5
        and len(zero_valid_test) == 1
        and "official S2Hand chip valid-data mask"
        in preparation.get("reference_quality", "")
        and provider_counts["earth-search"] > 0
        and provider_counts["planetary-computer"] > 0
    )
    return passed, {
        "records": len(records),
        "unique_records": len(selected_by_id),
        "split_counts": split_counts,
        "events": len(event_counts),
        "provider_counts": dict(provider_counts),
        "selected_manifest_sha256": hashes.get(selected_name),
        "preparation_status": preparation.get("status"),
        "preparation_records_valid": preparation_records_valid,
        "split_aggregates_valid": aggregate_valid,
        "zero_valid_target_chips": zero_valid,
        "zero_valid_test_chips": zero_valid_test,
        "reference_quality": preparation.get("reference_quality"),
    }, split_ids, chip_metadata


def _audit_training(
    archive: ZipFile,
    root: str,
    seed: int,
    hashes: dict[str, str],
    split_ids: dict[str, list[str]],
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/runs/smagnet_official_seed{seed}"
    config = _read_json(archive, f"{prefix}/config.json")
    checkpoint = _read_json(archive, f"{prefix}/checkpoint_manifest.json")
    threshold = _read_json(archive, f"{prefix}/threshold_selection.json")
    fallback = _read_json(archive, f"{prefix}/fallback_boundary.json")
    normalization = _read_json(archive, f"{prefix}/normalization.json")
    splits = _read_json(archive, f"{prefix}/splits.json")
    rows = _read_csv(archive, f"{prefix}/metrics.csv")
    best_checkpoint_name = f"{prefix}/best_validation_loss.pt"
    val_losses = [float(row["val_loss"]) for row in rows]
    best_index = val_losses.index(min(val_losses))
    metric_fields = (
        "train_loss",
        "val_loss",
        "val_iou_at_0p5",
        "val_f1_at_0p5",
        "val_precision_at_0p5",
        "val_recall_at_0p5",
        "val_accuracy_at_0p5",
        "elapsed_seconds",
    )
    trajectory_valid = (
        len(rows) == EXPECTED_EPOCHS
        and [int(row["epoch"]) for row in rows]
        == list(range(1, EXPECTED_EPOCHS + 1))
        and _all_finite(rows, metric_fields)
        and all(
            float(row["train_loss"]) >= 0.0
            and float(row["val_loss"]) >= 0.0
            and all(
                0.0 <= float(row[name]) <= 1.0
                for name in (
                    "val_iou_at_0p5",
                    "val_f1_at_0p5",
                    "val_precision_at_0p5",
                    "val_recall_at_0p5",
                    "val_accuracy_at_0p5",
                )
            )
            for row in rows
        )
        and all(
            float(rows[index]["elapsed_seconds"])
            > float(rows[index - 1]["elapsed_seconds"])
            for index in range(1, len(rows))
        )
    )
    normalization_valid = (
        normalization.get("schema")
        == "geoai-sen1floods11-smagnet-normalization-v1"
        and normalization.get("source") == "frozen training records only"
        and normalization.get("pixels") == 252 * 512 * 512
        and normalization.get("optical_order")
        == ["B4_red", "B3_green", "B2_blue", "B8_nir"]
        and normalization.get("radar_order") == ["VV", "VH"]
        and all(
            math.isfinite(float(value))
            for name in ("optical_mean", "radar_mean")
            for value in normalization.get(name, [])
        )
        and all(
            math.isfinite(float(value)) and float(value) > 0.0
            for name in ("optical_std", "radar_std")
            for value in normalization.get(name, [])
        )
        and len(normalization.get("optical_mean", [])) == 4
        and len(normalization.get("optical_std", [])) == 4
        and len(normalization.get("radar_mean", [])) == 2
        and len(normalization.get("radar_std", [])) == 2
    )
    threshold_iou = (
        float(threshold["precision"])
        * float(threshold["recall"])
        / (
            float(threshold["precision"])
            + float(threshold["recall"])
            - float(threshold["precision"]) * float(threshold["recall"])
        )
    )
    passed = (
        config.get("architecture") == "official_smagnet"
        and config.get("route") == "smagnet_official"
        and config.get("source_commit") == EXPECTED_SOURCE_COMMIT
        and config.get("run_name") == f"smagnet_official_seed{seed}"
        and config.get("seed") == seed
        and config.get("loader_seed") == seed + 200_000
        and config.get("augmentation_seed") == seed + 300_000
        and config.get("epochs") == EXPECTED_EPOCHS
        and config.get("train_count") == EXPECTED_SPLIT_COUNTS["train"]
        and config.get("validation_count")
        == EXPECTED_SPLIT_COUNTS["validation"]
        and config.get("validation_patches") == 356
        and config.get("micro_batch_size") == 4
        and config.get("gradient_accumulation") == 4
        and config.get("effective_batch_size") == 16
        and config.get("model_parameters") == 56_035_958
        and config.get("lr") == 0.0005
        and config.get("weight_decay") == 0.0
        and config.get("device") == "cuda"
        and config.get("amp") is True
        and config.get("amp_effective") is True
        and config.get("segmentation_models_pytorch") == "0.5.0"
        and config.get("official_model_configuration") == EXPECTED_ARCHITECTURE
        and config.get("official_source_manifest_sha256")
        == EXPECTED_OFFICIAL_MANIFEST_SHA256
        and config.get("manifest_sha256")
        == EXPECTED_SELECTED_MANIFEST_SHA256
        and config.get("normalization") == normalization
        and normalization_valid
        and len(splits.get("train", []))
        == len(set(splits.get("train", [])))
        == EXPECTED_SPLIT_COUNTS["train"]
        and len(splits.get("validation", []))
        == len(set(splits.get("validation", [])))
        == EXPECTED_SPLIT_COUNTS["validation"]
        and set(splits.get("train", [])) == set(split_ids["train"])
        and set(splits.get("validation", []))
        == set(split_ids["validation"])
        and trajectory_valid
        and checkpoint.get("best_checkpoint_sha256")
        == hashes.get(best_checkpoint_name)
        and checkpoint.get("best_validation_loss_epoch")
        == int(rows[best_index]["epoch"])
        and _close(checkpoint.get("best_validation_loss"), min(val_losses))
        and checkpoint.get("threshold_selection_sha256")
        == hashes.get(f"{prefix}/threshold_selection.json")
        and checkpoint.get("fallback_boundary_sha256")
        == hashes.get(f"{prefix}/fallback_boundary.json")
        and _close(checkpoint.get("threshold"), threshold.get("threshold"))
        and threshold.get("selection_split") == "validation"
        and threshold.get("selection_rule")
        == "precision_recall_threshold_maximizing_pixel_iou"
        and threshold.get("positive_pixels") == 2_237_605
        and threshold.get("valid_pixels") == 20_294_725
        and math.isfinite(float(threshold.get("threshold")))
        and 0.0 <= float(threshold.get("threshold")) <= 1.0
        and _close(float(threshold.get("iou")), threshold_iou)
        and fallback.get("status") == "pass"
        and fallback.get("maximum_fused_sar_logit_difference") == 0.0
        and fallback.get("maximum_masked_gate") == 0.0
    )
    return passed, {
        "epochs": len(rows),
        "trajectory_valid": trajectory_valid,
        "best_epoch": checkpoint.get("best_validation_loss_epoch"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "threshold": threshold.get("threshold"),
        "checkpoint_sha256": hashes.get(best_checkpoint_name),
        "parameter_count": config.get("model_parameters"),
        "effective_batch_size": config.get("effective_batch_size"),
        "normalization_valid": normalization_valid,
        "amp_effective": config.get("amp_effective"),
        "elapsed_seconds": float(rows[-1]["elapsed_seconds"]),
        "fallback": fallback,
    }


def _row_metrics_valid(row: dict[str, str]) -> bool:
    try:
        counts = {name: int(row[name]) for name in ("tp", "fp", "fn", "tn")}
        expected = _metrics(**counts)
        return all(
            _close(float(row[name]), expected[name]) for name in METRIC_NAMES
        )
    except (KeyError, TypeError, ValueError):
        return False


def _condition_fields_match(
    row: dict[str, str], condition: dict[str, Any]
) -> bool:
    try:
        return (
            row["quality_mode"] == condition["quality_mode"]
            and _close(
                float(row["false_available_rate"]),
                float(condition["false_available_rate"]),
            )
            and _close(
                float(row["false_unavailable_rate"]),
                float(condition["false_unavailable_rate"]),
            )
            and int(row["shift_y"]) == int(condition["shift_y"])
            and int(row["shift_x"]) == int(condition["shift_x"])
            and int(row["radius"]) == int(condition["radius"])
            and row["matched_source_mode"]
            == condition["matched_source_mode"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _audit_evaluation_split(
    archive: ZipFile,
    root: str,
    seed: int,
    split: str,
    split_ids: dict[str, list[str]],
    chip_metadata: dict[str, dict[str, str]],
    conditions: list[dict[str, Any]],
    hashes: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    prefix = f"{root}/evaluations/seed{seed}/{split}"
    run_prefix = f"{root}/runs/smagnet_official_seed{seed}"
    config = _read_json(archive, f"{prefix}/evaluation_config.json")
    training_config = _read_json(archive, f"{run_prefix}/config.json")
    normalization = _read_json(archive, f"{run_prefix}/normalization.json")
    threshold = _read_json(archive, f"{run_prefix}/threshold_selection.json")
    fallback = _read_json(archive, f"{run_prefix}/fallback_boundary.json")
    summaries = _read_csv(archive, f"{prefix}/summary_metrics.csv")
    chips = _read_csv(archive, f"{prefix}/per_chip_metrics.csv")
    events = _read_csv(archive, f"{prefix}/per_event_metrics.csv")
    condition_by_id = {
        str(condition["condition_id"]): condition for condition in conditions
    }
    expected_keys = {
        (identifier, repetition)
        for identifier in condition_by_id
        for repetition in range(EXPECTED_REPETITIONS)
    }
    summary_by_key = {
        (row["condition_id"], int(row["repetition"])): row
        for row in summaries
    }
    chips_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    events_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in chips:
        chips_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
    for row in events:
        events_by_key[(row["condition_id"], int(row["repetition"]))].append(row)
    expected_chip_ids = set(split_ids[split])
    expected_events = {
        chip_metadata[chip_id]["event"] for chip_id in expected_chip_ids
    }
    issues: list[str] = []
    independent_max_rate_drift = 0.0
    for key in expected_keys:
        if key not in summary_by_key or key not in chips_by_key or key not in events_by_key:
            continue
        summary = summary_by_key[key]
        key_chips = chips_by_key[key]
        key_events = events_by_key[key]
        condition = condition_by_id[key[0]]
        aggregate_ok, reason = _validate_aggregate_metrics(
            summary, key_chips, key_events
        )
        identity_ok = (
            summary.get("route") == "smagnet_official"
            and summary.get("split") == split
            and int(summary.get("model_seed", -1)) == seed
            and int(summary.get("perturb_seed", -1)) == EXPECTED_PERTURB_SEED
            and _condition_fields_match(summary, condition)
            and int(summary.get("samples", -1)) == len(expected_chip_ids)
            and int(summary.get("events", -1)) == len(expected_events)
            and len(key_chips) == len(expected_chip_ids)
            and {row.get("chip_id") for row in key_chips} == expected_chip_ids
            and len({row.get("chip_id") for row in key_chips})
            == len(key_chips)
            and {row.get("event") for row in key_events} == expected_events
            and len({row.get("event") for row in key_events}) == len(key_events)
            and all(
                row.get("route") == "smagnet_official"
                and row.get("split") == split
                and int(row.get("model_seed", -1)) == seed
                and int(row.get("perturb_seed", -1))
                == EXPECTED_PERTURB_SEED
                and _condition_fields_match(row, condition)
                and chip_metadata.get(row.get("chip_id"), {}).get("event")
                == row.get("event")
                for row in key_chips
            )
            and all(
                row.get("route") == "smagnet_official"
                and row.get("split") == split
                and int(row.get("model_seed", -1)) == seed
                and int(row.get("perturb_seed", -1))
                == EXPECTED_PERTURB_SEED
                and _condition_fields_match(row, condition)
                for row in key_events
            )
        )
        if not aggregate_ok or not identity_ok:
            issues.append(f"{key}: {reason if not aggregate_ok else 'identity mismatch'}")
        if condition["quality_mode"] == "independent":
            differences = (
                abs(
                    float(summary["false_available_rate"])
                    - float(summary["realized_false_available_rate"])
                ),
                abs(
                    float(summary["false_unavailable_rate"])
                    - float(summary["realized_false_unavailable_rate"])
                ),
                abs(
                    float(summary["false_available_rate"])
                    - float(summary["valid_realized_false_available_rate"])
                ),
                abs(
                    float(summary["false_unavailable_rate"])
                    - float(summary["valid_realized_false_unavailable_rate"])
                ),
            )
            independent_max_rate_drift = max(
                independent_max_rate_drift, *differences
            )
    if not all(_row_metrics_valid(row) for row in chips):
        issues.append("per-chip metrics do not reconstruct from confusion counts")
    if not all(_row_metrics_valid(row) for row in events):
        issues.append("per-event metrics do not reconstruct from confusion counts")
    zero_valid_rows = [
        row for row in chips if int(row.get("valid_target_pixels", -1)) == 0
    ]
    expected_zero_valid_rows = (
        EXPECTED_REPETITIONS * len(conditions) if split == "test" else 0
    )
    if len(zero_valid_rows) != expected_zero_valid_rows or any(
        row.get("has_valid_target") != "False"
        or row.get("mean_probability") != ""
        or any(int(row[name]) != 0 for name in ("tp", "fp", "fn", "tn"))
        or any(float(row[name]) != 0.0 for name in METRIC_NAMES)
        for row in zero_valid_rows
    ):
        issues.append("zero-valid-target rows are not encoded fail-closed")
    valid_rows = [
        row for row in chips if int(row.get("valid_target_pixels", -1)) > 0
    ]
    if any(
        row.get("has_valid_target") != "True"
        or not row.get("mean_probability")
        or not math.isfinite(float(row["mean_probability"]))
        for row in valid_rows
    ):
        issues.append("valid-target rows have invalid probability metadata")
    rows_by_identity = {
        (row["condition_id"], row["chip_id"], int(row["repetition"])): row
        for row in chips
    }
    matched_max_rate_difference = 0.0
    for condition in conditions:
        if condition["quality_mode"] not in {"translate", "dilate", "erode"}:
            continue
        identifier = str(condition["condition_id"])
        matched_identifier = f"matched_random__{identifier}"
        for repetition in range(EXPECTED_REPETITIONS):
            for chip_id in expected_chip_ids:
                structured = rows_by_identity[(identifier, chip_id, repetition)]
                matched = rows_by_identity[
                    (matched_identifier, chip_id, repetition)
                ]
                for field in (
                    "quality_false_available_rate",
                    "quality_false_unavailable_rate",
                    "valid_quality_false_available_rate",
                    "valid_quality_false_unavailable_rate",
                ):
                    matched_max_rate_difference = max(
                        matched_max_rate_difference,
                        abs(float(structured[field]) - float(matched[field])),
                    )
    absence_rows = [
        row
        for row in summaries
        if row["condition_id"] == "complete_absence"
    ]
    absence_valid = len(absence_rows) == EXPECTED_REPETITIONS and all(
        _close(float(row["realized_false_available_rate"]), 0.0)
        and _close(float(row["realized_false_unavailable_rate"]), 1.0)
        and _close(float(row["valid_realized_false_available_rate"]), 0.0)
        and _close(float(row["valid_realized_false_unavailable_rate"]), 1.0)
        for row in absence_rows
    )
    if not absence_valid:
        issues.append("complete-absence realized rates are invalid")
    finite_fields = (
        *METRIC_NAMES,
        "event_equal_iou",
        "realized_false_available_rate",
        "realized_false_unavailable_rate",
        "valid_realized_false_available_rate",
        "valid_realized_false_unavailable_rate",
    )
    finite = _all_finite(summaries, finite_fields)
    expected_summary_rows = len(conditions) * EXPECTED_REPETITIONS
    expected_chip_rows = expected_summary_rows * len(expected_chip_ids)
    expected_event_rows = expected_summary_rows * len(expected_events)
    config_valid = (
        config.get("split") == split
        and config.get("model_seed") == seed
        and config.get("sample_count") == len(expected_chip_ids)
        and config.get("patches_per_chip") == 4
        and config.get("repetitions") == EXPECTED_REPETITIONS
        and config.get("perturb_seed") == EXPECTED_PERTURB_SEED
        and config.get("patch_size") == 256
        and config.get("batch_size") == 4
        and config.get("amp") is True
        and config.get("amp_effective") is True
        and config.get("conditions") == conditions
        and config.get("checkpoint_config") == training_config
        and config.get("normalization") == normalization
        and config.get("threshold") == threshold
        and config.get("fallback_boundary") == fallback
        and config.get("checkpoint_sha256")
        == hashes.get(f"{run_prefix}/best_validation_loss.pt")
        and config.get("manifest_sha256")
        == EXPECTED_SELECTED_MANIFEST_SHA256
    )
    coverage_valid = (
        set(summary_by_key) == expected_keys
        and set(chips_by_key) == expected_keys
        and set(events_by_key) == expected_keys
        and len(summaries) == len(summary_by_key) == expected_summary_rows
        and len(chips) == expected_chip_rows
        and len(events) == expected_event_rows
    )
    passed = (
        config_valid
        and coverage_valid
        and finite
        and independent_max_rate_drift <= 0.005
        and matched_max_rate_difference <= 1e-12
        and absence_valid
        and not issues
    )
    return passed, {
        "config_valid": config_valid,
        "coverage_valid": coverage_valid,
        "summary_rows": len(summaries),
        "per_chip_rows": len(chips),
        "per_event_rows": len(events),
        "samples_per_condition": len(expected_chip_ids),
        "events_per_condition": len(expected_events),
        "zero_valid_target_rows": len(zero_valid_rows),
        "independent_max_rate_drift": independent_max_rate_drift,
        "matched_control_max_rate_difference": matched_max_rate_difference,
        "complete_absence_valid": absence_valid,
        "finite_metrics": finite,
        "issues": issues,
    }


def _audit_logs_and_payload(
    archive: ZipFile,
    root: str,
    seed: int,
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
        and f"official SMAGNet quality-uncertainty Full seed {seed}" in log
    )
    return passed, {
        "raw_rasters": raw_rasters,
        "anomalies": anomalies,
        "command_counts": command_counts,
        "final_pass_marker": '"status": "pass"' in log,
        "log_lines": len(log.splitlines()),
    }


def audit_smagnet_full_shard_artifact(
    archive_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Fail-closed audit of one returned official-SMAGNet Full seed shard."""

    archive_path = Path(archive_path)
    expected_root = f"quality_uncertainty_smagnet_full_seed{seed}"
    checks: list[dict[str, Any]] = []
    artifact_bytes = 0
    artifact_sha256 = ""
    member_count = 0
    if seed not in EXPECTED_FULL_SEEDS:
        raise ValueError(f"Full seed must be one of {EXPECTED_FULL_SEEDS}")
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
            manifest_ok, manifest_detail, hashes = _audit_manifest(
                archive, expected_root
            )
            checks.append(
                _check("artifact_manifest_integrity", manifest_ok, manifest_detail)
            )
            if manifest_ok:
                try:
                    runtime_ok, runtime_detail = _audit_runtime_and_source(
                        archive, expected_root, hashes
                    )
                except Exception as exc:  # fail-closed audit boundary
                    runtime_ok, runtime_detail = False, {"error": str(exc)}
                checks.append(
                    _check("runtime_and_official_source", runtime_ok, runtime_detail)
                )
                try:
                    protocol_ok, protocol_detail, conditions = _audit_protocol(
                        archive, expected_root, seed
                    )
                except Exception as exc:  # fail-closed audit boundary
                    protocol_ok, protocol_detail, conditions = (
                        False,
                        {"error": str(exc)},
                        [],
                    )
                checks.append(
                    _check(
                        "frozen_protocol_and_boundary",
                        protocol_ok,
                        protocol_detail,
                    )
                )
                try:
                    data_ok, data_detail, split_ids, chip_metadata = (
                        _audit_data_preparation(archive, expected_root, hashes)
                    )
                except Exception as exc:  # fail-closed audit boundary
                    data_ok, data_detail = False, {"error": str(exc)}
                    split_ids = defaultdict(list)
                    chip_metadata = {}
                checks.append(_check("data_preparation", data_ok, data_detail))
                try:
                    training_ok, training_detail = _audit_training(
                        archive, expected_root, seed, hashes, split_ids
                    )
                except Exception as exc:  # fail-closed audit boundary
                    training_ok, training_detail = False, {"error": str(exc)}
                checks.append(
                    _check(
                        "training_and_checkpoint", training_ok, training_detail
                    )
                )
                if protocol_ok and data_ok and training_ok:
                    for split in ("test", "bolivia"):
                        try:
                            passed, detail = _audit_evaluation_split(
                                archive,
                                expected_root,
                                seed,
                                split,
                                split_ids,
                                chip_metadata,
                                conditions,
                                hashes,
                            )
                        except Exception as exc:  # fail-closed audit boundary
                            passed, detail = False, {"error": str(exc)}
                        checks.append(
                            _check(
                                f"evaluation_{split}_integrity", passed, detail
                            )
                        )
                else:
                    for split in ("test", "bolivia"):
                        checks.append(
                            _check(
                                f"evaluation_{split}_integrity",
                                False,
                                {"error": "protocol, data, or training audit failed"},
                            )
                        )
                try:
                    log_ok, log_detail = _audit_logs_and_payload(
                        archive, expected_root, seed
                    )
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
                _check("artifact_manifest_integrity", False, {"error": str(exc)}),
            ]
        )
    shard_accepted = bool(checks) and all(
        check["status"] == "pass" for check in checks
    )
    seed_index = EXPECTED_FULL_SEEDS.index(seed)
    next_seed = (
        EXPECTED_FULL_SEEDS[seed_index + 1]
        if seed_index + 1 < len(EXPECTED_FULL_SEEDS)
        else None
    )
    return {
        "schema": "geoai-quality-map-uncertainty-smagnet-full-shard-audit-v1",
        "artifact": {
            "path": str(archive_path.resolve()),
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
            "members": member_count,
        },
        "shard": {
            "seed": seed,
            "planned_seeds": list(EXPECTED_FULL_SEEDS),
            "next_seed": next_seed,
        },
        "checks": checks,
        "decision": {
            "status": "pass" if shard_accepted else "fail",
            "shard_accepted": shard_accepted,
            "next_seed_authorized": shard_accepted and next_seed is not None,
            "shard_scores_publishable": False,
            "scientific_interpretation_allowed": False,
            "claim_boundary": (
                "One SMAGNet Full shard is execution evidence only. Scores remain "
                "locked until all five official-architecture seeds pass audit and "
                "are paired offline with the frozen seed-matched Sentinel-1 "
                "reference."
            ),
        },
    }


def render_smagnet_full_shard_audit_markdown(report: dict[str, Any]) -> str:
    artifact = report["artifact"]
    shard = report["shard"]
    decision = report["decision"]
    lines = [
        f"# Official SMAGNet Full seed-{shard['seed']} artifact audit",
        "",
        f"- Decision: **{decision['status'].upper()}**",
        f"- Shard accepted: **{str(decision['shard_accepted']).lower()}**",
        f"- Next seed: **{shard['next_seed']}**",
        "- Shard scores publishable: **false**",
        "- Scientific interpretation allowed: **false**",
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
        ]
    )
    return "\n".join(lines)
