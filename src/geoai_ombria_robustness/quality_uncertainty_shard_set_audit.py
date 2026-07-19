from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from .quality_uncertainty_full_audit import (
    FULL_SEEDS,
    FULL_SOURCE_COMMIT,
    audit_quality_uncertainty_full_shard_artifact,
)


INVARIANT_MEMBERS = {
    "quality_uncertainty_smoke_authorization.json": (
        "dd049298d7a83604a1bd709c82bba82175df10a5dd4f4af9c79943e135b7eddc"
    ),
    "quality_uncertainty_core_equivalence.json": (
        "a972448e62df6484688061a869502dd7007766deeef2ea7782fc00a816bee52b"
    ),
    "sen1floods11_scl_manifest.json": (
        "738037cdd69b300037552793fc81d4e454a2bc14f16561c4a71d9f86bcf509c9"
    ),
    "sen1floods11/sen1floods11_selected_manifest.json": (
        "3e75c42d31ead77bced230b287622faa7620f8278eca66b318424041f4a8ed27"
    ),
}
EXPECTED_EXTERNAL_SPLIT_FILES = 8
EXPECTED_EXTERNAL_SPLIT_COUNTS = {"train": 252, "validation": 89}
EXPECTED_OMBRIA_SPLIT_FILES = 5
EXPECTED_OMBRIA_SPLIT_COUNTS = {"train": 530, "val": 94, "test": 70}
EXPECTED_ROWS_PER_SHARD = {
    "external_per_chip_rows": 86625,
    "external_raw_summary_rows": 1650,
    "external_seed_summary_rows": 550,
    "ombria_raw_summary_rows": 301,
    "ombria_response_surface_rows": 101,
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _check(check_id: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _canonical_ombria_split_rows(
    split_payload: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    chip_identities: list[tuple[str, str]] = []
    for partition in sorted(split_payload):
        records = split_payload[partition]
        if not isinstance(records, list):
            raise ValueError(f"OMBRIA split partition {partition!r} is not a list")
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"OMBRIA split record in {partition!r} is not an object"
                )
            try:
                source_split = str(record["split"])
                chip_id = str(record["chip_id"])
            except KeyError as exc:
                raise ValueError(
                    f"OMBRIA split record in {partition!r} lacks {exc.args[0]!r}"
                ) from exc
            rows.append((str(partition), source_split, chip_id))
            chip_identities.append((source_split, chip_id))
    if len(chip_identities) != len(set(chip_identities)):
        raise ValueError("OMBRIA semantic split contains duplicate chip assignments")
    return sorted(rows)


def ombria_semantic_split_signature(split_payload: Mapping[str, Any]) -> str:
    """Hash OMBRIA chip assignments while deliberately ignoring absolute paths."""

    rows = _canonical_ombria_split_rows(split_payload)
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    return _sha256(encoded)


def _check_detail(report: Mapping[str, Any], check_id: str) -> Mapping[str, Any]:
    for check in report.get("checks", []):
        if check.get("id") == check_id:
            detail = check.get("detail", {})
            return detail if isinstance(detail, Mapping) else {}
    return {}


def _inspect_archive(seed: int, archive_path: Path) -> dict[str, Any]:
    root = f"quality_uncertainty_full_seed{seed}"
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        invariant_hashes = {
            relative: _sha256(archive.read(f"{root}/{relative}"))
            for relative in INVARIANT_MEMBERS
        }

        external_names = sorted(
            name
            for name in names
            if name.startswith(f"{root}/sen1floods11/runs/")
            and name.endswith("/splits.json")
        )
        external_hashes = sorted({_sha256(archive.read(name)) for name in external_names})
        external_counts: dict[str, int] = {}
        if external_names:
            external_payload = json.loads(archive.read(external_names[0]))
            external_counts = {
                str(partition): len(chips)
                for partition, chips in external_payload.items()
            }

        ombria_names = sorted(
            name
            for name in names
            if name.startswith(f"{root}/ombria/seed{seed}/runs/")
            and name.endswith("/splits.json")
        )
        ombria_raw_hashes: set[str] = set()
        ombria_semantic_signatures: set[str] = set()
        ombria_counts: set[tuple[tuple[str, int], ...]] = set()
        for name in ombria_names:
            payload_bytes = archive.read(name)
            payload = json.loads(payload_bytes)
            ombria_raw_hashes.add(_sha256(payload_bytes))
            ombria_semantic_signatures.add(
                ombria_semantic_split_signature(payload)
            )
            ombria_counts.add(
                tuple(sorted((str(key), len(value)) for key, value in payload.items()))
            )

    return {
        "invariant_hashes": invariant_hashes,
        "external_split_file_count": len(external_names),
        "external_split_hashes": external_hashes,
        "external_split_counts": external_counts,
        "ombria_split_file_count": len(ombria_names),
        "ombria_raw_split_hashes": sorted(ombria_raw_hashes),
        "ombria_semantic_split_signatures": sorted(ombria_semantic_signatures),
        "ombria_split_counts": [dict(counts) for counts in sorted(ombria_counts)],
    }


def _row_counts(report: Mapping[str, Any]) -> dict[str, int]:
    def count(detail: Mapping[str, Any], key: str) -> int:
        try:
            return int(detail.get(key, -1))
        except (TypeError, ValueError):
            return -1

    external_evaluation = _check_detail(report, "external_evaluation_integrity")
    external_summary = _check_detail(report, "external_seed_summary")
    ombria = _check_detail(report, "ombria_integrity")
    return {
        "external_per_chip_rows": count(external_evaluation, "per_chip_rows"),
        "external_raw_summary_rows": count(
            external_evaluation, "raw_summary_rows"
        ),
        "external_seed_summary_rows": count(external_summary, "rows"),
        "ombria_raw_summary_rows": count(ombria, "raw_summary_rows"),
        "ombria_response_surface_rows": count(ombria, "response_surface_rows"),
    }


def audit_quality_uncertainty_shard_set_artifacts(
    artifacts: Sequence[tuple[int, Path]],
    *,
    code_root: Path | None = None,
) -> dict[str, Any]:
    """Audit a partial or complete set of frozen Full seed shards.

    A passing partial-set preflight authorizes only the still-missing frozen
    shards. Full core merge authorization is emitted only for all five seeds.
    """

    requested = [(int(seed), Path(path)) for seed, path in artifacts]
    requested_seeds = [seed for seed, _ in requested]
    unique_seed_paths = {seed: path for seed, path in requested}
    audited_seeds = sorted(unique_seed_paths)
    missing_seeds = [seed for seed in FULL_SEEDS if seed not in unique_seed_paths]
    checks: list[dict[str, Any]] = []

    valid_seed_set = (
        bool(requested)
        and len(requested_seeds) == len(set(requested_seeds))
        and all(seed in FULL_SEEDS for seed in requested_seeds)
    )
    checks.append(
        _check(
            "frozen_seed_subset",
            valid_seed_set,
            {
                "requested_seeds": requested_seeds,
                "frozen_seeds": list(FULL_SEEDS),
                "unique": len(requested_seeds) == len(set(requested_seeds)),
                "missing_seeds": missing_seeds,
            },
        )
    )

    shard_reports: dict[int, dict[str, Any]] = {}
    if valid_seed_set:
        for seed in audited_seeds:
            shard_reports[seed] = audit_quality_uncertainty_full_shard_artifact(
                unique_seed_paths[seed], seed=seed, code_root=code_root
            )
    all_individual_pass = (
        valid_seed_set
        and len(shard_reports) == len(requested)
        and all(
            report.get("decision", {}).get("status") == "pass"
            for report in shard_reports.values()
        )
    )
    checks.append(
        _check(
            "individual_shard_audits",
            all_individual_pass,
            {
                str(seed): {
                    "status": report.get("decision", {}).get("status"),
                    "sha256": report.get("artifact", {}).get("sha256"),
                    "checks": len(report.get("checks", [])),
                }
                for seed, report in sorted(shard_reports.items())
            },
        )
    )

    artifact_hashes = {
        seed: str(report.get("artifact", {}).get("sha256", ""))
        for seed, report in shard_reports.items()
    }
    artifact_identity_pass = (
        all_individual_pass
        and all(artifact_hashes.values())
        and len(set(artifact_hashes.values())) == len(artifact_hashes)
    )
    checks.append(
        _check(
            "artifact_identity",
            artifact_identity_pass,
            {str(seed): value for seed, value in sorted(artifact_hashes.items())},
        )
    )

    source_commits: dict[int, dict[str, Any]] = {}
    for seed, report in shard_reports.items():
        protocol = _check_detail(report, "protocol_and_claim_boundary")
        runtime = _check_detail(report, "runtime_source_and_hotfix")
        source_commits[seed] = {
            "plan": protocol.get("source_commit"),
            "runtime": runtime.get("repository_commit"),
        }
    source_pass = bool(source_commits) and all(
        values["plan"] == FULL_SOURCE_COMMIT
        and values["runtime"] == FULL_SOURCE_COMMIT
        for values in source_commits.values()
    )
    checks.append(
        _check(
            "source_commit_consistency",
            source_pass,
            {
                "expected": FULL_SOURCE_COMMIT,
                "by_seed": {
                    str(seed): value for seed, value in sorted(source_commits.items())
                },
            },
        )
    )

    inspections: dict[int, dict[str, Any]] = {}
    inspection_errors: dict[int, str] = {}
    if valid_seed_set:
        for seed in audited_seeds:
            try:
                inspections[seed] = _inspect_archive(seed, unique_seed_paths[seed])
            except Exception as exc:
                inspection_errors[seed] = str(exc)

    invariant_pass = bool(inspections) and not inspection_errors
    for relative, expected in INVARIANT_MEMBERS.items():
        values = {
            seed: detail["invariant_hashes"].get(relative)
            for seed, detail in inspections.items()
        }
        invariant_pass = invariant_pass and bool(values) and all(
            value == expected for value in values.values()
        )
    checks.append(
        _check(
            "invariant_file_consistency",
            invariant_pass,
            {
                "expected": INVARIANT_MEMBERS,
                "by_seed": {
                    str(seed): detail["invariant_hashes"]
                    for seed, detail in sorted(inspections.items())
                },
                "errors": {
                    str(seed): error for seed, error in sorted(inspection_errors.items())
                },
            },
        )
    )

    external_hash_union = {
        value
        for detail in inspections.values()
        for value in detail["external_split_hashes"]
    }
    external_pass = bool(inspections) and not inspection_errors and all(
        detail["external_split_file_count"] == EXPECTED_EXTERNAL_SPLIT_FILES
        and len(detail["external_split_hashes"]) == 1
        and detail["external_split_counts"] == EXPECTED_EXTERNAL_SPLIT_COUNTS
        for detail in inspections.values()
    )
    external_pass = external_pass and len(external_hash_union) == 1
    checks.append(
        _check(
            "external_split_consistency",
            external_pass,
            {
                "raw_hashes_across_shards": sorted(external_hash_union),
                "by_seed": {
                    str(seed): {
                        "files": detail["external_split_file_count"],
                        "raw_hashes": detail["external_split_hashes"],
                        "partition_counts": detail["external_split_counts"],
                    }
                    for seed, detail in sorted(inspections.items())
                },
            },
        )
    )

    ombria_semantic_union = {
        value
        for detail in inspections.values()
        for value in detail["ombria_semantic_split_signatures"]
    }
    ombria_pass = bool(inspections) and not inspection_errors and all(
        detail["ombria_split_file_count"] == EXPECTED_OMBRIA_SPLIT_FILES
        and len(detail["ombria_semantic_split_signatures"]) == 1
        and detail["ombria_split_counts"] == [EXPECTED_OMBRIA_SPLIT_COUNTS]
        for detail in inspections.values()
    )
    ombria_pass = ombria_pass and len(ombria_semantic_union) == 1
    checks.append(
        _check(
            "ombria_semantic_split_consistency",
            ombria_pass,
            {
                "normalization": ["partition", "record.split", "chip_id"],
                "ignored_fields": "absolute image and mask paths",
                "semantic_signatures_across_shards": sorted(ombria_semantic_union),
                "by_seed": {
                    str(seed): {
                        "files": detail["ombria_split_file_count"],
                        "raw_hashes": detail["ombria_raw_split_hashes"],
                        "semantic_signatures": detail[
                            "ombria_semantic_split_signatures"
                        ],
                        "partition_counts": detail["ombria_split_counts"],
                    }
                    for seed, detail in sorted(inspections.items())
                },
            },
        )
    )

    row_counts_by_seed = {
        seed: _row_counts(report) for seed, report in shard_reports.items()
    }
    row_totals = {
        name: sum(counts.get(name, -1) for counts in row_counts_by_seed.values())
        for name in EXPECTED_ROWS_PER_SHARD
    }
    expected_totals = {
        name: value * len(shard_reports)
        for name, value in EXPECTED_ROWS_PER_SHARD.items()
    }
    row_count_pass = bool(row_counts_by_seed) and all(
        counts == EXPECTED_ROWS_PER_SHARD for counts in row_counts_by_seed.values()
    )
    row_count_pass = row_count_pass and row_totals == expected_totals
    checks.append(
        _check(
            "row_count_totals",
            row_count_pass,
            {
                "expected_per_shard": EXPECTED_ROWS_PER_SHARD,
                "by_seed": {
                    str(seed): counts
                    for seed, counts in sorted(row_counts_by_seed.items())
                },
                "observed_totals": row_totals,
                "expected_totals": expected_totals,
            },
        )
    )

    subset_preflight_pass = bool(checks) and all(
        check["status"] == "pass" for check in checks
    )
    complete = tuple(audited_seeds) == FULL_SEEDS
    core_merge_authorized = subset_preflight_pass and complete
    return {
        "schema": "geoai-quality-map-uncertainty-shard-set-audit-v1",
        "expected_seeds": list(FULL_SEEDS),
        "audited_seeds": audited_seeds,
        "missing_seeds": missing_seeds,
        "source_commit": FULL_SOURCE_COMMIT,
        "checks": checks,
        "decision": {
            "status": "pass" if subset_preflight_pass else "fail",
            "subset_preflight_authorized": subset_preflight_pass,
            "remaining_core_seed_execution_authorized": (
                subset_preflight_pass and bool(missing_seeds)
            ),
            "all_core_shards_present": complete,
            "core_merge_authorized": core_merge_authorized,
            "manuscript_results_authorized": False,
            "scientific_interpretation_allowed": False,
            "claim_boundary": (
                "A passing partial-set preflight establishes archive, source, "
                "split, and row-count compatibility only. Core merge requires all "
                "five frozen seeds; manuscript interpretation additionally requires "
                "the SMAGNet gate and post-merge scientific audit."
            ),
        },
    }


def render_quality_uncertainty_shard_set_audit_markdown(
    report: Mapping[str, Any],
) -> str:
    decision = report["decision"]
    audited = ", ".join(str(seed) for seed in report["audited_seeds"]) or "none"
    missing = ", ".join(str(seed) for seed in report["missing_seeds"]) or "none"
    lines = [
        "# Quality-map uncertainty cross-shard merge preflight",
        "",
        f"- Decision: **{str(decision['status']).upper()}**",
        f"- Audited frozen seeds: **{audited}**",
        f"- Missing frozen seeds: **{missing}**",
        "- Partial-set compatibility preflight: "
        f"**{str(decision['subset_preflight_authorized']).lower()}**",
        "- Full core merge authorized: "
        f"**{str(decision['core_merge_authorized']).lower()}**",
        "- Manuscript results authorized: **false**",
        f"- Frozen source commit: `{report['source_commit']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence |",
        "|---|---:|---|",
    ]
    for check in report["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True, ensure_ascii=False)
        if len(detail) > 720:
            detail = detail[:717] + "..."
        detail = detail.replace("|", "\\|")
        lines.append(
            f"| `{check['id']}` | {check['status']} | `{detail}` |"
        )
    lines.extend(
        [
            "",
            "## Semantic split result",
            "",
            "Sen1Floods11 split files must be byte-identical across shards. OMBRIA "
            "split files are compared by partition, source split, and chip ID; "
            "absolute Kaggle image and mask paths are intentionally ignored. A raw "
            "OMBRIA hash difference caused only by workspace paths is therefore not "
            "treated as a data-partition difference.",
            "",
            "## Claim boundary",
            "",
            str(decision["claim_boundary"]),
            "",
            "No seed-level score or partial-seed aggregate in this preflight is "
            "authorized for the manuscript Results section.",
            "",
        ]
    )
    return "\n".join(lines)
