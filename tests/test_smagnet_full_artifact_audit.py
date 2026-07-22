from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from geoai_ombria_robustness.smagnet_full_artifact_audit import (
    audit_smagnet_full_shard_artifact,
    expected_full_conditions,
    render_smagnet_full_shard_audit_markdown,
)


class SmagnetFullArtifactAuditTests(unittest.TestCase):
    def test_frozen_conditions_are_complete_and_unique(self) -> None:
        conditions = expected_full_conditions()
        identifiers = [str(item["condition_id"]) for item in conditions]
        self.assertEqual(len(conditions), 54)
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual(
            sum(item["quality_mode"] == "independent" for item in conditions),
            25,
        )
        self.assertEqual(
            sum(item["quality_mode"] == "matched-random" for item in conditions),
            14,
        )
        self.assertEqual(identifiers[-1], "complete_absence")

    def test_unsafe_member_blocks_shard_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../escaped.txt", "unsafe")

            report = audit_smagnet_full_shard_artifact(archive_path, 7)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["archive_path_safety"]["status"], "fail")
        self.assertFalse(report["decision"]["shard_accepted"])

    def test_manifest_hash_mismatch_blocks_shard_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "tampered.zip"
            root = "quality_uncertainty_smagnet_full_seed7"
            manifest = {
                "schema": "geoai-quality-map-uncertainty-artifacts-v1",
                "root": root,
                "files": [
                    {"path": "payload.txt", "bytes": 7, "sha256": "0" * 64}
                ],
            }
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{root}/payload.txt", "payload")
                archive.writestr(
                    f"{root}/artifact_manifest.json", json.dumps(manifest)
                )

            report = audit_smagnet_full_shard_artifact(archive_path, 7)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(
            checks["artifact_manifest_integrity"]["status"], "fail"
        )
        self.assertFalse(report["decision"]["shard_accepted"])

    def test_markdown_preserves_single_shard_claim_boundary(self) -> None:
        report = {
            "artifact": {
                "path": "/tmp/full.zip",
                "sha256": "0" * 64,
                "bytes": 1,
                "members": 1,
            },
            "shard": {"seed": 7, "next_seed": 13},
            "checks": [],
            "decision": {
                "status": "pass",
                "shard_accepted": True,
                "shard_scores_publishable": False,
                "scientific_interpretation_allowed": False,
                "claim_boundary": "One shard is not scientific evidence.",
            },
        }

        rendered = render_smagnet_full_shard_audit_markdown(report)

        self.assertIn("Shard scores publishable: **false**", rendered)
        self.assertIn("Scientific interpretation allowed: **false**", rendered)
        self.assertIn("One shard is not scientific evidence.", rendered)


if __name__ == "__main__":
    unittest.main()
