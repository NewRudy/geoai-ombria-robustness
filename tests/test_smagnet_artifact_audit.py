from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from geoai_ombria_robustness.smagnet_artifact_audit import (
    audit_smagnet_smoke_artifact,
    render_smagnet_smoke_audit_markdown,
)


class SmagnetArtifactAuditTests(unittest.TestCase):
    def test_unsafe_member_blocks_full_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../escaped.txt", "unsafe")

            report = audit_smagnet_smoke_artifact(archive_path)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["archive_path_safety"]["status"], "fail")
        self.assertFalse(report["decision"]["full_authorized"])

    def test_manifest_hash_mismatch_blocks_full_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "tampered.zip"
            manifest = {
                "schema": "geoai-quality-map-uncertainty-artifacts-v1",
                "root": "quality_uncertainty_smagnet_smoke",
                "files": [
                    {
                        "path": "payload.txt",
                        "bytes": 7,
                        "sha256": "0" * 64,
                    }
                ],
            }
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "quality_uncertainty_smagnet_smoke/payload.txt", "payload"
                )
                archive.writestr(
                    "quality_uncertainty_smagnet_smoke/artifact_manifest.json",
                    json.dumps(manifest),
                )

            report = audit_smagnet_smoke_artifact(archive_path)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(
            checks["artifact_manifest_integrity"]["status"], "fail"
        )
        self.assertFalse(report["decision"]["full_authorized"])

    def test_markdown_preserves_smoke_claim_boundary(self) -> None:
        report = {
            "artifact": {
                "path": "/tmp/smoke.zip",
                "sha256": "0" * 64,
                "bytes": 1,
                "members": 1,
            },
            "checks": [],
            "decision": {
                "status": "pass",
                "full_authorized": True,
                "smoke_scores_publishable": False,
                "scientific_interpretation_allowed": False,
                "claim_boundary": "Pipeline only.",
            },
        }

        rendered = render_smagnet_smoke_audit_markdown(report)

        self.assertIn("Smoke scores publishable: **false**", rendered)
        self.assertIn("Scientific interpretation allowed: **false**", rendered)
        self.assertIn("Pipeline only.", rendered)


if __name__ == "__main__":
    unittest.main()
