from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from geoai_ombria_robustness.quality_uncertainty_artifact_audit import (
    audit_quality_uncertainty_smoke_artifact,
)
from geoai_ombria_robustness.quality_uncertainty_full_audit import (
    render_quality_uncertainty_full_shard_audit_markdown,
)


class QualityUncertaintyArtifactAuditTests(unittest.TestCase):
    def test_full_audit_markdown_uses_active_seed_label(self) -> None:
        report = {
            "artifact": {
                "path": "/tmp/seed13.zip",
                "sha256": "0" * 64,
                "bytes": 1,
                "members": 1,
            },
            "seed": 13,
            "checks": [],
            "decision": {
                "status": "pass",
                "remaining_core_seeds_authorized": True,
                "claim_boundary": "QC only.",
            },
        }

        rendered = render_quality_uncertainty_full_shard_audit_markdown(report)

        self.assertIn("Seed-13 score previews", rendered)
        self.assertNotIn("Seed-7 score previews", rendered)

    def test_unsafe_archive_member_blocks_full_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "unsafe.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("../escaped.txt", "unsafe")

            report = audit_quality_uncertainty_smoke_artifact(archive_path)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["archive_path_safety"]["status"], "fail")
        self.assertFalse(report["decision"]["full_authorized"])

    def test_manifest_hash_mismatch_blocks_full_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "tampered.zip"
            manifest = {
                "schema": "geoai-quality-map-uncertainty-artifacts-v1",
                "root": "quality_uncertainty_smoke",
                "files": [
                    {
                        "path": "payload.txt",
                        "bytes": 7,
                        "sha256": "0" * 64,
                    }
                ],
            }
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("quality_uncertainty_smoke/payload.txt", "payload")
                archive.writestr(
                    "quality_uncertainty_smoke/artifact_manifest.json",
                    json.dumps(manifest),
                )

            report = audit_quality_uncertainty_smoke_artifact(archive_path)

        checks = {check["id"]: check for check in report["checks"]}
        self.assertEqual(checks["artifact_manifest_integrity"]["status"], "fail")
        self.assertFalse(report["decision"]["full_authorized"])


if __name__ == "__main__":
    unittest.main()
