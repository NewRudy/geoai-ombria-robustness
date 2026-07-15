from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from package_confirmatory_artifacts import main as package_main  # noqa: E402


class QualityGatedProtocolTests(unittest.TestCase):
    def test_v3_packager_uses_manifest_route_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = project / "results" / "quality_gated_v3"
            routes = (
                "matched_control",
                "quality_concat",
                "quality_gated",
                "gated_misaligned",
            )
            templates = {route: f"{route}_seed{{seed}}" for route in routes}
            root.mkdir(parents=True)
            (root / "experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "geoai-ombria-confirmatory-experiment-v3",
                        "protocol": "quality-gated-v3",
                        "model_seeds": [7],
                        "routes": routes,
                        "run_directory_templates": templates,
                        "checkpoint_policies": ["clean"],
                        "evaluation_modes": ["none"],
                    }
                )
            )
            (root / "decision_gate.json").write_text(
                json.dumps(
                    {
                        "schema": "geoai-ombria-quality-gated-decision-v1",
                        "decision": {"status": "pipeline_only"},
                    }
                )
            )
            for route in routes:
                run_dir = root / "runs" / templates[route].format(seed=7)
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "best_clean.pt"
                checkpoint.write_bytes(f"checkpoint:{route}".encode())
                for name, content in {
                    "config.json": "{}",
                    "splits.json": "{}",
                    "metrics.csv": "epoch,val_iou\n1,0.5\n",
                }.items():
                    (run_dir / name).write_text(content)
                evaluation = root / "evaluations" / "clean" / route / "seed7" / "none"
                evaluation.mkdir(parents=True)
                (evaluation / "evaluation_config.json").write_text(
                    json.dumps(
                        {
                            "route": route,
                            "model_seed": 7,
                            "checkpoint_policy": "clean",
                            "degrade_s2": "none",
                            "checkpoint": str(checkpoint.resolve()),
                            "checkpoint_bytes": checkpoint.stat().st_size,
                            "checkpoint_sha256": hashlib.sha256(
                                checkpoint.read_bytes()
                            ).hexdigest(),
                        }
                    )
                )
                (evaluation / "summary_metrics.csv").write_text("event,iou\nALL,0.5\n")
                (evaluation / "per_chip_metrics.csv").write_text("chip_id,iou\n1,0.5\n")
            archive = project / "v3.zip"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "package_confirmatory_artifacts.py",
                    "--root",
                    str(root),
                    "--out",
                    str(archive),
                    "--include-checkpoints",
                ],
            ):
                package_main()
            with ZipFile(archive) as packaged:
                self.assertIsNone(packaged.testzip())
                manifest_name = next(
                    name
                    for name in packaged.namelist()
                    if name.endswith("artifact_manifest.json")
                )
                manifest = json.loads(packaged.read(manifest_name))
                self.assertEqual(manifest["protocol"], "quality-gated-v3")
                self.assertTrue(manifest["schema"].endswith("v3"))
                self.assertEqual(manifest["completeness_checks"]["decision_gate"], [1, 1])
                decision_name = next(
                    name
                    for name in packaged.namelist()
                    if name.endswith("decision_gate.json")
                )
                decision = json.loads(packaged.read(decision_name))
                self.assertEqual(decision["decision"]["status"], "pipeline_only")

            (root / "decision_gate.json").unlink()
            with mock.patch.object(
                sys,
                "argv",
                [
                    "package_confirmatory_artifacts.py",
                    "--root",
                    str(root),
                    "--out",
                    str(project / "missing-decision.zip"),
                    "--include-checkpoints",
                ],
            ):
                with self.assertRaisesRegex(RuntimeError, "decision_gate"):
                    package_main()


if __name__ == "__main__":
    unittest.main()
