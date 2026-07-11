from __future__ import annotations

import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from geoai_ombria_robustness.ombria import (  # noqa: E402
    OmbriaSample,
    degrade_s2_pair_with_quality,
    load_sample,
    mislocalize_quality_channels,
)
from train_ombria_unet import (  # noqa: E402
    choose_train_degrade_s2,
    stable_stream_seed,
)
from summarize_confirmatory_events import ci95  # noqa: E402
from package_confirmatory_artifacts import (  # noqa: E402
    RUN_DIR_TEMPLATES,
    main as package_main,
)


class V2ProtocolTests(unittest.TestCase):
    def test_v2_packager_preserves_selected_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            root = project / "results" / "sensor_state_v2"
            routes = list(RUN_DIR_TEMPLATES)
            root.mkdir(parents=True)
            (root / "experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "model_seeds": [7],
                        "routes": routes,
                        "checkpoint_policies": ["clean"],
                        "evaluation_modes": ["none"],
                    }
                )
            )
            for route in routes:
                run_dir = root / "runs" / RUN_DIR_TEMPLATES[route].format(seed=7)
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
            archive = project / "v2.zip"
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
                names = packaged.namelist()
                self.assertEqual(
                    len([name for name in names if name.endswith("best_clean.pt")]),
                    len(routes),
                )
                self.assertTrue(
                    any(name.endswith("checkpoint_manifest.json") for name in names)
                )

    def test_five_run_interval_uses_df4_student_t(self) -> None:
        values = [0.30, 0.35, 0.40, 0.45, 0.50]
        expected = 2.7764451051977987 * np.std(values, ddof=1) / np.sqrt(5)
        self.assertAlmostEqual(ci95(values), expected, places=12)

    def test_quality_uses_applied_mask_not_zero_pixel_inference(self) -> None:
        before = np.ones((32, 32, 3), dtype=np.float32)
        after = np.ones((32, 32, 3), dtype=np.float32)
        after[:3, :3] = 0.0  # legitimate source zeros
        result = degrade_s2_pair_with_quality(
            before, after, "patch_after", np.random.default_rng(17)
        )
        source_zero = after.sum(axis=2) == 0
        self.assertTrue(np.any(result.quality[:, :, 1][source_zero] == 1.0))
        unavailable = result.quality[:, :, 1] == 0.0
        self.assertTrue(np.any(unavailable))
        self.assertTrue(np.all(result.after[unavailable] == 0.0))

    def test_mislocalized_quality_preserves_prevalence(self) -> None:
        quality = np.ones((32, 32, 2), dtype=np.float32)
        quality[4:12, 6:18, 1] = 0.0
        shifted = mislocalize_quality_channels(quality)
        np.testing.assert_array_equal(
            quality.sum(axis=(0, 1)), shifted.sum(axis=(0, 1))
        )
        self.assertFalse(np.array_equal(quality, shifted))

    def test_stream_seed_is_call_order_independent(self) -> None:
        first = stable_stream_seed(300007, 4, "0012", "s2_corruption")
        repeated = stable_stream_seed(300007, 4, "0012", "s2_corruption")
        next_epoch = stable_stream_seed(300007, 5, "0012", "s2_corruption")
        other_stream = stable_stream_seed(300007, 4, "0012", "loader")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, next_epoch)
        self.assertNotEqual(first, other_stream)

    def test_quality_and_mislocalized_routes_share_corrupted_imagery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rgb = np.full((32, 32, 3), 180, dtype=np.uint8)
            gray = np.full((32, 32), 90, dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            paths = {}
            for name, array in {
                "s1_before": gray,
                "s1_after": gray,
                "s1_mask": mask,
                "s2_before": rgb,
                "s2_after": rgb,
                "s2_mask": mask,
            }.items():
                path = folder / f"{name}_0001.png"
                Image.fromarray(array).save(path)
                paths[name] = path
            sample = OmbriaSample(split="train", chip_id="0001", **paths)
            seed = stable_stream_seed(300007, 3, sample.chip_id, "s2_corruption")
            binary_rng = np.random.default_rng(seed)
            binary_mode = choose_train_degrade_s2("quality_matched_light", binary_rng)
            binary_x, _ = load_sample(
                sample,
                "multimodal",
                binary_mode,
                binary_rng,
                s2_quality="binary",
            )
            mislocalized_rng = np.random.default_rng(seed)
            mislocalized_mode = choose_train_degrade_s2(
                "quality_matched_light", mislocalized_rng
            )
            mislocalized_x, _ = load_sample(
                sample,
                "multimodal",
                mislocalized_mode,
                mislocalized_rng,
                s2_quality="mislocalized",
            )
            self.assertEqual(binary_mode, mislocalized_mode)
            np.testing.assert_array_equal(binary_x[:, :, :8], mislocalized_x[:, :, :8])
            np.testing.assert_array_equal(
                binary_x[:, :, 8:].sum(axis=(0, 1)),
                mislocalized_x[:, :, 8:].sum(axis=(0, 1)),
            )


if __name__ == "__main__":
    unittest.main()
