from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_cuda_compat_module():
    path = ROOT / "scripts" / "ensure_cuda_compat.py"
    specification = importlib.util.spec_from_file_location(
        "test_ensure_cuda_compat", path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class CudaCompatibilityTests(unittest.TestCase):
    def test_nvidia_smi_snapshot_parses_compute_capability(self) -> None:
        module = load_cuda_compat_module()
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="Tesla P100-PCIE-16GB, 6.0\n",
            stderr="",
        )
        with patch.object(module.subprocess, "run", return_value=completed):
            snapshot = module.nvidia_gpu_snapshot()
        self.assertEqual(
            snapshot,
            {
                "device": "Tesla P100-PCIE-16GB",
                "capability": [6, 0],
                "nvidia_smi_record": "Tesla P100-PCIE-16GB, 6.0",
            },
        )

    def test_nvidia_smi_snapshot_handles_cpu_image_without_binary(self) -> None:
        module = load_cuda_compat_module()
        with patch.object(
            module.subprocess,
            "run",
            side_effect=FileNotFoundError("nvidia-smi"),
        ):
            self.assertIsNone(module.nvidia_gpu_snapshot())

    def test_p100_repairs_a_cpu_only_or_broken_preinstalled_torch(self) -> None:
        module = load_cuda_compat_module()
        p100 = {
            "device": "Tesla P100-PCIE-16GB",
            "capability": [6, 0],
        }
        with (
            patch.object(
                module,
                "runtime_info",
                side_effect=RuntimeError("CUDA is unavailable from torch"),
            ),
            patch.object(
                module,
                "nvidia_gpu_snapshot",
                return_value=p100,
                create=True,
            ),
            patch.object(module, "install_legacy_stack") as install,
        ):
            module.main()
        install.assert_called_once()

    def test_missing_gpu_fails_before_modifying_the_python_environment(self) -> None:
        module = load_cuda_compat_module()
        with (
            patch.object(module, "nvidia_gpu_snapshot", return_value=None),
            patch.object(module, "install_legacy_stack") as install,
        ):
            with self.assertRaisesRegex(RuntimeError, "set Accelerator to GPU"):
                module.main()
        install.assert_not_called()

    def test_legacy_install_uninstalls_existing_stack_before_downloading(self) -> None:
        module = load_cuda_compat_module()
        expected_versions = module.REQUIRED_RUNTIME_DISTRIBUTIONS

        def installed_version(name: str) -> str | None:
            return expected_versions.get(name)

        with (
            patch.object(
                module,
                "installed_torch_stack_distributions",
                return_value=["nvidia-cublas-cu13", "torch", "torchvision"],
            ),
            patch.object(module, "installed_base_version", side_effect=installed_version),
            patch.object(module, "disk_snapshot", return_value={}),
            patch.object(module.subprocess, "run") as run,
        ):
            module.install_legacy_stack()

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][3:5], ["uninstall", "-y"])
        self.assertIn("nvidia-cublas-cu13", commands[0])
        self.assertEqual(commands[1][3:5], ["cache", "purge"])
        self.assertEqual(commands[2][3], "install")
        self.assertNotIn("--force-reinstall", commands[2])


if __name__ == "__main__":
    unittest.main()
