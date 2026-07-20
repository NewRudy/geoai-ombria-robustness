from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any


LEGACY_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
LEGACY_PACKAGES = (
    "torch==2.7.1",
    "torchvision==0.22.1",
    "torchaudio==2.7.1",
)
REQUIRED_RUNTIME_DISTRIBUTIONS = {
    "nvidia-cuda-nvrtc-cu12": "12.6.77",
    "nvidia-cuda-runtime-cu12": "12.6.77",
    "nvidia-cuda-cupti-cu12": "12.6.80",
    "nvidia-cudnn-cu12": "9.5.1.17",
    "nvidia-cublas-cu12": "12.6.4.1",
    "nvidia-cufft-cu12": "11.3.0.4",
    "nvidia-curand-cu12": "10.3.7.77",
    "nvidia-cusolver-cu12": "11.7.1.2",
    "nvidia-cusparse-cu12": "12.5.4.2",
    "nvidia-cusparselt-cu12": "0.6.3",
    "nvidia-nccl-cu12": "2.26.2",
    "nvidia-nvtx-cu12": "12.6.77",
    "nvidia-nvjitlink-cu12": "12.6.85",
    "nvidia-cufile-cu12": "1.11.1.6",
    "triton": "3.3.1",
}
LEGACY_CAPABILITY_MINIMUM = (6, 0)
LEGACY_CAPABILITY_MAXIMUM = (7, 5)


def _normalized_distribution_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def nvidia_gpu_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    line = completed.stdout.splitlines()[0]
    try:
        device, capability_text = [part.strip() for part in line.rsplit(",", 1)]
        major_text, minor_text = capability_text.split(".", 1)
        capability = [int(major_text), int(minor_text)]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot parse nvidia-smi GPU record: {line!r}") from exc
    return {
        "device": device,
        "capability": capability,
        "nvidia_smi_record": line,
    }


def disk_snapshot() -> dict[str, dict[str, float]]:
    paths = {Path(tempfile.gettempdir()), Path("/")}
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        paths.add(kaggle_working)
    snapshot: dict[str, dict[str, float]] = {}
    for path in sorted(paths, key=str):
        usage = shutil.disk_usage(path)
        snapshot[str(path)] = {
            "total_gib": round(usage.total / 1024**3, 2),
            "used_gib": round(usage.used / 1024**3, 2),
            "free_gib": round(usage.free / 1024**3, 2),
        }
    return snapshot


def installed_torch_stack_distributions() -> list[str]:
    names: set[str] = set()
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_distribution_name(raw_name)
        if name in {"torch", "torchvision", "torchaudio", "triton"} or (
            name.startswith("nvidia-")
            and (name.endswith("-cu12") or name.endswith("-cu13"))
        ):
            names.add(name)
    return sorted(names)


def runtime_info() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Enable a Kaggle GPU accelerator before running the experiment."
        )
    capability = torch.cuda.get_device_capability(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "required_arch": f"sm_{capability[0]}{capability[1]}",
        "compiled_arches": torch.cuda.get_arch_list(),
    }


def installed_base_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution).split("+", 1)[0]
    except metadata.PackageNotFoundError:
        return None


def install_legacy_stack() -> None:
    existing = installed_torch_stack_distributions()
    print("Disk before compatibility installation:")
    print(json.dumps(disk_snapshot(), indent=2))
    if existing:
        print(
            "Removing the preinstalled CUDA/PyTorch stack first to avoid the "
            "temporary disk spike caused by force-reinstall:"
        )
        print(json.dumps(existing, indent=2))
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", *existing],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "cache", "purge"],
        check=False,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-cache-dir",
            "--no-compile",
            *LEGACY_PACKAGES,
            "--index-url",
            LEGACY_CUDA_INDEX,
        ],
        check=True,
    )
    missing = {
        distribution: (expected, installed_base_version(distribution))
        for distribution, expected in REQUIRED_RUNTIME_DISTRIBUTIONS.items()
        if installed_base_version(distribution) != expected
    }
    if missing:
        raise RuntimeError(
            f"CUDA 12.6 runtime dependency installation is incomplete: {missing}"
        )
    print("Disk after compatibility installation:")
    print(json.dumps(disk_snapshot(), indent=2))
    print(
        "Compatibility build and CUDA runtime dependencies installed. The next "
        "command runs in a fresh Python process and verifies an actual CUDA convolution."
    )


def main() -> None:
    hardware = nvidia_gpu_snapshot()
    print(
        json.dumps(
            {
                "nvidia_hardware": hardware,
                "installed_torch_distribution": installed_base_version("torch"),
                "disk": disk_snapshot(),
            },
            indent=2,
        )
    )
    if hardware is None:
        raise RuntimeError(
            "No NVIDIA GPU is visible to nvidia-smi. In Kaggle, open Settings, "
            "set Accelerator to GPU, save, and start a fresh session before "
            "running this notebook."
        )
    try:
        info = runtime_info()
    except Exception as exc:
        capability = tuple(hardware["capability"])
        if LEGACY_CAPABILITY_MINIMUM <= capability <= LEGACY_CAPABILITY_MAXIMUM:
            print(
                "NVIDIA hardware is present but the preinstalled PyTorch runtime "
                "cannot use it. Installing the complete CUDA 12.6 compatibility "
                f"stack for {hardware['device']}. Import/runtime error: {exc!r}"
            )
            install_legacy_stack()
            return
        raise RuntimeError(
            "NVIDIA hardware is visible, but PyTorch CUDA initialization failed. "
            "The automatic CUDA 12.6 repair is restricted to compute capability "
            f"{LEGACY_CAPABILITY_MINIMUM} through {LEGACY_CAPABILITY_MAXIMUM}; "
            f"observed {capability}. Original error: {exc!r}"
        ) from exc
    print("CUDA runtime before compatibility check:")
    print(json.dumps(info, indent=2))
    if info["required_arch"] in info["compiled_arches"]:
        print("CUDA compatibility: current PyTorch build already supports this GPU")
        return

    capability = tuple(info["capability"])
    if not LEGACY_CAPABILITY_MINIMUM <= capability <= LEGACY_CAPABILITY_MAXIMUM:
        raise RuntimeError(
            "The current PyTorch build does not contain this GPU architecture, and "
            "the CUDA 12.6 fallback is only intended for Pascal, Volta, and Turing. "
            f"Observed device={hardware['device']!r}, capability={capability}."
        )

    print(
        "Current PyTorch does not contain the GPU architecture; installing the "
        "official CUDA 12.6 compatibility build for Pascal/Volta/Turing."
    )
    install_legacy_stack()


if __name__ == "__main__":
    main()
