from __future__ import annotations

import json
import subprocess
import sys


LEGACY_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
LEGACY_PACKAGES = (
    "torch==2.7.1",
    "torchvision==0.22.1",
    "torchaudio==2.7.1",
)


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


def main() -> None:
    info = runtime_info()
    print("CUDA runtime before compatibility check:")
    print(json.dumps(info, indent=2))
    if info["required_arch"] in info["compiled_arches"]:
        print("CUDA compatibility: current PyTorch build already supports this GPU")
        return

    capability = tuple(info["capability"])
    if not (6, 0) <= capability <= (7, 5):
        raise RuntimeError(
            "The current PyTorch build does not contain this GPU architecture, and "
            "the CUDA 12.6 fallback is only intended for Pascal, Volta, and Turing."
        )

    print(
        "Current PyTorch does not contain the GPU architecture; installing the "
        "official CUDA 12.6 compatibility build for Pascal/Volta/Turing."
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            "--no-cache-dir",
            *LEGACY_PACKAGES,
            "--index-url",
            LEGACY_CUDA_INDEX,
        ],
        check=True,
    )
    print(
        "Compatibility build installed. The next command runs in a fresh Python "
        "process and verifies an actual CUDA convolution."
    )


if __name__ == "__main__":
    main()
