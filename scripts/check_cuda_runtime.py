from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def command_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def main() -> None:
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Enable a Kaggle GPU accelerator before running the experiment."
        )
    capability = torch.cuda.get_device_capability(0)
    info = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(capability),
        "required_arch": f"sm_{capability[0]}{capability[1]}",
        "compiled_arches": torch.cuda.get_arch_list(),
        "cudnn": torch.backends.cudnn.version(),
        "driver": command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "repository_commit": command_output(["git", "rev-parse", "HEAD"]),
        "repository_release": command_output(
            ["git", "describe", "--tags", "--exact-match"]
        ),
        "repository_dirty_tracked": bool(
            command_output(["git", "status", "--porcelain", "--untracked-files=no"])
        ),
    }
    print(json.dumps(info, indent=2))

    device = torch.device("cuda")
    layer = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1).to(device)
    sample = torch.randn(1, 1, 16, 16, device=device)
    output = layer(sample)
    torch.cuda.synchronize()
    if output.shape != (1, 2, 16, 16):
        raise RuntimeError(f"Unexpected CUDA convolution output: {tuple(output.shape)}")
    info["cuda_conv2d_gate"] = "pass"
    info["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(info, indent=2) + "\n")
        print(f"runtime manifest: {args.json_out}")
    print("CUDA Conv2d compatibility gate: pass")


if __name__ == "__main__":
    main()
