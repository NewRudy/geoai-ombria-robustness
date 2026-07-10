from __future__ import annotations

import json


def main() -> None:
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
    }
    print(json.dumps(info, indent=2))

    device = torch.device("cuda")
    layer = torch.nn.Conv2d(1, 2, kernel_size=3, padding=1).to(device)
    sample = torch.randn(1, 1, 16, 16, device=device)
    output = layer(sample)
    torch.cuda.synchronize()
    if output.shape != (1, 2, 16, 16):
        raise RuntimeError(f"Unexpected CUDA convolution output: {tuple(output.shape)}")
    print("CUDA Conv2d compatibility gate: pass")


if __name__ == "__main__":
    main()
