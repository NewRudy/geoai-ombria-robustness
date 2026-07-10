from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from geoai_ombria_robustness.ombria import load_sample, read_mask, variant_channels  # noqa: E402
from evaluate_ombria_2021_events import (  # noqa: E402
    EVENTS,
    collect_event_samples,
    stable_sample_seed,
)
from train_ombria_unet import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external/OMBRIA"))
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--light-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--quality-checkpoint", type=Path, required=True)
    parser.add_argument("--s1-checkpoint", type=Path, required=True)
    parser.add_argument("--perturb-seed", type=int, default=20260710)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("none", "cloud_after_50", "cloud_after_70", "noise_after", "zero_after", "zero_all"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_model(checkpoint: Path, variant: str, s2_quality: str, device):
    import torch

    config = json.loads((checkpoint.parent / "config.json").read_text())
    model = build_model(variant_channels(variant, s2_quality), int(config["base_channels"]))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def predict(model, image: np.ndarray, device) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.moveaxis(image, 2, 0)[None]).to(device)
    with torch.no_grad():
        probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return probability


def select_median_positive_sample(samples):
    ranked = sorted(
        ((float(read_mask(sample.s2_mask).mean()), sample) for sample in samples),
        key=lambda item: (item[0], int(item[1].chip_id)),
    )
    positive = [item for item in ranked if item[0] > 0.0]
    candidates = positive or ranked
    return candidates[len(candidates) // 2][1]


def false_color(s2_after: np.ndarray) -> np.ndarray:
    # The stored channels are green, NIR, and SWIR. Display SWIR/NIR/green as RGB.
    return np.clip(s2_after[:, :, [2, 1, 0]] * 2.5, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "clean": load_model(args.clean_checkpoint, "multimodal", "none", device),
        "light": load_model(args.light_checkpoint, "multimodal", "none", device),
        "control": load_model(args.control_checkpoint, "multimodal", "none", device),
        "quality": load_model(args.quality_checkpoint, "multimodal", "binary", device),
        "s1": load_model(args.s1_checkpoint, "s1_bitemporal", "none", device),
    }
    selected = {
        event: select_median_positive_sample(collect_event_samples(args.root, event))
        for event in EVENTS
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "selection_rule": "median positive flood fraction within each released 2021 event folder",
        "selection_uses_model_outputs": False,
        "display": "post-event S2 fixed-contrast false color: SWIR/NIR/green mapped to RGB",
        "probability_scale": [0.0, 1.0],
        "chips": {event: sample.chip_id for event, sample in selected.items()},
        "checkpoints": {
            "clean": str(args.clean_checkpoint),
            "light": str(args.light_checkpoint),
            "matched_control": str(args.control_checkpoint),
            "quality": str(args.quality_checkpoint),
            "s1": str(args.s1_checkpoint),
        },
    }
    (args.out_dir / "panel_manifest.json").write_text(json.dumps(manifest, indent=2))

    for mode in args.modes:
        fig, axes = plt.subplots(len(EVENTS), 7, figsize=(21, 11.5), constrained_layout=True)
        probability_image = None
        for row, event in enumerate(EVENTS):
            sample = selected[event]
            seed = stable_sample_seed(args.perturb_seed, 0, event, sample.chip_id)

            multimodal, mask = load_sample(
                sample,
                "multimodal",
                mode,
                np.random.default_rng(seed),
                s2_quality="none",
            )
            quality_input, _ = load_sample(
                sample,
                "multimodal",
                mode,
                np.random.default_rng(seed),
                s2_quality="binary",
            )
            s1_input, _ = load_sample(
                sample,
                "s1_bitemporal",
                "none",
                np.random.default_rng(seed),
                s2_quality="none",
            )
            predictions = [
                predict(models["clean"], multimodal, device),
                predict(models["light"], multimodal, device),
                predict(models["control"], multimodal, device),
                predict(models["quality"], quality_input, device),
                predict(models["s1"], s1_input, device),
            ]

            axes[row, 0].imshow(false_color(multimodal[:, :, 3:6]))
            axes[row, 1].imshow(mask, cmap="Blues", vmin=0.0, vmax=1.0)
            for column, probability in enumerate(predictions, start=2):
                probability_image = axes[row, column].imshow(
                    probability,
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                )
            axes[row, 0].set_ylabel(
                f"{event.title()}\nchip {sample.chip_id}",
                fontsize=10,
                fontweight="bold",
            )
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])

        titles = [
            "Post-S2 false color\n(SWIR/NIR/green)",
            "Reference flood mask",
            "Clean training",
            "Light degradation training",
            "Matched training\n(no quality map)",
            "Matched quality-map route",
            "S1-only reference",
        ]
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, fontsize=11, fontweight="bold")
        assert probability_image is not None
        colorbar = fig.colorbar(
            probability_image,
            ax=axes[:, 2:],
            location="bottom",
            shrink=0.55,
            pad=0.02,
        )
        colorbar.set_label("Predicted flood probability (0–1)")
        mode_labels = {
            "none": "No optical degradation",
            "cloud_after_50": "50% cloud-like post-event occlusion",
            "cloud_after_70": "70% cloud-like post-event occlusion",
            "noise_after": "Post-event optical noise",
            "zero_after": "Complete post-event S2 absence",
            "zero_all": "Complete bitemporal S2 absence",
        }
        fig.suptitle(
            f"OMBRIA 2021 event-held-out qualitative comparison — {mode_labels.get(mode, mode)}",
            fontsize=16,
            fontweight="bold",
        )
        out = args.out_dir / f"confirmatory_panel_{mode}.png"
        fig.savefig(out, dpi=220, facecolor="white")
        plt.close(fig)
        print(out)


if __name__ == "__main__":
    main()
