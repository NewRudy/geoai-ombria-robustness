from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from evaluate_ombria_2021_events import (  # noqa: E402
    EVENTS,
    collect_event_samples,
    stable_sample_seed,
)
from geoai_ombria_robustness.ombria import (  # noqa: E402
    load_sample,
    read_mask,
    variant_channels,
)
from train_ombria_unet import build_model  # noqa: E402


ROUTES = {
    "clean": ("multimodal_none_seed{seed}", "multimodal", "none"),
    "light": (
        "multimodal_none_train-modality_dropout_light_seed{seed}",
        "multimodal",
        "none",
    ),
    "matched_control": (
        "multimodal_none_train-quality_matched_light_seed{seed}",
        "multimodal",
        "none",
    ),
    "matched_quality": (
        "multimodal_quality-binary_none_train-quality_matched_light_seed{seed}",
        "multimodal",
        "binary",
    ),
    "mislocalized_quality": (
        "multimodal_quality-mislocalized_none_train-quality_matched_light_seed{seed}",
        "multimodal",
        "mislocalized",
    ),
    "s1_reference": ("s1_bitemporal_none_seed{seed}", "s1_bitemporal", "none"),
    "s2_reference": ("s2_bitemporal_none_seed{seed}", "s2_bitemporal", "none"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--checkpoint-policy", choices=("clean", "robust"), default="clean"
    )
    parser.add_argument("--perturb-seed", type=int, default=20260710)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("cloud_after_50", "zero_after", "zero_all"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def select_sample(samples):
    ranked = sorted(
        ((float(read_mask(sample.s2_mask).mean()), sample) for sample in samples),
        key=lambda item: (item[0], int(item[1].chip_id)),
    )
    positive = [item for item in ranked if item[0] > 0]
    candidates = positive or ranked
    return candidates[len(candidates) // 2][1]


def load_model(checkpoint: Path, variant: str, quality: str, device):
    import torch

    config = json.loads((checkpoint.parent / "config.json").read_text())
    model = build_model(
        variant_channels(variant, quality), int(config["base_channels"])
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def predict(model, image: np.ndarray, device) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.moveaxis(image, 2, 0)[None]).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(tensor))[0, 0].cpu().numpy()


def error_map(probability: np.ndarray, truth: np.ndarray) -> np.ndarray:
    prediction = probability > 0.5
    target = truth > 0.5
    image = np.full((*truth.shape, 3), 0.08, dtype=np.float32)
    image[np.logical_and(prediction, target)] = (0.12, 0.45, 0.95)  # TP
    image[np.logical_and(prediction, ~target)] = (1.00, 0.55, 0.05)  # FP
    image[np.logical_and(~prediction, target)] = (0.85, 0.12, 0.55)  # FN
    return image


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {
        route: args.runs_dir
        / template.format(seed=args.seed)
        / f"best_{args.checkpoint_policy}.pt"
        for route, (template, _, _) in ROUTES.items()
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    models = {
        route: load_model(checkpoints[route], variant, quality, device)
        for route, (_, variant, quality) in ROUTES.items()
    }
    selected = {
        event: select_sample(collect_event_samples(args.root, event))
        for event in EVENTS
    }
    route_names = list(ROUTES)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        truth_rows: list[np.ndarray] = []
        probability_rows: list[np.ndarray] = []
        for event in EVENTS:
            sample = selected[event]
            seed = stable_sample_seed(args.perturb_seed, 0, event, sample.chip_id)
            truth = read_mask(sample.s2_mask)
            truth_rows.append(truth)
            event_probabilities: list[np.ndarray] = []
            for route, (_, variant, quality) in ROUTES.items():
                effective_mode = "none" if route == "s1_reference" else mode
                image, _ = load_sample(
                    sample,
                    variant,
                    effective_mode,
                    np.random.default_rng(seed),
                    s2_quality=quality,
                )
                event_probabilities.append(predict(models[route], image, device))
            probability_rows.append(np.stack(event_probabilities))

        probabilities = np.stack(probability_rows).astype(np.float16)
        truths = np.stack(truth_rows).astype(np.uint8)
        np.savez_compressed(
            args.out_dir
            / f"selected_probabilities_{args.checkpoint_policy}_{mode}.npz",
            probabilities=probabilities,
            truths=truths,
            events=np.asarray(EVENTS),
            routes=np.asarray(route_names),
            chip_ids=np.asarray([selected[event].chip_id for event in EVENTS]),
        )

        figure, axes = plt.subplots(
            len(EVENTS),
            len(route_names) + 1,
            figsize=(18, 9.5),
            constrained_layout=True,
        )
        for row_index, event in enumerate(EVENTS):
            axes[row_index, 0].imshow(truths[row_index], cmap="Blues", vmin=0, vmax=1)
            axes[row_index, 0].set_ylabel(
                f"{event.title()}\nchip {selected[event].chip_id}",
                fontweight="bold",
            )
            for route_index, route in enumerate(route_names):
                axes[row_index, route_index + 1].imshow(
                    error_map(
                        probabilities[row_index, route_index].astype(np.float32),
                        truths[row_index],
                    )
                )
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        titles = ["Reference"] + [
            name.replace("_", " ").title() for name in route_names
        ]
        for axis, title in zip(axes[0], titles, strict=True):
            axis.set_title(title, fontsize=9, fontweight="bold")
        figure.suptitle(
            f"Selected-chip segmentation errors: {mode} / {args.checkpoint_policy}-selected",
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.005,
            "Blue: true positive | orange: false positive | magenta: false negative | dark: true negative",
            ha="center",
        )
        figure.savefig(
            args.out_dir / f"error_map_{args.checkpoint_policy}_{mode}.png",
            dpi=300,
            facecolor="white",
        )
        plt.close(figure)

    (args.out_dir / f"probability_export_{args.checkpoint_policy}.json").write_text(
        json.dumps(
            {
                "schema": "geoai-ombria-selected-probabilities-v1",
                "selection_rule": "median positive flood fraction within each event; model outputs not used",
                "seed": args.seed,
                "checkpoint_policy": args.checkpoint_policy,
                "modes": list(args.modes),
                "events": list(EVENTS),
                "routes": route_names,
                "chip_ids": {event: selected[event].chip_id for event in EVENTS},
                "probability_dtype": "float16 archive converted from float32 inference output",
                "checkpoints": {
                    route: str(path) for route, path in checkpoints.items()
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
