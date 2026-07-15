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
from export_sensor_state_v2_probabilities import (  # noqa: E402
    error_map,
    select_sample,
)
from geoai_ombria_robustness.ombria import load_sample, read_mask  # noqa: E402
from train_ombria_unet import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--checkpoint-policy", choices=("clean", "robust"), default="clean"
    )
    parser.add_argument("--perturb-seed", type=int, default=20260710)
    parser.add_argument("--mode", default="cloud_after_50")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_model(checkpoint: Path, device):
    import torch

    config = json.loads((checkpoint.parent / "config.json").read_text())
    if config.get("architecture") != "quality_gated_fusion":
        raise ValueError(f"Not a quality-gated checkpoint: {checkpoint}")
    model = build_model(
        10,
        int(config["base_channels"]),
        architecture="quality_gated_fusion",
        quality_branch_channels=int(config["quality_branch_channels"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    return model


def infer(model, image: np.ndarray, device):
    import torch

    tensor = torch.from_numpy(np.moveaxis(image, 2, 0)[None]).to(device)
    with torch.no_grad():
        logits, gates = model(tensor, return_gate_maps=True)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
        post_gate = gates["after"][0][0, 0].cpu().numpy()
    return probability, post_gate


def main() -> None:
    args = parse_args()
    import matplotlib.pyplot as plt
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {
        "aligned": args.runs_dir
        / f"quality_gated_seed{args.seed}"
        / f"best_{args.checkpoint_policy}.pt",
        "mislocalized": args.runs_dir
        / f"gated_misaligned_seed{args.seed}"
        / f"best_{args.checkpoint_policy}.pt",
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing quality-gated checkpoints: {missing}")
    models = {
        name: load_model(checkpoint, device)
        for name, checkpoint in checkpoints.items()
    }
    selected = {
        event: select_sample(collect_event_samples(args.root, event))
        for event in EVENTS
    }

    records: dict[str, list[np.ndarray]] = {
        "truth": [],
        "s2_after": [],
        "aligned_quality": [],
        "aligned_gate": [],
        "aligned_probability": [],
        "mislocalized_quality": [],
        "mislocalized_gate": [],
        "mislocalized_probability": [],
    }
    for event in EVENTS:
        sample = selected[event]
        seed = stable_sample_seed(args.perturb_seed, 0, event, sample.chip_id)
        aligned_image, _ = load_sample(
            sample,
            "multimodal",
            args.mode,
            np.random.default_rng(seed),
            s2_quality="binary",
        )
        mislocalized_image, _ = load_sample(
            sample,
            "multimodal",
            args.mode,
            np.random.default_rng(seed),
            s2_quality="mislocalized",
        )
        np.testing.assert_array_equal(aligned_image[:, :, :8], mislocalized_image[:, :, :8])
        aligned_probability, aligned_gate = infer(
            models["aligned"], aligned_image, device
        )
        mislocalized_probability, mislocalized_gate = infer(
            models["mislocalized"], mislocalized_image, device
        )
        records["truth"].append(read_mask(sample.s2_mask))
        records["s2_after"].append(aligned_image[:, :, 3:6])
        records["aligned_quality"].append(aligned_image[:, :, 9])
        records["aligned_gate"].append(aligned_gate)
        records["aligned_probability"].append(aligned_probability)
        records["mislocalized_quality"].append(mislocalized_image[:, :, 9])
        records["mislocalized_gate"].append(mislocalized_gate)
        records["mislocalized_probability"].append(mislocalized_probability)

    stacked = {key: np.stack(values) for key, values in records.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"quality_gate_{args.checkpoint_policy}_{args.mode}"
    np.savez_compressed(
        args.out_dir / f"{stem}.npz",
        **{key: value.astype(np.float16) for key, value in stacked.items()},
        events=np.asarray(EVENTS),
        chip_ids=np.asarray([selected[event].chip_id for event in EVENTS]),
    )

    figure, axes = plt.subplots(
        len(EVENTS), 7, figsize=(15.5, 9.5), constrained_layout=True
    )
    for row, event in enumerate(EVENTS):
        truth = stacked["truth"][row]
        panels = (
            stacked["s2_after"][row],
            stacked["aligned_quality"][row],
            stacked["aligned_gate"][row],
            error_map(stacked["aligned_probability"][row], truth),
            stacked["mislocalized_quality"][row],
            stacked["mislocalized_gate"][row],
            error_map(stacked["mislocalized_probability"][row], truth),
        )
        for column, panel in enumerate(panels):
            if column in {1, 2, 4, 5}:
                axes[row, column].imshow(panel, cmap="viridis", vmin=0, vmax=1)
            else:
                axes[row, column].imshow(panel)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(
            f"{event.title()}\nchip {selected[event].chip_id}", fontweight="bold"
        )
    titles = (
        "Degraded S2 after",
        "Aligned quality",
        "Aligned gate",
        "Aligned errors",
        "Shifted quality",
        "Shifted gate",
        "Shifted errors",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, fontsize=9, fontweight="bold")
    figure.suptitle(
        f"Quality localization mechanism: {args.mode} / {args.checkpoint_policy}-selected",
        fontweight="bold",
    )
    figure.savefig(args.out_dir / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(figure)

    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "schema": "geoai-ombria-quality-gate-panel-v1",
                "selection_rule": "median positive flood fraction within each event; model outputs not used",
                "mode": args.mode,
                "seed": args.seed,
                "checkpoint_policy": args.checkpoint_policy,
                "events": list(EVENTS),
                "chip_ids": {
                    event: selected[event].chip_id for event in EVENTS
                },
                "checkpoints": {
                    name: str(path) for name, path in checkpoints.items()
                },
                "boundary": "Synthetic cloud-like occlusion with oracle applied-mask quality maps.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
