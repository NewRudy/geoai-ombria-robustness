from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

from geoai_ombria_robustness.ombria import collect_ombria_samples, load_sample, read_mask  # noqa: E402
from train_ombria_unet import build_model, variant_channels  # noqa: E402


MODES = (
    "none",
    "patch_after",
    "cloud_after_30",
    "cloud_after_50",
    "cloud_after_70",
    "noise_after",
    "zero_after",
    "zero_all",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external/OMBRIA"))
    parser.add_argument(
        "--clean-checkpoint",
        type=Path,
        default=Path("results/runs/ombria/multimodal_none_seed7/best_model.pt"),
    )
    parser.add_argument(
        "--robust-checkpoint",
        type=Path,
        default=Path(
            "results/runs/ombria/multimodal_none_train-modality_dropout_seed7/best_model.pt"
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/figures/ombria_qualitative")
    )
    return parser.parse_args()


def checkpoint_config(path: Path) -> dict[str, object]:
    config_path = path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def load_model(checkpoint: Path, device):
    import torch

    config = checkpoint_config(checkpoint)
    base_channels = int(config.get("base_channels", 16))
    model = build_model(variant_channels("multimodal"), base_channels).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def predict(model, image: np.ndarray, device) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.moveaxis(image, 2, 0))[None].to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return pred


def to_rgb(array: np.ndarray) -> Image.Image:
    arr = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return Image.fromarray(arr[:, :, :3])


def mask_image(mask: np.ndarray) -> Image.Image:
    arr = np.zeros((*mask.shape, 3), dtype=np.uint8)
    arr[mask > 0.5] = [32, 144, 255]
    return Image.fromarray(arr)


def probability_image(prob: np.ndarray) -> Image.Image:
    prob = np.clip(prob, 0, 1)
    arr = np.zeros((*prob.shape, 3), dtype=np.uint8)
    arr[:, :, 0] = (prob * 255).astype(np.uint8)
    arr[:, :, 1] = (np.maximum(0, prob - 0.35) * 180).astype(np.uint8)
    arr[:, :, 2] = ((1 - prob) * 80).astype(np.uint8)
    return Image.fromarray(arr)


def labeled(tile: Image.Image, label: str) -> Image.Image:
    tile = tile.resize((160, 160), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (160, 184), "white")
    canvas.paste(tile, (0, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), label, fill=(20, 20, 20))
    return canvas


def make_panel(
    sample_id: str,
    mode: str,
    image: np.ndarray,
    mask: np.ndarray,
    clean_pred: np.ndarray,
    robust_pred: np.ndarray,
) -> Image.Image:
    s2_after = image[:, :, 3:6]
    s1_after = image[:, :, 7]
    tiles = [
        labeled(to_rgb(s2_after), f"{sample_id} S2 {mode}"),
        labeled(to_rgb(s1_after), "S1 after"),
        labeled(mask_image(mask), "Mask"),
        labeled(probability_image(clean_pred), "Clean-train pred"),
        labeled(probability_image(robust_pred), "Robust-train pred"),
    ]
    panel = Image.new("RGB", (160 * len(tiles), 184), "white")
    for idx, tile in enumerate(tiles):
        panel.paste(tile, (idx * 160, 0))
    return panel


def main() -> None:
    args = parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_model = load_model(args.clean_checkpoint, device)
    robust_model = load_model(args.robust_checkpoint, device)

    samples = collect_ombria_samples(args.root, "test")
    ranked = sorted(
        samples,
        key=lambda sample: float(read_mask(sample.s2_mask).mean()),
        reverse=True,
    )[: args.num_samples]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample_index, sample in enumerate(ranked):
        rows = []
        for mode in args.modes:
            rng = np.random.default_rng(args.seed + sample_index)
            image, mask = load_sample(sample, "multimodal", mode, rng)
            clean_pred = predict(clean_model, image, device)
            robust_pred = predict(robust_model, image, device)
            rows.append(
                make_panel(sample.chip_id, mode, image, mask, clean_pred, robust_pred)
            )
        panel = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
        for idx, row in enumerate(rows):
            panel.paste(row, (0, idx * row.height))
        out = args.out_dir / f"sample_{sample.chip_id}.png"
        panel.save(out)
        written.append(out)

    print("wrote qualitative panels:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
