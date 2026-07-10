from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from time import time

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.ombria import (  # noqa: E402
    VARIANTS,
    OmbriaSample,
    collect_ombria_samples,
    load_sample,
    summarize_samples,
    variant_channels,
)


TRAIN_S2_DEGRADATIONS = (
    "none",
    "modality_dropout",
    "modality_dropout_light",
    "modality_dropout_balanced",
    "modality_dropout_patch",
    "quality_matched_light",
    "quality_dropout_light",
    "sar_anchor_light",
    "sar_anchor_severe_w010",
    "sar_anchor_severe_w020",
    "sar_anchor_severe_w025",
)

ANCHOR_DEGRADATION_WEIGHT = {
    "none": 0.0,
    "patch_after": 0.35,
    "cloud_after_30": 0.35,
    "cloud_after_50": 0.55,
    "cloud_after_70": 0.75,
    "noise_after": 0.55,
    "zero_after": 0.75,
    "zero_all": 1.0,
}

SEVERE_ANCHOR_DEGRADATION_WEIGHT = {
    "none": 0.0,
    "patch_after": 0.0,
    "cloud_after_30": 0.0,
    "cloud_after_50": 0.0,
    "cloud_after_70": 0.65,
    "noise_after": 0.45,
    "zero_after": 0.75,
    "zero_all": 1.0,
}


def is_sar_anchor_mode(mode: str) -> bool:
    return mode.startswith("sar_anchor")


def anchor_degradation_weight(train_mode: str, degrade_s2: str) -> float:
    if train_mode.startswith("sar_anchor_severe"):
        return SEVERE_ANCHOR_DEGRADATION_WEIGHT.get(degrade_s2, 0.0)
    return ANCHOR_DEGRADATION_WEIGHT.get(degrade_s2, 0.0)


def choose_train_degrade_s2(mode: str, rng: np.random.Generator) -> str:
    if mode == "none":
        return "none"
    schedules = {
        "modality_dropout": [0.50, 0.20, 0.10, 0.10, 0.10],
        "modality_dropout_light": [0.65, 0.12, 0.08, 0.08, 0.07],
        "modality_dropout_balanced": [0.60, 0.12, 0.08, 0.13, 0.07],
        "modality_dropout_patch": [0.50, 0.15, 0.10, 0.10, 0.15],
        "quality_matched_light": [0.55, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07],
        "quality_dropout_light": [0.55, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07],
        "sar_anchor_light": [0.45, 0.10, 0.08, 0.07, 0.08, 0.08, 0.07, 0.07],
        "sar_anchor_severe_w010": [0.60, 0.08, 0.08, 0.08, 0.05, 0.05, 0.06],
        "sar_anchor_severe_w020": [0.60, 0.08, 0.08, 0.08, 0.05, 0.05, 0.06],
        "sar_anchor_severe_w025": [0.60, 0.08, 0.08, 0.08, 0.05, 0.05, 0.06],
    }
    if mode in schedules:
        choices = ["none", "zero_after", "zero_all", "noise_after", "patch_after"]
        if mode in {"quality_matched_light", "quality_dropout_light"}:
            choices.extend(["cloud_after_30", "cloud_after_50"])
        elif mode == "sar_anchor_light":
            choices.extend(["cloud_after_30", "cloud_after_50", "cloud_after_70"])
        elif mode.startswith("sar_anchor_severe"):
            choices.extend(["cloud_after_50", "cloud_after_70"])
        return str(
            rng.choice(
                choices,
                p=schedules[mode],
            )
        )
    raise ValueError(
        f"Unknown training S2 degradation {mode!r}; choose from {TRAIN_S2_DEGRADATIONS}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external/OMBRIA"))
    parser.add_argument("--variant", choices=VARIANTS, default="s2_after")
    parser.add_argument("--degrade-s2", default="none")
    parser.add_argument(
        "--train-degrade-s2", choices=TRAIN_S2_DEGRADATIONS, default="none"
    )
    parser.add_argument("--s2-quality", choices=("none", "binary"), default="none")
    parser.add_argument("--out-dir", type=Path, default=Path("results/runs/ombria"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20260710,
        help="Fixed train/validation split seed, independent of the model seed.",
    )
    parser.add_argument(
        "--eval-perturb-seed",
        type=int,
        default=20260710,
        help="Fixed evaluation-perturbation seed, independent of the model seed.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--eval-checkpoint", type=Path, default=None)
    parser.add_argument("--anchor-checkpoint", type=Path, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser.parse_args()


class OmbriaTorchDataset:
    def __init__(
        self,
        samples: list[OmbriaSample],
        variant: str,
        degrade_s2: str = "none",
        seed: int = 7,
        s2_quality: str = "none",
        return_anchor: bool = False,
        train_degrade_s2: str = "none",
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __init__(self_inner) -> None:
                self_inner.calls = 0

            def __len__(self_inner) -> int:
                return len(samples)

            def __getitem__(self_inner, idx: int):
                if degrade_s2 in TRAIN_S2_DEGRADATIONS and degrade_s2 != "none":
                    call_id = self_inner.calls
                    self_inner.calls += 1
                    rng = np.random.default_rng(seed + idx + call_id * 1_000_003)
                    sample_degrade_s2 = choose_train_degrade_s2(degrade_s2, rng)
                else:
                    rng = np.random.default_rng(seed + idx)
                    sample_degrade_s2 = degrade_s2
                image, mask = load_sample(
                    samples[idx],
                    variant,
                    sample_degrade_s2,
                    rng,
                    s2_quality=s2_quality,
                )
                x = torch.from_numpy(np.moveaxis(image, 2, 0))
                y = torch.from_numpy(mask[None, :, :])
                if return_anchor:
                    anchor_image, _ = load_sample(
                        samples[idx],
                        "s1_bitemporal",
                        "none",
                        rng,
                        s2_quality="none",
                    )
                    anchor_x = torch.from_numpy(np.moveaxis(anchor_image, 2, 0))
                    anchor_weight = torch.tensor(
                        anchor_degradation_weight(train_degrade_s2, sample_degrade_s2),
                        dtype=torch.float32,
                    )
                    return x, y, anchor_x, anchor_weight
                return x, y

        self.dataset = _Dataset()


def split_train_val(
    samples: list[OmbriaSample],
    val_fraction: float,
    seed: int,
    max_train_samples: int,
) -> tuple[list[OmbriaSample], list[OmbriaSample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_samples = sorted(shuffled[:val_count], key=lambda sample: sample.chip_id)
    train_samples = sorted(shuffled[val_count:], key=lambda sample: sample.chip_id)
    if max_train_samples > 0:
        train_samples = train_samples[:max_train_samples]
    return train_samples, val_samples


def build_model(in_channels: int, base_channels: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.enc1 = DoubleConv(in_channels, c)
            self.enc2 = DoubleConv(c, c * 2)
            self.enc3 = DoubleConv(c * 2, c * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(c * 4, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = DoubleConv(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = DoubleConv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = DoubleConv(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear")
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    return SmallUNet()


def binary_metrics(logits, target) -> dict[str, float]:
    import torch

    pred = torch.sigmoid(logits) > 0.5
    truth = target > 0.5
    tp = torch.logical_and(pred, truth).sum().item()
    fp = torch.logical_and(pred, ~truth).sum().item()
    fn = torch.logical_and(~pred, truth).sum().item()
    tn = torch.logical_and(~pred, ~truth).sum().item()
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def evaluate(model, loader, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    model.eval()
    loss_total = 0.0
    pixels = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_total += float(
                F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
            )
            pixels += int(y.numel())
            pred = torch.sigmoid(logits) > 0.5
            truth = y > 0.5
            tp += int(torch.logical_and(pred, truth).sum().item())
            fp += int(torch.logical_and(pred, ~truth).sum().item())
            fn += int(torch.logical_and(~pred, truth).sum().item())
            tn += int(torch.logical_and(~pred, ~truth).sum().item())
    if pixels == 0:
        raise RuntimeError("Evaluation loader is empty")
    eps = 1e-9
    return {
        "loss": loss_total / pixels,
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def main() -> None:
    args = parse_args()
    if args.train_degrade_s2 != "none" and args.variant != "multimodal":
        raise ValueError("--train-degrade-s2 is only supported for --variant multimodal")
    if args.anchor_checkpoint is not None and not is_sar_anchor_mode(args.train_degrade_s2):
        raise ValueError("--anchor-checkpoint is only used with SAR-anchor train modes")
    if (
        is_sar_anchor_mode(args.train_degrade_s2)
        and args.eval_checkpoint is None
        and args.anchor_checkpoint is None
    ):
        raise ValueError("SAR-anchor train modes require --anchor-checkpoint")

    train_all = collect_ombria_samples(args.root, "train")
    test_samples = collect_ombria_samples(args.root, "test")
    train_samples, val_samples = split_train_val(
        train_all, args.val_fraction, args.split_seed, args.max_train_samples
    )

    print(
        "variant",
        args.variant,
        "channels",
        variant_channels(args.variant, args.s2_quality),
        "s2_quality",
        args.s2_quality,
    )
    print("train", summarize_samples(train_samples))
    print("val", summarize_samples(val_samples))
    print("test", summarize_samples(test_samples))

    if args.dry_run:
        image, mask = load_sample(
            train_samples[0],
            args.variant,
            args.degrade_s2,
            s2_quality=args.s2_quality,
        )
        print("sample_image_shape", image.shape, "sample_mask_shape", mask.shape)
        return

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(
        variant_channels(args.variant, args.s2_quality), args.base_channels
    ).to(device)
    train_suffix = (
        "" if args.train_degrade_s2 == "none" else f"_train-{args.train_degrade_s2}"
    )
    if args.eval_checkpoint is None:
        quality_suffix = (
            "" if args.s2_quality == "none" else f"_quality-{args.s2_quality}"
        )
        run_name = (
            f"{args.variant}{quality_suffix}_{args.degrade_s2}"
            f"{train_suffix}_seed{args.seed}"
        )
    else:
        run_name = f"{args.eval_checkpoint.parent.name}_eval-{args.degrade_s2}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.eval_checkpoint is not None:
        test_ds = OmbriaTorchDataset(
            test_samples,
            args.variant,
            args.degrade_s2,
            args.eval_perturb_seed,
            s2_quality=args.s2_quality,
        ).dataset
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
        test = evaluate(model, test_loader, device)
        checkpoint_config_path = args.eval_checkpoint.parent / "config.json"
        checkpoint_config = {}
        if checkpoint_config_path.exists():
            with checkpoint_config_path.open() as f:
                checkpoint_config = json.load(f)
        out = {
            "checkpoint": str(args.eval_checkpoint),
            "checkpoint_base_channels": checkpoint_config.get("base_channels"),
            "checkpoint_batch_size": checkpoint_config.get("batch_size"),
            "checkpoint_epochs": checkpoint_config.get("epochs"),
            "checkpoint_train_degrade_s2": checkpoint_config.get("train_degrade_s2"),
            "checkpoint_s2_quality": checkpoint_config.get("s2_quality"),
            "variant": args.variant,
            "degrade_s2": args.degrade_s2,
            "train_degrade_s2": args.train_degrade_s2,
            "s2_quality": args.s2_quality,
            **{f"test_{key}": value for key, value in test.items()},
        }
        with (run_dir / "eval_metrics.json").open("w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, sort_keys=True))
        return

    train_ds = OmbriaTorchDataset(
        train_samples,
        args.variant,
        args.train_degrade_s2,
        args.seed,
        s2_quality=args.s2_quality,
        return_anchor=is_sar_anchor_mode(args.train_degrade_s2),
        train_degrade_s2=args.train_degrade_s2,
    ).dataset
    val_ds = OmbriaTorchDataset(
        val_samples,
        args.variant,
        "none",
        args.seed,
        s2_quality=args.s2_quality,
    ).dataset

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    anchor_model = None
    if is_sar_anchor_mode(args.train_degrade_s2):
        anchor_model = build_model(variant_channels("s1_bitemporal"), args.base_channels).to(device)
        anchor_model.load_state_dict(torch.load(args.anchor_checkpoint, map_location=device))
        anchor_model.eval()
        for param in anchor_model.parameters():
            param.requires_grad_(False)

    with (run_dir / "splits.json").open("w") as f:
        json.dump(
            {
                "train": [asdict(sample) for sample in train_samples],
                "val": [asdict(sample) for sample in val_samples],
                "test": [asdict(sample) for sample in test_samples],
            },
            f,
            indent=2,
            default=str,
        )

    metrics_path = run_dir / "metrics.csv"
    best_val_iou = -1.0
    start = time()
    with metrics_path.open("w", newline="") as f:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_batches = 0
            for batch in train_loader:
                if is_sar_anchor_mode(args.train_degrade_s2):
                    x, y, anchor_x, anchor_weight = batch
                else:
                    x, y = batch
                    anchor_x = None
                    anchor_weight = None
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                if anchor_model is not None:
                    assert anchor_x is not None
                    assert anchor_weight is not None
                    anchor_x = anchor_x.to(device)
                    anchor_weight = anchor_weight.to(device).view(-1, 1, 1, 1)
                    with torch.no_grad():
                        anchor_prob = torch.sigmoid(anchor_model(anchor_x))
                    anchor_loss = F.mse_loss(
                        torch.sigmoid(logits),
                        anchor_prob,
                        reduction="none",
                    )
                    loss = loss + args.anchor_weight * (anchor_loss * anchor_weight).mean()
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_batches += 1

            val = evaluate(model, val_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(train_batches, 1),
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_f1": val["f1"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "val_accuracy": val["accuracy"],
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, sort_keys=True))

            if val["iou"] > best_val_iou and not args.no_checkpoint:
                best_val_iou = val["iou"]
                torch.save(model.state_dict(), run_dir / "best_model.pt")

    elapsed = time() - start
    print(f"finished run_dir={run_dir} elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
