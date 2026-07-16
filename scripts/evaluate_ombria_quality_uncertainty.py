from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.models import build_model  # noqa: E402
from geoai_ombria_robustness.ombria import (  # noqa: E402
    collect_ombria_samples,
    load_multimodal_quality_uncertainty_sample,
    load_sample,
    variant_channels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate OMBRIA while separating optical content degradation "
            "from supplied quality-map errors."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("external/OMBRIA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--content-degradation", default="cloud_after_50")
    parser.add_argument("--false-available-rate", type=float, required=True)
    parser.add_argument("--false-unavailable-rate", type=float, required=True)
    parser.add_argument("--perturb-seed", type=int, default=20260716)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def stable_sample_seed(
    base_seed: int,
    repetition: int,
    chip_id: str,
    stream: str,
) -> int:
    token = f"{base_seed}:{repetition}:{chip_id}:{stream}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn


def metrics(counts: Counts) -> dict[str, float]:
    eps = 1e-9
    return {
        "iou": counts.tp / (counts.tp + counts.fp + counts.fn + eps),
        "f1": (2 * counts.tp) / (2 * counts.tp + counts.fp + counts.fn + eps),
        "precision": counts.tp / (counts.tp + counts.fp + eps),
        "recall": counts.tp / (counts.tp + counts.fn + eps),
        "accuracy": (counts.tp + counts.tn)
        / (counts.tp + counts.fp + counts.fn + counts.tn + eps),
    }


class QualityUncertaintyDataset:
    def __init__(
        self,
        samples,
        args: argparse.Namespace,
        variant: str,
        s2_quality: str,
        repetition: int,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(samples)

            def __getitem__(self_inner, index: int):
                sample = samples[index]
                if variant == "s1_bitemporal":
                    image, target = load_sample(
                        sample,
                        variant,
                        "none",
                        np.random.default_rng(
                            stable_sample_seed(
                                args.perturb_seed,
                                repetition,
                                sample.chip_id,
                                "s1_reference",
                            )
                        ),
                        s2_quality="none",
                    )
                    quality_values = np.zeros(7, dtype=np.float64)
                else:
                    if variant != "multimodal" or s2_quality == "none":
                        raise ValueError(
                            "Quality uncertainty evaluation requires a "
                            "multimodal route with explicit quality input"
                        )
                    loaded = load_multimodal_quality_uncertainty_sample(
                        sample,
                        args.content_degradation,
                        degradation_rng=np.random.default_rng(
                            stable_sample_seed(
                                args.perturb_seed,
                                repetition,
                                sample.chip_id,
                                "content",
                            )
                        ),
                        quality_rng=np.random.default_rng(
                            stable_sample_seed(
                                args.perturb_seed,
                                repetition,
                                sample.chip_id,
                                "quality",
                            )
                        ),
                        false_available_rate=args.false_available_rate,
                        false_unavailable_rate=args.false_unavailable_rate,
                    )
                    image, target = loaded.image, loaded.target
                    confusion = loaded.quality_confusion
                    quality_values = np.array(
                        [
                            confusion.available_pixels,
                            confusion.unavailable_pixels,
                            confusion.false_available,
                            confusion.false_unavailable,
                            confusion.false_available_rate,
                            confusion.false_unavailable_rate,
                            confusion.quality_iou,
                        ],
                        dtype=np.float64,
                    )
                return (
                    torch.from_numpy(np.moveaxis(image, 2, 0)),
                    torch.from_numpy(target[None, :, :]),
                    index,
                    torch.from_numpy(quality_values),
                )

        self.dataset = _Dataset()


def evaluate(
    model,
    device,
    loader,
    samples,
    args: argparse.Namespace,
    repetition: int,
) -> tuple[Counts, list[dict[str, object]], dict[str, float]]:
    import torch

    pooled = Counts()
    rows: list[dict[str, object]] = []
    quality_totals = np.zeros(4, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for x, y, indexes, quality_values in loader:
            probabilities = torch.sigmoid(model(x.to(device))).cpu()
            predictions = probabilities > 0.5
            truth = y > 0.5
            for batch_index, sample_index in enumerate(indexes.tolist()):
                pred = predictions[batch_index]
                target = truth[batch_index]
                counts = Counts(
                    tp=int(torch.logical_and(pred, target).sum().item()),
                    fp=int(torch.logical_and(pred, ~target).sum().item()),
                    fn=int(torch.logical_and(~pred, target).sum().item()),
                    tn=int(torch.logical_and(~pred, ~target).sum().item()),
                )
                pooled.add(counts)
                quality = quality_values[batch_index].numpy()
                quality_totals += quality[:4]
                sample = samples[sample_index]
                rows.append(
                    {
                        "route": args.route,
                        "content_degradation": args.content_degradation,
                        "requested_false_available_rate": (
                            args.false_available_rate
                        ),
                        "requested_false_unavailable_rate": (
                            args.false_unavailable_rate
                        ),
                        "realized_false_available_rate": quality[4],
                        "realized_false_unavailable_rate": quality[5],
                        "quality_iou": quality[6],
                        "perturb_seed": args.perturb_seed,
                        "repetition": repetition,
                        "chip_id": sample.chip_id,
                        "mean_probability": float(
                            probabilities[batch_index].mean().item()
                        ),
                        "tp": counts.tp,
                        "fp": counts.fp,
                        "fn": counts.fn,
                        "tn": counts.tn,
                        **metrics(counts),
                    }
                )
    available, unavailable, false_available, false_unavailable = quality_totals
    quality_summary = {
        "realized_false_available_rate": (
            float(false_available / unavailable) if unavailable else 0.0
        ),
        "realized_false_unavailable_rate": (
            float(false_unavailable / available) if available else 0.0
        ),
    }
    return pooled, rows, quality_summary


def main() -> None:
    args = parse_args()
    for name in ("false_available_rate", "false_unavailable_rate"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie within [0, 1]")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")

    import torch
    from torch.utils.data import DataLoader

    config_path = args.checkpoint.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    variant = str(config["variant"])
    s2_quality = str(config.get("s2_quality", "none"))
    architecture = str(config.get("architecture", "early_fusion_unet"))
    base_channels = int(config["base_channels"])
    quality_branch_channels = config.get("quality_branch_channels")
    model = build_model(
        variant_channels(variant, s2_quality),
        base_channels,
        architecture=architecture,
        quality_branch_channels=(
            int(quality_branch_channels)
            if quality_branch_channels is not None
            else None
        ),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = collect_ombria_samples(args.root, "test")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "variant": variant,
                "s2_quality": s2_quality,
                "architecture": architecture,
                "base_channels": base_channels,
                "quality_branch_channels": quality_branch_channels,
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "sample_count": len(samples),
                "reference_availability": (
                    "exact controlled OMBRIA degradation mask"
                ),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    all_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        dataset = QualityUncertaintyDataset(
            samples,
            args,
            variant,
            s2_quality,
            repetition,
        ).dataset
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        counts, rows, quality_summary = evaluate(
            model,
            device,
            loader,
            samples,
            args,
            repetition,
        )
        all_rows.extend(rows)
        summary_rows.append(
            {
                "route": args.route,
                "content_degradation": args.content_degradation,
                "requested_false_available_rate": args.false_available_rate,
                "requested_false_unavailable_rate": (
                    args.false_unavailable_rate
                ),
                **quality_summary,
                "perturb_seed": args.perturb_seed,
                "repetition": repetition,
                "samples": len(samples),
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "tn": counts.tn,
                **metrics(counts),
            }
        )

    for filename, rows in (
        ("summary_metrics.csv", summary_rows),
        ("per_chip_metrics.csv", all_rows),
    ):
        with (args.out_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
