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
sys.path.append(str(Path(__file__).resolve().parent))

from geoai_ombria_robustness.ombria import (  # noqa: E402
    OmbriaSample,
    load_sample,
    read_mask,
    variant_channels,
)
from train_ombria_unet import build_model  # noqa: E402


EVENTS = ("ALBANIA", "FRANCE", "GUYANA", "TIMOR")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked evaluation on the four OMBRIA 2021 event folders."
    )
    parser.add_argument("--root", type=Path, default=Path("external/OMBRIA"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--s2-quality", choices=("none", "binary"), default="none")
    parser.add_argument("--degrade-s2", default="none")
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--perturb-seed", type=int, default=20260710)
    parser.add_argument("--perturb-repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _chip_id(path: Path) -> str:
    return path.stem.split("_")[-1]


def _index_pngs(folder: Path) -> dict[str, Path]:
    return {_chip_id(path): path for path in folder.glob("*.png")}


def collect_event_samples(root: Path, event: str) -> list[OmbriaSample]:
    event_root = root / "2021" / event
    s1_base = event_root / "Sentinel1"
    s2_base = event_root / "Sentinel2"
    indexes = {
        "s1_before": _index_pngs(s1_base / "BEFORE"),
        "s1_after": _index_pngs(s1_base / "AFTER"),
        "s1_mask": _index_pngs(s1_base / "MASK"),
        "s2_before": _index_pngs(s2_base / "BEFORE"),
        "s2_after": _index_pngs(s2_base / "AFTER"),
        "s2_mask": _index_pngs(s2_base / "MASK"),
    }
    id_sets = [set(index) for index in indexes.values()]
    if not id_sets or not all(id_sets):
        raise FileNotFoundError(f"Incomplete event folders for {event}: {event_root}")
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        counts = {name: len(index) for name, index in indexes.items()}
        raise RuntimeError(f"Unmatched modality IDs for {event}: {counts}")
    return [
        OmbriaSample(
            split=f"2021/{event}",
            chip_id=chip_id,
            s1_before=indexes["s1_before"][chip_id],
            s1_after=indexes["s1_after"][chip_id],
            s1_mask=indexes["s1_mask"][chip_id],
            s2_before=indexes["s2_before"][chip_id],
            s2_after=indexes["s2_after"][chip_id],
            s2_mask=indexes["s2_mask"][chip_id],
        )
        for chip_id in sorted(id_sets[0], key=lambda value: int(value))
    ]


def stable_sample_seed(
    base_seed: int, repetition: int, event: str, chip_id: str
) -> int:
    token = f"{event}:{chip_id}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
    return (base_seed + repetition * 1_000_003 + offset) % (2**32)


class EventDataset:
    def __init__(
        self,
        samples: list[OmbriaSample],
        event: str,
        variant: str,
        degrade_s2: str,
        s2_quality: str,
        perturb_seed: int,
        repetition: int,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(samples)

            def __getitem__(self_inner, idx: int):
                sample = samples[idx]
                rng = np.random.default_rng(
                    stable_sample_seed(perturb_seed, repetition, event, sample.chip_id)
                )
                image, mask = load_sample(
                    sample,
                    variant,
                    degrade_s2,
                    rng,
                    s2_quality=s2_quality,
                )
                x = torch.from_numpy(np.moveaxis(image, 2, 0))
                y = torch.from_numpy(mask[None, :, :])
                return x, y, idx

        self.dataset = _Dataset()


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


def evaluate_event(
    model,
    device,
    samples: list[OmbriaSample],
    event: str,
    args: argparse.Namespace,
    repetition: int,
) -> tuple[Counts, list[dict[str, object]]]:
    import torch
    from torch.utils.data import DataLoader

    dataset = EventDataset(
        samples,
        event,
        args.variant,
        args.degrade_s2,
        args.s2_quality,
        args.perturb_seed,
        repetition,
    ).dataset
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    event_counts = Counts()
    chip_rows: list[dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for x, y, indexes in loader:
            logits = model(x.to(device))
            probabilities = torch.sigmoid(logits).cpu()
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
                event_counts.add(counts)
                sample = samples[sample_index]
                chip_rows.append(
                    {
                        "route": args.route,
                        "variant": args.variant,
                        "s2_quality": args.s2_quality,
                        "degrade_s2": args.degrade_s2,
                        "model_seed": args.model_seed,
                        "perturb_seed": args.perturb_seed,
                        "repetition": repetition,
                        "event": event,
                        "chip_id": sample.chip_id,
                        "flood_fraction": float(read_mask(sample.s2_mask).mean()),
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
    return event_counts, chip_rows


def main() -> None:
    args = parse_args()
    if args.perturb_repetitions < 1:
        raise ValueError("--perturb-repetitions must be positive")

    import torch

    config_path = args.checkpoint.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    config = json.loads(config_path.read_text())
    checkpoint_sha256 = file_sha256(args.checkpoint)
    checkpoint_bytes = args.checkpoint.stat().st_size
    base_channels = int(config["base_channels"])
    expected_channels = variant_channels(args.variant, args.s2_quality)
    model = build_model(expected_channels, base_channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    event_samples = {event: collect_event_samples(args.root, event) for event in EVENTS}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "checkpoint": str(args.checkpoint),
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_sha256": checkpoint_sha256,
                "out_dir": str(args.out_dir),
                "event_counts": {
                    event: len(samples) for event, samples in event_samples.items()
                },
                "metric_aggregation": "global confusion counts plus per-chip rows",
            },
            indent=2,
            default=str,
        )
    )

    summary_rows: list[dict[str, object]] = []
    chip_rows: list[dict[str, object]] = []
    for repetition in range(args.perturb_repetitions):
        pooled = Counts()
        for event, samples in event_samples.items():
            counts, rows = evaluate_event(
                model, device, samples, event, args, repetition
            )
            pooled.add(counts)
            chip_rows.extend(rows)
            summary_rows.append(
                {
                    "route": args.route,
                    "variant": args.variant,
                    "s2_quality": args.s2_quality,
                    "degrade_s2": args.degrade_s2,
                    "model_seed": args.model_seed,
                    "perturb_seed": args.perturb_seed,
                    "repetition": repetition,
                    "event": event,
                    "samples": len(samples),
                    "tp": counts.tp,
                    "fp": counts.fp,
                    "fn": counts.fn,
                    "tn": counts.tn,
                    **metrics(counts),
                }
            )
        summary_rows.append(
            {
                "route": args.route,
                "variant": args.variant,
                "s2_quality": args.s2_quality,
                "degrade_s2": args.degrade_s2,
                "model_seed": args.model_seed,
                "perturb_seed": args.perturb_seed,
                "repetition": repetition,
                "event": "ALL",
                "samples": sum(len(samples) for samples in event_samples.values()),
                "tp": pooled.tp,
                "fp": pooled.fp,
                "fn": pooled.fn,
                "tn": pooled.tn,
                **metrics(pooled),
            }
        )

    for path, rows in [
        (args.out_dir / "summary_metrics.csv", summary_rows),
        (args.out_dir / "per_chip_metrics.csv", chip_rows),
    ]:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary_rows[-1], sort_keys=True))


if __name__ == "__main__":
    main()
