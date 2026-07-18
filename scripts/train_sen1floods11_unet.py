from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from time import time

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.models import count_trainable_parameters  # noqa: E402
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    local_paths,
    manifest_records,
)
from geoai_ombria_robustness.sen1floods11_protocol import (  # noqa: E402
    SEN1FLOODS11_ROUTES,
    augment_spatially,
    build_observed_quality,
    build_route_input,
    route_config,
    route_manifest,
)
from geoai_ombria_robustness.single_time_models import (  # noqa: E402
    build_single_time_model,
)


def stable_stream_seed(base_seed: int, epoch: int, chip_id: str, stream: str) -> int:
    token = f"{base_seed}:{epoch}:{chip_id}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train capacity-controlled Sen1Floods11 comparison routes."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--route", choices=tuple(SEN1FLOODS11_ROUTES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--quality-branch-channels", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loader-seed", type=int, default=None)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    parser.add_argument("--quality-error-seed", type=int, default=None)
    parser.add_argument(
        "--train-quality-error-rates",
        type=float,
        nargs="*",
        default=(0.0, 0.05, 0.10, 0.20, 0.40),
    )
    parser.add_argument("--modality-dropout-rate", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _limit_records(
    records: list[dict],
    maximum: int,
    salt: str,
) -> list[dict]:
    if maximum <= 0 or maximum >= len(records):
        return sorted(records, key=lambda record: str(record["chip_id"]))
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record['chip_id']}".encode("utf-8")
        ).hexdigest(),
    )[:maximum]


def assert_prepared(records: list[dict], data_root: Path) -> None:
    missing = []
    for record in records:
        paths = local_paths(data_root, record)
        for path in (paths.s1, paths.s2, paths.label):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Sen1Floods11 assets are not prepared. Missing examples: "
            + ", ".join(missing[:3])
        )


class Sen1Floods11TorchDataset:
    def __init__(
        self,
        records: list[dict],
        data_root: Path,
        route: str,
        training: bool,
        augmentation_seed: int,
        quality_error_seed: int,
        quality_error_rates: tuple[float, ...],
        modality_dropout_rate: float,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        config = route_config(route)

        class _Dataset(Dataset):
            def __init__(self_inner) -> None:
                self_inner.epoch = 0

            def set_epoch(self_inner, epoch: int) -> None:
                self_inner.epoch = int(epoch)

            def __len__(self_inner) -> int:
                return len(records)

            def __getitem__(self_inner, index: int):
                record = records[index]
                chip_id = str(record["chip_id"])
                chip = load_hand_labeled_chip(record, data_root)
                epoch = self_inner.epoch if training else 0
                observed_quality = None
                if training and config.error_aware_training:
                    rate_rng = np.random.default_rng(
                        stable_stream_seed(
                            quality_error_seed,
                            epoch,
                            chip_id,
                            "quality_error_rates",
                        )
                    )
                    false_available_rate = float(rate_rng.choice(quality_error_rates))
                    false_unavailable_rate = float(rate_rng.choice(quality_error_rates))
                    observed_quality = build_observed_quality(
                        chip.reference_quality,
                        false_available_rate=false_available_rate,
                        false_unavailable_rate=false_unavailable_rate,
                        rng=np.random.default_rng(
                            stable_stream_seed(
                                quality_error_seed,
                                epoch,
                                chip_id,
                                "quality_error_pixels",
                            )
                        ),
                    ).observed

                complete_absence = False
                if training and config.modality_dropout_training:
                    dropout_rng = np.random.default_rng(
                        stable_stream_seed(
                            quality_error_seed,
                            epoch,
                            chip_id,
                            "modality_dropout",
                        )
                    )
                    complete_absence = bool(
                        dropout_rng.random() < modality_dropout_rate
                    )
                image = build_route_input(
                    chip,
                    route,
                    observed_quality=observed_quality,
                    complete_optical_absence=complete_absence,
                )
                target = chip.target
                valid_target = chip.valid_target
                if training:
                    image, target, valid_target = augment_spatially(
                        image,
                        target,
                        valid_target,
                        np.random.default_rng(
                            stable_stream_seed(
                                augmentation_seed,
                                epoch,
                                chip_id,
                                "spatial_augmentation",
                            )
                        ),
                    )
                return (
                    torch.from_numpy(image),
                    torch.from_numpy(target[None].astype(np.float32)),
                    torch.from_numpy(valid_target[None].astype(bool)),
                    chip_id,
                )

        self.dataset = _Dataset()


def masked_bce_with_logits(logits, target, valid_target):
    import torch.nn.functional as F

    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    valid = valid_target.to(pixel_loss.dtype)
    return (pixel_loss * valid).sum() / valid.sum().clamp_min(1.0)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
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
    loss_sum = 0.0
    valid_pixels = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for image, target, valid, _ in loader:
            image = image.to(device)
            target = target.to(device)
            valid = valid.to(device)
            logits = model(image)
            pixel_loss = F.binary_cross_entropy_with_logits(
                logits,
                target,
                reduction="none",
            )
            loss_sum += float((pixel_loss * valid).sum().item())
            valid_pixels += int(valid.sum().item())
            prediction = torch.sigmoid(logits) > 0.5
            truth = target > 0.5
            tp += int((prediction & truth & valid).sum().item())
            fp += int((prediction & ~truth & valid).sum().item())
            fn += int((~prediction & truth & valid).sum().item())
            tn += int((~prediction & ~truth & valid).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("Evaluation split has no valid target pixels")
    return {
        "loss": loss_sum / valid_pixels,
        "valid_pixels": float(valid_pixels),
        **metrics_from_counts(tp, fp, fn, tn),
    }


def main() -> None:
    args = parse_args()
    config = route_config(args.route)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.loader_seed is None:
        args.loader_seed = args.seed + 200_000
    if args.augmentation_seed is None:
        args.augmentation_seed = args.seed + 300_000
    if args.quality_error_seed is None:
        args.quality_error_seed = args.seed + 400_000
    rates = tuple(float(rate) for rate in args.train_quality_error_rates)
    if not rates or any(rate < 0.0 or rate > 1.0 for rate in rates):
        raise ValueError("train quality-error rates must lie within [0, 1]")
    if not 0.0 <= args.modality_dropout_rate <= 1.0:
        raise ValueError("modality-dropout-rate must lie within [0, 1]")

    document = load_sen1floods11_manifest(args.manifest)
    train_records = _limit_records(
        manifest_records(document, ["train"]),
        args.max_train_samples,
        "train",
    )
    validation_records = _limit_records(
        manifest_records(document, ["validation"]),
        args.max_validation_samples,
        "validation",
    )
    assert_prepared(train_records + validation_records, args.data_root)

    if args.dry_run:
        chip = load_hand_labeled_chip(train_records[0], args.data_root)
        image = build_route_input(chip, args.route)
        print(
            json.dumps(
                {
                    "route": args.route,
                    "architecture": config.architecture,
                    "image_shape": list(image.shape),
                    "target_shape": list(chip.target.shape),
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                },
                indent=2,
            )
        )
        return

    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_single_time_model(
        base_channels=args.base_channels,
        architecture=config.architecture,
        quality_branch_channels=args.quality_branch_channels,
    ).to(device)
    args.quality_branch_channels = getattr(model, "quality_branch_channels", None)
    model_parameters = count_trainable_parameters(model)

    run_name = args.run_name or f"{args.route}_seed{args.seed}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        **vars(args),
        "architecture": config.architecture,
        "route_definition": route_manifest()[args.route],
        "model_parameters": model_parameters,
        "manifest_sha256": file_sha256(args.manifest),
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "device": str(device),
    }
    (run_dir / "config.json").write_text(
        json.dumps(configuration, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (run_dir / "splits.json").write_text(
        json.dumps(
            {
                "train": [record["chip_id"] for record in train_records],
                "validation": [record["chip_id"] for record in validation_records],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    train_dataset = Sen1Floods11TorchDataset(
        train_records,
        args.data_root,
        args.route,
        training=True,
        augmentation_seed=args.augmentation_seed,
        quality_error_seed=args.quality_error_seed,
        quality_error_rates=rates,
        modality_dropout_rate=args.modality_dropout_rate,
    ).dataset
    validation_dataset = Sen1Floods11TorchDataset(
        validation_records,
        args.data_root,
        args.route,
        training=False,
        augmentation_seed=args.augmentation_seed,
        quality_error_seed=args.quality_error_seed,
        quality_error_rates=rates,
        modality_dropout_rate=args.modality_dropout_rate,
    ).dataset
    loader_generator = torch.Generator().manual_seed(args.loader_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    metrics_path = run_dir / "metrics.csv"
    best_iou = -1.0
    best_epoch = 0
    started = time()
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            train_loss = 0.0
            train_batches = 0
            for image, target, valid, _ in train_loader:
                image = image.to(device)
                target = target.to(device)
                valid = valid.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = masked_bce_with_logits(model(image), target, valid)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_batches += 1
            validation = evaluate(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_batches),
                "val_loss": validation["loss"],
                "val_iou": validation["iou"],
                "val_f1": validation["f1"],
                "val_precision": validation["precision"],
                "val_recall": validation["recall"],
                "val_accuracy": validation["accuracy"],
                "elapsed_seconds": time() - started,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True))
            if validation["iou"] > best_iou:
                best_iou = validation["iou"]
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "best_clean.pt")
        torch.save(model.state_dict(), run_dir / "last.pt")

    checkpoint_manifest = {
        "best_clean_epoch": best_epoch,
        "best_clean_iou": best_iou,
        "best_clean_sha256": file_sha256(run_dir / "best_clean.pt"),
        "last_sha256": file_sha256(run_dir / "last.pt"),
    }
    (run_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint_manifest, sort_keys=True))


if __name__ == "__main__":
    main()
