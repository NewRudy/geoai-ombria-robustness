from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from time import time
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.models import count_trainable_parameters  # noqa: E402
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    local_paths,
    manifest_records,
)
from geoai_ombria_robustness.smagnet_adapter import (  # noqa: E402
    OFFICIAL_SMAGNET_MODEL,
    build_official_smagnet,
    dual_path_masked_bce,
    file_sha256,
    forward_official_smagnet,
    normalize_sen1floods11_for_official_smagnet,
    verify_complete_absence_equivalence,
)


def stable_stream_seed(base_seed: int, epoch: int, chip_id: str, stream: str) -> int:
    token = f"{base_seed}:{epoch}:{chip_id}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen official SMAGNet architecture on Sen1Floods11."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smagnet-source", type=Path, required=True)
    parser.add_argument("--official-source-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loader-seed", type=int, default=None)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--encoder-weights-msi",
        choices=("imagenet", "none"),
        default="imagenet",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _limit_records(
    records: Sequence[dict[str, Any]], maximum: int, salt: str
) -> list[dict[str, Any]]:
    records = list(records)
    if maximum <= 0 or maximum >= len(records):
        return sorted(records, key=lambda record: str(record["chip_id"]))
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            f"{salt}:{record['chip_id']}".encode("utf-8")
        ).hexdigest(),
    )[:maximum]


def assert_prepared(records: Sequence[dict[str, Any]], data_root: Path) -> None:
    missing: list[str] = []
    for record in records:
        paths = local_paths(data_root, record)
        for path in (paths.s1, paths.s2, paths.label, paths.quality):
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Sen1Floods11 assets are not prepared. Missing examples: "
            + ", ".join(missing[:3])
        )


class TrainingDataset:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        data_root: Path,
        crop_size: int,
        augmentation_seed: int,
        normalization: dict[str, Any],
    ) -> None:
        self.records = list(records)
        self.data_root = data_root
        self.crop_size = crop_size
        self.augmentation_seed = augmentation_seed
        self.normalization = normalization
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        chip = load_hand_labeled_chip(record, self.data_root)
        image = normalize_sen1floods11_for_official_smagnet(
            chip.image, self.normalization
        )
        target = chip.target.copy()
        valid = chip.valid_target.copy()
        height, width = target.shape
        if self.crop_size > min(height, width):
            raise ValueError("crop size exceeds the Sen1Floods11 chip dimensions")
        rng = np.random.default_rng(
            stable_stream_seed(
                self.augmentation_seed,
                self.epoch,
                str(record["chip_id"]),
                "official_random_crop_and_flip",
            )
        )
        top = int(rng.integers(0, height - self.crop_size + 1))
        left = int(rng.integers(0, width - self.crop_size + 1))
        rows = slice(top, top + self.crop_size)
        columns = slice(left, left + self.crop_size)
        image = image[:, rows, columns]
        target = target[rows, columns]
        valid = valid[rows, columns]
        if bool(rng.integers(0, 2)):
            image = np.flip(image, axis=2)
            target = np.flip(target, axis=1)
            valid = np.flip(valid, axis=1)
        if bool(rng.integers(0, 2)):
            image = np.flip(image, axis=1)
            target = np.flip(target, axis=0)
            valid = np.flip(valid, axis=0)
        return (
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(target[None]).astype(np.float32)),
            torch.from_numpy(np.ascontiguousarray(valid[None]).astype(bool)),
            str(record["chip_id"]),
        )


class ValidationPatchDataset:
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        data_root: Path,
        crop_size: int,
        normalization: dict[str, Any],
    ) -> None:
        self.records = list(records)
        self.data_root = data_root
        self.crop_size = crop_size
        self.normalization = normalization
        if 512 % crop_size:
            raise ValueError("validation crop size must divide 512 exactly")
        positions = range(0, 512, crop_size)
        self.index = [
            (record_index, top, left)
            for record_index in range(len(self.records))
            for top in positions
            for left in positions
        ]
        self._cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        import torch

        record_index, top, left = self.index[index]
        if record_index not in self._cache:
            if len(self._cache) >= 2:
                self._cache.clear()
            chip = load_hand_labeled_chip(
                self.records[record_index], self.data_root
            )
            image = normalize_sen1floods11_for_official_smagnet(
                chip.image, self.normalization
            )
            self._cache[record_index] = (chip, image)
        chip, image = self._cache[record_index]
        rows = slice(top, top + self.crop_size)
        columns = slice(left, left + self.crop_size)
        return (
            torch.from_numpy(image[:, rows, columns].copy()),
            torch.from_numpy(chip.target[None, rows, columns].copy()),
            torch.from_numpy(chip.valid_target[None, rows, columns].copy()),
            str(self.records[record_index]["chip_id"]),
        )


def counts_to_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    eps = 1e-9
    return {
        "iou": tp / (tp + fp + fn + eps),
        "f1": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def compute_training_normalization(
    records: Sequence[dict[str, Any]], data_root: Path
) -> dict[str, Any]:
    optical_sum = np.zeros(4, dtype=np.float64)
    optical_squared_sum = np.zeros(4, dtype=np.float64)
    radar_sum = np.zeros(2, dtype=np.float64)
    radar_squared_sum = np.zeros(2, dtype=np.float64)
    pixels = 0
    for record in records:
        chip = load_hand_labeled_chip(record, data_root)
        optical = chip.image[[2, 1, 0, 3]].astype(np.float64)
        radar = chip.image[4:6].astype(np.float64)
        optical_sum += optical.sum(axis=(1, 2))
        optical_squared_sum += np.square(optical).sum(axis=(1, 2))
        radar_sum += radar.sum(axis=(1, 2))
        radar_squared_sum += np.square(radar).sum(axis=(1, 2))
        pixels += int(optical.shape[1] * optical.shape[2])
    if pixels <= 0:
        raise RuntimeError("cannot compute SMAGNet normalization without pixels")
    optical_mean = optical_sum / pixels
    radar_mean = radar_sum / pixels
    optical_variance = optical_squared_sum / pixels - np.square(optical_mean)
    radar_variance = radar_squared_sum / pixels - np.square(radar_mean)
    optical_std = np.sqrt(np.maximum(optical_variance, 0.0))
    radar_std = np.sqrt(np.maximum(radar_variance, 0.0))
    if np.any(optical_std <= 0) or np.any(radar_std <= 0):
        raise RuntimeError("SMAGNet training normalization has a constant channel")
    return {
        "schema": "geoai-sen1floods11-smagnet-normalization-v1",
        "source": "frozen training records only",
        "input_source_scaling": (
            "S1 clipped to [-50,1] dB then [0,1]; S2 clipped to [0,10000] "
            "then [0,1]"
        ),
        "optical_order": ["B4_red", "B3_green", "B2_blue", "B8_nir"],
        "radar_order": ["VV", "VH"],
        "pixels": pixels,
        "optical_mean": optical_mean.tolist(),
        "optical_std": optical_std.tolist(),
        "radar_mean": radar_mean.tolist(),
        "radar_std": radar_std.tolist(),
    }


def evaluate(model, loader, device, *, amp: bool) -> dict[str, float]:
    import torch

    model.eval()
    loss_sum = 0.0
    valid_pixels = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for image, target, valid, _chip_id in loader:
            image = image.to(device)
            target = target.to(device)
            valid = valid.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                fused, sar, _gates = forward_official_smagnet(model, image)
                loss = dual_path_masked_bce(fused, sar, target, valid)
            batch_valid = int(valid.sum().item())
            loss_sum += float(loss.item()) * batch_valid
            valid_pixels += batch_valid
            prediction = torch.sigmoid(fused) > 0.5
            truth = target > 0.5
            tp += int((prediction & truth & valid).sum().item())
            fp += int((prediction & ~truth & valid).sum().item())
            fn += int((~prediction & truth & valid).sum().item())
            tn += int((~prediction & ~truth & valid).sum().item())
    if valid_pixels == 0:
        raise RuntimeError("validation split has no valid target pixels")
    return {
        "loss": loss_sum / valid_pixels,
        "valid_pixels": float(valid_pixels),
        **counts_to_metrics(tp, fp, fn, tn),
    }


def select_validation_threshold(model, loader, device, *, amp: bool) -> dict[str, Any]:
    import torch
    from sklearn.metrics import precision_recall_curve

    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for image, target, valid, _chip_id in loader:
            image = image.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                fused, _sar, _gates = forward_official_smagnet(model, image)
            probability = torch.sigmoid(fused).float().cpu()
            valid = valid.bool()
            probabilities.append(probability[valid].numpy().astype(np.float32))
            targets.append((target[valid] > 0.5).numpy().astype(np.uint8))
    probability_values = np.concatenate(probabilities)
    target_values = np.concatenate(targets)
    if not target_values.any():
        raise RuntimeError("validation threshold selection has no positive pixels")
    precision, recall, thresholds = precision_recall_curve(
        target_values,
        probability_values,
        drop_intermediate=True,
    )
    if not len(thresholds):
        raise RuntimeError("validation precision-recall curve has no thresholds")
    denominator = precision[:-1] + recall[:-1] - precision[:-1] * recall[:-1]
    iou = np.divide(
        precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(iou))
    return {
        "selection_split": "validation",
        "selection_rule": "precision_recall_threshold_maximizing_pixel_iou",
        "valid_pixels": int(target_values.size),
        "positive_pixels": int(target_values.sum()),
        "retained_threshold_candidates": int(len(thresholds)),
        "threshold": float(thresholds[best_index]),
        "iou": float(iou[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }


def current_source_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if min(
        args.epochs,
        args.micro_batch_size,
        args.gradient_accumulation,
        args.crop_size,
    ) < 1:
        raise ValueError("epochs, batch settings, and crop size must be positive")
    if args.loader_seed is None:
        args.loader_seed = args.seed + 200_000
    if args.augmentation_seed is None:
        args.augmentation_seed = args.seed + 300_000
    source_manifest = json.loads(
        args.official_source_manifest.read_text(encoding="utf-8")
    )
    if source_manifest.get("source_sha256") != file_sha256(args.smagnet_source):
        raise ValueError("official source manifest does not match SMAGNet source")

    document = load_sen1floods11_manifest(args.manifest)
    train_records = _limit_records(
        manifest_records(document, ["train"]), args.max_train_samples, "train"
    )
    validation_records = _limit_records(
        manifest_records(document, ["validation"]),
        args.max_validation_samples,
        "validation",
    )
    assert_prepared(train_records + validation_records, args.data_root)
    if args.dry_run:
        chip = load_hand_labeled_chip(train_records[0], args.data_root)
        print(
            json.dumps(
                {
                    "architecture": "official_smagnet",
                    "image_shape": list(chip.image.shape),
                    "train_records": len(train_records),
                    "validation_records": len(validation_records),
                    "effective_batch_size": (
                        args.micro_batch_size * args.gradient_accumulation
                    ),
                },
                indent=2,
            )
        )
        return

    import segmentation_models_pytorch as smp
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    normalization = compute_training_normalization(train_records, args.data_root)
    encoder_weights_msi = (
        None if args.encoder_weights_msi == "none" else args.encoder_weights_msi
    )
    model = build_official_smagnet(
        args.smagnet_source,
        encoder_weights_msi=encoder_weights_msi,
    ).to(device)
    fallback_check = verify_complete_absence_equivalence(model, device=device)
    model_parameters = count_trainable_parameters(model)

    run_name = args.run_name or f"smagnet_official_seed{args.seed}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        **vars(args),
        "architecture": "official_smagnet",
        "route": "smagnet_official",
        "source_commit": current_source_commit(),
        "manifest_sha256": file_sha256(args.manifest),
        "official_source_manifest_sha256": file_sha256(
            args.official_source_manifest
        ),
        "official_model_configuration": {
            **OFFICIAL_SMAGNET_MODEL,
            "encoder_weights_msi": encoder_weights_msi,
        },
        "model_parameters": model_parameters,
        "train_count": len(train_records),
        "validation_count": len(validation_records),
        "validation_patches": len(validation_records) * (512 // args.crop_size) ** 2,
        "effective_batch_size": (
            args.micro_batch_size * args.gradient_accumulation
        ),
        "optimizer": "Adam",
        "loss": "0.5 * masked BCE(SAR) + 0.5 * masked BCE(fused)",
        "checkpoint_rule": "minimum clean validation dual-path BCE",
        "threshold_rule": "validation PR threshold maximizing pixel IoU",
        "normalization": normalization,
        "quality_proxy": (
            "SCL availability intersected with official S2 valid-data support"
        ),
        "device": str(device),
        "amp_effective": amp,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "segmentation_models_pytorch": smp.__version__,
        "paper_default_deviations": [
            "Sen1Floods11 frozen splits replace C2S-MS Floods splits",
            "Sen1Floods11 training-only band statistics replace C2S-MS statistics",
            "gradient accumulation preserves effective batch 16 on 16 GB GPU",
            "invalid target pixels are excluded from both BCE paths",
            "input SCL availability replaces source-image zero-value mask",
            *(["automatic mixed precision enabled"] if amp else []),
        ],
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
    (run_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "fallback_boundary.json").write_text(
        json.dumps(fallback_check, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_dataset = TrainingDataset(
        train_records,
        args.data_root,
        args.crop_size,
        args.augmentation_seed,
        normalization,
    )
    validation_dataset = ValidationPatchDataset(
        validation_records, args.data_root, args.crop_size, normalization
    )
    loader_generator = torch.Generator().manual_seed(args.loader_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    metrics_path = run_dir / "metrics.csv"
    checkpoint_path = run_dir / "best_validation_loss.pt"
    best_loss = float("inf")
    best_epoch = 0
    started = time()
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "val_iou_at_0p5",
            "val_f1_at_0p5",
            "val_precision_at_0p5",
            "val_recall_at_0p5",
            "val_accuracy_at_0p5",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_dataset.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = 0.0
            train_batches = 0
            for batch_index, (image, target, valid, _chip_id) in enumerate(
                train_loader, start=1
            ):
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                valid = valid.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=amp):
                    fused, sar, _gates = forward_official_smagnet(model, image)
                    loss = dual_path_masked_bce(fused, sar, target, valid)
                scaler.scale(loss / args.gradient_accumulation).backward()
                if (
                    batch_index % args.gradient_accumulation == 0
                    or batch_index == len(train_loader)
                ):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_loss += float(loss.item())
                train_batches += 1
            validation = evaluate(model, validation_loader, device, amp=amp)
            row = {
                "epoch": epoch,
                "train_loss": train_loss / max(1, train_batches),
                "val_loss": validation["loss"],
                "val_iou_at_0p5": validation["iou"],
                "val_f1_at_0p5": validation["f1"],
                "val_precision_at_0p5": validation["precision"],
                "val_recall_at_0p5": validation["recall"],
                "val_accuracy_at_0p5": validation["accuracy"],
                "elapsed_seconds": time() - started,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if validation["loss"] < best_loss:
                best_loss = float(validation["loss"])
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    threshold = select_validation_threshold(model, validation_loader, device, amp=amp)
    threshold_path = run_dir / "threshold_selection.json"
    threshold_path.write_text(
        json.dumps(threshold, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checkpoint_manifest = {
        "best_validation_loss_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_checkpoint": checkpoint_path.name,
        "best_checkpoint_sha256": file_sha256(checkpoint_path),
        "threshold": threshold["threshold"],
        "threshold_selection_sha256": file_sha256(threshold_path),
        "fallback_boundary_sha256": file_sha256(
            run_dir / "fallback_boundary.json"
        ),
    }
    (run_dir / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(checkpoint_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
