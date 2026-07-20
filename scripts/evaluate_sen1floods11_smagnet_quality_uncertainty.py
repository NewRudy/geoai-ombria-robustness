from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_sen1floods11_quality_uncertainty import (  # noqa: E402
    Counts,
    event_equal_iou,
    file_sha256,
    load_evaluation_conditions,
    metrics,
    quality_confusion_on_valid_target,
    stable_sample_seed,
    write_csv,
)
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    manifest_records,
)
from geoai_ombria_robustness.sen1floods11_protocol import (  # noqa: E402
    build_quality_condition,
)
from geoai_ombria_robustness.smagnet_adapter import (  # noqa: E402
    build_official_smagnet,
    forward_official_smagnet,
    normalize_sen1floods11_for_official_smagnet,
    verify_complete_absence_equivalence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official SMAGNet adaptation under frozen "
            "quality-map errors."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--smagnet-source", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "bolivia"), required=True)
    parser.add_argument("--conditions-json", type=Path, required=True)
    parser.add_argument("--quality-mode", default=None)
    parser.add_argument("--false-available-rate", type=float, default=0.0)
    parser.add_argument("--false-unavailable-rate", type=float, default=0.0)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument("--matched-source-mode", default="translate")
    parser.add_argument("--perturb-seed", type=int, default=20260716)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def build_smagnet_image(
    chip,
    observed_quality,
    normalization: dict[str, Any],
    *,
    complete_absence: bool,
):
    optical = chip.image[:4].copy()
    radar = chip.image[4:6].copy()
    quality = np.asarray(observed_quality, dtype=np.float32).copy()
    if complete_absence:
        optical.fill(0.0)
        quality.fill(0.0)
    image = np.concatenate([optical, radar, quality[None]], axis=0).astype(
        np.float32, copy=False
    )
    return normalize_sen1floods11_for_official_smagnet(image, normalization)


def _flush_batch(
    model,
    device,
    *,
    amp: bool,
    threshold: float,
    images: list[np.ndarray],
    targets: list[np.ndarray],
    valid_masks: list[np.ndarray],
    record_indices: list[int],
    chip_counts: list[Counts],
    chip_probability_sum: list[float],
    chip_valid_pixels: list[int],
) -> None:
    import torch

    if not images:
        return
    image = torch.from_numpy(np.stack(images)).to(device)
    target = torch.from_numpy(np.stack(targets)) > 0.5
    valid = torch.from_numpy(np.stack(valid_masks)).bool()
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=amp):
        fused, _sar, _gates = forward_official_smagnet(model, image)
    probability = torch.sigmoid(fused).float().cpu()
    prediction = probability > threshold
    for batch_index, record_index in enumerate(record_indices):
        pred = prediction[batch_index]
        truth = target[batch_index]
        valid_mask = valid[batch_index]
        counts = Counts(
            tp=int((pred & truth & valid_mask).sum().item()),
            fp=int((pred & ~truth & valid_mask).sum().item()),
            fn=int((~pred & truth & valid_mask).sum().item()),
            tn=int((~pred & ~truth & valid_mask).sum().item()),
        )
        chip_counts[record_index].add(counts)
        chip_probability_sum[record_index] += float(
            probability[batch_index][valid_mask].sum().item()
        )
        chip_valid_pixels[record_index] += int(valid_mask.sum().item())


def evaluate_repetition(
    model,
    device,
    *,
    chips,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    repetition: int,
    threshold: float,
    amp: bool,
    normalization: dict[str, Any],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if 512 % args.patch_size:
        raise ValueError("patch size must divide the 512-pixel chips exactly")
    chip_counts = [Counts() for _record in records]
    chip_probability_sum = [0.0 for _record in records]
    chip_valid_pixels = [0 for _record in records]
    quality_values_by_chip: list[np.ndarray] = []
    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    record_indices: list[int] = []

    def flush() -> None:
        _flush_batch(
            model,
            device,
            amp=amp,
            threshold=threshold,
            images=images,
            targets=targets,
            valid_masks=valid_masks,
            record_indices=record_indices,
            chip_counts=chip_counts,
            chip_probability_sum=chip_probability_sum,
            chip_valid_pixels=chip_valid_pixels,
        )
        images.clear()
        targets.clear()
        valid_masks.clear()
        record_indices.clear()

    for record_index, (record, chip) in enumerate(zip(records, chips, strict=True)):
        condition = build_quality_condition(
            chip.reference_quality,
            mode=args.quality_mode,
            rng=np.random.default_rng(
                stable_sample_seed(
                    args.perturb_seed,
                    repetition,
                    str(record["chip_id"]),
                    "quality_condition",
                )
            ),
            false_available_rate=args.false_available_rate,
            false_unavailable_rate=args.false_unavailable_rate,
            shift_y=args.shift_y,
            shift_x=args.shift_x,
            radius=args.radius,
            matched_source_mode=args.matched_source_mode,
            comparison_mask=chip.valid_target,
        )
        confusion = condition.perturbation.confusion
        valid_confusion = quality_confusion_on_valid_target(
            chip.reference_quality, condition.observed, chip.valid_target
        )
        quality_values_by_chip.append(
            np.asarray(
                [
                    confusion.available_pixels,
                    confusion.unavailable_pixels,
                    confusion.false_available,
                    confusion.false_unavailable,
                    confusion.false_available_rate,
                    confusion.false_unavailable_rate,
                    confusion.quality_iou,
                    valid_confusion.available_pixels,
                    valid_confusion.unavailable_pixels,
                    valid_confusion.false_available,
                    valid_confusion.false_unavailable,
                    valid_confusion.false_available_rate,
                    valid_confusion.false_unavailable_rate,
                    valid_confusion.quality_iou,
                ],
                dtype=np.float64,
            )
        )
        image = build_smagnet_image(
            chip,
            condition.observed,
            normalization,
            complete_absence=condition.complete_optical_absence,
        )
        for top in range(0, 512, args.patch_size):
            for left in range(0, 512, args.patch_size):
                rows = slice(top, top + args.patch_size)
                columns = slice(left, left + args.patch_size)
                images.append(image[:, rows, columns])
                targets.append(chip.target[None, rows, columns])
                valid_masks.append(chip.valid_target[None, rows, columns])
                record_indices.append(record_index)
                if len(images) == args.batch_size:
                    flush()
    flush()

    pooled = Counts()
    event_counts: dict[str, Counts] = defaultdict(Counts)
    chip_rows: list[dict[str, object]] = []
    quality_totals = np.zeros(8, dtype=np.float64)
    for record_index, record in enumerate(records):
        counts = chip_counts[record_index]
        pooled.add(counts)
        event_counts[str(record["event"])].add(counts)
        quality = quality_values_by_chip[record_index]
        quality_totals += quality[[0, 1, 2, 3, 7, 8, 9, 10]]
        valid_pixels = chip_valid_pixels[record_index]
        chip_rows.append(
            {
                "route": "smagnet_official",
                "model_seed": args.model_seed,
                "split": args.split,
                "condition_id": args.condition_id,
                "quality_mode": args.quality_mode,
                "false_available_rate": args.false_available_rate,
                "false_unavailable_rate": args.false_unavailable_rate,
                "shift_y": args.shift_y,
                "shift_x": args.shift_x,
                "radius": args.radius,
                "matched_source_mode": args.matched_source_mode,
                "perturb_seed": args.perturb_seed,
                "repetition": repetition,
                "chip_id": record["chip_id"],
                "event": record["event"],
                "quality_false_available_rate": quality[4],
                "quality_false_unavailable_rate": quality[5],
                "quality_iou": quality[6],
                "valid_quality_false_available_rate": quality[11],
                "valid_quality_false_unavailable_rate": quality[12],
                "valid_quality_iou": quality[13],
                "valid_target_pixels": valid_pixels,
                "has_valid_target": valid_pixels > 0,
                "mean_probability": (
                    chip_probability_sum[record_index] / valid_pixels
                    if valid_pixels
                    else None
                ),
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "tn": counts.tn,
                **metrics(counts),
            }
        )
    event_rows = [
        {
            "route": "smagnet_official",
            "model_seed": args.model_seed,
            "split": args.split,
            "condition_id": args.condition_id,
            "quality_mode": args.quality_mode,
            "false_available_rate": args.false_available_rate,
            "false_unavailable_rate": args.false_unavailable_rate,
            "shift_y": args.shift_y,
            "shift_x": args.shift_x,
            "radius": args.radius,
            "matched_source_mode": args.matched_source_mode,
            "perturb_seed": args.perturb_seed,
            "repetition": repetition,
            "event": event,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "tn": counts.tn,
            **metrics(counts),
        }
        for event, counts in sorted(event_counts.items())
    ]
    (
        available,
        unavailable,
        false_available,
        false_unavailable,
        valid_available,
        valid_unavailable,
        valid_false_available,
        valid_false_unavailable,
    ) = quality_totals
    summary = {
        "route": "smagnet_official",
        "model_seed": args.model_seed,
        "split": args.split,
        "condition_id": args.condition_id,
        "quality_mode": args.quality_mode,
        "false_available_rate": args.false_available_rate,
        "false_unavailable_rate": args.false_unavailable_rate,
        "shift_y": args.shift_y,
        "shift_x": args.shift_x,
        "radius": args.radius,
        "matched_source_mode": args.matched_source_mode,
        "perturb_seed": args.perturb_seed,
        "repetition": repetition,
        "samples": len(records),
        "events": len(event_counts),
        "threshold": threshold,
        "realized_false_available_rate": (
            float(false_available / unavailable) if unavailable else 0.0
        ),
        "realized_false_unavailable_rate": (
            float(false_unavailable / available) if available else 0.0
        ),
        "valid_realized_false_available_rate": (
            float(valid_false_available / valid_unavailable)
            if valid_unavailable
            else 0.0
        ),
        "valid_realized_false_unavailable_rate": (
            float(valid_false_unavailable / valid_available) if valid_available else 0.0
        ),
        "event_equal_iou": event_equal_iou(event_counts),
        "tp": pooled.tp,
        "fp": pooled.fp,
        "fn": pooled.fn,
        "tn": pooled.tn,
        **metrics(pooled),
    }
    return summary, chip_rows, event_rows


def main() -> None:
    args = parse_args()
    if min(args.repetitions, args.batch_size, args.patch_size) < 1:
        raise ValueError("repetitions, batch size, and patch size must be positive")
    conditions = load_evaluation_conditions(args)
    config_path = args.checkpoint.parent / "config.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if configuration.get("architecture") != "official_smagnet":
        raise ValueError("checkpoint config is not the official SMAGNet adaptation")
    if file_sha256(args.checkpoint) != json.loads(
        (args.checkpoint.parent / "checkpoint_manifest.json").read_text(
            encoding="utf-8"
        )
    )["best_checkpoint_sha256"]:
        raise ValueError("SMAGNet checkpoint hash does not match its manifest")
    threshold_document = json.loads(
        (args.checkpoint.parent / "threshold_selection.json").read_text(
            encoding="utf-8"
        )
    )
    threshold = float(threshold_document["threshold"])
    normalization = configuration["normalization"]
    args.model_seed = int(configuration["seed"])

    document = load_sen1floods11_manifest(args.manifest)
    records = sorted(
        manifest_records(document, [args.split]),
        key=lambda record: str(record["chip_id"]),
    )
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError(f"no records found for split {args.split}")
    chips = [load_hand_labeled_chip(record, args.data_root) for record in records]

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(args.amp and device.type == "cuda")
    model = build_official_smagnet(
        args.smagnet_source, encoder_weights_msi=None
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    fallback_check = verify_complete_absence_equivalence(model, device=device)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "evaluation_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "conditions": conditions,
                "manifest_sha256": file_sha256(args.manifest),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "checkpoint_config": configuration,
                "sample_count": len(records),
                "patches_per_chip": (512 // args.patch_size) ** 2,
                "threshold": threshold_document,
                "normalization": normalization,
                "fallback_boundary": fallback_check,
                "reference_quality": (
                    "available SCL class intersected with official S2 valid data"
                ),
                "amp_effective": amp,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    summaries: list[dict[str, object]] = []
    chip_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for condition in conditions:
        condition_args = argparse.Namespace(**vars(args))
        for name, value in condition.items():
            setattr(condition_args, name, value)
        for repetition in range(args.repetitions):
            summary, per_chip, per_event = evaluate_repetition(
                model,
                device,
                chips=chips,
                records=records,
                args=condition_args,
                repetition=repetition,
                threshold=threshold,
                amp=amp,
                normalization=normalization,
            )
            summaries.append(summary)
            chip_rows.extend(per_chip)
            event_rows.extend(per_event)
            print(json.dumps(summary, sort_keys=True), flush=True)
    write_csv(args.out_dir / "summary_metrics.csv", summaries)
    write_csv(args.out_dir / "per_chip_metrics.csv", chip_rows)
    write_csv(args.out_dir / "per_event_metrics.csv", event_rows)


if __name__ == "__main__":
    main()
