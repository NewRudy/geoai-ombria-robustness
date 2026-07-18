from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_maps import (  # noqa: E402
    QualityMapConfusion,
    quality_map_confusion,
)
from geoai_ombria_robustness.sen1floods11 import (  # noqa: E402
    load_hand_labeled_chip,
    load_sen1floods11_manifest,
    manifest_records,
)
from geoai_ombria_robustness.sen1floods11_protocol import (  # noqa: E402
    build_quality_condition,
    build_route_input,
    route_config,
)
from geoai_ombria_robustness.single_time_models import (  # noqa: E402
    build_single_time_model,
)


QUALITY_MODES = (
    "reference",
    "independent",
    "translate",
    "dilate",
    "erode",
    "matched-random",
    "complete-absence",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one Sen1Floods11 route while perturbing the supplied "
            "quality map independently of optical content."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("manifests/sen1floods11_scl_manifest.json"),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--route", default=None)
    parser.add_argument(
        "--split",
        choices=("validation", "test", "bolivia"),
        default="test",
    )
    parser.add_argument("--quality-mode", choices=QUALITY_MODES, default=None)
    parser.add_argument("--conditions-json", type=Path, default=None)
    parser.add_argument("--false-available-rate", type=float, default=0.0)
    parser.add_argument("--false-unavailable-rate", type=float, default=0.0)
    parser.add_argument("--shift-y", type=int, default=0)
    parser.add_argument("--shift-x", type=int, default=0)
    parser.add_argument("--radius", type=int, default=0)
    parser.add_argument(
        "--matched-source-mode",
        choices=("translate", "dilate", "erode"),
        default="translate",
    )
    parser.add_argument("--perturb-seed", type=int, default=20260716)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def normalize_evaluation_conditions(
    raw_conditions: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not raw_conditions:
        raise ValueError("At least one evaluation condition is required")
    defaults: dict[str, object] = {
        "false_available_rate": 0.0,
        "false_unavailable_rate": 0.0,
        "shift_y": 0,
        "shift_x": 0,
        "radius": 0,
        "matched_source_mode": "translate",
    }
    conditions: list[dict[str, object]] = []
    identifiers: list[str] = []
    for raw in raw_conditions:
        condition = {**defaults, **raw}
        identifier = str(condition.get("condition_id", "")).strip()
        mode = str(condition.get("quality_mode", "")).strip()
        if not identifier:
            raise ValueError("Every condition needs a non-empty condition_id")
        if mode not in QUALITY_MODES:
            raise ValueError(f"Unknown quality_mode {mode!r}")
        condition["condition_id"] = identifier
        condition["quality_mode"] = mode
        condition["false_available_rate"] = float(condition["false_available_rate"])
        condition["false_unavailable_rate"] = float(condition["false_unavailable_rate"])
        condition["shift_y"] = int(condition["shift_y"])
        condition["shift_x"] = int(condition["shift_x"])
        condition["radius"] = int(condition["radius"])
        for name in ("false_available_rate", "false_unavailable_rate"):
            value = float(condition[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie within [0, 1]")
        if int(condition["radius"]) < 0:
            raise ValueError("radius must be non-negative")
        if condition["matched_source_mode"] not in {
            "translate",
            "dilate",
            "erode",
        }:
            raise ValueError("matched_source_mode is invalid")
        identifiers.append(identifier)
        conditions.append(condition)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("condition_id values must be unique")
    return conditions


def load_evaluation_conditions(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.conditions_json is not None:
        document = json.loads(args.conditions_json.read_text(encoding="utf-8"))
        raw = document.get("conditions") if isinstance(document, dict) else document
        if not isinstance(raw, list):
            raise ValueError("conditions-json must contain a list of conditions")
        return normalize_evaluation_conditions(raw)
    if args.quality_mode is None:
        raise ValueError("Provide --quality-mode or --conditions-json")
    return normalize_evaluation_conditions(
        [
            {
                "condition_id": "single",
                "quality_mode": args.quality_mode,
                "false_available_rate": args.false_available_rate,
                "false_unavailable_rate": args.false_unavailable_rate,
                "shift_y": args.shift_y,
                "shift_x": args.shift_x,
                "radius": args.radius,
                "matched_source_mode": args.matched_source_mode,
            }
        ]
    )


def stable_sample_seed(
    base_seed: int,
    repetition: int,
    chip_id: str,
    stream: str,
) -> int:
    token = f"{base_seed}:{repetition}:{chip_id}:{stream}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:4], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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


def event_equal_iou(event_counts: dict[str, Counts]) -> float:
    if not event_counts:
        raise ValueError("At least one event is required")
    return float(np.mean([metrics(counts)["iou"] for counts in event_counts.values()]))


def quality_confusion_on_valid_target(
    reference: np.ndarray,
    observed: np.ndarray,
    valid_target: np.ndarray,
) -> QualityMapConfusion:
    """Measure quality-map errors only where segmentation labels are valid.

    Some official Sen1Floods11 chips contain no valid hand-label pixels. They
    remain in the frozen 446-chip manifest for provenance and coverage, but
    contribute zero counts to valid-domain statistics. Quality IoU follows the
    existing empty-union identity convention (1.0); callers retain the valid
    pixel count so this value is never mistaken for observed evidence.
    """

    reference = np.asarray(reference)
    observed = np.asarray(observed)
    valid_target = np.asarray(valid_target, dtype=bool)
    if reference.shape != observed.shape or reference.shape != valid_target.shape:
        raise ValueError(
            "reference, observed, and valid_target must have equal shapes"
        )
    if valid_target.size == 0:
        raise ValueError("valid_target map must not be empty")
    if not valid_target.any():
        return QualityMapConfusion(
            available_pixels=0,
            unavailable_pixels=0,
            true_available=0,
            true_unavailable=0,
            false_available=0,
            false_unavailable=0,
            false_available_rate=0.0,
            false_unavailable_rate=0.0,
            quality_iou=1.0,
        )
    return quality_map_confusion(
        reference[valid_target][None, :],
        observed[valid_target][None, :],
    )


def masked_mean_probability(values, valid_target) -> float | None:
    """Return no value, rather than NaN, for an empty labeled domain."""

    valid_pixels = int(valid_target.sum().item())
    if valid_pixels == 0:
        return None
    return float(values[valid_target].mean().item())


class EvaluationDataset:
    def __init__(
        self,
        records: list[dict],
        data_root: Path,
        route: str,
        args: argparse.Namespace,
        repetition: int,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(records)

            def __getitem__(self_inner, index: int):
                record = records[index]
                chip = load_hand_labeled_chip(record, data_root)
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
                image = build_route_input(
                    chip,
                    route,
                    observed_quality=condition.observed,
                    complete_optical_absence=(condition.complete_optical_absence),
                )
                confusion = condition.perturbation.confusion
                valid_confusion = quality_confusion_on_valid_target(
                    chip.reference_quality,
                    condition.observed,
                    chip.valid_target,
                )
                quality_values = np.array(
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
                return (
                    torch.from_numpy(image),
                    torch.from_numpy(chip.target[None].astype(np.float32)),
                    torch.from_numpy(chip.valid_target[None].astype(bool)),
                    index,
                    torch.from_numpy(quality_values),
                )

        self.dataset = _Dataset()


def evaluate_repetition(
    model,
    device,
    loader,
    records: list[dict],
    args: argparse.Namespace,
    repetition: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    import torch

    pooled = Counts()
    event_counts: dict[str, Counts] = defaultdict(Counts)
    chip_rows: list[dict[str, object]] = []
    quality_totals = np.zeros(8, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for image, target, valid, indices, quality_values in loader:
            probability = torch.sigmoid(model(image.to(device))).cpu()
            prediction = probability > 0.5
            truth = target > 0.5
            for batch_index, record_index in enumerate(indices.tolist()):
                record = records[record_index]
                pred = prediction[batch_index]
                target_mask = truth[batch_index]
                valid_mask = valid[batch_index]
                valid_target_pixels = int(valid_mask.sum().item())
                counts = Counts(
                    tp=int((pred & target_mask & valid_mask).sum().item()),
                    fp=int((pred & ~target_mask & valid_mask).sum().item()),
                    fn=int((~pred & target_mask & valid_mask).sum().item()),
                    tn=int((~pred & ~target_mask & valid_mask).sum().item()),
                )
                pooled.add(counts)
                event_counts[str(record["event"])].add(counts)
                quality = quality_values[batch_index].numpy()
                quality_totals += quality[[0, 1, 2, 3, 7, 8, 9, 10]]
                chip_rows.append(
                    {
                        "route": args.route,
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
                        "valid_target_pixels": valid_target_pixels,
                        "has_valid_target": valid_target_pixels > 0,
                        "mean_probability": masked_mean_probability(
                            probability[batch_index], valid_mask
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
            "route": args.route,
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
        "route": args.route,
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.repetitions < 1 or args.batch_size < 1:
        raise ValueError("repetitions and batch-size must be positive")
    conditions = load_evaluation_conditions(args)

    config_path = args.checkpoint.parent / "config.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    configured_route = str(configuration["route"])
    if args.route is not None and args.route != configured_route:
        raise ValueError("--route disagrees with checkpoint config")
    args.route = configured_route
    route = route_config(args.route)
    if str(configuration["architecture"]) != route.architecture:
        raise ValueError("Checkpoint architecture disagrees with route definition")
    model_seed = int(configuration["seed"])

    document = load_sen1floods11_manifest(args.manifest)
    records = manifest_records(document, [args.split])
    records = sorted(records, key=lambda record: str(record["chip_id"]))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    if not records:
        raise RuntimeError(f"No records found for split {args.split}")

    import torch
    from torch.utils.data import DataLoader

    model = build_single_time_model(
        base_channels=int(configuration["base_channels"]),
        architecture=route.architecture,
        quality_branch_channels=(
            int(configuration["quality_branch_channels"])
            if configuration.get("quality_branch_channels") is not None
            else None
        ),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
                "reference_quality": (
                    "available SCL class intersected with official S2 valid data"
                ),
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
        condition_args.model_seed = model_seed
        for repetition in range(args.repetitions):
            dataset = EvaluationDataset(
                records,
                args.data_root,
                args.route,
                condition_args,
                repetition,
            ).dataset
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            summary, per_chip, per_event = evaluate_repetition(
                model,
                device,
                loader,
                records,
                condition_args,
                repetition,
            )
            summaries.append(summary)
            chip_rows.extend(per_chip)
            event_rows.extend(per_event)

    write_csv(args.out_dir / "summary_metrics.csv", summaries)
    write_csv(args.out_dir / "per_chip_metrics.csv", chip_rows)
    write_csv(args.out_dir / "per_event_metrics.csv", event_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
