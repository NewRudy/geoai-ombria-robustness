from __future__ import annotations

from collections import defaultdict
from typing import Iterable


AVERAGED_FIELDS = (
    "iou",
    "f1",
    "precision",
    "recall",
    "event_equal_iou",
    "realized_false_available_rate",
    "realized_false_unavailable_rate",
    "valid_realized_false_available_rate",
    "valid_realized_false_unavailable_rate",
)

CONDITION_FIELDS = (
    "quality_mode",
    "false_available_rate",
    "false_unavailable_rate",
    "shift_y",
    "shift_x",
    "radius",
    "matched_source_mode",
)


def _mean(rows: list[dict[str, str]], name: str) -> float:
    values = [float(row[name]) for row in rows if row.get(name, "") != ""]
    return sum(values) / len(values) if values else float("nan")


def summarize_seed_conditions(
    rows: Iterable[dict[str, str]],
) -> list[dict[str, object]]:
    """Average perturbation repeats and attach a paired S1 reference."""

    grouped: dict[tuple[int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["model_seed"]),
            str(row["split"]),
            str(row["route"]),
            str(row["condition_id"]),
        )
        grouped[key].append(row)
    if not grouped:
        raise RuntimeError("No evaluation rows were supplied")

    s1_by_seed_split: dict[tuple[int, str], float] = {}
    for (seed, split, route, _condition), values in grouped.items():
        if route == "s1_reference":
            s1_by_seed_split[(seed, split)] = _mean(values, "iou")

    summaries: list[dict[str, object]] = []
    for (seed, split, route, condition), values in sorted(grouped.items()):
        reference_key = (seed, split)
        if reference_key not in s1_by_seed_split:
            raise RuntimeError(
                f"Missing S1 reference for model seed {seed} and split {split}"
            )
        first = values[0]
        iou = _mean(values, "iou")
        summary: dict[str, object] = {
            "model_seed": seed,
            "split": split,
            "route": route,
            "condition_id": condition,
            **{name: first.get(name, "") for name in CONDITION_FIELDS},
            **{name: _mean(values, name) for name in AVERAGED_FIELDS},
            "s1_reference_iou": s1_by_seed_split[reference_key],
            "delta_s1_iou": iou - s1_by_seed_split[reference_key],
            "repetitions": len(values),
        }
        summaries.append(summary)
    return summaries
