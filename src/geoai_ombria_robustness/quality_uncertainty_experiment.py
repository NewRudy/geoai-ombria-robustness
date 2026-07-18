from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Iterable


CORE_ROUTES = (
    "s1_reference",
    "early_fusion",
    "early_fusion_dropout",
    "quality_concat",
    "quality_concat_error_aware",
    "hard_quality_gate",
    "hard_quality_gate_error_aware",
    "soft_quality_prior_error_aware",
)

QUALITY_ROUTES = (
    "quality_concat",
    "quality_concat_error_aware",
    "hard_quality_gate",
    "hard_quality_gate_error_aware",
    "soft_quality_prior_error_aware",
)

NO_QUALITY_ROUTES = tuple(route for route in CORE_ROUTES if route not in QUALITY_ROUTES)

MODEL_SEEDS = (7, 13, 21, 29, 37)
SMOKE_ERROR_RATES = (0.0, 0.2, 0.4)
FULL_ERROR_RATES = (0.0, 0.05, 0.1, 0.2, 0.4)
SELECTION_SALT = "sen1floods11-quality-uncertainty-v1"


@dataclass(frozen=True)
class IndependentCondition:
    false_available_rate: float
    false_unavailable_rate: float


@dataclass(frozen=True)
class StructuredCondition:
    name: str
    mode: str
    shift_y: int = 0
    shift_x: int = 0
    radius: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"translate", "dilate", "erode"}:
            raise ValueError(f"Unsupported structured mode: {self.mode}")
        if self.radius < 0:
            raise ValueError("radius must be non-negative")


@dataclass(frozen=True)
class ExperimentPlan:
    mode: str
    pipeline_only: bool
    seeds: tuple[int, ...]
    epochs: int
    error_rates: tuple[float, ...]
    routes: tuple[str, ...]
    quality_routes: tuple[str, ...]
    evaluation_splits: tuple[str, ...]
    sample_limits: dict[str, int]
    perturbation_repetitions: int
    structured_conditions: tuple[StructuredCondition, ...]
    batch_size: int = 4
    base_channels: int = 16
    perturb_seed: int = 20260716

    @property
    def independent_conditions(self) -> tuple[IndependentCondition, ...]:
        return tuple(
            IndependentCondition(false_available, false_unavailable)
            for false_available, false_unavailable in product(
                self.error_rates,
                repeat=2,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "geoai-quality-uncertainty-experiment-plan-v1",
            "mode": self.mode,
            "pipeline_only": self.pipeline_only,
            "seeds": list(self.seeds),
            "epochs": self.epochs,
            "error_rates": list(self.error_rates),
            "routes": list(self.routes),
            "quality_routes": list(self.quality_routes),
            "evaluation_splits": list(self.evaluation_splits),
            "sample_limits": dict(self.sample_limits),
            "perturbation_repetitions": self.perturbation_repetitions,
            "structured_conditions": [
                asdict(condition) for condition in self.structured_conditions
            ],
            "batch_size": self.batch_size,
            "base_channels": self.base_channels,
            "perturb_seed": self.perturb_seed,
            "claim_boundary": (
                "Smoke validates execution only. Full results remain bounded "
                "to controlled OMBRIA cloud-like occlusion and a "
                "Sen1Floods11 SCL-derived operational quality proxy."
            ),
        }


def _smoke_structured_conditions() -> tuple[StructuredCondition, ...]:
    return (
        StructuredCondition("translate_east_5pct", "translate", shift_x=26),
        StructuredCondition("dilate_unavailable_r8", "dilate", radius=8),
        StructuredCondition("erode_unavailable_r8", "erode", radius=8),
    )


def _full_structured_conditions() -> tuple[StructuredCondition, ...]:
    conditions: list[StructuredCondition] = []
    for pixels, label in ((26, "5pct"), (51, "10pct")):
        conditions.extend(
            [
                StructuredCondition(
                    f"translate_north_{label}",
                    "translate",
                    shift_y=-pixels,
                ),
                StructuredCondition(
                    f"translate_south_{label}",
                    "translate",
                    shift_y=pixels,
                ),
                StructuredCondition(
                    f"translate_west_{label}",
                    "translate",
                    shift_x=-pixels,
                ),
                StructuredCondition(
                    f"translate_east_{label}",
                    "translate",
                    shift_x=pixels,
                ),
            ]
        )
    for radius in (4, 8, 16):
        conditions.extend(
            [
                StructuredCondition(
                    f"dilate_unavailable_r{radius}",
                    "dilate",
                    radius=radius,
                ),
                StructuredCondition(
                    f"erode_unavailable_r{radius}",
                    "erode",
                    radius=radius,
                ),
            ]
        )
    return tuple(conditions)


def build_experiment_plan(mode: str) -> ExperimentPlan:
    if mode == "smoke":
        return ExperimentPlan(
            mode=mode,
            pipeline_only=True,
            seeds=(MODEL_SEEDS[0],),
            epochs=2,
            error_rates=SMOKE_ERROR_RATES,
            routes=CORE_ROUTES,
            quality_routes=QUALITY_ROUTES,
            evaluation_splits=("test", "bolivia"),
            sample_limits={
                "train": 24,
                "validation": 12,
                "test": 12,
                "bolivia": 4,
            },
            perturbation_repetitions=1,
            structured_conditions=_smoke_structured_conditions(),
        )
    if mode == "full":
        return ExperimentPlan(
            mode=mode,
            pipeline_only=False,
            seeds=MODEL_SEEDS,
            epochs=25,
            error_rates=FULL_ERROR_RATES,
            routes=CORE_ROUTES,
            quality_routes=QUALITY_ROUTES,
            evaluation_splits=("test", "bolivia"),
            sample_limits={},
            perturbation_repetitions=3,
            structured_conditions=_full_structured_conditions(),
        )
    raise ValueError("mode must be 'smoke' or 'full'")


def _selection_score(record: dict[str, Any], split: str, event: str) -> str:
    token = (f"{SELECTION_SALT}:{split}:{event}:{record['chip_id']}").encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def select_split_records(
    records: Iterable[dict[str, Any]],
    split: str,
    maximum: int = 0,
) -> list[dict[str, Any]]:
    """Select a deterministic event-stratified subset for one split.

    A maximum of zero means all records. Positive limits use round-robin
    sampling across events after a stable within-event hash ranking. Outcome
    values and model predictions never enter the selection rule.
    """

    maximum = int(maximum)
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    candidates = [
        record
        for record in records
        if str(record.get("split")) == split and record.get("scl_assets")
    ]
    if not candidates:
        raise ValueError(f"No records available for split {split!r}")
    if maximum == 0 or maximum >= len(candidates):
        return sorted(candidates, key=lambda record: str(record["chip_id"]))

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        by_event[str(record["event"])].append(record)
    for event, values in by_event.items():
        values.sort(key=lambda record: _selection_score(record, split, event))

    selected: list[dict[str, Any]] = []
    events = sorted(by_event)
    offset = 0
    while len(selected) < maximum:
        made_progress = False
        for event in events:
            values = by_event[event]
            if offset < len(values):
                selected.append(values[offset])
                made_progress = True
                if len(selected) == maximum:
                    break
        if not made_progress:
            break
        offset += 1
    return sorted(
        selected,
        key=lambda record: (str(record["event"]), str(record["chip_id"])),
    )


def build_selected_manifest(
    document: dict[str, Any],
    plan: ExperimentPlan,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Freeze exactly the records that one experiment run may access."""

    if document.get("schema") != "geoai-sen1floods11-scl-manifest-v1":
        raise ValueError("Unsupported Sen1Floods11 source manifest schema")
    records = list(document.get("records", []))
    selected: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "bolivia"):
        selected.extend(
            select_split_records(
                records,
                split,
                maximum=plan.sample_limits.get(split, 0),
            )
        )
    selected.sort(
        key=lambda record: (
            str(record["split"]),
            str(record["event"]),
            str(record["chip_id"]),
        )
    )
    chip_ids = [str(record["chip_id"]) for record in selected]
    if len(chip_ids) != len(set(chip_ids)):
        raise ValueError("Selected manifest contains duplicate chip IDs")

    split_counts: dict[str, int] = defaultdict(int)
    event_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        split_counts[str(record["split"])] += 1
        event_counts[str(record["event"])] += 1
    return {
        "schema": "geoai-sen1floods11-scl-manifest-v1",
        "source_manifest_sha256": source_manifest_sha256,
        "selection_schema": "event-stratified-outcome-independent-v1",
        "experiment_mode": plan.mode,
        "pipeline_only": plan.pipeline_only,
        "summary": {
            "record_count": len(selected),
            "split_counts": dict(sorted(split_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
        },
        "records": selected,
    }


def _rate_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _evaluation_condition(
    condition_id: str,
    quality_mode: str,
    **overrides: object,
) -> dict[str, Any]:
    return {
        "condition_id": condition_id,
        "quality_mode": quality_mode,
        "false_available_rate": 0.0,
        "false_unavailable_rate": 0.0,
        "shift_y": 0,
        "shift_x": 0,
        "radius": 0,
        "matched_source_mode": "translate",
        **overrides,
    }


def evaluation_conditions_for_route(
    plan: ExperimentPlan,
    route: str,
) -> list[dict[str, Any]]:
    """Return the frozen, non-redundant evaluation matrix for one route."""

    if route not in plan.routes:
        raise ValueError(f"Route {route!r} is not in the experiment plan")
    if route == "s1_reference":
        return [_evaluation_condition("reference", "reference")]
    if route not in plan.quality_routes:
        return [
            _evaluation_condition("reference", "reference"),
            _evaluation_condition("complete_absence", "complete-absence"),
        ]

    conditions: list[dict[str, Any]] = []
    for condition in plan.independent_conditions:
        conditions.append(
            _evaluation_condition(
                (
                    "independent_fa"
                    f"{_rate_key(condition.false_available_rate)}_fu"
                    f"{_rate_key(condition.false_unavailable_rate)}"
                ),
                "independent",
                false_available_rate=condition.false_available_rate,
                false_unavailable_rate=condition.false_unavailable_rate,
            )
        )
    for condition in plan.structured_conditions:
        parameters = {
            "shift_y": condition.shift_y,
            "shift_x": condition.shift_x,
            "radius": condition.radius,
        }
        conditions.append(
            _evaluation_condition(
                condition.name,
                condition.mode,
                **parameters,
            )
        )
        conditions.append(
            _evaluation_condition(
                f"matched_random__{condition.name}",
                "matched-random",
                matched_source_mode=condition.mode,
                **parameters,
            )
        )
    conditions.append(_evaluation_condition("complete_absence", "complete-absence"))
    identifiers = [condition["condition_id"] for condition in conditions]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Evaluation condition identifiers must be unique")
    return conditions
