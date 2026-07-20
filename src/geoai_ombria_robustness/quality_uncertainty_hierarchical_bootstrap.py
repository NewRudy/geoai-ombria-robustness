from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

from .quality_uncertainty_full_audit import FULL_SEEDS


CORNER_RATES = ((0.0, 0.0), (0.0, 0.4), (0.4, 0.0), (0.4, 0.4))
EXTERNAL_HARD_ROUTES = (
    "hard_quality_gate",
    "hard_quality_gate_error_aware",
)
OMBRIA_HARD_ROUTES = ("hard_oracle", "hard_error_aware")
COUNT_FIELDS = ("tp", "fp", "fn")


@dataclass(frozen=True)
class StateKey:
    dataset: str
    split: str
    route: str
    condition: str
    model_seed: int


@dataclass(frozen=True)
class UnitTable:
    keys: tuple[tuple[str, str], ...]
    repetitions: tuple[int, ...]
    counts: np.ndarray

    def event_indices(self) -> dict[str, np.ndarray]:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, (event, _chip_id) in enumerate(self.keys):
            grouped[event].append(index)
        return {
            event: np.asarray(indices, dtype=np.int64)
            for event, indices in sorted(grouped.items())
        }


@dataclass(frozen=True)
class Estimand:
    estimand_id: str
    family: str
    dataset: str
    split: str
    route: str
    condition_a: str
    condition_b: str
    state_a_route: str
    state_a_condition: str
    state_b_route: str
    state_b_condition: str
    false_available_rate: float | None = None
    false_unavailable_rate: float | None = None


def _rate_key(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def independent_condition_id(false_available: float, false_unavailable: float) -> str:
    return (
        f"independent_fa{_rate_key(false_available)}_"
        f"fu{_rate_key(false_unavailable)}"
    )


def ombria_condition_id(false_available: float, false_unavailable: float) -> str:
    return (
        f"fa{_rate_key(false_available)}_fu{_rate_key(false_unavailable)}"
    )


def _read_csv(archive: ZipFile, name: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8"))))


def _aggregate_unit_table(
    rows: Sequence[Mapping[str, str]],
    *,
    event_name: str | None = None,
    expected_repetitions: frozenset[int] = frozenset({0, 1, 2}),
) -> UnitTable:
    totals: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    repetitions: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in rows:
        event = event_name if event_name is not None else str(row["event"])
        key = (event, str(row["chip_id"]))
        counts = np.asarray([int(row[field]) for field in COUNT_FIELDS], dtype=np.int64)
        if np.any(counts < 0):
            raise ValueError("confusion counts must be non-negative")
        repetition = int(row["repetition"])
        if repetition in totals[key]:
            raise ValueError("duplicate chip and repetition row")
        totals[key][repetition] = counts
        repetitions[key].add(repetition)
    if not totals:
        raise ValueError("cannot aggregate an empty per-chip table")
    if any(reps != expected_repetitions for reps in repetitions.values()):
        raise ValueError(
            "selected chip repetitions do not match the frozen table contract"
        )
    keys = tuple(sorted(totals))
    repetition_ids = tuple(sorted(expected_repetitions))
    return UnitTable(
        keys=keys,
        repetitions=repetition_ids,
        counts=np.stack(
            [
                np.stack([totals[key][repetition] for repetition in repetition_ids])
                for key in keys
            ]
        ),
    )


def _store_state(
    states: dict[StateKey, UnitTable], key: StateKey, table: UnitTable
) -> None:
    if key in states:
        raise ValueError(f"duplicate per-chip state: {key}")
    states[key] = table


def _load_external_seed(
    archive: ZipFile,
    *,
    root: str,
    seed: int,
    states: dict[StateKey, UnitTable],
) -> set[str]:
    structured_conditions: set[str] = set()
    corner_conditions = {
        independent_condition_id(false_available, false_unavailable)
        for false_available, false_unavailable in CORNER_RATES
    }
    for split, expected_units in (("test", 90), ("bolivia", 15)):
        reference_name = (
            f"{root}/sen1floods11/evaluations/s1_reference/seed{seed}/"
            f"{split}/per_chip_metrics.csv"
        )
        reference = _aggregate_unit_table(_read_csv(archive, reference_name))
        if len(reference.keys) != expected_units:
            raise ValueError(f"seed {seed} {split}: unexpected S1 chip count")
        _store_state(
            states,
            StateKey("sen1floods11", split, "s1_reference", "reference", seed),
            reference,
        )

        for route in EXTERNAL_HARD_ROUTES:
            name = (
                f"{root}/sen1floods11/evaluations/{route}/seed{seed}/"
                f"{split}/per_chip_metrics.csv"
            )
            rows = _read_csv(archive, name)
            selected: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                condition = str(row["condition_id"])
                is_structured_pair = split == "test" and (
                    row["quality_mode"] in {"translate", "dilate", "erode"}
                    or condition.startswith("matched_random__")
                )
                if condition in corner_conditions or is_structured_pair:
                    selected[condition].append(row)
                if split == "test" and row["quality_mode"] in {
                    "translate",
                    "dilate",
                    "erode",
                }:
                    structured_conditions.add(condition)
            expected_conditions = 32 if split == "test" else 4
            if len(selected) != expected_conditions:
                raise ValueError(
                    f"seed {seed} {split} {route}: expected "
                    f"{expected_conditions} selected conditions, received "
                    f"{len(selected)}"
                )
            for condition, condition_rows in sorted(selected.items()):
                table = _aggregate_unit_table(condition_rows)
                if len(table.keys) != expected_units:
                    raise ValueError(
                        f"seed {seed} {split} {route} {condition}: "
                        "unexpected chip count"
                    )
                _store_state(
                    states,
                    StateKey("sen1floods11", split, route, condition, seed),
                    table,
                )
    return structured_conditions


def _load_ombria_seed(
    archive: ZipFile,
    *,
    root: str,
    seed: int,
    states: dict[StateKey, UnitTable],
) -> None:
    reference_name = (
        f"{root}/ombria/seed{seed}/evaluations/s1_reference/fa0_fu0/"
        "per_chip_metrics.csv"
    )
    reference = _aggregate_unit_table(
        _read_csv(archive, reference_name),
        event_name="OMBRIA",
        expected_repetitions=frozenset({0}),
    )
    if len(reference.keys) != 70:
        raise ValueError(f"seed {seed}: unexpected OMBRIA reference chip count")
    _store_state(
        states,
        StateKey("ombria", "test", "s1_reference", "reference", seed),
        reference,
    )
    for route in OMBRIA_HARD_ROUTES:
        for false_available, false_unavailable in CORNER_RATES:
            condition = ombria_condition_id(false_available, false_unavailable)
            name = (
                f"{root}/ombria/seed{seed}/evaluations/{route}/{condition}/"
                "per_chip_metrics.csv"
            )
            table = _aggregate_unit_table(
                _read_csv(archive, name), event_name="OMBRIA"
            )
            if len(table.keys) != 70:
                raise ValueError(
                    f"seed {seed} {route} {condition}: unexpected OMBRIA chip count"
                )
            _store_state(
                states,
                StateKey("ombria", "test", route, condition, seed),
                table,
            )


def load_selected_unit_tables(
    artifacts: Sequence[tuple[int, Path]],
) -> tuple[dict[StateKey, UnitTable], tuple[str, ...]]:
    seeds = tuple(sorted(int(seed) for seed, _path in artifacts))
    if seeds != FULL_SEEDS:
        raise ValueError(f"expected frozen seeds {FULL_SEEDS}, received {seeds}")
    states: dict[StateKey, UnitTable] = {}
    structured_by_seed: dict[int, set[str]] = {}
    for seed, path in sorted((int(seed), Path(path)) for seed, path in artifacts):
        root = f"quality_uncertainty_full_seed{seed}"
        with ZipFile(path) as archive:
            structured_by_seed[seed] = _load_external_seed(
                archive, root=root, seed=seed, states=states
            )
            _load_ombria_seed(
                archive, root=root, seed=seed, states=states
            )
    structured_sets = {tuple(sorted(value)) for value in structured_by_seed.values()}
    if len(structured_sets) != 1:
        raise ValueError("structured-condition identities differ across seed shards")
    structured = next(iter(structured_sets))
    if len(structured) != 14:
        raise ValueError("expected 14 frozen structured conditions")
    return states, structured


def build_selected_estimands(structured_conditions: Sequence[str]) -> list[Estimand]:
    estimands: list[Estimand] = []

    dataset_routes = (
        (
            "ombria",
            "test",
            "hard_oracle",
            "hard_error_aware",
            ombria_condition_id,
        ),
        (
            "sen1floods11",
            "test",
            "hard_quality_gate",
            "hard_quality_gate_error_aware",
            independent_condition_id,
        ),
        (
            "sen1floods11",
            "bolivia",
            "hard_quality_gate",
            "hard_quality_gate_error_aware",
            independent_condition_id,
        ),
    )
    for dataset, split, baseline, intervention, condition_fn in dataset_routes:
        fa_condition = condition_fn(0.4, 0.0)
        fu_condition = condition_fn(0.0, 0.4)
        estimands.append(
            Estimand(
                estimand_id=f"h1__{dataset}__{split}__{baseline}",
                family="H1_endpoint_asymmetry",
                dataset=dataset,
                split=split,
                route=baseline,
                condition_a=fu_condition,
                condition_b=fa_condition,
                state_a_route=baseline,
                state_a_condition=fu_condition,
                state_b_route=baseline,
                state_b_condition=fa_condition,
            )
        )
        for route in (baseline, intervention):
            for false_available, false_unavailable in CORNER_RATES:
                condition = condition_fn(false_available, false_unavailable)
                estimands.append(
                    Estimand(
                        estimand_id=(
                            f"h3__{dataset}__{split}__{route}__{condition}"
                        ),
                        family="H3_fusion_minus_s1",
                        dataset=dataset,
                        split=split,
                        route=route,
                        condition_a=condition,
                        condition_b="reference",
                        state_a_route=route,
                        state_a_condition=condition,
                        state_b_route="s1_reference",
                        state_b_condition="reference",
                        false_available_rate=false_available,
                        false_unavailable_rate=false_unavailable,
                    )
                )
        for false_available, false_unavailable in ((0.0, 0.0), (0.4, 0.4)):
            condition = condition_fn(false_available, false_unavailable)
            estimands.append(
                Estimand(
                    estimand_id=(
                        f"h4__{dataset}__{split}__{intervention}_minus_"
                        f"{baseline}__{condition}"
                    ),
                    family="H4_error_aware_minus_baseline",
                    dataset=dataset,
                    split=split,
                    route=f"{intervention}_minus_{baseline}",
                    condition_a=condition,
                    condition_b=condition,
                    state_a_route=intervention,
                    state_a_condition=condition,
                    state_b_route=baseline,
                    state_b_condition=condition,
                    false_available_rate=false_available,
                    false_unavailable_rate=false_unavailable,
                )
            )

    for route in EXTERNAL_HARD_ROUTES:
        for condition in sorted(structured_conditions):
            matched = f"matched_random__{condition}"
            estimands.append(
                Estimand(
                    estimand_id=f"h2__sen1floods11__test__{route}__{condition}",
                    family="H2_structured_minus_matched_random",
                    dataset="sen1floods11",
                    split="test",
                    route=route,
                    condition_a=condition,
                    condition_b=matched,
                    state_a_route=route,
                    state_a_condition=condition,
                    state_b_route=route,
                    state_b_condition=matched,
                )
            )
    if len(estimands) != 61 or len({row.estimand_id for row in estimands}) != 61:
        raise ValueError("selected estimand registry is incomplete or duplicated")
    return estimands


def _iou(counts: np.ndarray) -> float:
    denominator = int(counts[0] + counts[1] + counts[2])
    if denominator <= 0:
        raise ValueError("IoU denominator must be positive")
    return float(counts[0] / denominator)


def _mean_repetition_iou(counts: np.ndarray) -> float:
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError("expected repetition by confusion-count matrix")
    return float(np.mean([_iou(repetition) for repetition in counts]))


def _paired_tables(
    states: Mapping[StateKey, UnitTable], estimand: Estimand, seed: int
) -> tuple[UnitTable, UnitTable]:
    table_a = states[
        StateKey(
            estimand.dataset,
            estimand.split,
            estimand.state_a_route,
            estimand.state_a_condition,
            seed,
        )
    ]
    table_b = states[
        StateKey(
            estimand.dataset,
            estimand.split,
            estimand.state_b_route,
            estimand.state_b_condition,
            seed,
        )
    ]
    if table_a.keys != table_b.keys:
        raise ValueError(f"paired unit identities differ for {estimand.estimand_id}")
    return table_a, table_b


def hierarchical_contrast_summary(
    states: Mapping[StateKey, UnitTable],
    estimand: Estimand,
    *,
    replicates: int,
    random_seed: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("hierarchical bootstrap requires at least 100 replicates")
    seed_values: dict[int, float] = {}
    event_groups: dict[int, dict[str, np.ndarray]] = {}
    for seed in FULL_SEEDS:
        table_a, table_b = _paired_tables(states, estimand, seed)
        seed_values[seed] = _mean_repetition_iou(
            table_a.counts.sum(axis=0)
        ) - _mean_repetition_iou(table_b.counts.sum(axis=0))
        event_groups[seed] = table_a.event_indices()

    digest = hashlib.sha256(
        f"{random_seed}:{estimand.estimand_id}".encode("utf-8")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled_seeds = rng.choice(np.asarray(FULL_SEEDS), size=len(FULL_SEEDS))
        seed_contrasts = np.empty(len(FULL_SEEDS), dtype=np.float64)
        for position, sampled_seed in enumerate(sampled_seeds):
            seed = int(sampled_seed)
            table_a, table_b = _paired_tables(states, estimand, seed)
            groups = event_groups[seed]
            events = tuple(groups)
            sampled_event_positions = rng.integers(0, len(events), size=len(events))
            total_a = np.zeros((len(table_a.repetitions), 3), dtype=np.int64)
            total_b = np.zeros((len(table_b.repetitions), 3), dtype=np.int64)
            for event_position in sampled_event_positions:
                indices = groups[events[int(event_position)]]
                sampled_indices = rng.choice(indices, size=len(indices))
                total_a += table_a.counts[sampled_indices].sum(axis=0)
                total_b += table_b.counts[sampled_indices].sum(axis=0)
            seed_contrasts[position] = _mean_repetition_iou(
                total_a
            ) - _mean_repetition_iou(total_b)
        draws[replicate] = float(seed_contrasts.mean())

    observed = float(np.mean(list(seed_values.values())))
    lower, upper = np.percentile(draws, [2.5, 97.5], method="linear")
    return {
        "estimand_id": estimand.estimand_id,
        "family": estimand.family,
        "dataset": estimand.dataset,
        "split": estimand.split,
        "route": estimand.route,
        "condition_a": estimand.condition_a,
        "condition_b": estimand.condition_b,
        "contrast_definition": "pooled_iou_a_minus_pooled_iou_b",
        "false_available_rate": estimand.false_available_rate,
        "false_unavailable_rate": estimand.false_unavailable_rate,
        "model_seeds": len(FULL_SEEDS),
        "events": len(next(iter(event_groups.values()))),
        "units_per_seed": len(_paired_tables(states, estimand, FULL_SEEDS[0])[0].keys),
        "bootstrap_replicates": replicates,
        "observed_mean": observed,
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_bias": float(draws.mean() - observed),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "positive_draw_fraction": float(np.mean(draws > 0)),
        "negative_draw_fraction": float(np.mean(draws < 0)),
        "seed_values_json": json.dumps(
            {str(seed): seed_values[seed] for seed in FULL_SEEDS},
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_selected_hierarchical_bootstrap(
    artifacts: Sequence[tuple[int, Path]],
    *,
    replicates: int = 5000,
    random_seed: int = 20260720,
) -> dict[str, Any]:
    states, structured = load_selected_unit_tables(artifacts)
    estimands = build_selected_estimands(structured)
    rows = [
        hierarchical_contrast_summary(
            states,
            estimand,
            replicates=replicates,
            random_seed=random_seed,
        )
        for estimand in estimands
    ]
    h1_rows = [row for row in rows if row["family"] == "H1_endpoint_asymmetry"]
    if len(h1_rows) != 3:
        raise ValueError("hierarchical H1 registry is incomplete")
    return {
        "report": {
            "schema": "geoai-quality-map-uncertainty-hierarchical-bootstrap-v1",
            "frozen_seeds": list(FULL_SEEDS),
            "artifact_sha256": {
                str(seed): _sha256(Path(path)) for seed, path in sorted(artifacts)
            },
            "random_seed": random_seed,
            "bootstrap_replicates": replicates,
            "estimands": len(rows),
            "selection": {
                "H1_endpoint_asymmetry": 3,
                "H2_structured_minus_matched_random": 28,
                "H3_fusion_minus_s1": 24,
                "H4_error_aware_minus_baseline": 6,
                "selection_rule": (
                    "Frozen hard-gate hypotheses; H3 uses the four prespecified "
                    "corner cells, H4 uses clean and joint-high cells, and H2 "
                    "uses all 14 frozen structures for the two hard-gate routes."
                ),
            },
            "hierarchy": (
                "Resample model seeds; within each sampled seed resample events; "
                "within each sampled event resample paired chips. OMBRIA and "
                "Bolivia contain one event group, so their inner level is chip."
            ),
            "primary_h1": {
                row["estimand_id"]: {
                    key: row[key]
                    for key in ("observed_mean", "ci95_lower", "ci95_upper")
                }
                for row in h1_rows
            },
            "boundary": (
                "Percentile intervals describe the frozen datasets and selected "
                "hierarchy. Five model seeds, one OMBRIA event group, and one "
                "Bolivia event group limit population-level interpretation."
            ),
        },
        "rows": rows,
    }
