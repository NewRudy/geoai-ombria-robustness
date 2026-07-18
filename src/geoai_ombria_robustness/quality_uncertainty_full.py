from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .quality_uncertainty_experiment import (
    build_experiment_plan,
    evaluation_conditions_for_route,
)


OMBRIA_ROUTES = (
    "hard_oracle",
    "hard_error_aware",
    "concat_error_aware",
    "soft_error_aware",
    "s1_reference",
)


@dataclass(frozen=True)
class FullShardPlan:
    planned_seeds: tuple[int, ...]
    active_seed: int
    epochs: int
    error_rates: tuple[float, ...]
    perturbation_repetitions: int
    external_routes: tuple[str, ...]
    ombria_routes: tuple[str, ...]
    external_seed_condition_rows: int
    external_raw_summary_rows: int
    ombria_evaluation_cells: int
    ombria_raw_summary_rows: int
    scientific_interpretation_allowed: bool = False

    @property
    def shard_id(self) -> str:
        return f"seed-{self.active_seed}"

    @property
    def artifact_name(self) -> str:
        return f"quality_map_uncertainty_full_seed{self.active_seed}_artifacts.zip"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "geoai-quality-map-uncertainty-full-shard-plan-v1",
            **asdict(self),
            "shard_id": self.shard_id,
            "artifact_name": self.artifact_name,
            "claim_boundary": (
                "One Full seed shard is incomplete scientific evidence. All "
                "five frozen seeds, the published-architecture baseline gate, "
                "and a post-run audit are required before manuscript use."
            ),
        }


def build_full_shard_plan(seed: int) -> FullShardPlan:
    plan = build_experiment_plan("full")
    seed = int(seed)
    if seed not in plan.seeds:
        raise ValueError(f"Seed {seed} is outside the frozen Full plan")
    external_conditions = sum(
        len(evaluation_conditions_for_route(plan, route))
        for route in plan.routes
    )
    external_seed_condition_rows = external_conditions * len(
        plan.evaluation_splits
    )
    ombria_evaluation_cells = 1 + 4 * len(plan.independent_conditions)
    return FullShardPlan(
        planned_seeds=plan.seeds,
        active_seed=seed,
        epochs=plan.epochs,
        error_rates=plan.error_rates,
        perturbation_repetitions=plan.perturbation_repetitions,
        external_routes=plan.routes,
        ombria_routes=OMBRIA_ROUTES,
        external_seed_condition_rows=external_seed_condition_rows,
        external_raw_summary_rows=(
            external_seed_condition_rows * plan.perturbation_repetitions
        ),
        ombria_evaluation_cells=ombria_evaluation_cells,
        ombria_raw_summary_rows=(
            1 + (ombria_evaluation_cells - 1) * plan.perturbation_repetitions
        ),
    )


def _rates_arguments(rates: Iterable[float]) -> tuple[str, ...]:
    return tuple(f"{float(rate):g}" for rate in rates)


def ombria_training_route_args(
    route: str,
    rates: Iterable[float],
) -> tuple[str, ...]:
    """Return the prespecified OMBRIA route contrast behind one small interface."""

    common = (
        "--variant",
        "multimodal",
        "--s2-quality",
        "binary",
        "--train-degrade-s2",
        "quality_matched_light",
    )
    if route == "hard_oracle":
        return (*common, "--architecture", "quality_gated_fusion")
    if route == "hard_error_aware":
        return (
            *common,
            "--architecture",
            "quality_gated_fusion",
            "--train-quality-error-rates",
            *_rates_arguments(rates),
        )
    if route == "concat_error_aware":
        return (
            *common,
            "--architecture",
            "early_fusion_unet",
            "--train-quality-error-rates",
            *_rates_arguments(rates),
        )
    if route == "soft_error_aware":
        return (
            *common,
            "--architecture",
            "soft_quality_prior_fusion",
            "--train-quality-error-rates",
            *_rates_arguments(rates),
        )
    if route == "s1_reference":
        return ("--variant", "s1_bitemporal")
    raise ValueError(f"Unknown OMBRIA route: {route}")
