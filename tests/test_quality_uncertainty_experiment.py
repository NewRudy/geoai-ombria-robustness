from __future__ import annotations

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_experiment import (
    CORE_ROUTES,
    QUALITY_ROUTES,
    build_selected_manifest,
    build_experiment_plan,
    evaluation_conditions_for_route,
    resolve_active_seeds,
    select_split_records,
)


def record(chip_id: str, event: str, split: str) -> dict[str, object]:
    return {
        "chip_id": chip_id,
        "event": event,
        "split": split,
        "scl_assets": [{"provider": "earth-search"}],
    }


class QualityUncertaintyExperimentTests(unittest.TestCase):
    def test_smoke_plan_exercises_every_full_route_and_error_axis(self) -> None:
        plan = build_experiment_plan("smoke")
        self.assertTrue(plan.pipeline_only)
        self.assertEqual(plan.seeds, (7,))
        self.assertEqual(plan.epochs, 2)
        self.assertEqual(plan.error_rates, (0.0, 0.2, 0.4))
        self.assertEqual(plan.routes, CORE_ROUTES)
        self.assertEqual(plan.quality_routes, QUALITY_ROUTES)
        self.assertEqual(len(plan.independent_conditions), 9)
        self.assertEqual(
            set(plan.evaluation_splits),
            {"test", "bolivia"},
        )
        self.assertGreater(len(plan.structured_conditions), 0)

    def test_full_plan_freezes_five_seeds_and_dense_error_grid(self) -> None:
        plan = build_experiment_plan("full")
        self.assertFalse(plan.pipeline_only)
        self.assertEqual(plan.seeds, (7, 13, 21, 29, 37))
        self.assertEqual(plan.epochs, 25)
        self.assertEqual(plan.error_rates, (0.0, 0.05, 0.1, 0.2, 0.4))
        self.assertEqual(len(plan.independent_conditions), 25)
        self.assertEqual(plan.sample_limits, {})
        self.assertEqual(plan.perturbation_repetitions, 3)

    def test_full_seed_shard_is_a_frozen_subset_not_a_new_protocol(self) -> None:
        plan = build_experiment_plan("full")

        self.assertEqual(resolve_active_seeds(plan, [21, 7]), (7, 21))
        self.assertEqual(resolve_active_seeds(plan, None), plan.seeds)
        with self.assertRaisesRegex(ValueError, "frozen Full plan"):
            resolve_active_seeds(plan, [999])
        with self.assertRaisesRegex(ValueError, "at least one"):
            resolve_active_seeds(plan, [])

    def test_split_selection_is_order_independent_and_event_stratified(self) -> None:
        records = [
            record("A1", "A", "train"),
            record("A2", "A", "train"),
            record("B1", "B", "train"),
            record("B2", "B", "train"),
            record("C1", "C", "validation"),
        ]
        first = select_split_records(records, "train", maximum=2)
        second = select_split_records(list(reversed(records)), "train", maximum=2)
        self.assertEqual(
            [value["chip_id"] for value in first],
            [value["chip_id"] for value in second],
        )
        self.assertEqual({value["event"] for value in first}, {"A", "B"})

    def test_split_selection_rejects_impossible_or_invalid_requests(self) -> None:
        records = [record("A1", "A", "train")]
        with self.assertRaisesRegex(ValueError, "maximum"):
            select_split_records(records, "train", maximum=-1)
        with self.assertRaisesRegex(ValueError, "No records"):
            select_split_records(records, "test", maximum=1)

    def test_selected_manifest_records_exact_frozen_split_counts(self) -> None:
        source_records = []
        for split, count in (
            ("train", 30),
            ("validation", 15),
            ("test", 15),
            ("bolivia", 6),
        ):
            for index in range(count):
                source_records.append(
                    record(
                        f"{split}-{index}",
                        f"event-{index % 3}",
                        split,
                    )
                )
        selected = build_selected_manifest(
            {
                "schema": "geoai-sen1floods11-scl-manifest-v1",
                "records": source_records,
            },
            build_experiment_plan("smoke"),
            source_manifest_sha256="abc123",
        )
        self.assertEqual(selected["schema"], "geoai-sen1floods11-scl-manifest-v1")
        self.assertEqual(selected["source_manifest_sha256"], "abc123")
        self.assertEqual(
            selected["summary"]["split_counts"],
            {"train": 24, "validation": 12, "test": 12, "bolivia": 4},
        )
        chip_ids = [value["chip_id"] for value in selected["records"]]
        self.assertEqual(len(chip_ids), len(set(chip_ids)))

    def test_evaluation_conditions_avoid_redundant_quality_grids(self) -> None:
        plan = build_experiment_plan("smoke")
        s1 = evaluation_conditions_for_route(plan, "s1_reference")
        early = evaluation_conditions_for_route(plan, "early_fusion")
        quality = evaluation_conditions_for_route(plan, "quality_concat")

        self.assertEqual([value["condition_id"] for value in s1], ["reference"])
        self.assertEqual(
            {value["condition_id"] for value in early},
            {"reference", "complete_absence"},
        )
        self.assertEqual(len(quality), 16)
        self.assertEqual(len({value["condition_id"] for value in quality}), 16)
        self.assertEqual(
            sum(value["quality_mode"] == "independent" for value in quality),
            9,
        )
        self.assertEqual(
            sum(value["quality_mode"] == "matched-random" for value in quality),
            3,
        )


if __name__ == "__main__":
    unittest.main()
