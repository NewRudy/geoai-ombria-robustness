from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoai_ombria_robustness.models import (  # noqa: E402
    build_model,
    count_trainable_parameters,
)


class QualityGatedFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest(f"PyTorch is unavailable: {exc}") from exc
        cls.torch = torch

    def test_output_and_multiscale_gate_shapes(self) -> None:
        torch = self.torch
        model = build_model(10, 16, "quality_gated_fusion").eval()
        inputs = torch.rand(2, 10, 64, 64)
        logits, gates = model(inputs, return_gate_maps=True)
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))
        self.assertEqual(
            [tuple(gate.shape) for gate in gates["before"]],
            [(2, 1, 64, 64), (2, 1, 32, 32), (2, 1, 16, 16)],
        )

    def test_complete_s2_absence_makes_output_invariant_to_s2_values(self) -> None:
        torch = self.torch
        torch.manual_seed(17)
        model = build_model(10, 16, "quality_gated_fusion").eval()
        first = torch.rand(2, 10, 64, 64)
        second = first.clone()
        first[:, 8:10] = 0.0
        second[:, 8:10] = 0.0
        second[:, 0:6] = torch.rand_like(second[:, 0:6]) * 100.0
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        torch.testing.assert_close(first_logits, second_logits, rtol=0.0, atol=0.0)

    def test_gate_is_zero_where_quality_is_unavailable(self) -> None:
        torch = self.torch
        model = build_model(10, 16, "quality_gated_fusion").eval()
        inputs = torch.rand(1, 10, 64, 64)
        inputs[:, 8:10] = 1.0
        inputs[:, 9:10, 8:40, 12:44] = 0.0
        _, gates = model(inputs, return_gate_maps=True)
        self.assertTrue(torch.all(gates["after"][0][:, :, 8:40, 12:44] == 0.0))
        for gate in (*gates["before"], *gates["after"]):
            self.assertGreaterEqual(float(gate.detach().min()), 0.0)
            self.assertLessEqual(float(gate.detach().max()), 1.0)

    def test_available_gate_has_trust_preserving_initialization(self) -> None:
        torch = self.torch
        model = build_model(10, 16, "quality_gated_fusion").eval()
        inputs = torch.rand(1, 10, 64, 64)
        inputs[:, 8:10] = 1.0
        with torch.no_grad():
            _, gates = model(inputs, return_gate_maps=True)
        expected = torch.sigmoid(torch.tensor(2.0))
        for gate in (*gates["before"], *gates["after"]):
            torch.testing.assert_close(gate, torch.full_like(gate, expected))

    def test_soft_quality_prior_can_recover_from_false_unavailable_map(self) -> None:
        torch = self.torch
        torch.manual_seed(29)
        model = build_model(10, 8, "soft_quality_prior_fusion").eval()
        first = torch.rand(1, 10, 64, 64)
        first[:, 8:10] = 0.0
        second = first.clone()
        second[:, 0:6] = torch.rand_like(second[:, 0:6])
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        self.assertFalse(torch.equal(first_logits, second_logits))

    def test_soft_quality_prior_preserves_complete_absence_boundary(self) -> None:
        torch = self.torch
        model = build_model(10, 8, "soft_quality_prior_fusion").eval()
        inputs = torch.rand(1, 10, 64, 64)
        inputs[:, 0:6] = 0.0
        inputs[:, 8:10] = 0.0
        with torch.no_grad():
            _, gates = model(inputs, return_gate_maps=True)
        for gate in (*gates["before"], *gates["after"]):
            self.assertTrue(torch.all(gate == 0.0))

    def test_default_width_is_capacity_controlled(self) -> None:
        baseline = build_model(10, 16, "early_fusion_unet")
        proposed = build_model(10, 16, "quality_gated_fusion")
        baseline_parameters = count_trainable_parameters(baseline)
        proposed_parameters = count_trainable_parameters(proposed)
        relative_gap = abs(proposed_parameters - baseline_parameters) / baseline_parameters
        self.assertLess(relative_gap, 0.05)

    def test_quality_gated_model_rejects_noncanonical_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "10-channel"):
            build_model(8, 16, "quality_gated_fusion")


if __name__ == "__main__":
    unittest.main()
