from __future__ import annotations

import unittest

from geoai_ombria_robustness.models import count_trainable_parameters
from geoai_ombria_robustness.single_time_models import build_single_time_model


class SingleTimeModelTests(unittest.TestCase):
    def test_s1_only_model_accepts_two_radar_channels(self) -> None:
        torch = self.torch
        model = build_single_time_model(8, "s1_only_unet")
        inputs = torch.rand(1, 2, 64, 64)
        self.assertEqual(tuple(model(inputs).shape), (1, 1, 64, 64))

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch
        except ImportError as exc:
            raise unittest.SkipTest(f"PyTorch is unavailable: {exc}") from exc
        cls.torch = torch

    def test_single_time_gate_shapes(self) -> None:
        torch = self.torch
        model = build_single_time_model(
            base_channels=12,
            architecture="hard_quality_gate",
        ).eval()
        inputs = torch.rand(2, 7, 64, 64)
        logits, gates = model(inputs, return_gate_maps=True)
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))
        self.assertEqual(
            [tuple(gate.shape) for gate in gates],
            [(2, 1, 64, 64), (2, 1, 32, 32), (2, 1, 16, 16)],
        )

    def test_hard_gate_treats_quality_as_oracle(self) -> None:
        torch = self.torch
        torch.manual_seed(31)
        model = build_single_time_model(
            base_channels=12,
            architecture="hard_quality_gate",
        ).eval()
        first = torch.rand(1, 7, 64, 64)
        first[:, 6:7] = 0.0
        second = first.clone()
        second[:, :4] = torch.rand_like(second[:, :4]) * 100.0
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        torch.testing.assert_close(first_logits, second_logits, rtol=0, atol=0)

    def test_soft_prior_uses_content_when_quality_is_false_unavailable(self) -> None:
        torch = self.torch
        torch.manual_seed(37)
        model = build_single_time_model(
            base_channels=12,
            architecture="soft_quality_prior",
        ).eval()
        first = torch.rand(1, 7, 64, 64)
        first[:, 6:7] = 0.0
        second = first.clone()
        second[:, :4] = torch.rand_like(second[:, :4])
        with torch.no_grad():
            first_logits = model(first)
            second_logits = model(second)
        self.assertFalse(torch.equal(first_logits, second_logits))

    def test_soft_prior_complete_absence_gate_is_exactly_zero(self) -> None:
        torch = self.torch
        model = build_single_time_model(
            base_channels=12,
            architecture="soft_quality_prior",
        ).eval()
        inputs = torch.rand(1, 7, 64, 64)
        inputs[:, :4] = 0.0
        inputs[:, 6:7] = 0.0
        with torch.no_grad():
            _, gates = model(inputs, return_gate_maps=True)
        for gate in gates:
            self.assertTrue(torch.all(gate == 0.0))

    def test_default_gate_capacity_is_reported(self) -> None:
        baseline = build_single_time_model(12, "quality_concat_unet")
        gated = build_single_time_model(12, "hard_quality_gate")
        baseline_parameters = count_trainable_parameters(baseline)
        gated_parameters = count_trainable_parameters(gated)
        relative_gap = abs(gated_parameters - baseline_parameters) / baseline_parameters
        self.assertLess(relative_gap, 0.15)


if __name__ == "__main__":
    unittest.main()
