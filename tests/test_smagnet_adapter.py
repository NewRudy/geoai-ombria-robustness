from __future__ import annotations

import unittest

import torch
import numpy as np

from geoai_ombria_robustness.smagnet_adapter import (
    dual_path_masked_bce,
    normalize_sen1floods11_for_official_smagnet,
    split_smagnet_input,
    verify_complete_absence_equivalence,
)


class _FallbackContractModel(torch.nn.Module):
    def forward(self, radar, optical, invalid):
        gate = 1.0 - invalid
        sar_logits = radar[:, :1]
        fused_logits = sar_logits + gate * optical[:, :1]
        return fused_logits, sar_logits, [gate]


class SMAGNetAdapterTests(unittest.TestCase):
    def test_sen1floods11_optical_channels_are_reordered_to_rgb_nir(self) -> None:
        image = np.stack(
            [np.full((2, 2), value, dtype=np.float32) for value in range(7)]
        )
        normalization = {
            "optical_mean": [0.0, 0.0, 0.0, 0.0],
            "optical_std": [1.0, 1.0, 1.0, 1.0],
            "radar_mean": [0.0, 0.0],
            "radar_std": [1.0, 1.0],
        }

        result = normalize_sen1floods11_for_official_smagnet(
            image, normalization
        )

        self.assertEqual(result[:, 0, 0].tolist(), [2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0])

    def test_input_contract_maps_availability_to_official_invalid_mask(self) -> None:
        optical = torch.full((2, 4, 32, 32), 2.0)
        radar = torch.full((2, 2, 32, 32), 3.0)
        available = torch.zeros((2, 1, 32, 32))
        image = torch.cat([optical, radar, available], dim=1)

        actual_radar, actual_optical, invalid = split_smagnet_input(image)

        self.assertTrue(torch.equal(actual_radar, radar))
        self.assertTrue(torch.equal(actual_optical, optical))
        self.assertTrue(torch.equal(invalid, torch.ones_like(available)))

    def test_dual_path_loss_uses_equal_weights_and_valid_mask(self) -> None:
        fused = torch.tensor([[[[0.0, 4.0]]]])
        sar = torch.tensor([[[[0.0, -4.0]]]])
        target = torch.tensor([[[[1.0, 1.0]]]])
        valid = torch.tensor([[[[True, False]]]])

        loss = dual_path_masked_bce(fused, sar, target, valid)

        self.assertAlmostEqual(float(loss), 0.69314718056, places=6)

    def test_complete_absence_contract_requires_fused_sar_equivalence(self) -> None:
        result = verify_complete_absence_equivalence(
            _FallbackContractModel(), device=torch.device("cpu"), size=32
        )

        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["maximum_fused_sar_logit_difference"], 0.0)
        self.assertEqual(result["maximum_masked_gate"], 0.0)


if __name__ == "__main__":
    unittest.main()
