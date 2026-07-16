from __future__ import annotations

from .models import _build_early_fusion_unet


SINGLE_TIME_MODEL_ARCHITECTURES = (
    "early_fusion_unet",
    "quality_concat_unet",
    "hard_quality_gate",
    "soft_quality_prior",
)


def build_single_time_model(
    base_channels: int,
    architecture: str,
    optical_channels: int = 4,
    radar_channels: int = 2,
    quality_branch_channels: int | None = None,
):
    """Build a single-time S1/S2 model for Sen1Floods11.

    Quality-aware layouts are fixed as optical, radar, and one quality channel.
    """

    if architecture not in SINGLE_TIME_MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; "
            f"choose from {SINGLE_TIME_MODEL_ARCHITECTURES}"
        )
    if optical_channels < 1 or radar_channels < 1:
        raise ValueError("optical_channels and radar_channels must be positive")
    if architecture == "early_fusion_unet":
        return _build_early_fusion_unet(
            optical_channels + radar_channels,
            base_channels,
        )
    if architecture == "quality_concat_unet":
        return _build_early_fusion_unet(
            optical_channels + radar_channels + 1,
            base_channels,
        )
    branch_channels = (
        max(4, int(round(base_channels / 3)))
        if quality_branch_channels is None
        else int(quality_branch_channels)
    )
    if branch_channels < 2:
        raise ValueError("quality_branch_channels must be at least 2")
    return _build_single_time_quality_fusion(
        base_channels=base_channels,
        branch_channels=branch_channels,
        optical_channels=optical_channels,
        radar_channels=radar_channels,
        gate_mode=(
            "hard" if architecture == "hard_quality_gate" else "soft_prior"
        ),
    )


def _build_single_time_quality_fusion(
    base_channels: int,
    branch_channels: int,
    optical_channels: int,
    radar_channels: int,
    gate_mode: str,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class QualityGate(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            hidden = max(4, channels // 2)
            self.net = nn.Sequential(
                nn.Conv2d(channels * 2 + 1, hidden, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(
                self.net[-1].bias,
                2.0 if gate_mode == "hard" else 0.0,
            )

        def forward(self, radar, optical, quality, structural_present):
            if quality.shape[-2:] != optical.shape[-2:]:
                quality = F.interpolate(
                    quality,
                    size=optical.shape[-2:],
                    mode="area",
                )
            quality = quality.clamp(0.0, 1.0)
            correction = self.net(
                torch.cat([radar, optical, quality], dim=1)
            )
            if gate_mode == "hard":
                return quality * torch.sigmoid(correction)
            prior = quality.clamp(0.05, 0.95)
            return (
                torch.sigmoid(torch.logit(prior) + correction)
                * structural_present
            )

    class SingleTimeQualityFusionUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            b = branch_channels
            self.architecture = (
                "hard_quality_gate"
                if gate_mode == "hard"
                else "soft_quality_prior"
            )
            self.gate_mode = gate_mode
            self.quality_branch_channels = b
            self.optical_channels = optical_channels
            self.radar_channels = radar_channels
            self.pool = nn.MaxPool2d(2)

            self.s1_enc1 = DoubleConv(radar_channels, b)
            self.s1_enc2 = DoubleConv(b, b * 2)
            self.s1_enc3 = DoubleConv(b * 2, b * 4)

            self.s2_enc1 = DoubleConv(optical_channels, b)
            self.s2_enc2 = DoubleConv(b, b * 2)
            self.s2_enc3 = DoubleConv(b * 2, b * 4)

            self.gate1 = QualityGate(b)
            self.gate2 = QualityGate(b * 2)
            self.gate3 = QualityGate(b * 4)

            self.fuse1 = DoubleConv(b * 2, c)
            self.fuse2 = DoubleConv(b * 4, c * 2)
            self.fuse3 = DoubleConv(b * 8, c * 4)

            self.bottleneck = DoubleConv(c * 4, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = DoubleConv(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = DoubleConv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = DoubleConv(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1)

        @staticmethod
        def _fuse(radar, optical, gate, module):
            return module(torch.cat([radar, optical * gate], dim=1))

        def forward(self, x, return_gate_maps: bool = False):
            expected_channels = (
                self.optical_channels + self.radar_channels + 1
            )
            if x.ndim != 4 or x.shape[1] != expected_channels:
                raise ValueError(
                    f"{self.architecture} expects input shaped "
                    f"[N, {expected_channels}, H, W]"
                )

            optical_end = self.optical_channels
            radar_end = optical_end + self.radar_channels
            raw_s2 = x[:, :optical_end]
            s1 = x[:, optical_end:radar_end]
            quality = x[:, radar_end : radar_end + 1].clamp(0.0, 1.0)
            s2 = raw_s2 * quality if gate_mode == "hard" else raw_s2
            structural_present = (
                (raw_s2.abs().amax(dim=(1, 2, 3), keepdim=True) > 0)
                | (quality.amax(dim=(1, 2, 3), keepdim=True) > 0)
            ).to(x.dtype)

            radar1 = self.s1_enc1(s1)
            optical1 = self.s2_enc1(s2)
            gate1 = self.gate1(
                radar1,
                optical1,
                quality,
                structural_present,
            )
            fused1 = self._fuse(
                radar1,
                optical1,
                gate1,
                self.fuse1,
            )

            radar2 = self.s1_enc2(self.pool(radar1))
            optical2 = self.s2_enc2(self.pool(optical1))
            gate2 = self.gate2(
                radar2,
                optical2,
                quality,
                structural_present,
            )
            fused2 = self._fuse(
                radar2,
                optical2,
                gate2,
                self.fuse2,
            )

            radar3 = self.s1_enc3(self.pool(radar2))
            optical3 = self.s2_enc3(self.pool(optical2))
            gate3 = self.gate3(
                radar3,
                optical3,
                quality,
                structural_present,
            )
            fused3 = self._fuse(
                radar3,
                optical3,
                gate3,
                self.fuse3,
            )

            bottleneck = self.bottleneck(self.pool(fused3))
            decoded3 = self.up3(bottleneck)
            decoded3 = self.dec3(torch.cat([decoded3, fused3], dim=1))
            decoded2 = self.up2(decoded3)
            decoded2 = self.dec2(torch.cat([decoded2, fused2], dim=1))
            decoded1 = self.up1(decoded2)
            if decoded1.shape[-2:] != fused1.shape[-2:]:
                decoded1 = F.interpolate(
                    decoded1,
                    size=fused1.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            decoded1 = self.dec1(torch.cat([decoded1, fused1], dim=1))
            logits = self.out(decoded1)
            if not return_gate_maps:
                return logits
            return logits, (gate1, gate2, gate3)

    return SingleTimeQualityFusionUNet()
