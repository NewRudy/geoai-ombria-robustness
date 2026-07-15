from __future__ import annotations

from typing import Any


MODEL_ARCHITECTURES = (
    "early_fusion_unet",
    "quality_gated_fusion",
)


def resolve_quality_branch_channels(
    base_channels: int, requested_channels: int | None = None
) -> int:
    """Resolve a capacity-controlled modality-branch width.

    One quarter of the decoder width keeps the quality-gated model within five
    percent of the early-fusion U-Net parameter count at the default width.
    """
    if base_channels < 4:
        raise ValueError("base_channels must be at least 4")
    if requested_channels is not None:
        if requested_channels < 2:
            raise ValueError("quality_branch_channels must be at least 2")
        return requested_channels
    return max(4, int(round(base_channels / 4)))


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model(
    in_channels: int,
    base_channels: int,
    architecture: str = "early_fusion_unet",
    quality_branch_channels: int | None = None,
):
    if architecture not in MODEL_ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture {architecture!r}; choose from {MODEL_ARCHITECTURES}"
        )
    if architecture == "early_fusion_unet":
        return _build_early_fusion_unet(in_channels, base_channels)
    if in_channels != 10:
        raise ValueError(
            "quality_gated_fusion requires the 10-channel multimodal layout "
            "(S2 before/after, S1 before/after, quality before/after)"
        )
    return _build_quality_gated_fusion(
        base_channels,
        resolve_quality_branch_channels(base_channels, quality_branch_channels),
    )


def _build_early_fusion_unet(in_channels: int, base_channels: int):
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

    class SmallUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            self.architecture = "early_fusion_unet"
            self.quality_branch_channels = None
            self.enc1 = DoubleConv(in_channels, c)
            self.enc2 = DoubleConv(c, c * 2)
            self.enc3 = DoubleConv(c * 2, c * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(c * 4, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = DoubleConv(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = DoubleConv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = DoubleConv(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-2:] != e1.shape[-2:]:
                d1 = F.interpolate(
                    d1, size=e1.shape[-2:], mode="bilinear", align_corners=False
                )
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    return SmallUNet()


def _build_quality_gated_fusion(base_channels: int, branch_channels: int):
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
        def __init__(self, radar_ch: int, optical_ch: int) -> None:
            super().__init__()
            hidden = max(4, optical_ch // 2)
            self.net = nn.Sequential(
                nn.Conv2d(radar_ch + optical_ch + 1, hidden, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(hidden, 1, 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, 2.0)

        def forward(self, radar, optical, availability):
            if availability.shape[-2:] != optical.shape[-2:]:
                availability = F.interpolate(
                    availability, size=optical.shape[-2:], mode="area"
                )
            availability = availability.clamp(0.0, 1.0)
            learned = torch.sigmoid(
                self.net(torch.cat([radar, optical, availability], dim=1))
            )
            return availability * learned

    class QualityGatedFusionUNet(nn.Module):
        """S1-preserving fusion with explicit bitemporal S2 availability gates.

        Input channels are fixed as S2-before RGB (0:3), S2-after RGB (3:6),
        S1-before/after (6:8), and S2-before/after availability (8:10).
        """

        def __init__(self) -> None:
            super().__init__()
            c = base_channels
            b = branch_channels
            self.architecture = "quality_gated_fusion"
            self.quality_branch_channels = b
            self.pool = nn.MaxPool2d(2)

            self.s1_enc1 = DoubleConv(2, b)
            self.s1_enc2 = DoubleConv(b, b * 2)
            self.s1_enc3 = DoubleConv(b * 2, b * 4)

            # The same optical encoder is applied to pre- and post-event S2.
            self.s2_enc1 = DoubleConv(3, b)
            self.s2_enc2 = DoubleConv(b, b * 2)
            self.s2_enc3 = DoubleConv(b * 2, b * 4)

            self.pre_gate1 = QualityGate(b, b)
            self.post_gate1 = QualityGate(b, b)
            self.pre_gate2 = QualityGate(b * 2, b * 2)
            self.post_gate2 = QualityGate(b * 2, b * 2)
            self.pre_gate3 = QualityGate(b * 4, b * 4)
            self.post_gate3 = QualityGate(b * 4, b * 4)

            self.fuse1 = DoubleConv(b * 3, c)
            self.fuse2 = DoubleConv(b * 6, c * 2)
            self.fuse3 = DoubleConv(b * 12, c * 4)

            self.bottleneck = DoubleConv(c * 4, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
            self.dec3 = DoubleConv(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
            self.dec2 = DoubleConv(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
            self.dec1 = DoubleConv(c * 2, c)
            self.out = nn.Conv2d(c, 1, 1)

        @staticmethod
        def _shared_temporal(module, before, after):
            paired = module(torch.cat([before, after], dim=0))
            return paired.chunk(2, dim=0)

        @staticmethod
        def _fuse(radar, before, after, before_gate, after_gate, module):
            return module(
                torch.cat(
                    [radar, before * before_gate, after * after_gate], dim=1
                )
            )

        def forward(self, x, return_gate_maps: bool = False):
            if x.ndim != 4 or x.shape[1] != 10:
                raise ValueError(
                    "quality_gated_fusion expects input shaped [N, 10, H, W]"
                )

            quality_before = x[:, 8:9].clamp(0.0, 1.0)
            quality_after = x[:, 9:10].clamp(0.0, 1.0)
            # Sanitization prevents explicitly unavailable optical values from
            # entering the feature extractor; multi-scale gates then enforce
            # the same availability boundary after spatial aggregation.
            s2_before = x[:, 0:3] * quality_before
            s2_after = x[:, 3:6] * quality_after
            s1 = x[:, 6:8]

            radar1 = self.s1_enc1(s1)
            before1, after1 = self._shared_temporal(
                self.s2_enc1, s2_before, s2_after
            )
            pre_gate1 = self.pre_gate1(radar1, before1, quality_before)
            post_gate1 = self.post_gate1(radar1, after1, quality_after)
            fused1 = self._fuse(
                radar1,
                before1,
                after1,
                pre_gate1,
                post_gate1,
                self.fuse1,
            )

            radar2 = self.s1_enc2(self.pool(radar1))
            before2, after2 = self._shared_temporal(
                self.s2_enc2, self.pool(before1), self.pool(after1)
            )
            pre_gate2 = self.pre_gate2(radar2, before2, quality_before)
            post_gate2 = self.post_gate2(radar2, after2, quality_after)
            fused2 = self._fuse(
                radar2,
                before2,
                after2,
                pre_gate2,
                post_gate2,
                self.fuse2,
            )

            radar3 = self.s1_enc3(self.pool(radar2))
            before3, after3 = self._shared_temporal(
                self.s2_enc3, self.pool(before2), self.pool(after2)
            )
            pre_gate3 = self.pre_gate3(radar3, before3, quality_before)
            post_gate3 = self.post_gate3(radar3, after3, quality_after)
            fused3 = self._fuse(
                radar3,
                before3,
                after3,
                pre_gate3,
                post_gate3,
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
            return logits, {
                "before": (pre_gate1, pre_gate2, pre_gate3),
                "after": (post_gate1, post_gate2, post_gate3),
            }

    return QualityGatedFusionUNet()
