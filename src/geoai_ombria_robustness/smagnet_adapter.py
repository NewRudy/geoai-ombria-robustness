from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


OFFICIAL_SMAGNET_REPOSITORY = "https://github.com/ASUcicilab/SMAGNet"
OFFICIAL_SMAGNET_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
OFFICIAL_SMAGNET_SOURCE_SHA256 = (
    "daf00d0533ca7865b4bd7b47404f1c0fa42e4a0bdc70706dee45bedcc1420f25"
)
OFFICIAL_SMAGNET_LICENSE_SHA256 = (
    "4261bd84b3a36788cb1bb4e25d3f59a2cf2ac79abb93cb45cf09fc043b39265c"
)
OFFICIAL_SMAGNET_PAPER_DOI = "10.1016/j.isprsjprs.2025.12.023"
OFFICIAL_SMAGNET_MODEL = {
    "encoder_name": "resnet50",
    "encoder_depth": 5,
    "encoder_weights_sar": None,
    "encoder_weights_msi": "imagenet",
    "decoder_use_batchnorm": False,
    "decoder_channels": [256, 128, 64, 32, 16],
    "classes": 1,
    "activation": None,
    "sarmsiff_method": "sar_msi_gated",
    "enable_spatial_mask": True,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_official_smagnet_checkout(checkout: Path) -> dict[str, Any]:
    source = checkout / "src" / "smagnet.py"
    license_path = checkout / "LICENSE"
    if not source.is_file() or not license_path.is_file():
        raise FileNotFoundError("official SMAGNet source or license is missing")
    source_sha256 = file_sha256(source)
    license_sha256 = file_sha256(license_path)
    if source_sha256 != OFFICIAL_SMAGNET_SOURCE_SHA256:
        raise ValueError("official SMAGNet source hash does not match frozen commit")
    if license_sha256 != OFFICIAL_SMAGNET_LICENSE_SHA256:
        raise ValueError("official SMAGNet license hash does not match frozen commit")
    return {
        "schema": "geoai-official-smagnet-source-v1",
        "repository": OFFICIAL_SMAGNET_REPOSITORY,
        "commit": OFFICIAL_SMAGNET_COMMIT,
        "paper_doi": OFFICIAL_SMAGNET_PAPER_DOI,
        "license": "MIT",
        "source_path": str(source),
        "source_sha256": source_sha256,
        "license_path": str(license_path),
        "license_sha256": license_sha256,
        "model_configuration": OFFICIAL_SMAGNET_MODEL,
    }


def load_official_smagnet_module(source: Path) -> ModuleType:
    if file_sha256(source) != OFFICIAL_SMAGNET_SOURCE_SHA256:
        raise ValueError("refusing to import unverified official SMAGNet source")
    specification = importlib.util.spec_from_file_location(
        "geoai_frozen_official_smagnet", source
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load official SMAGNet module from {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not hasattr(module, "SMAGNet"):
        raise ImportError("verified official source does not define SMAGNet")
    return module


def build_official_smagnet(
    source: Path,
    *,
    encoder_weights_msi: str | None = "imagenet",
):
    module = load_official_smagnet_module(source)
    configuration = {
        **OFFICIAL_SMAGNET_MODEL,
        "encoder_weights_msi": encoder_weights_msi,
    }
    return module.SMAGNet(**configuration)


def normalize_sen1floods11_for_official_smagnet(
    image: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    """Map the frozen B/G/R/NIR layout to official R/G/B/NIR standardization."""

    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != 7:
        raise ValueError("Sen1Floods11 SMAGNet image must have shape [7, H, W]")
    optical = image[[2, 1, 0, 3]].copy()
    radar = image[4:6].copy()
    quality = image[6:7].copy()
    optical_mean = np.asarray(normalization["optical_mean"], dtype=np.float32)
    optical_std = np.asarray(normalization["optical_std"], dtype=np.float32)
    radar_mean = np.asarray(normalization["radar_mean"], dtype=np.float32)
    radar_std = np.asarray(normalization["radar_std"], dtype=np.float32)
    expected_shapes = (
        (optical_mean, 4),
        (optical_std, 4),
        (radar_mean, 2),
        (radar_std, 2),
    )
    if any(values.shape != (size,) for values, size in expected_shapes):
        raise ValueError("official SMAGNet normalization channel counts are invalid")
    if np.any(optical_std <= 0) or np.any(radar_std <= 0):
        raise ValueError("official SMAGNet normalization standard deviations must be positive")
    optical = (optical - optical_mean[:, None, None]) / optical_std[:, None, None]
    radar = (radar - radar_mean[:, None, None]) / radar_std[:, None, None]
    return np.concatenate([optical, radar, quality], axis=0).astype(
        np.float32, copy=False
    )


def split_smagnet_input(image):
    if image.ndim != 4 or image.shape[1] != 7:
        raise ValueError("SMAGNet input must have shape [N, 7, H, W]")
    optical = image[:, :4]
    radar = image[:, 4:6]
    available = image[:, 6:7]
    if bool(((available < 0) | (available > 1)).any().item()):
        raise ValueError("SMAGNet availability channel must lie within [0, 1]")
    invalid = 1.0 - available
    return radar, optical, invalid


def forward_official_smagnet(model, image):
    radar, optical, invalid = split_smagnet_input(image)
    return model(radar, optical, invalid)


def masked_bce_with_logits(logits, target, valid_target):
    import torch.nn.functional as F

    pixel_loss = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    valid = valid_target.to(pixel_loss.dtype)
    return (pixel_loss * valid).sum() / valid.sum().clamp_min(1.0)


def dual_path_masked_bce(fused_logits, sar_logits, target, valid_target):
    fused = masked_bce_with_logits(fused_logits, target, valid_target)
    sar = masked_bce_with_logits(sar_logits, target, valid_target)
    return 0.5 * sar + 0.5 * fused


def verify_complete_absence_equivalence(
    model,
    *,
    device,
    size: int = 64,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    import torch

    if size < 32 or size % 32:
        raise ValueError("equivalence check size must be a multiple of 32")
    generator = torch.Generator(device="cpu").manual_seed(20260720)
    radar = torch.randn((1, 2, size, size), generator=generator)
    optical = torch.zeros((1, 4, size, size))
    quality = torch.zeros((1, 1, size, size))
    image = torch.cat([optical, radar, quality], dim=1).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        fused, sar, gates = forward_official_smagnet(model, image)
    maximum_difference = float((fused - sar).abs().max().item())
    maximum_gate = max(float(gate.abs().max().item()) for gate in gates)
    model.train(was_training)
    if maximum_difference > tolerance or maximum_gate > tolerance:
        raise RuntimeError("official SMAGNet complete-absence boundary failed")
    return {
        "status": "pass",
        "input_size": size,
        "tolerance": tolerance,
        "maximum_fused_sar_logit_difference": maximum_difference,
        "maximum_masked_gate": maximum_gate,
    }


def render_source_manifest(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True) + "\n"
