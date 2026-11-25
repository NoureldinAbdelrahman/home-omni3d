"""
Shared checkpoint loading helpers for Pix2Vox++ and AtlasNet.
Automatically inspects saved architecture configs for both legacy and upgraded checkpoints.
"""

import os
from typing import Optional
import torch

from src.models.pix2vox import Pix2Vox
from src.models.atlasnet import AtlasNet


def load_pix2vox_checkpoint(
    weights_path: Optional[str],
    device: torch.device,
    use_refiner: bool = True,
) -> Pix2Vox:
    """Loads a Pix2Vox or Pix2Vox++ checkpoint, inferring arch_config if present."""
    ckpt = None
    arch = {}
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        arch = ckpt.get("arch_config", {}) if isinstance(ckpt, dict) else {}

    model = Pix2Vox(
        pretrained=False,
        use_refiner=arch.get("use_refiner", use_refiner),
        backbone=arch.get("backbone", "resnet18"),
        spatial_pool=arch.get("spatial_pool", 1),
        feat_dim=arch.get("feat_dim", 512),
        deep_refiner=arch.get("deep_refiner", False),
    ).to(device)

    if ckpt is not None:
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_atlasnet_checkpoint(
    weights_path: Optional[str],
    device: torch.device,
) -> AtlasNet:
    """Loads an AtlasNet checkpoint, inferring spherical/square template and pooling."""
    ckpt = None
    arch = {}
    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        arch = ckpt.get("arch_config", {}) if isinstance(ckpt, dict) else {}

    model = AtlasNet(
        num_patches=arch.get("num_patches", 25),
        latent_dim=arch.get("latent_dim", 1024),
        hidden_dim=arch.get("hidden_dim", 256),
        pretrained=False,
        template=arch.get("template", "square"),
        view_pooling=arch.get("view_pooling", "max"),
    ).to(device)

    if ckpt is not None:
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model
