"""Attention visualisation for ViT encoders.

Captures the self-attention weights of the last transformer layer, averages
across heads and renders the CLS-token attention over the patch grid, resized
to the input resolution. Only meaningful for transformer backbones (for CNN
encoders use :mod:`src.xai.gradcam`).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.encoder import ModalityAdaptiveEncoder


def last_attention_module(model: nn.Module) -> nn.Module:
    """Return the last transformer self-attention module in the backbone."""
    layers = getattr(getattr(getattr(model, "backbone", None), "backbone", None), "encoder", None)
    if layers is None or not hasattr(layers, "layers") or not len(layers.layers):
        raise ValueError(
            "no transformer layers found; attention visualisation requires a "
            "ViT / transformer backbone (use Grad-CAM for CNNs)."
        )
    return layers.layers[-1].self_attention


def attention_map(
    model: ModalityAdaptiveEncoder,
    x: torch.Tensor,
    modality: str,
    layer: int = -1,
    head: Optional[int] = None,
) -> np.ndarray:
    """CLS-to-patch attention map for one image, shape (H, W) in [0, 1]."""
    layers = model.backbone.backbone.encoder.layers
    attn_module = layers[layer].self_attention
    captured: dict = {}

    def fwd_hook(module, args, output):
        # Remove this hook first, then re-run with need_weights=True to obtain
        # the attention weights (torchvision discards them by default).
        h.remove()
        out, attn = module(*args, need_weights=True, average_attn_weights=False)
        captured["attn"] = attn.detach()
        return out, attn  # torchvision's EncoderBlock unpacks (x, _)

    h = attn_module.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.no_grad():
            model.encode_features(x.to(next(model.parameters()).device), modality)
    finally:
        h.remove()

    attn = captured["attn"]  # (B, heads, L, L) with batch_first=True
    a = attn[0]
    if head is None:
        a = a.mean(dim=0)             # average heads -> (L, L)
    else:
        a = a[head]
    cls_attn = a[0, 1:]               # CLS token -> patch tokens (L-1,)
    n = int(round(math.sqrt(cls_attn.shape[0])))
    grid = cls_attn.reshape(n, n).float()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    size = x.shape[-2:]
    grid = F.interpolate(grid[None, None], size=size, mode="bilinear", align_corners=False)
    return grid[0, 0].numpy()


def attention_maps(model: ModalityAdaptiveEncoder, x: torch.Tensor, modality: str,
                   layer: int = -1) -> np.ndarray:
    """All-head attention for a batch image, shape (heads, H, W)."""
    layers = model.backbone.backbone.encoder.layers
    attn_module = layers[layer].self_attention
    captured: dict = {}

    def fwd_hook(module, args, output):
        h.remove()
        out, attn = module(*args, need_weights=True, average_attn_weights=False)
        captured["attn"] = attn.detach()
        return out, attn

    h = attn_module.register_forward_hook(fwd_hook)
    try:
        model.eval()
        with torch.no_grad():
            model.encode_features(x.to(next(model.parameters()).device), modality)
    finally:
        h.remove()
    a = captured["attn"][0]  # (heads, L, L)
    cls_attn = a[:, 0, 1:]
    n = int(round(math.sqrt(cls_attn.shape[-1])))
    grid = cls_attn.reshape(-1, n, n).float()
    grid = (grid - grid.min(dim=(1, 2), keepdim=True).values) / (
        grid.max(dim=(1, 2), keepdim=True).values - grid.min(dim=(1, 2), keepdim=True).values + 1e-8
    )
    size = x.shape[-2:]
    grid = F.interpolate(grid[:, None], size=size, mode="bilinear", align_corners=False)
    return grid[:, 0].numpy()
