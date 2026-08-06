"""Grad-CAM saliency for CNN (ResNet) encoders.

Grad-CAM localises which image regions drive a prediction by weighting the last
convolutional feature maps by their gradient w.r.t. the target class score:

    CAM = ReLU(sum_c alpha_c * A_c),   alpha_c = mean spatial gradient of A_c

Works on frozen backbones (gradients flow to the feature maps, not the
parameters). For transformer encoders this is not technically appropriate --
use :mod:`src.xai.attention` instead.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.encoder import ModalityAdaptiveEncoder


def last_conv_layer(model: nn.Module) -> nn.Conv2d:
    """Return the last Conv2d in the backbone (the Grad-CAM target layer)."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError(
            "no Conv2d layer found; Grad-CAM requires a CNN backbone "
            "(ResNet). For transformer encoders use attention maps."
        )
    return convs[-1]


def _class_scores(
    model: ModalityAdaptiveEncoder, x: torch.Tensor, modality: str
) -> torch.Tensor:
    """Per-class scores for Grad-CAM (classifier logits, else raw embedding)."""
    if model.classifier is not None:
        feat = model.encode_features(x, modality)
        emb = feat if model.embedding_mode == "raw" else model.project(feat, modality)
        return model.classifier(emb)
    return model.embed(x, modality, normalize=False)


def gradcam(
    model: ModalityAdaptiveEncoder,
    x: torch.Tensor,
    modality: str,
    class_idx: Optional[int] = None,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    """Compute a Grad-CAM saliency map for one image, shape (H, W) in [0, 1].

    ``class_idx`` selects the target class; when None the argmax class is used.
    """
    model.eval()
    layer = target_layer or last_conv_layer(model)

    activations: dict = {}
    gradients: dict = {}

    def fwd_hook(_m, _i, out):
        activations["a"] = out.detach()

    def bwd_hook(_m, _gi, go):
        gradients["g"] = go[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        x = x.detach().requires_grad_(True)
        scores = _class_scores(model, x, modality)
        if class_idx is None:
            class_idx = int(scores[0].argmax(dim=0).item())
        model.zero_grad()
        scores[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()

    a = activations["a"][0]   # (C, H, W)
    g = gradients["g"][0]     # (C, H, W)
    weights = g.mean(dim=(1, 2), keepdim=True)          # (C, 1, 1)
    cam = F.relu((weights * a).sum(dim=0))              # (H, W)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear",
                        align_corners=False)
    return cam[0, 0].detach().cpu().numpy()


def overlay_saliency(image_rgb: np.ndarray, saliency: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a saliency map (H, W, [0,1]) on an (H, W, 3) image (jet)."""
    from matplotlib import colormaps

    cmap = colormaps["jet"]
    heat = (cmap(saliency)[:, :, :3] * 255.0).astype(np.uint8)
    img = image_rgb.astype(np.float32)
    return (alpha * heat + (1.0 - alpha) * img).astype(np.uint8)
