"""Modality-adaptive encoder that maps every sensor view into a shared space.

Architecture (in order):
  1. A per-modality **input adapter** -- a 1x1 Conv + BN + ReLU that maps the
     modality's raw band count onto the 3-channel space expected by the shared
     backbone (initialised to a sensible spectral projection of the bands).
  2. A **shared** ResNet backbone (frozen or fine-tunable) that extracts generic
     spatial features, agnostic to the *choice* of sensor.
  3. A MLP **projection head** that projects features into a low-dimensional,
     L2-normalised embedding where cross-modal contrastive loss is applied.
  4. An optional linear **classifier** used as an auxiliary supervised signal.

By sharing the backbone and aligning all modalities with contrastive losses,
the final embedding space places semantically-similar scenes (same land cover)
close together regardless of which sensor captured them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

BACKBONE_CATALOG = {
    "resnet18": {
        "factory": torchvision.models.resnet18,
        "weights": torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
        "feat_dim": 512,
    },
    "resnet34": {
        "factory": torchvision.models.resnet34,
        "weights": torchvision.models.ResNet34_Weights.IMAGENET1K_V1,
        "feat_dim": 512,
    },
    "resnet50": {
        "factory": torchvision.models.resnet50,
        "weights": torchvision.models.ResNet50_Weights.IMAGENET1K_V1,
        "feat_dim": 2048,
    },
}

# Which multispectral channel index feeds each of the 3 RGB output slots.
_MS_TO_RGB = {"R": 2, "G": 1, "B": 0}


def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    if name not in BACKBONE_CATALOG:
        raise ValueError(f"Unknown backbone '{name}'; choose from {sorted(BACKBONE_CATALOG)}")
    spec = BACKBONE_CATALOG[name]
    weights = spec["weights"] if pretrained else None
    net = spec["factory"](weights=weights)
    net.fc = nn.Identity()  # drop the ImageNet classifier
    return net


class _AdaptConv2d(nn.Module):
    """1x1 conv with a spectrally-meaningful initialisation.

    * 3-channel input  -> identity init.
    * multispectral    -> maps the Red/Green/Blue bands onto the 3 output slots.
    * single-band SAR  -> replicates the channel across the 3 slots.
    """

    def __init__(self, in_channels: int, modality: str = "optical") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self._init_spectral(in_channels, modality)

    def _init_spectral(self, in_ch: int, modality: str) -> None:
        with torch.no_grad():
            w = torch.zeros((3, in_ch, 1, 1))
            if modality == "multispectral":
                for slot, band in zip(range(3), ["R", "G", "B"]):
                    w[slot, _MS_TO_RGB[band], 0, 0] = 1.0
            elif in_ch == 1:
                w[:, 0, 0, 0] = 1.0  # replicate the single SAR channel
            else:  # 3-channel -> identity
                for slot in range(3):
                    w[slot, slot, 0, 0] = 1.0
            self.conv.weight.copy_(w)
            self.conv.bias.copy_(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def _unfreeze_blocks(net: nn.Module, stage: str) -> None:
    """Unfreeze the deepest ResNet blocks named in ``stage`` (none/stage4/stage3)."""
    if stage in (None, "none", ""):
        return
    target = {"stage4": ["layer4"], "stage3": ["layer4", "layer3"]}.get(stage, [])
    for name, module in net.named_modules():
        if name in target:  # matches the top-level blocks; children are covered too
            for p in module.parameters():
                p.requires_grad = True


class ModalityAdaptiveEncoder(nn.Module):
    """Shared-embedding encoder: adapters + backbone + projection + classifier."""

    def __init__(
        self,
        modalities: Sequence[str],
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 128,
        n_classes: Optional[int] = None,
        freeze_backbone: bool = True,
        unfreeze_stage: str = "none",
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        feat_dim = BACKBONE_CATALOG[backbone]["feat_dim"]

        # Per-modality input adapters.
        self.adapters = nn.ModuleDict()
        for m in self.modalities:
            n_in = 3 if m == "optical" else (8 if m == "multispectral" else 1)
            self.adapters[m] = _AdaptConv2d(n_in, modality=m)

        self.backbone = _build_backbone(backbone, pretrained)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes) if n_classes else None

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Optionally unfreeze the deepest block(s) so the shared trunk can
            # adapt to the specific sensor statistics while staying CPU-cheap.
            _unfreeze_blocks(self.backbone, unfreeze_stage)

    def encode_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Raw backbone features (pre-projection), shape (B, feat_dim)."""
        x = self.adapters[modality](x)
        return self.backbone(x)

    def embed(self, x: torch.Tensor, modality: str, normalize: bool = True) -> torch.Tensor:
        """Projected embedding in the shared space."""
        feat = self.encode_features(x, modality)
        e = self.projection(feat)
        if normalize:
            e = F.normalize(e, dim=1)
        return e


def build_model(
    modalities: Sequence[str],
    backbone: str = "resnet18",
    pretrained: bool = True,
    embedding_dim: int = 128,
    n_classes: Optional[int] = None,
    freeze_backbone: bool = True,
    unfreeze_stage: str = "none",
) -> ModalityAdaptiveEncoder:
    return ModalityAdaptiveEncoder(
        modalities=modalities,
        backbone=backbone,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
        freeze_backbone=freeze_backbone,
        unfreeze_stage=unfreeze_stage,
    )