"""Modality-adaptive encoders that map every sensor view into a shared space.

Architecture (in order):
  1. A per-modality **input adapter** -- a 1x1 Conv + BN + ReLU that maps the
     modality's raw band count onto the 3-channel space expected by the shared
     backbone (initialised to a sensible spectral projection of the bands).
  2. A **shared backbone** selected through :class:`BaseEncoder`:

     * :class:`ResNetEncoder`  -- torchvision ResNet-18/34/50 (frozen or
       fine-tunable by stage),
     * :class:`ViTEncoder`     -- torchvision ViT-B/16 (with ImageNet pos-embed
       interpolated to the patch size used),
     * :class:`SatMAEEncoder` / :class:`PrithviEncoder` -- foundation-model
       adapters that require a user-supplied local checkpoint (we do not fake
       compatibility when the weights are absent).

  3. A MLP **projection head** (shared across modalities, or one per modality)
     that projects features into a low-dimensional, L2-normalised embedding
     where cross-modal contrastive loss is applied.
  4. An optional linear **classifier** used as an auxiliary supervised signal.

By sharing the backbone and aligning all modalities with contrastive losses,
the final embedding space places semantically-similar scenes (same land cover)
close together regardless of which sensor captured them.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ---------------------------------------------------------------------------
# Backbone catalog + encoder interface
# ---------------------------------------------------------------------------

RESNET_CATALOG = {
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


class BaseEncoder(nn.Module):
    """Adapter interface for the shared backbone.

    Every backend exposes ``feature_dim`` and ``forward_features(x) -> (B, D)``
    so the projection/classifier layers are backbone-agnostic.
    """

    feature_dim: int = 0
    name: str = "base"

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def freeze(self, freeze: bool = True) -> None:
        for p in self.parameters():
            p.requires_grad = not freeze

    def unfreeze_stage(self, stage: str) -> None:
        """Unfreeze the deepest blocks named by ``stage`` (no-op by default)."""
        return


class ResNetEncoder(BaseEncoder):
    name = "resnet"

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if backbone not in RESNET_CATALOG:
            raise ValueError(f"Unknown backbone '{backbone}'; choose from {sorted(RESNET_CATALOG)}")
        spec = RESNET_CATALOG[backbone]
        net = spec["factory"](weights=spec["weights"] if pretrained else None)
        net.fc = nn.Identity()  # drop the ImageNet classifier
        self.backbone = net
        self.feature_dim = spec["feat_dim"]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_stage(self, stage: str) -> None:
        target = {"stage4": ["layer4"], "stage3": ["layer4", "layer3"]}.get(stage, [])
        for name, module in self.backbone.named_modules():
            if name in target:
                for p in module.parameters():
                    p.requires_grad = True


def _interpolate_pos_embed(state: Dict, n_tokens: int) -> torch.Tensor:
    """Bilinearly interpolate a ViT positional embedding to a new token grid.

    ``state`` is the ImageNet pos_embed (1, 1 + n, D); the cls token is kept
    verbatim and the patch tokens are interpolated to ``n_tokens``.
    """
    cls_token = state[:, :1]
    tokens = state[:, 1:]
    n = int(round(math.sqrt(tokens.shape[1])))
    new_n = int(round(math.sqrt(n_tokens)))
    tokens = tokens.reshape(1, n, n, -1).permute(0, 3, 1, 2).contiguous()
    tokens = F.interpolate(tokens, size=(new_n, new_n), mode="bilinear", align_corners=False)
    tokens = tokens.permute(0, 2, 3, 1).reshape(1, n_tokens, -1)
    return torch.cat([cls_token, tokens], dim=1)


class ViTEncoder(BaseEncoder):
    name = "vit"

    def __init__(
        self, pretrained: bool = True, image_size: int = 64, variant: str = "vit_b_16"
    ) -> None:
        super().__init__()
        if variant != "vit_b_16":
            raise ValueError(f"unsupported ViT variant '{variant}' (only vit_b_16)")
        net = torchvision.models.vit_b_16(image_size=int(image_size))
        net.heads = nn.Identity()
        self.backbone = net
        self.feature_dim = 768
        self.image_size = int(image_size)
        if pretrained:
            _load_vit_pretrained(self.backbone, self.image_size)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def unfreeze_stage(self, stage: str) -> None:
        if stage in ("stage4", "stage3", "last"):
            blocks = list(self.backbone.encoder.layers.children())
            for blk in blocks[-2:]:  # unfreeze the last 2 transformer blocks
                for p in blk.parameters():
                    p.requires_grad = True


def _load_vit_pretrained(net: nn.Module, image_size: int) -> None:
    """Load ImageNet ViT weights, interpolating pos_embed for the patch grid."""
    try:
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        state = weights.get_state_dict(progress=True)
    except Exception:
        return
    n_tokens = (int(image_size) // 16) ** 2
    pe_key = next((k for k in state if k.endswith("pos_embed")), None)
    if pe_key and state[pe_key].shape[1] - 1 != n_tokens:
        state[pe_key] = _interpolate_pos_embed(state[pe_key], n_tokens)
    net.load_state_dict(state, strict=False)


class _ExternalEncoder(BaseEncoder):
    """Best-effort wrapper for user-supplied foundation-model checkpoints.

    ``backbone`` is an already-built module (e.g. a ViT). ``checkpoint_path``,
    when provided, loads weights with ``strict=False`` so partial/renamed
    checkpoints can be applied. Raises a clear error when the checkpoint is
    configured but unreadable -- we never silently fake a foundation model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise ValueError(
                    f"foundation-model checkpoint not found: '{checkpoint_path}'"
                )
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            sd = state.get("state_dict", state)
            sd = {k: v for k, v in sd.items() if not k.startswith("module.")}
            try:
                self.backbone.load_state_dict(sd, strict=False)
            except Exception as exc:  # pragma: no cover - depends on the checkpoint
                raise ValueError(
                    f"could not load weights from '{checkpoint_path}' into the "
                    f"foundation backbone: {exc}"
                ) from exc

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SatMAEEncoder(_ExternalEncoder):
    """SatMAE foundation-model adapter (ViT backbone, 3-channel input).

    SatMAE is trained on the fMoW dataset and is *not bundled*. To use it,
    build a ViT backbone matching the SatMAE encoder and provide a checkpoint
    via ``model.foundation.satmae.path`` (+ ``feature_dim``).
    """

    name = "satmae"


class PrithviEncoder(_ExternalEncoder):
    """Prithvi-EO foundation-model adapter (single-frame spatial ViT).

    Prithvi is a temporal ViT; only its single-frame spatial encoder is used
    here. Weights are *not bundled*: provide a checkpoint via
    ``model.foundation.prithvi.path`` (+ ``feature_dim``) to enable it.
    """

    name = "prithvi"


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def build_backbone(
    name: str,
    pretrained: bool = True,
    image_size: Optional[int] = None,
    foundation: Optional[Dict] = None,
) -> BaseEncoder:
    """Construct a :class:`BaseEncoder` for ``name``.

    ``foundation`` is the ``model.foundation`` config dict with per-model
    ``path``/``feature_dim`` entries used by SatMAE/Prithvi.
    """
    foundation = foundation or {}
    if name in RESNET_CATALOG:
        return ResNetEncoder(name, pretrained)
    if name in ("vit_b_16", "vit"):
        return ViTEncoder(pretrained=pretrained, image_size=int(image_size or 64))
    if name in ("satmae", "prithvi"):
        cfg = foundation.get(name, {}) or {}
        path = cfg.get("path")
        feature_dim = int(cfg.get("feature_dim") or 768)
        if not path:
            raise ValueError(
                f"Foundation model '{name}' requires a local checkpoint: "
                f"set model.foundation.{name}.path (and .feature_dim). "
                f"Weights are not bundled and cannot be faked."
            )
        backbone = torchvision.models.vit_b_16(image_size=int(image_size or 64))
        backbone.heads = nn.Identity()
        cls = {"satmae": SatMAEEncoder, "prithvi": PrithviEncoder}[name]
        return cls(backbone, feature_dim, checkpoint_path=path)
    raise ValueError(
        f"Unknown backbone '{name}'; choose from "
        f"{sorted(list(RESNET_CATALOG)) + ['vit_b_16', 'satmae', 'prithvi']}"
    )


# ---------------------------------------------------------------------------
# Input adapters + main encoder
# ---------------------------------------------------------------------------

def _legacy_nbands(modality: str) -> int:
    """Backward-compatible band counts for the built-in modalities."""
    return 3 if modality == "optical" else (8 if modality == "multispectral" else 1)


class _AdaptConv2d(nn.Module):
    """1x1 conv with a spectrally-meaningful initialisation."""

    def __init__(self, in_channels: int, modality: str = "optical") -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self._init_spectral(in_channels, modality)

    def _init_spectral(self, in_ch: int, modality: str) -> None:
        with torch.no_grad():
            w = torch.zeros((3, in_ch, 1, 1))
            if modality == "multispectral" and in_ch >= 3:
                for slot, band in zip(range(3), ["R", "G", "B"]):
                    w[slot, _MS_TO_RGB[band], 0, 0] = 1.0
            else:
                # Map the available input channels onto the 3 RGB slots.
                # 1 channel -> replicate it; 2 channels (VV+VH) -> VV, VH, VH;
                # >=3 channels -> first 3 channels identity.
                for slot in range(3):
                    src = min(slot, in_ch - 1)
                    w[slot, src, 0, 0] = 1.0
            self.conv.weight.copy_(w)
            self.conv.bias.copy_(torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


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
        modality_channels: Optional[Dict[str, int]] = None,
        image_size: Optional[int] = None,
        projection_heads: str = "shared",
        foundation: Optional[Dict] = None,
        embedding_mode: str = "projected",
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.projection_heads = projection_heads
        self.embedding_mode = embedding_mode  # "projected" | "raw"
        # Real datasets may use non-default band counts (e.g. 2-band SAR).
        self.modality_channels = dict(modality_channels or {})
        if not self.modality_channels:
            self.modality_channels = {m: _legacy_nbands(m) for m in self.modalities}

        # Per-modality input adapters.
        self.adapters = nn.ModuleDict()
        for m in self.modalities:
            n_in = int(self.modality_channels.get(m, _legacy_nbands(m)))
            self.adapters[m] = _AdaptConv2d(n_in, modality=m)

        self.backbone = build_backbone(backbone, pretrained, image_size, foundation)
        feat_dim = int(self.backbone.feature_dim)

        # Projection head: shared across modalities, or one per modality.
        if projection_heads == "per_modality":
            self.projection = nn.ModuleDict(
                {m: self._make_projection(feat_dim, embedding_dim) for m in self.modalities}
            )
        else:
            self.projection = self._make_projection(feat_dim, embedding_dim)

        # In "raw" embedding mode the (pretrained) backbone features are the
        # embeddings, so the classifier reads the backbone dimension.
        embed_out_dim = feat_dim if embedding_mode == "raw" else int(embedding_dim)
        self.classifier = nn.Linear(embed_out_dim, n_classes) if n_classes else None

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.freeze(True)
            self.backbone.unfreeze_stage(unfreeze_stage)

    @staticmethod
    def _make_projection(feat_dim: int, embedding_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )

    def project(self, feat: torch.Tensor, modality: str) -> torch.Tensor:
        """Project backbone features through the (possibly per-modality) head."""
        if isinstance(self.projection, nn.ModuleDict):
            return self.projection[modality](feat)
        return self.projection(feat)

    def encode_features(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """Raw backbone features (pre-projection), shape (B, feat_dim)."""
        x = self.adapters[modality](x)
        return self.backbone.forward_features(x)

    def embed(self, x: torch.Tensor, modality: str, normalize: bool = True) -> torch.Tensor:
        """Embedding in the shared space (projected or raw backbone features)."""
        feat = self.encode_features(x, modality)
        if self.embedding_mode == "raw":
            e = feat
        else:
            e = self.project(feat, modality)
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
    modality_channels: Optional[Dict[str, int]] = None,
    image_size: Optional[int] = None,
    projection_heads: str = "shared",
    foundation: Optional[Dict] = None,
    embedding_mode: str = "projected",
) -> ModalityAdaptiveEncoder:
    return ModalityAdaptiveEncoder(
        modalities=modalities,
        backbone=backbone,
        pretrained=pretrained,
        embedding_dim=embedding_dim,
        n_classes=n_classes,
        freeze_backbone=freeze_backbone,
        unfreeze_stage=unfreeze_stage,
        modality_channels=modality_channels,
        image_size=image_size,
        projection_heads=projection_heads,
        foundation=foundation,
        embedding_mode=embedding_mode,
    )
