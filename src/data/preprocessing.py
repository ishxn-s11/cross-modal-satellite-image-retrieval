"""Per-modality preprocessing and normalisation utilities.

Each modality is stored in its own native dynamic range:

  * ``optical``       -- uint8 RGB, 0..255
  * ``multispectral`` -- float reflectance, ~0..1
  * ``sar``           -- float SAR intensity, ~0..2.5

``normalize_scale`` maps every modality onto a common [0,1] scale; per-modality
mean/std statistics are then computed on that scale and used to standardise the
inputs before feeding the network.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

# Range caps used to bring each raw modality into a comparable [0,1] scale.
_SCALE_CAPS: Dict[str, float] = {
    "optical": 255.0,
    "multispectral": 1.0,
    "sar": 1.0,
}


def normalize_scale(patches: np.ndarray, modality: str) -> np.ndarray:
    """Map a modality's raw array onto a common [0,1] scale."""
    cap = _SCALE_CAPS.get(modality, 1.0)
    out = patches.astype(np.float32) / cap
    if modality == "sar":
        # SAR intensity can exceed 1; keep it bounded for stable statistics.
        out = np.clip(out, 0.0, 2.0)
    return out


def compute_normalization_stats(
    patches_by_modality: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-modality (mean, std) computed over all pixels on the [0,1] scale.

    Returns {modality: {"mean": (C,), "std": (C,)}}.
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for modality, arr in patches_by_modality.items():
        unit = normalize_scale(arr, modality)  # (N, C, H, W)
        mean = unit.mean(axis=(0, 2, 3)).astype(np.float32)
        std = unit.std(axis=(0, 2, 3)).astype(np.float32) + 1e-6
        stats[modality] = {"mean": mean, "std": std}
    return stats


def get_transform(
    modality: str, stats: Dict[str, np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable standardising a (C,H,W) [0,1]-scale array."""
    mean = stats["mean"]
    std = stats["std"]

    def _transform(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return x

    return _transform


def build_transforms(
    stats: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {modality: get_transform(modality, stats[modality]) for modality in stats}
