"""Helpers to render a modality patch as a displayable RGB image."""

from __future__ import annotations

from typing import Dict

import numpy as np


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return ((x - lo) / (hi - lo) * 255.0).astype(np.uint8)


def render_patch(patches: Dict[str, np.ndarray], idx: int, modality: str) -> np.ndarray:
    """Return an (H, W, 3) uint8 image for a given patch/modality."""
    arr = patches[modality][idx]
    if arr.ndim != 3:
        raise ValueError("expected (C,H,W)")
    c, h, w = arr.shape  # noqa: F841 (c kept for clarity)

    if modality == "optical":
        rgb = arr.transpose(1, 2, 0)[:, :, :3]
        return np.clip(rgb.astype(np.float32), 0, 255).astype(np.uint8)
    if modality == "multispectral":
        # Display the Red / Green / Blue composite via bands [2, 1, 0].
        rgb = arr[[2, 1, 0]].transpose(1, 2, 0)
        return _to_uint8(rgb)
    # sar or any other single/multi-channel modality -> grayscale heatmap.
    gray = _to_uint8(arr[0])
    return np.stack([gray, gray, gray], axis=-1)