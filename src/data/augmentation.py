"""Remote-sensing-aware data augmentation (numpy, applied on the [0,1] scale).

Only augmentations that preserve the semantics of satellite imagery are offered:

* random crop + resize back to the original size (scale-preserving),
* horizontal / vertical flip,
* rotation by multiples of 90 degrees (arbitrary angles would corrupt the
  axis-aligned geometry of remote-sensing scenes),
* additive gaussian noise,
* per-band multiplicative spectral jitter (multispectral only).

Each transform is optional and configured through the ``augmentation`` section.
With every setting at its default the chain is a no-op and the legacy training
behaviour is preserved.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np


def _random_crop(x: np.ndarray, crop_frac: float) -> np.ndarray:
    """Crop a random window of (1 - crop_frac) of the image, then resize back."""
    if crop_frac <= 0:
        return x
    _, h, w = x.shape
    new_h = max(1, int(round(h * (1.0 - crop_frac))))
    new_w = max(1, int(round(w * (1.0 - crop_frac))))
    y0 = int(np.random.randint(0, max(1, h - new_h + 1)))
    x0 = int(np.random.randint(0, max(1, w - new_w + 1)))
    crop = x[:, y0 : y0 + new_h, x0 : x0 + new_w]
    return _resize(crop, h, w)


def _resize(x: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image

    out = np.empty((x.shape[0], h, w), dtype=x.dtype)
    for i in range(x.shape[0]):
        pil = Image.fromarray(np.ascontiguousarray(x[i]), mode="F")
        out[i] = np.asarray(pil.resize((w, h), resample=Image.BILINEAR), dtype=x.dtype)
    return out


def _add_noise(x: np.ndarray, std: float) -> np.ndarray:
    if std <= 0:
        return x
    return np.clip(x + np.random.normal(0.0, std, size=x.shape).astype(np.float32), 0.0, 2.0)


def _spectral_jitter(x: np.ndarray, std: float) -> np.ndarray:
    """Per-band multiplicative jitter: ``x * (1 + N(0, std))`` per channel."""
    if std <= 0:
        return x
    scale = np.random.normal(1.0, std, size=(x.shape[0], 1, 1)).astype(np.float32)
    return np.clip(x * scale, 0.0, 2.0)


def build_augmentation(
    modality: str, aug_cfg: Dict
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Build the augmentation chain for one modality, or None if all disabled."""
    if not aug_cfg or not aug_cfg.get("enabled", False):
        return None

    crop_frac = float(aug_cfg.get("random_crop", 0.0) or 0.0)
    hflip = bool(aug_cfg.get("hflip", False))
    vflip = bool(aug_cfg.get("vflip", False))
    rot90 = bool(aug_cfg.get("rotation_90", False))
    noise_std = float(aug_cfg.get("noise_std", 0.0) or 0.0)
    spectral = float(aug_cfg.get("spectral_jitter", 0.0) or 0.0)

    steps = 0
    if crop_frac > 0:
        steps += 1
    if hflip or vflip or rot90 or noise_std > 0:
        steps += 1
    if modality == "multispectral" and spectral > 0:
        steps += 1
    if steps == 0:
        return None

    def _aug(x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32)
        if crop_frac > 0:
            out = _random_crop(out, crop_frac)
        if hflip and bool(np.random.rand() < 0.5):
            out = out[:, :, ::-1]
        if vflip and bool(np.random.rand() < 0.5):
            out = out[:, ::-1, :]
        if rot90:
            k = int(np.random.randint(0, 4))
            if k:
                out = np.rot90(out, k, axes=(1, 2))
        if noise_std > 0:
            out = _add_noise(out, noise_std)
        if modality == "multispectral" and spectral > 0:
            out = _spectral_jitter(out, spectral)
        return out

    return _aug
