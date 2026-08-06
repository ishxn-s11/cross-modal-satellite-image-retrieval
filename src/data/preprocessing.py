"""Per-modality preprocessing and input normalisation.

There are two stages:

1. **Patch preprocessing** (:func:`preprocess_patches`) -- a configurable,
   modality-aware pass over the *raw* loaded arrays. SAR-specific handling
   (log transform, clipping, speckle reduction, invalid-value repair),
   optical invalid-pixel repair, multispectral band selection and resizing all
   live here. With the default (identity) configuration the arrays pass through
   unchanged, so legacy behaviour is preserved.

2. **Input normalisation** (:func:`build_transforms`) -- maps every modality
   onto a shared [0,1] scale and standardises it with per-modality mean/std
   statistics. The transform always scales to [0,1] *first* so the values the
   network sees are consistent with the statistics they were normalised with
   (this was a latent defect before -- optical uint8 values were being fed
   directly against stats computed on the [0,1] scale).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

# Range caps used to bring each raw modality into a comparable [0,1] scale.
_SCALE_CAPS: Dict[str, float] = {
    "optical": 255.0,
    "multispectral": 1.0,
    "sar": 1.0,
}

# Optional CLI surface: default configuration for the preprocessing pass.
_DEFAULTS: Dict = {
    "resize": None,
    "cloud_max": None,
    "sar": {"log_transform": False, "clip_min": None, "clip_max": None,
            "invalid_value": None, "invalid_fill": "zero",
            "speckle_filter": "none", "speckle_window": 3},
    "optical": {"clip_min": None, "clip_max": None,
                "invalid_value": None, "invalid_fill": "zero"},
    "multispectral": {"band_selection": None, "missing_bands": "raise",
                      "clip_min": None, "clip_max": None,
                      "invalid_value": None, "invalid_fill": "zero"},
}


def scale_to_unit(arr: np.ndarray, modality: str) -> np.ndarray:
    """Map a raw modality array onto the shared [0,1] scale used for stats.

    Mirrors the scaling applied when statistics are computed so the transform
    and the statistics are always consistent.
    """
    cap = _SCALE_CAPS.get(modality, 1.0)
    out = arr.astype(np.float32) / cap
    if modality == "sar":
        # SAR intensity can exceed 1; keep it bounded for stable statistics.
        out = np.clip(out, 0.0, 2.0)
    return out


def normalize_scale(patches: np.ndarray, modality: str) -> np.ndarray:
    """Map a modality's raw array onto a common [0,1] scale (legacy name)."""
    return scale_to_unit(patches, modality)


def compute_normalization_stats(
    patches_by_modality: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Per-modality (mean, std) computed over all pixels on the [0,1] scale.

    Returns {modality: {"mean": (C,), "std": (C,)}}.
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for modality, arr in patches_by_modality.items():
        unit = scale_to_unit(arr, modality)  # (N, C, H, W)
        mean = unit.mean(axis=(0, 2, 3)).astype(np.float32)
        std = unit.std(axis=(0, 2, 3)).astype(np.float32) + 1e-6
        stats[modality] = {"mean": mean, "std": std}
    return stats


def get_transform(
    modality: str,
    stats: Dict[str, np.ndarray],
    augmentation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable standardising a raw (C,H,W) array for the network.

    Optionally applies ``augmentation`` on the [0,1] scale between scaling and
    standardisation (used for the training set only).
    """
    mean = stats["mean"]
    std = stats["std"]

    def _transform(x: np.ndarray) -> np.ndarray:
        x = scale_to_unit(x, modality)
        if augmentation is not None:
            x = augmentation(x)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return x

    return _transform


def build_transforms(
    stats: Dict[str, Dict[str, np.ndarray]],
    augmentation: Optional[Dict[str, Callable[[np.ndarray], np.ndarray]]] = None,
) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Per-modality standardising transforms (with optional augmentation map)."""
    aug = augmentation or {}
    return {
        modality: get_transform(modality, stats[modality], aug.get(modality))
        for modality in stats
    }


# ---------------------------------------------------------------------------
# Stage 1: modality-aware patch preprocessing
# ---------------------------------------------------------------------------


def _replace_invalid(arr: np.ndarray, value, fill: str) -> np.ndarray:
    """Replace a sentinel value (or NaN) with a stable fill.

    Works on (N, C, H, W) arrays (channels are axis 1). ``fill`` in
    {"zero", "median", "nan_to_num"}.
    """
    if value is None:
        return arr
    out = arr.astype(np.float32, copy=True)
    if isinstance(value, str) and value.lower() == "nan":
        mask = np.isnan(out)
    else:
        mask = np.isclose(out, float(value), atol=0.0)
        if not mask.any():
            return arr
    if fill == "median":
        for c in range(out.shape[1]):
            vals = out[:, c]
            m = mask[:, c]
            valid = vals[~m]
            if valid.size:
                out[:, c][m] = float(np.median(valid))
    elif fill == "nan_to_num":
        out = np.nan_to_num(out, nan=0.0)
    else:  # zero
        out[mask] = 0.0
    return out


def _clip(arr: np.ndarray, lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    if lo is None and hi is None:
        return arr
    return np.clip(arr, lo if lo is not None else arr.min(), hi if hi is not None else arr.max())


def _resize(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize a (N, C, H, W) array to (N, C, size, size) with PIL bilinear."""
    from PIL import Image

    n, c = arr.shape[0], arr.shape[1]
    out = np.empty((n, c, size, size), dtype=arr.dtype)
    for i in range(n):
        for j in range(c):
            band = np.ascontiguousarray(arr[i, j]).astype(np.float32)
            pil = Image.fromarray(band, mode="F")
            out[i, j] = np.asarray(
                pil.resize((size, size), resample=Image.BILINEAR), dtype=arr.dtype
            )
    return out


def _lee_filter(band: np.ndarray, window: int = 3) -> np.ndarray:
    """3x3 (or window x window) Lee speckle filter on a 2-D SAR intensity band.

    The Lee filter estimates the "true" backscatter as
    ``k * observed + (1 - k) * local_mean`` where ``k`` shrinks with local
    variance -- smoothing speckle while preserving edges.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if band.ndim != 2:
        return band
    pad = window // 2
    padded = np.pad(band, pad, mode="reflect")
    views = sliding_window_view(padded, (window, window))
    local_mean = views.mean(axis=(2, 3))
    local_var = views.var(axis=(2, 3))
    global_var = float(band.var())
    if global_var < 1e-12:
        return band
    k = np.clip(local_var - global_var, 0.0, None) / np.clip(local_var, 1e-12, None)
    return (k * band + (1.0 - k) * local_mean).astype(band.dtype)


def _preprocess_sar(arr: np.ndarray, cfg: Dict) -> np.ndarray:
    out = _replace_invalid(arr, cfg.get("invalid_value"), cfg.get("invalid_fill", "zero"))
    speckle = cfg.get("speckle_filter", "none")
    if speckle == "lee":
        window = int(cfg.get("speckle_window", 3))
        bands = [_lee_filter(out[i, j], window) for i in range(out.shape[0]) for j in range(out.shape[1])]
        out = np.stack(bands, axis=0).reshape(out.shape)
    elif speckle not in ("none", ""):
        warnings.warn(f"unknown speckle_filter '{speckle}'; ignoring")
    if cfg.get("log_transform"):
        out = np.log1p(np.maximum(out, 0.0))
    return _clip(out, cfg.get("clip_min"), cfg.get("clip_max"))


def _preprocess_multispectral(arr: np.ndarray, cfg: Dict) -> np.ndarray:
    selection = cfg.get("band_selection")
    missing = cfg.get("missing_bands", "raise")
    if selection is not None:
        sel = [int(b) for b in selection]
        n_bands = arr.shape[1]
        invalid = [b for b in sel if not (0 <= b < n_bands)]
        if invalid:
            if missing == "raise":
                raise ValueError(
                    f"band_selection {sel} out of range for a {n_bands}-band stack; "
                    f"invalid band(s): {invalid}"
                )
            if missing == "warn":
                warnings.warn(f"dropping out-of-range bands {invalid} from band_selection")
            sel = [b for b in sel if 0 <= b < n_bands]
        if not sel:
            raise ValueError("band_selection produced an empty band set")
        arr = arr[:, sel]
    out = _replace_invalid(arr, cfg.get("invalid_value"), cfg.get("invalid_fill", "zero"))
    return _clip(out, cfg.get("clip_min"), cfg.get("clip_max"))


def _preprocess_optical(arr: np.ndarray, cfg: Dict) -> np.ndarray:
    out = _replace_invalid(arr, cfg.get("invalid_value"), cfg.get("invalid_fill", "zero"))
    return _clip(out, cfg.get("clip_min"), cfg.get("clip_max"))


def _modality_preprocess(arr: np.ndarray, modality: str, cfg: Dict) -> np.ndarray:
    if modality == "sar":
        return _preprocess_sar(arr, cfg)
    if modality == "multispectral":
        return _preprocess_multispectral(arr, cfg)
    if modality == "optical":
        return _preprocess_optical(arr, cfg)
    return arr


def preprocess_patches(
    patches: Dict[str, np.ndarray], cfg: Optional[Dict]
) -> Dict[str, np.ndarray]:
    """Apply the configurable modality-aware preprocessing pass to raw patches.

    Inputs are (N, C, H, W) per modality. With a ``None``/empty config (or
    default identity settings) the arrays are returned unchanged.
    """
    if not cfg:
        return patches
    resize = cfg.get("resize")
    out: Dict[str, np.ndarray] = {}
    for modality, arr in patches.items():
        arr = _modality_preprocess(arr, modality, cfg.get(modality, {}))
        if resize:
            arr = _resize(arr, int(resize))
        out[modality] = arr
    return out


def filter_images_by_cloud(
    patches: Dict[str, np.ndarray],
    labels: np.ndarray,
    metadata: Sequence,
    cloud_max: float,
) -> tuple:
    """Drop images whose cloud cover exceeds ``cloud_max`` (metadata-based).

    Images with unknown cloud cover (None) are kept. Returns
    ``(patches, labels, metadata)`` with the retained subset.
    """
    keep = [
        i for i, m in enumerate(metadata)
        if getattr(m, "cloud_cover", None) is None or float(m.cloud_cover) <= float(cloud_max)
    ]
    if len(keep) == len(labels):
        return patches, np.asarray(labels), list(metadata)
    kept = np.asarray(keep, dtype=np.int64)
    return (
        {m: arr[kept] for m, arr in patches.items()},
        np.asarray(labels)[kept],
        [metadata[i] for i in keep],
    )


@dataclass
class PreprocessingConfig:
    """Typed view of the ``preprocessing`` config section."""

    resize: Optional[int] = None
    cloud_max: Optional[float] = None
    sar: Dict = field(default_factory=dict)
    optical: Dict = field(default_factory=dict)
    multispectral: Dict = field(default_factory=dict)

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict]) -> "PreprocessingConfig":
        c = cfg or {}
        merged = {
            k: {**_DEFAULTS.get(k, {}), **(c.get(k, {}) or {})}
            for k in ("sar", "optical", "multispectral")
        }
        return cls(
            resize=int(c["resize"]) if c.get("resize") else None,
            cloud_max=float(c["cloud_max"]) if c.get("cloud_max") is not None else None,
            sar=merged["sar"],
            optical=merged["optical"],
            multispectral=merged["multispectral"],
        )
