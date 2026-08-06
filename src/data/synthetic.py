"""Self-contained multi-sensor satellite image generator.

Generates a paired multi-modal dataset in which every patch is a 2-D scene
described by the *same* underlying land-cover layout, rendered through three
distinct sensor models:

  * **optical**       -- a true-colour RGB composite (sRGB tone response,
                         atmosphere + sensor noise),
  * **multispectral** -- an 8-band reflectance stack (Blue..SWIR1) with
                         per-band radiance noise and spatial resolution blur,
  * **sar**           -- a single-channel SAR *intensity* image built from the
                         class-dependent backscatter coefficient, multiplicative
                         (Gamma) speckle and an incidence-angle shading.

All three modalities therefore share the exact same scene semantics while
carrying genuinely different radiometric information. This is the kind of
*aligned* multi-sensor data the retrieval task assumes (e.g. same/nearby
geographic location observed with optical, multispectral and SAR sensors).

The generator is deterministic (seeded) and caches its output to disk so that
repeated pipeline runs reuse the same dataset.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .interface import DatasetInterface, register_dataset
from .metadata import (
    metadata_from_arrays,
    synthetic_date_strings,
    synthetic_geo_dates,
)
from .modalities import MODALITIES, DEFAULT_MODALITIES, validate_modalities

# ---------------------------------------------------------------------------
# Spectral prior tables
# ---------------------------------------------------------------------------

CLASS_NAMES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Mean spectral reflectance (0..1) per class for the 8 multispectral bands.
_SPECTRAL_SIGNATURES: Dict[str, List[float]] = {
    "AnnualCrop": [0.06, 0.12, 0.10, 0.28, 0.55, 0.60, 0.58, 0.30],
    "Forest": [0.04, 0.08, 0.05, 0.18, 0.45, 0.52, 0.50, 0.20],
    "HerbaceousVegetation": [0.05, 0.10, 0.08, 0.24, 0.50, 0.55, 0.52, 0.25],
    "Highway": [0.14, 0.15, 0.15, 0.16, 0.16, 0.17, 0.17, 0.16],
    "Industrial": [0.18, 0.19, 0.19, 0.20, 0.21, 0.22, 0.22, 0.24],
    "Pasture": [0.05, 0.11, 0.09, 0.26, 0.52, 0.57, 0.54, 0.26],
    "PermanentCrop": [0.05, 0.10, 0.08, 0.22, 0.42, 0.47, 0.45, 0.22],
    "Residential": [0.16, 0.17, 0.16, 0.20, 0.24, 0.27, 0.28, 0.30],
    "River": [0.10, 0.14, 0.12, 0.10, 0.07, 0.05, 0.04, 0.03],
    "SeaLake": [0.08, 0.11, 0.09, 0.07, 0.04, 0.03, 0.02, 0.02],
}

# Mean SAR backscatter *intensity* (linear) per class.
_SAR_BACKSCATTER: Dict[str, float] = {
    "AnnualCrop": 0.45,
    "Forest": 0.60,
    "HerbaceousVegetation": 0.40,
    "Highway": 0.08,
    "Industrial": 0.70,
    "Pasture": 0.35,
    "PermanentCrop": 0.50,
    "Residential": 0.65,
    "River": 0.05,
    "SeaLake": 0.03,
}

# Per-class spatial texture strength (relative variance of reflectance).
_TEXTURE_STD: Dict[str, float] = {
    "AnnualCrop": 0.12,
    "Forest": 0.20,
    "HerbaceousVegetation": 0.14,
    "Highway": 0.06,
    "Industrial": 0.18,
    "Pasture": 0.10,
    "PermanentCrop": 0.16,
    "Residential": 0.22,
    "River": 0.04,
    "SeaLake": 0.03,
}

_MS_BANDS = MODALITIES["multispectral"]["channels"]  # 8 band names


# ---------------------------------------------------------------------------
# Low-level noise helpers
# ---------------------------------------------------------------------------

def _smooth_noise(rng: np.random.RandomState, size: int, low_res: int = 12) -> np.ndarray:
    """Bilinearly-upsampled random field -> smooth low-frequency noise."""
    coarse = rng.normal(0.0, 1.0, size=(low_res, low_res))
    img = Image.fromarray(coarse.astype(np.float32), mode="F").resize(
        (size, size), resample=Image.BILINEAR
    )
    return np.asarray(img, dtype=np.float32)


def _blocky_noise(rng: np.random.RandomState, size: int, block: int = 6) -> np.ndarray:
    """Tiled random field -> blocky / structured (urban-like) noise."""
    n = max(1, math.ceil(size / block))
    coarse = rng.normal(0.0, 1.0, size=(n, n))
    up = np.kron(coarse, np.ones((block, block), dtype=np.float32))
    return up[:size, :size]


def _crop_rows(rng: np.random.RandomState, size: int) -> np.ndarray:
    """Row-like striping to mimic agricultural row crops."""
    period = int(rng.uniform(4, 10))
    phase = rng.uniform(0, period)
    rows = np.cos(2.0 * np.pi * (np.arange(size) + phase) / period).astype(np.float32)
    return np.tile(rows[:, None], (1, size))


def _class_texture(class_name: str, size: int, rng: np.random.RandomState) -> np.ndarray:
    """Per-class spatial pattern multiplier (mean 1)."""
    std = _TEXTURE_STD.get(class_name, 0.1)
    if class_name in ("Residential", "Industrial", "Highway"):
        pat = _blocky_noise(rng, size, block=5)
    elif class_name in ("AnnualCrop", "PermanentCrop"):
        pat = _crop_rows(rng, size) * 0.6 + _smooth_noise(rng, size, 10) * 0.4
    else:
        pat = _smooth_noise(rng, size, 12)
    # Normalize so the mean multiplier is 1.
    pat = pat - pat.mean()
    return 1.0 + std * pat


def _illumination_gradient(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Smooth cross-track illumination shading shared by optical/MS views."""
    a = rng.uniform(0.85, 1.15)
    b = rng.uniform(0.85, 1.15)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    grad = a + (b - a) * (0.5 * yy + 0.5 * xx)
    return grad


def _incidence_shading(size: int, rng: np.random.RandomState) -> np.ndarray:
    """Range-direction fall-off applied to SAR intensity (look direction = x)."""
    xx = np.linspace(0.0, 1.0, size).astype(np.float32)
    shading = 1.0 - rng.uniform(0.1, 0.4) * xx
    return np.tile(shading[None, :], (size, 1))


# ---------------------------------------------------------------------------
# Patch layout
# ---------------------------------------------------------------------------

def _make_label_map(
    dominant_class: int,
    size: int,
    rng: np.random.RandomState,
    n_classes: int,
) -> np.ndarray:
    """A per-pixel class map whose majority (dominant) class labels the patch."""
    label_map = np.full((size, size), dominant_class, dtype=np.int32)
    if rng.rand() < 0.75:
        n_sub = int(rng.randint(1, 4))
        for _ in range(n_sub):
            other = int(rng.randint(0, n_classes - 1))
            other = other if other != dominant_class else (other + 1) % n_classes
            w, h = int(rng.uniform(0.15, 0.4) * size), int(rng.uniform(0.15, 0.4) * size)
            x0, y0 = int(rng.uniform(0, size - w)), int(rng.uniform(0, size - h))
            if rng.rand() < 0.4:  # rotated ellipse region
                yy, xx = np.mgrid[0:size, 0:size]
                cx, cy = x0 + w / 2, y0 + h / 2
                a, b = w / 2, h / 2
                angle = rng.uniform(0, np.pi)
                xr = np.cos(angle) * (xx - cx) + np.sin(angle) * (yy - cy)
                yr = -np.sin(angle) * (xx - cx) + np.cos(angle) * (yy - cy)
                mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
                label_map[mask] = other
            else:  # axis-aligned rectangle
                label_map[y0 : y0 + h, x0 : x0 + w] = other
    return label_map


# ---------------------------------------------------------------------------
# Per-modality rendering
# ---------------------------------------------------------------------------

def _render_multispectral(
    label_map: np.ndarray, size: int, rng: np.random.RandomState
) -> np.ndarray:
    """8-band reflectance stack, shape (8, size, size), values in [0, 1]."""
    signature = np.array([_SPECTRAL_SIGNATURES[CLASS_NAMES[c]] for c in range(len(CLASS_NAMES))])
    base = signature[label_map]  # (H, W, 8) reflectance per class
    # spatial texture per pixel
    dominant = int(np.bincount(label_map.ravel()).argmax())
    tex = _class_texture(CLASS_NAMES[dominant], size, rng)[..., None]  # (H, W, 1)
    illu = _illumination_gradient(size, rng)[..., None]
    # per-band sensor noise + small per-band radiometric offset
    noise = rng.normal(0.0, 0.012, size=base.shape).astype(np.float32)
    img = base * tex * illu + noise
    img = np.clip(img, 0.0, 1.0)
    return img.transpose(2, 0, 1).astype(np.float32)  # (8, H, W)


def _render_optical(ms: np.ndarray, size: int, rng: np.random.RandomState) -> np.ndarray:
    """True-colour RGB composite (uint8, 0..255), shape (3, size, size).

    Built from the multispectral Red/Green/Blue bands with a simple linear
    stretch, an sRGB-like gamma and mild atmospheric haze.
    """
    r = ms[2]
    g = ms[1]
    b = ms[0]
    rgb = np.stack([r, g, b], axis=0)  # (3, H, W)
    rgb = (rgb - 0.03) * 1.45  # contrast stretch
    # atmospheric path radiance (haze)
    haze = rng.uniform(0.02, 0.08)
    rgb = rgb + haze
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = np.power(rgb, 1.0 / 2.2)  # gamma -> display space
    rgb = np.clip(rgb * 255.0 + rng.normal(0, 2.0, size=rgb.shape), 0, 255)
    return rgb.astype(np.uint8)


def _render_sar(
    label_map: np.ndarray, size: int, rng: np.random.RandomState
) -> np.ndarray:
    """Single-channel SAR intensity, shape (1, size, size), float32 >= 0.

    intensity = backscatter * texture * speckle * incidence_shading
    Speckle is multiplicative Gamma noise with `looks` = 1 (fully developed).
    """
    mu = np.array([_SAR_BACKSCATTER[CLASS_NAMES[c]] for c in range(len(CLASS_NAMES))])
    base = mu[label_map]  # (H, W)
    dominant = int(np.bincount(label_map.ravel()).argmax())
    tex = _class_texture(CLASS_NAMES[dominant], size, rng)
    speckle = rng.gamma(1.0, 1.0, size=base.shape).astype(np.float32)  # exponential mean 1
    incidence = _incidence_shading(size, rng)
    img = base * tex * speckle * incidence
    img = np.clip(img, 0.0, 3.0)
    return img[None].astype(np.float32)  # (1, H, W)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    num_patches: int = 2000,
    image_size: int = 64,
    seed: int = 42,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    with_geo: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate `num_patches` paired multi-modal patches.

    Returns a dict {modality: ndarray of shape (N, bands, H, W), ...} plus a
    "labels" entry of shape (N,) with integer class ids. Per-class balanced.

    When ``with_geo`` is true the dict also carries deterministic *synthetic*
    scene placement arrays (``latitudes``, ``longitudes``, ``date_index``,
    ``region_index``) used to demo geographic/temporal evaluation and the
    interactive map without requiring any real imagery.
    """
    validate_modalities(list(modalities))
    rng = np.random.RandomState(seed)
    n_classes = len(CLASS_NAMES)

    out: Dict[str, np.ndarray] = {}
    for m in modalities:
        n_b = int(MODALITIES[m]["bands"])
        out[m] = np.zeros((num_patches, n_b, image_size, image_size), dtype=np.float32)
    out["labels"] = np.zeros((num_patches,), dtype=np.int64)

    # Balanced: assign dominant classes cyclically so each class is present.
    dominants = np.arange(num_patches) % n_classes
    rng.shuffle(dominants)

    for i in range(num_patches):
        label_map = _make_label_map(int(dominants[i]), image_size, rng, n_classes)
        ms = _render_multispectral(label_map, image_size, rng)
        if "multispectral" in modalities:
            out["multispectral"][i] = ms
        if "optical" in modalities:
            out["optical"][i] = _render_optical(ms, image_size, rng)
        if "sar" in modalities:
            out["sar"][i] = _render_sar(label_map, image_size, rng)
        out["labels"][i] = int(dominants[i])

    if with_geo:
        lat, lon, date_idx, region_idx = synthetic_geo_dates(num_patches, seed=seed)
        out["latitudes"] = lat
        out["longitudes"] = lon
        out["date_index"] = date_idx
        out["region_index"] = region_idx

    return out


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

# Keys in the cached npz that are not per-modality image stacks.
_NON_IMAGE_KEYS = {"labels", "latitudes", "longitudes", "date_index", "region_index"}


def save_synthetic_dataset(
    path: str, dataset: Dict[str, np.ndarray], class_names: Sequence[str]
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **dataset)
    first_key = next(k for k in dataset if k not in _NON_IMAGE_KEYS and np.ndim(dataset[k]) == 4)
    image_size = int(dataset[first_key].shape[-1])
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "class_names": list(class_names),
                "modalities": sorted(
                    k for k in dataset if k not in _NON_IMAGE_KEYS and np.ndim(dataset[k]) == 4
                ),
                "num_patches": int(dataset["labels"].shape[0]),
                "image_size": image_size,
            },
            fh,
            indent=2,
        )
    return meta_path


def load_synthetic_dataset(path: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    data = dict(np.load(path, allow_pickle=True))
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    class_names = CLASS_NAMES
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            class_names = json.load(fh)["class_names"]
    return data, class_names


def load_or_generate_synthetic(
    root: str,
    num_patches: int = 2000,
    image_size: int = 64,
    seed: int = 42,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Load cached synthetic data or generate + cache it."""
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, "synthetic", f"synthetic_{num_patches}_{image_size}_{seed}.npz")
    if os.path.exists(cache):
        return load_synthetic_dataset(cache)
    data = generate_synthetic_dataset(num_patches, image_size, seed, modalities)
    save_synthetic_dataset(cache, data, CLASS_NAMES)
    return data, CLASS_NAMES


# ---------------------------------------------------------------------------
# DatasetInterface backend
# ---------------------------------------------------------------------------

# Per-modality sensor labels used in result cards / database rows. Everything is
# *simulated* -- the synthetic generator renders sensor-like views of a scene.
_SYNTH_MODALITY_SENSOR = {
    "optical": "Sentinel-2 (simulated)",
    "multispectral": "Sentinel-2 (simulated)",
    "sar": "Sentinel-1 (simulated)",
}


@register_dataset("synthetic")
class SyntheticDataset(DatasetInterface):
    """Self-contained, fully offline synthetic multi-sensor dataset."""

    name = "synthetic"
    dataset_id = "synthetic"
    sensor = "simulated"
    downloads_required = False

    @classmethod
    def load(cls, cfg: Dict, logger=None) -> "SyntheticDataset":
        ds_cfg = cfg.get("dataset", {})
        modalities = list(cfg.get("modalities", DEFAULT_MODALITIES))
        data, class_names = load_or_generate_synthetic(
            root=ds_cfg.get("root", "data/raw"),
            num_patches=int(ds_cfg.get("num_patches", 2000)),
            image_size=int(ds_cfg.get("image_size", 64)),
            seed=int(ds_cfg.get("seed", 42)),
            modalities=modalities,
        )
        labels = np.asarray(data["labels"], dtype=np.int64)
        patches = {m: data[m] for m in modalities}

        # Optional deterministic synthetic scene placement.
        lat = data.get("latitudes")
        lon = data.get("longitudes")
        date_idx = data.get("date_index")
        region_idx = data.get("region_index")
        dates = None
        if date_idx is not None and region_idx is not None:
            dates = synthetic_date_strings(
                np.asarray(date_idx), np.asarray(region_idx)
            )
        metadata = metadata_from_arrays(
            image_id=np.arange(len(labels)),
            class_names=class_names,
            labels=labels,
            dataset="synthetic",
            sensor=cls.sensor,
            modality=None,
            latitude=lat,
            longitude=lon,
            acquisition_date=dates,
            resolution=10.0,  # nominal simulated GSD
        )
        ds = cls(patches, labels, class_names, metadata)
        ds.modality_sensor = {
            m: _SYNTH_MODALITY_SENSOR.get(m, cls.sensor) for m in modalities
        }
        return ds
