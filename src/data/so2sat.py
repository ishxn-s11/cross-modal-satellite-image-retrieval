"""So2Sat LCZ42 loader -- paired Sentinel-1 / Sentinel-2 urban land-use data.

So2Sat LCZ42 provides 424,331 co-registered 256x256 image patches over the
world's 42 largest cities with 17 local-climate-zone labels. Each patch has a
Sentinel-1 (2-band) and a Sentinel-2 (8-band) view -- aligned cross-modal data.

Downloaded by the user (not automatically). Official HDF5 split files
``training.h5`` / ``validation.h5`` / ``testing.h5`` contain:

    sen1      (N, 256, 256, 2)  Sentinel-1 VV/VH magnitude, uint16
    sen2      (N, 256, 256, 8)  Sentinel-2 B2,B3,B4,B8,B11,B12,B5,B6 (uint16)
    label_idx (N,)              17 LCZ classes (0-16)

Requires ``h5py`` (optional dependency; see requirements-real-data.txt).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .interface import DatasetInterface, DatasetNotFound, register_dataset

SO2SAT_CLASS_NAMES: List[str] = [
    "Compact high-rise",
    "Compact mid-rise",
    "Compact low-rise",
    "Open high-rise",
    "Open mid-rise",
    "Open low-rise",
    "Lightweight low-rise",
    "Large low-rise",
    "Sparsely built",
    "Heavy industry",
    "Dense trees",
    "Scattered trees",
    "Bush/scrub",
    "Low plants",
    "Paved area",
    "Bare soil",
    "Water",
]

S2_DN_MAX = 10000.0
S1_DN_MAX = 32767.0

# Sentinel-2 band order inside So2Sat ``sen2``.
S2_RGB_BANDS = [1, 0, 0]  # not used directly: So2Sat has no true RGB bands
# So2Sat sen2 band list (official order): B2, B3, B4, B8, B11, B12, B5, B6.
# Map a sensible RGB composite from B4(red), B3(green), B2(blue).
S2_RGB_IDX = [2, 1, 0]
S2_MS_IDX = [0, 1, 2, 3, 4, 5, 6, 7]  # all 8


@register_dataset("so2sat")
class So2SatDataset(DatasetInterface):
    """So2Sat LCZ42 paired Sentinel-1/Sentinel-2 dataset."""

    name = "so2sat"
    dataset_id = "so2sat"
    sensor = "Sentinel-1 + Sentinel-2"
    downloads_required = True
    _MODALITY_SENSOR = {
        "optical": "Sentinel-2",
        "multispectral": "Sentinel-2",
        "sar": "Sentinel-1",
    }

    @classmethod
    def load(cls, cfg: Dict, logger=None) -> "So2SatDataset":
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "So2Sat loading requires 'h5py'. Install it with: "
                "pip install -r requirements-real-data.txt"
            ) from exc

        from .metadata import ImageMetadata

        ds_cfg = cfg.get("dataset", {})
        root = str(ds_cfg.get("root", "data/raw"))
        so_cfg = ds_cfg.get("so2sat", {}) or {}
        h5_name = str(so_cfg.get("h5_file", "training.h5"))
        h5_path = os.path.join(root, "so2sat", h5_name)
        if not os.path.exists(h5_path):
            raise DatasetNotFound(
                f"So2Sat LCZ42 not found: {h5_path}",
                hint=(
                    "So2Sat LCZ42 is ~55 GB and is never fetched automatically.\n"
                    "  * Download training.h5/validation.h5/testing.h5 (and prism.h5)\n"
                    "  * Place them under <dataset.root>/so2sat/\n"
                    "  * Set dataset.name to 'so2sat'\n"
                    "The loader falls back to the synthetic dataset until real data is present."
                ),
            )
        image_size = int(ds_cfg.get("image_size", 64))
        max_patches = int(so_cfg.get("max_patches") or 0) or None
        seed = int(ds_cfg.get("seed", 42))
        available = [m for m in cfg.get("modalities", ["optical", "sar"]) if m in ("optical", "multispectral", "sar")]

        with h5py.File(h5_path, "r") as f:
            sen1 = np.asarray(f["sen1"])
            sen2 = np.asarray(f["sen2"])
            label_key = "label_idx" if "label_idx" in f else ("label" if "label" in f else None)
            if label_key is None:
                raise DatasetNotFound(f"So2Sat {h5_path}: no label dataset found")
            labels = np.asarray(f[label_key]).ravel()

        n = min(sen1.shape[0], sen2.shape[0], labels.shape[0])
        if max_patches:
            rng = np.random.RandomState(seed)
            idx = rng.choice(n, size=min(max_patches, n), replace=False)
            sen1, sen2, labels = sen1[idx], sen2[idx], labels[idx]
            n = len(idx)

        def _crop(x: np.ndarray) -> np.ndarray:
            # (N, H, W, C) -> (N, C, H, W), centre-cropped to image_size.
            h, w = x.shape[1], x.shape[2]
            y0, x0 = (h - image_size) // 2, (w - image_size) // 2
            x = x[:, y0 : y0 + image_size, x0 : x0 + image_size, :]
            return np.transpose(x, (0, 3, 1, 2))

        sar_raw = _crop(sen1)[:, :, :, :] if sen1.ndim == 4 else None
        optical_raw = _crop(sen2)

        patches: Dict[str, np.ndarray] = {}
        metadata = []
        if "optical" in available:
            rgb = optical_raw[:, S2_RGB_IDX] / S2_DN_MAX
            patches["optical"] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        if "multispectral" in available:
            ms = optical_raw[:, S2_MS_IDX] / S2_DN_MAX
            patches["multispectral"] = np.clip(ms, 0, 1).astype(np.float32)
        if "sar" in available and sar_raw is not None:
            patches["sar"] = np.clip(sar_raw[:, :2] / S1_DN_MAX, 0, 2.5).astype(np.float32)

        class_names = SO2SAT_CLASS_NAMES
        for i in range(n):
            cls_idx = int(labels[i])
            metadata.append(
                ImageMetadata(
                    image_id=i,
                    dataset="so2sat",
                    sensor=cls.sensor,
                    modality=None,
                    latitude=None,
                    longitude=None,
                    acquisition_date=None,
                    land_cover=class_names[cls_idx] if 0 <= cls_idx < len(class_names) else None,
                    resolution=10.0,
                    cloud_cover=None,
                    orbit=None,
                    file_path=h5_path,
                )
            )
        ds = cls(patches, np.asarray(labels, dtype=np.int64), class_names, metadata)
        ds.modality_sensor = {m: cls._MODALITY_SENSOR[m] for m in available}
        return ds
