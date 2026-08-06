"""BigEarthNet-MM loader -- paired Sentinel-1 / Sentinel-2 multi-label patches.

BigEarthNet-MM contains 590,326 co-registered 120x120 Sentinel-1 (2-band) and
Sentinel-2 (12-band) patches over 10 European countries with multi-label CORINE
land-cover annotations (19-class or 43-class scheme).

Downloaded by the user (not automatically). Expected layout::

    <root>/BigEarthNet-S1/{patch_id}/{patch_id}_S1.tif     # 2-band SAR
    <root>/BigEarthNet-S2/{patch_id}/{patch_id}_S2.tif     # 12-band optical
    <root>/BigEarthNet_19_labels.csv                      # patch_id,label1,label2,...

Labels are **multi-label**; the system's single-label pipeline uses the first
present label in canonical class order as the primary label, and the full set
is preserved in ``metadata.extra["labels"]``.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np

from .interface import DatasetInterface, DatasetNotFound, register_dataset

# Canonical 19-class Level-3 CORINE labels (order defines "first present").
BIGEARTHNET_19_CLASSES: List[str] = [
    "Agroforestry areas",
    "Annual crops associated with permanent crops",
    "Broad-leaved forest",
    "Complex cultivation patterns",
    "Coniferous forest",
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Fruit trees",
    "Industrial or commercial units",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Mixed forest",
    "Moors and heathland",
    "Olive groves",
    "Pastures",
    "Permanent crops",
    "Rice fields",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Vineyards",
]

S2_DN_MAX = 10000.0
S1_DN_MAX = 32767.0

# Sentinel-2 band order in BigEarthNet-MM ``_S2.tif``: B1..B12.
S2_RGB_BANDS = [3, 2, 1]   # B4, B3, B2
S2_MS_BANDS_DEFAULT = [1, 2, 3, 4, 5, 7, 8, 11]  # B2,B3,B4,B5,B6,B8,B8A,B11


def _read_tif(path: str) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "BigEarthNet-MM loading requires 'tifffile'. Install it with: "
            "pip install -r requirements-real-data.txt"
        ) from exc
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        return arr[None].astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
            return arr.astype(np.float32)
        return arr.transpose(2, 0, 1).astype(np.float32)
    raise ValueError(f"unexpected tif shape {arr.shape} for {path}")


def _parse_labels_csv(path: str) -> Dict[str, List[str]]:
    """patch_id -> sorted label list (multi-label)."""
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or not row[0].strip():
                continue
            pid = row[0].strip()
            labels = [c.strip() for c in row[1:] if c and c.strip()]
            if pid:
                out[pid] = labels
    return out


@register_dataset("bigearthnet_mm")
class BigEarthNetMMDataset(DatasetInterface):
    """BigEarthNet-MM paired Sentinel-1 / Sentinel-2 multi-label dataset."""

    name = "bigearthnet_mm"
    dataset_id = "bigearthnet_mm"
    sensor = "Sentinel-1 + Sentinel-2"
    downloads_required = True
    _MODALITY_SENSOR = {
        "optical": "Sentinel-2",
        "multispectral": "Sentinel-2",
        "sar": "Sentinel-1",
    }

    @classmethod
    def load(cls, cfg: Dict, logger=None) -> "BigEarthNetMMDataset":
        from .metadata import ImageMetadata

        ds_cfg = cfg.get("dataset", {})
        root = str(ds_cfg.get("root", "data/raw"))
        ben_cfg = ds_cfg.get("bigearthnet_mm", {}) or {}
        s1_dir = os.path.join(root, "BigEarthNet-S1")
        s2_dir = os.path.join(root, "BigEarthNet-S2")
        labels_csv = str(ben_cfg.get("labels_csv") or os.path.join(root, "BigEarthNet_19_labels.csv"))
        image_size = int(ds_cfg.get("image_size", 64))
        max_patches = int(ben_cfg.get("max_patches") or 0) or None
        seed = int(ds_cfg.get("seed", 42))
        available = [m for m in cfg.get("modalities", ["optical", "sar"]) if m in ("optical", "multispectral", "sar")]

        if not (os.path.isdir(s1_dir) and os.path.isdir(s2_dir)):
            raise DatasetNotFound(
                f"BigEarthNet-MM data not found under '{root}'",
                hint=(
                    "BigEarthNet-MM is a ~20 GB download and is never fetched automatically.\n"
                    "  * Download from https://zenodo.org/record/6160062\n"
                    "  * Unpack so that <root>/BigEarthNet-S1 and <root>/BigEarthNet-S2 exist\n"
                    "  * Provide the 19-label CSV via dataset.bigearthnet_mm.labels_csv\n"
                    "The loader falls back to the synthetic dataset until real data is present."
                ),
            )
        if not os.path.exists(labels_csv):
            raise DatasetNotFound(
                f"BigEarthNet-MM labels CSV not found: {labels_csv}",
                hint="Set dataset.bigearthnet_mm.labels_csv to the BigEarthNet_19_labels.csv path.",
            )

        patch_map = _parse_labels_csv(labels_csv)
        class_names = BIGEARTHNET_19_CLASSES
        class_index = {c: i for i, c in enumerate(class_names)}

        # Discover patch ids that exist in both S1 and S2 folders.
        s2_ids = sorted(d for d in os.listdir(s2_dir) if os.path.isdir(os.path.join(s2_dir, d)))
        s1_ids = set(os.listdir(s1_dir))
        common = [pid for pid in s2_ids if pid in s1_ids and pid in patch_map]
        if max_patches:
            rng = np.random.RandomState(seed)
            common = list(rng.choice(common, size=min(max_patches, len(common)), replace=False))
        if not common:
            raise DatasetNotFound(
                f"BigEarthNet-MM: no paired labelled patches found under '{root}'",
                hint="Verify BigEarthNet-S1/S2 folder names match the labels CSV patch ids.",
            )

        optical: List[np.ndarray] = []
        multispectral: List[np.ndarray] = []
        sar: List[np.ndarray] = []
        labels: List[int] = []
        metadata: List[ImageMetadata] = []

        for pid in common:
            s2_path = os.path.join(s2_dir, pid, f"{pid}_S2.tif")
            s1_path = os.path.join(s1_dir, pid, f"{pid}_S1.tif")
            if not (os.path.exists(s2_path) and os.path.exists(s1_path)):
                continue
            s2 = _read_tif(s2_path)
            s1 = _read_tif(s1_path)
            # Downsample the 120x120 patch to image_size via centre crop+resize.
            h, w = s2.shape[1], s2.shape[2]
            y0, x0 = max(0, (h - image_size) // 2), max(0, (w - image_size) // 2)
            s2 = s2[:, y0 : y0 + image_size, x0 : x0 + image_size]
            s1 = s1[:, y0 : y0 + image_size, x0 : x0 + image_size]

            label_set = [l for l in patch_map.get(pid, []) if l in class_index]
            if not label_set:
                continue
            primary = min(label_set, key=lambda l: class_index[l])
            idx = int(np.argmin([class_index[l] for l in label_set]))
            label_idx = class_index[label_set[idx]]

            if "optical" in available:
                rgb = s2[S2_RGB_BANDS] / S2_DN_MAX
                optical.append(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
            if "multispectral" in available:
                ms = s2[S2_MS_BANDS_DEFAULT] / S2_DN_MAX
                multispectral.append(np.clip(ms, 0, 1).astype(np.float32))
            if "sar" in available and s1.shape[0] >= 2:
                sar.append(np.clip(s1[:2] / S1_DN_MAX, 0, 2.5).astype(np.float32))

            m = ImageMetadata(
                image_id=len(labels),
                dataset="bigearthnet_mm",
                sensor=cls.sensor,
                modality=None,
                latitude=None,
                longitude=None,
                acquisition_date=None,
                land_cover=class_names[label_idx],
                resolution=10.0,
                cloud_cover=None,
                orbit=None,
                file_path=s2_path,
                extra={"labels": label_set, "patch_id": pid},
            )
            labels.append(label_idx)
            metadata.append(m)

        if not labels:
            raise DatasetNotFound(
                f"BigEarthNet-MM: no readable paired patches under '{root}'",
                hint="Verify _S1.tif/_S2.tif band counts and the labels CSV.",
            )

        patches: Dict[str, np.ndarray] = {}
        if "optical" in available:
            patches["optical"] = np.stack(optical, axis=0)
        if "multispectral" in available:
            patches["multispectral"] = np.stack(multispectral, axis=0)
        if "sar" in available:
            patches["sar"] = np.stack(sar, axis=0)

        ds = cls(patches, np.asarray(labels, dtype=np.int64), class_names, metadata)
        ds.modality_sensor = {m: cls._MODALITY_SENSOR[m] for m in available}
        return ds
