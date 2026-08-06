"""SEN12MS (SEN1-2) loader -- Sentinel-1 / Sentinel-2 paired land-cover data.

SEN12MS provides **co-registered** Sentinel-1 SAR (VV/VH) and Sentinel-2
optical (13 bands) patches with land-cover maps -- exactly the aligned
multi-sensor data the retrieval task assumes. It is downloaded by the user
(**never** automatically by this package) and read from disk.

Expected layout (official distribution)::

    <root>/
      ROIs/
        ROIs1868_summer.csv        # one CSV per season; lists paired scenes
        ROIs1868_fall.csv
        ...
      s1/{s1_scene_id}/scene.tif          # 2-band SAR (VV, VH), 256x256, uint16
      s2/{s2_scene_id}/scene.tif          # 13-band optical, 256x256, uint16
      lulc/{lulc_scene_id}/scene.tif      # 1-band land-cover map, 256x256
      s1_meta/{s1_scene_id}/scene_meta.json
      s2_meta/{s2_scene_id}/scene_meta.json

The ROI CSV links the three scene ids that cover the same area on each row, so
patch ``i`` of the s1 view and patch ``i`` of the s2 view are the same scene.

Requires the optional dependency ``tifffile`` (``pip install -r
requirements-real-data.txt``). When the data directory is absent the loader
raises :class:`~src.data.interface.DatasetNotFound` with download instructions
and the factory falls back to the synthetic dataset (``allow_fallback: true``).
"""

from __future__ import annotations

import csv
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .interface import DatasetInterface, DatasetNotFound, register_dataset
from .metadata import ImageMetadata

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default 17-class IGBP-style legend used by SEN12MS ``lulc`` maps. The exact
# legend shipped with a given download can override this via
# ``dataset.sen12.class_names`` (a list of 17 names, index == tif value).
SEN12_CLASS_NAMES: List[str] = [
    "Evergreen Needleleaf Forests",
    "Evergreen Broadleaf Forests",
    "Deciduous Needleleaf Forests",
    "Deciduous Broadleaf Forests",
    "Mixed Forests",
    "Closed Shrublands",
    "Open Shrublands",
    "Woody Savannas",
    "Savannas",
    "Grasslands",
    "Permanent Wetlands",
    "Croplands",
    "Urban and Built-up Lands",
    "Cropland/Natural Vegetation Mosaics",
    "Snow and Ice",
    "Barren",
    "Water Bodies",
]

# Sentinel-2 band indices (0-based) mapped to the system's modalities.
#   RGB composite for ``optical``        : B4, B3, B2
#   8-band stack for ``multispectral``   : B2 B3 B4 B5 B6 B8 B8A B11
# (mirrors the synthetic 8-band Blue..SWIR1 semantics)
S2_RGB_BANDS = [3, 2, 1]
S2_MS_BANDS_DEFAULT = [1, 2, 3, 4, 5, 7, 8, 11]

# Sentinel-1 bands: [0] = VV, [1] = VH.
S1_BANDS = ["VV", "VH"]

# Radiometric scaling so values land in the ranges the pipeline expects
# (optical: uint8 0..255; multispectral: reflectance ~0..1; sar: intensity).
S2_DN_MAX = 10000.0   # S2 L1C reflectance DN scale
S1_DN_MAX = 32767.0   # S1 stored DN scale (16-bit)


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------


def _read_tif(path: str) -> np.ndarray:
    """Read a (possibly multi-band) tif into a (C, H, W) float32 array."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "SEN12MS loading requires 'tifffile'. Install it with: "
            "pip install -r requirements-real-data.txt"
        ) from exc
    arr = tifffile.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[None].astype(np.float32)  # (1, H, W)
    if arr.ndim == 3:
        # Planar (C, H, W) vs interleaved (H, W, C): assume planar when the
        # first axis is band-like (small, and smaller than the last axis).
        if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 16:
            return arr.astype(np.float32)
        return arr.transpose(2, 0, 1).astype(np.float32)
    raise ValueError(f"unexpected tif shape {arr.shape} for {path}")


def _crop_patches(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Non-overlapping grid crop of a (C, H, W) scene -> (K, C, P, P)."""
    c, h, w = arr.shape
    if h < patch_size or w < patch_size:
        raise ValueError(f"scene {arr.shape} smaller than patch {patch_size}")
    n_h, n_w = h // patch_size, w // patch_size
    out = np.zeros((n_h * n_w, c, patch_size, patch_size), dtype=arr.dtype)
    k = 0
    for i in range(n_h):
        for j in range(n_w):
            out[k] = arr[:, i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
            k += 1
    return out


def _find_roi_csv(root: str) -> str:
    """Locate a single ROI CSV under ``root/ROIs/`` (prefer summer)."""
    candidates = sorted(glob.glob(os.path.join(root, "ROIs", "*.csv")))
    if not candidates:
        raise DatasetNotFound(
            f"SEN12MS: no ROI CSV found under {os.path.join(root, 'ROIs')}",
            hint=("Place the SEN12MS download under the configured root. "
                  "Expected ROIs/ROIs*.csv + s1/ + s2/ + lulc/ folders."),
        )
    for preferred in ("summer",):
        for p in candidates:
            if preferred in os.path.basename(p).lower():
                return p
    return candidates[0]


def _parse_roi_csv(path: str) -> List[Tuple[str, str, str]]:
    """Parse an ROI CSV into (s1_scene_id, s2_scene_id, lulc_scene_id) rows.

    Values may be bare scene ids or paths ending in ``/scene.tif``; both are
    normalised to the scene folder name.
    """
    rows: List[Tuple[str, str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return rows
        col_idx: Dict[str, int] = {
            name.strip().lower(): i for i, name in enumerate(header)
        }

        def pick(row, key) -> Optional[str]:
            i = col_idx.get(key.lower())
            return _scene_id(row[i]) if i is not None and i < len(row) and row[i] else None

        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            s1 = pick(row, "s1_name") or pick(row, "s1")
            s2 = pick(row, "s2_name") or pick(row, "s2")
            lulc = pick(row, "lulc_name") or pick(row, "lulc")
            if s1 and s2 and lulc:
                rows.append((s1, s2, lulc))
    return rows


def _scene_id(value: str) -> str:
    """Normalise a CSV cell to a bare scene folder id."""
    v = value.strip().rstrip("/").replace("\\", "/")
    parts = [p for p in v.split("/") if p]
    if not parts:
        return v
    if parts[-1].endswith((".tif", ".npy", ".json")):
        parts = parts[:-1]
    return parts[-1]


def _dominant_class(lulc_patch: np.ndarray) -> int:
    vals, counts = np.unique(lulc_patch.astype(np.int64), return_counts=True)
    return int(vals[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# Metadata extraction (best effort -- any failure degrades to None)
# ---------------------------------------------------------------------------


def _scene_meta(root: str, s2_id: str) -> dict:
    """Read s2 scene_meta.json, tolerating a missing/invalid file."""
    path = os.path.join(root, "s2_meta", s2_id, "scene_meta.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _patch_centers(meta: dict, n_patches: int) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Approximate patch lat/lon centers from the scene's geometries."""
    lats: List[Optional[float]] = [None] * n_patches
    lons: List[Optional[float]] = [None] * n_patches
    geoms = meta.get("geometries") if isinstance(meta, dict) else None
    if not geoms:
        return lats, lons
    for g in geoms[:n_patches]:
        coords = (g or {}).get("coordinates")
        pid = ((g or {}).get("properties") or {}).get("patch_id")
        if not coords or pid is None:
            continue
        try:
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            lon, lat = float(np.mean(xs)), float(np.mean(ys))
            idx = int(pid)
            if 0 <= idx < n_patches:
                lats[idx], lons[idx] = lat, lon
        except Exception:
            continue
    return lats, lons


def _iso_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = str(raw).strip()
    # e.g. "20150601T170406" or "2015-06-01"
    s = s.replace("T", " ").split(" ")[0].replace("-", "")
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return raw if len(raw) >= 10 else None


# ---------------------------------------------------------------------------
# Dataset backend
# ---------------------------------------------------------------------------


@register_dataset("sen12ms")
class SEN12Dataset(DatasetInterface):
    """SEN12MS paired Sentinel-1 / Sentinel-2 retrieval dataset.

    Modalities exposed (subset via config, default ``[optical, sar]``):
      * ``optical``       -- S2 true-colour RGB composite (B4/B3/B2), uint8
      * ``multispectral`` -- configurable S2 band subset (default 8 bands), float
      * ``sar``           -- S1 VV/VH intensity (or a single band), float

    Labels are the dominant IGBP-style land-cover class per patch; metadata
    (lat/lon, acquisition date, cloud cover) is parsed from ``s2_meta`` when
    present and left ``None`` otherwise.
    """

    name = "sen12ms"
    dataset_id = "sen12ms"
    sensor = "Sentinel-1 + Sentinel-2"
    downloads_required = True
    _MODALITY_SENSOR = {
        "optical": "Sentinel-2",
        "multispectral": "Sentinel-2",
        "sar": "Sentinel-1",
    }

    @classmethod
    def load(cls, cfg: Dict, logger=None) -> "SEN12Dataset":
        ds_cfg = cfg.get("dataset", {})
        root = str(ds_cfg.get("root", "data/raw"))
        sen_cfg = ds_cfg.get("sen12", {}) or {}
        patch_size = int(sen_cfg.get("patch_size", 64))
        max_scenes = int(sen_cfg.get("max_scenes") or 0) or None
        requested = list(cfg.get("modalities", ["optical", "sar"]))
        available = [m for m in requested if m in ("optical", "multispectral", "sar")]
        if not available:
            raise ValueError(f"SEN12MS cannot provide any of the requested modalities {requested}")

        if not os.path.isdir(root) or not (os.path.isdir(os.path.join(root, "s2")) and os.path.isdir(os.path.join(root, "s1"))):
            raise DatasetNotFound(
                f"SEN12MS data not found under '{root}'",
                hint=(
                    "SEN12MS is a ~110 GB download and is never fetched automatically.\n"
                    "  * Download: https://mediatum.ub.tum.de/1474000 (register) or via Zenodo\n"
                    "  * Unpack so that <root>/s1, <root>/s2, <root>/lulc, <root>/ROIs exist\n"
                    "  * Set dataset.root to that folder and dataset.name to 'sen12ms'\n"
                    "The loader falls back to the synthetic dataset until real data is present."
                ),
            )

        # --- discover paired scenes ----------------------------------------
        roi_csv = sen_cfg.get("roi_csv") or _find_roi_csv(root)
        if not os.path.isabs(roi_csv):
            roi_csv = os.path.join(root, roi_csv)
        rows = _parse_roi_csv(roi_csv)
        if not rows:
            raise DatasetNotFound(
                f"SEN12MS: no paired scenes parsed from {roi_csv}",
                hint="Check the CSV column names (expected s1_name/s2_name/lulc_name).",
            )
        if max_scenes:
            rows = rows[:max_scenes]
        if logger is not None:
            logger.info(f"[sen12ms] {len(rows)} scenes from {os.path.basename(roi_csv)}")

        # --- class names ----------------------------------------------------
        class_names = list(sen_cfg.get("class_names") or SEN12_CLASS_NAMES)
        ms_bands = [int(b) for b in (sen_cfg.get("ms_bands") or S2_MS_BANDS_DEFAULT)]
        sar_bands = list(sen_cfg.get("sar_bands") or S1_BANDS)

        # --- per-scene patch accumulation ------------------------------------
        optical: List[np.ndarray] = []
        multispectral: List[np.ndarray] = []
        sar: List[np.ndarray] = []
        labels: List[int] = []
        metadata: List[ImageMetadata] = []

        image_id = 0
        for s1_id, s2_id, lulc_id in rows:
            s2_path = os.path.join(root, "s2", s2_id, "scene.tif")
            s1_path = os.path.join(root, "s1", s1_id, "scene.tif")
            lulc_path = os.path.join(root, "lulc", lulc_id, "scene.tif")
            if not all(os.path.exists(p) for p in (s2_path, s1_path, lulc_path)):
                continue
            s2 = _read_tif(s2_path)
            s1 = _read_tif(s1_path)
            lulc = _read_tif(lulc_path)
            if s2.shape[1:] != lulc.shape[1:]:
                continue
            n_patches = (s2.shape[1] // patch_size) * (s2.shape[2] // patch_size)
            s2_p = _crop_patches(s2, patch_size)
            s1_p = _crop_patches(s1, patch_size)
            lulc_p = _crop_patches(lulc, patch_size)
            n_patches = min(s2_p.shape[0], s1_p.shape[0], lulc_p.shape[0])

            lats, lons = _patch_centers(_scene_meta(root, s2_id), n_patches)
            meta = _scene_meta(root, s2_id)
            date = _iso_date(meta.get("acquisition_date"))
            cloud = meta.get("cloud_coverage")

            for k in range(n_patches):
                rgb = s2_p[k, S2_RGB_BANDS] / S2_DN_MAX          # (3, P, P) float
                rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
                if "optical" in available:
                    optical.append(rgb)
                if "multispectral" in available:
                    ms = s2_p[k, ms_bands] / S2_DN_MAX           # (n_ms, P, P) float
                    multispectral.append(np.clip(ms, 0.0, 1.0).astype(np.float32))
                if "sar" in available:
                    sar_b = s1_p[k, : len(sar_bands)] / S1_DN_MAX
                    sar.append(np.clip(sar_b, 0.0, 2.5).astype(np.float32))
                cls_idx = _dominant_class(lulc_p[k, 0])
                labels.append(cls_idx)
                metadata.append(
                    ImageMetadata(
                        image_id=image_id,
                        dataset="sen12ms",
                        sensor=cls.sensor,
                        modality=None,
                        latitude=lats[k] if k < len(lats) else None,
                        longitude=lons[k] if k < len(lons) else None,
                        acquisition_date=date,
                        land_cover=(class_names[cls_idx] if 0 <= cls_idx < len(class_names) else None),
                        resolution=10.0,
                        cloud_cover=float(cloud) if cloud is not None else None,
                        orbit=None,
                        file_path=s2_path,
                    )
                )
                image_id += 1

        if not labels:
            raise DatasetNotFound(
                f"SEN12MS: no readable patches from {len(rows)} scenes under '{root}'",
                hint="Verify the s1/s2/lulc subfolder layout and scene.tif names.",
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
