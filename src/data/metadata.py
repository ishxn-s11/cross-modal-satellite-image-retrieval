"""Standardized per-image metadata for the retrieval system.

Every dataset in the system exposes its patches as :class:`ImageMetadata`
records so that retrieval results, the database layer, geographic/temporal
learning and the UI can all rely on one representation.

All fields are nullable: datasets are not required to contain every field. Use
``None`` for anything the source does not provide (e.g. the synthetic generator
has no real cloud-cover; EuroSAT has no acquisition dates). The retrieval
result builders and the UI only show fields that are actually populated.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Metadata record
# ---------------------------------------------------------------------------


@dataclass
class ImageMetadata:
    """One record per image (patch = one geographic location).

    ``modality`` describes the *primary* sensor view for this image when the
    dataset is single-modality; for multi-modality datasets a separate
    per-modality map is kept on the :class:`DatasetInterface` (see
    ``modality_sensor``).
    """

    image_id: int = -1
    dataset: Optional[str] = None          # synthetic | eurosat | sen12ms | so2sat | bigearthnet_mm
    sensor: Optional[str] = None           # e.g. "Sentinel-1", "Sentinel-2", "simulated"
    modality: Optional[str] = None         # optical | multispectral | sar (primary view)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    acquisition_date: Optional[str] = None  # ISO date "YYYY-MM-DD" (or None)
    land_cover: Optional[str] = None        # class name / land-cover label
    resolution: Optional[float] = None       # metres per pixel (or None)
    cloud_cover: Optional[float] = None      # fraction 0..1 (or None)
    orbit: Optional[str] = None              # pass / track id (or None)
    file_path: Optional[str] = None          # on-disk source (or None)
    embedding_path: Optional[str] = None     # embedding artefact path (or None)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- dict / json helpers ------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("extra", None)
        d = {k: v for k, v in d.items() if v is not None}
        if self.extra:
            d.update(self.extra)
        return d

    def to_row(self) -> Dict[str, Any]:
        """SQLite-friendly row: all columns present, ``None`` for missing."""
        return {
            "image_id": self.image_id,
            "dataset": self.dataset,
            "sensor": self.sensor,
            "modality": self.modality,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "acquisition_date": self.acquisition_date,
            "land_cover": self.land_cover,
            "resolution": self.resolution,
            "cloud_cover": self.cloud_cover,
            "orbit": self.orbit,
            "file_path": self.file_path,
            "embedding_path": self.embedding_path,
        }

    def with_fields(self, **kwargs: Any) -> "ImageMetadata":
        return replace(self, **kwargs)


def metadata_fields() -> List[str]:
    return [f.name for f in fields(ImageMetadata)]


# ---------------------------------------------------------------------------
# Synthetic coordinates / dates
# ---------------------------------------------------------------------------

# A small set of plausible study regions. Coordinates are deterministic,
# documented *synthetic* scene placements used to demo geographic / temporal
# evaluation and the interactive map without requiring real data.
_SYNTHETIC_REGIONS = [
    # (name, lat_lo, lat_hi, lon_lo, lon_hi, acquisition_dates)
    ("Haryana", 28.6, 30.6, 75.4, 77.4, ["2018-05-14", "2018-09-02", "2019-02-10", "2019-11-21"]),
    ("MadhyaPradesh", 21.0, 23.5, 74.5, 78.0, ["2018-04-02", "2018-08-19", "2019-01-07", "2019-10-30"]),
    ("TamilNadu", 9.5, 13.0, 77.0, 80.5, ["2018-03-22", "2018-07-11", "2019-01-25", "2019-09-14"]),
]


def synthetic_geo_dates(
    n: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic (latitude, longitude, date_index, region) arrays.

    Returns ``(lat, lon, date_index, region_index)`` each shape (n,). Dates are
    drawn from a small per-region pool so the synthetic dataset demonstrates
    same-location / different-season sampling without any real imagery.
    """
    rng = np.random.RandomState(seed)
    region_idx = rng.randint(0, len(_SYNTHETIC_REGIONS), size=n)
    lat = np.zeros(n, dtype=np.float32)
    lon = np.zeros(n, dtype=np.float32)
    date_idx = np.zeros(n, dtype=np.int64)
    for i, ri in enumerate(region_idx):
        _name, lo_lat, hi_lat, lo_lon, hi_lon, dates = _SYNTHETIC_REGIONS[ri]
        lat[i] = rng.uniform(lo_lat, hi_lat)
        lon[i] = rng.uniform(lo_lon, hi_lon)
        date_idx[i] = rng.randint(0, len(dates))
    return lat, lon, date_idx, region_idx


def synthetic_date_strings(date_idx: np.ndarray, region_idx: np.ndarray) -> List[str]:
    dates: List[str] = []
    for i, di in enumerate(date_idx):
        _name, _a, _b, _c, _d, pool = _SYNTHETIC_REGIONS[int(region_idx[i])]
        dates.append(pool[int(di) % len(pool)])
    return dates


def metadata_from_arrays(
    image_id: np.ndarray,
    class_names: Sequence[str],
    labels: np.ndarray,
    dataset: str,
    sensor: Optional[str] = None,
    modality: Optional[str] = None,
    latitude: Optional[np.ndarray] = None,
    longitude: Optional[np.ndarray] = None,
    acquisition_date: Optional[Sequence[str]] = None,
    resolution: Optional[float] = None,
    cloud_cover: Optional[np.ndarray] = None,
    orbit: Optional[Sequence[str]] = None,
    file_path: Optional[Sequence[str]] = None,
) -> List[ImageMetadata]:
    """Build a metadata list from parallel arrays (all optional)."""
    n = int(len(image_id))
    out: List[ImageMetadata] = []
    for i in range(n):
        out.append(
            ImageMetadata(
                image_id=int(image_id[i]),
                dataset=dataset,
                sensor=sensor,
                modality=modality,
                latitude=float(latitude[i]) if latitude is not None else None,
                longitude=float(longitude[i]) if longitude is not None else None,
                acquisition_date=(
                    str(acquisition_date[i]) if acquisition_date is not None else None
                ),
                land_cover=str(class_names[int(labels[i])]),
                resolution=resolution,
                cloud_cover=float(cloud_cover[i]) if cloud_cover is not None else None,
                orbit=str(orbit[i]) if orbit is not None else None,
                file_path=str(file_path[i]) if file_path is not None else None,
            )
        )
    return out


def metadata_to_json(metadata: Sequence[ImageMetadata], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([m.to_dict() for m in metadata], fh, indent=2, default=_json_default)


def metadata_from_json(path: str) -> List[ImageMetadata]:
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [ImageMetadata(**{k: v for k, v in r.items() if k in ImageMetadata.__dataclass_fields__}) for r in rows]


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
