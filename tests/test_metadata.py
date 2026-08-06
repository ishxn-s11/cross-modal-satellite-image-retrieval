"""Unit tests for the standardized metadata representation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.metadata import (
    ImageMetadata,
    metadata_from_arrays,
    metadata_to_json,
    synthetic_date_strings,
    synthetic_geo_dates,
)


def test_to_dict_filters_none():
    m = ImageMetadata(image_id=3, dataset="synthetic", land_cover="Forest")
    d = m.to_dict()
    assert d["image_id"] == 3
    assert d["land_cover"] == "Forest"
    assert "latitude" not in d  # None fields are omitted


def test_to_row_keeps_none():
    m = ImageMetadata(image_id=1, dataset="x")
    r = m.to_row()
    assert r["image_id"] == 1
    assert r["latitude"] is None
    assert r["acquisition_date"] is None


def test_metadata_from_arrays():
    n = 10
    labels = np.arange(n) % 4
    names = ["A", "B", "C", "D"]
    lat = np.linspace(20, 30, n)
    lon = np.linspace(70, 80, n)
    meta = metadata_from_arrays(
        np.arange(n), names, labels, "synth", sensor="S1", latitude=lat, longitude=lon
    )
    assert len(meta) == n
    assert meta[2].latitude == float(lat[2])
    assert meta[2].longitude == float(lon[2])
    assert meta[2].land_cover == names[labels[2]]
    assert meta[2].acquisition_date is None  # missing -> nullable


def test_synthetic_geo_dates_deterministic():
    a1, b1, c1, d1 = synthetic_geo_dates(100, seed=1)
    a2, b2, c2, d2 = synthetic_geo_dates(100, seed=1)
    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
    assert np.array_equal(c1, c2)
    assert np.array_equal(d1, d2)
    assert len(c1) == 100 and len(d1) == 100
    dates = synthetic_date_strings(c1, d1)
    assert len(dates) == 100
    assert all("-" in s and s.count("-") == 2 for s in dates)


def test_metadata_json_roundtrip(tmp_path):
    meta = [ImageMetadata(image_id=0, dataset="synthetic", land_cover="Forest"),
            ImageMetadata(image_id=1, dataset="synthetic", land_cover="Water", latitude=12.5)]
    p = os.path.join(str(tmp_path), "meta.json")
    metadata_to_json(meta, p)
    import json

    with open(p, encoding="utf-8") as fh:
        rows = json.load(fh)
    assert len(rows) == 2
    assert rows[1]["latitude"] == 12.5


if __name__ == "__main__":
    test_to_dict_filters_none()
    test_to_row_keeps_none()
    test_metadata_from_arrays()
    test_synthetic_geo_dates_deterministic()
    test_metadata_json_roundtrip(__import__("tempfile").mkdtemp())
    print("test_metadata.py: all tests passed")
