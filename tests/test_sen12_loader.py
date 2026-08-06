"""Fixture-based test for the SEN12MS loader.

Builds a tiny SEN12MS-like directory tree (a single paired scene) and asserts
the loader produces correctly-shaped, correctly-paired patches and metadata.
"""

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.data.interface import DatasetNotFound, build_dataset
from src.utils.config import DEFAULT_CONFIG, deep_merge

S1_ID = "S1A_IW_GRDH_1SDV_20150601T170403"
S2_ID = "S2A_MSIL1C_20150601T170406"
LULC_ID = "LULC_T35JNP"


@pytest.fixture(scope="module")
def sen12_tree(tmp_path_factory):
    """Create a minimal SEN12MS tree and return its root."""
    root = tmp_path_factory.mktemp("sen12ms")
    for sub in ("ROIs", "s1", "s2", "lulc", "s2_meta"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    # ROI CSV (bare scene ids, no paths) -- links the three aligned views.
    with open(os.path.join(root, "ROIs", "ROIs1868_summer.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["s1_name", "s2_name", "lulc_name"])
        w.writerow([S1_ID, S2_ID, LULC_ID])

    try:
        import tifffile
    except ImportError:
        pytest.skip("tifffile not installed (requirements-real-data.txt)")

    # Sentinel-1: 2 bands (VV, VH), 128x128.
    rng = np.random.RandomState(7)
    s1 = (rng.rand(2, 128, 128) * 10000).astype(np.uint16)
    tifffile.imwrite(_path(root, "s1", S1_ID), s1)

    # Sentinel-2: 13 bands, 128x128.
    s2 = (rng.rand(13, 128, 128) * 10000).astype(np.uint16)
    tifffile.imwrite(_path(root, "s2", S2_ID), s2)

    # LULC: single-band class map, dominant class 5 everywhere.
    lulc = np.full((128, 128), 5, dtype=np.uint8)
    tifffile.imwrite(_path(root, "lulc", LULC_ID), lulc)

    # s2 scene_meta.json with per-patch geometry centers + date + cloud.
    meta_dir = os.path.join(root, "s2_meta", S2_ID)
    os.makedirs(meta_dir, exist_ok=True)
    geoms = []
    for pid in range(4):
        geoms.append({
            "properties": {"patch_id": pid},
            "coordinates": [
                [[10.0 + pid, 45.0 + pid], [10.0 + pid, 45.2 + pid],
                 [10.2 + pid, 45.2 + pid], [10.2 + pid, 45.0 + pid],
                 [10.0 + pid, 45.0 + pid]],
            ],
        })
    with open(os.path.join(meta_dir, "scene_meta.json"), "w") as fh:
        json.dump({"acquisition_date": "20150601T170406", "cloud_coverage": 0.12,
                   "geometries": geoms}, fh)
    return str(root)


def _path(root, sub, scene_id):
    p = os.path.join(root, sub, scene_id)
    os.makedirs(p, exist_ok=True)
    return os.path.join(p, "scene.tif")


def _cfg(root, **overrides):
    cfg = deep_merge({}, DEFAULT_CONFIG)
    cfg["dataset"]["name"] = "sen12ms"
    cfg["dataset"]["root"] = root
    cfg["dataset"]["allow_fallback"] = False
    cfg["dataset"]["sen12"]["patch_size"] = 64
    cfg["dataset"]["sen12"]["max_scenes"] = 1
    if "modalities" in overrides:
        cfg["modalities"] = overrides.pop("modalities")
    cfg["dataset"].update(overrides)
    return cfg


def test_sen12_loader_shapes_and_metadata(sen12_tree):
    cfg = _cfg(sen12_tree, modalities=["optical", "multispectral", "sar"])
    ds = build_dataset(cfg)

    # 128x128 scene / 64x64 patch -> 4 patches.
    assert ds.n == 4
    assert ds.patches["optical"].shape == (4, 3, 64, 64)      # S2 RGB composite
    assert ds.patches["multispectral"].shape == (4, 8, 64, 64)  # default 8-band subset
    assert ds.patches["sar"].shape == (4, 2, 64, 64)           # VV + VH
    assert ds.patches["optical"].dtype == np.uint8
    assert (ds.labels == 5).all()                              # dominant LULC class

    assert ds.has_metadata()
    md0 = ds.metadata_for(0)
    assert md0.dataset == "sen12ms"
    assert md0.latitude is not None
    assert md0.longitude is not None
    assert md0.acquisition_date == "2015-06-01"
    assert md0.cloud_cover == 0.12


def test_sen12_missing_dir_raises(tmp_path):
    cfg = _cfg(os.path.join(str(tmp_path), "absent"))
    with pytest.raises(DatasetNotFound):
        build_dataset(cfg)


if __name__ == "__main__":
    import tempfile

    tree = sen12_tree(tempfile.mkdtemp())
    test_sen12_loader_shapes_and_metadata(tree)
    test_sen12_missing_dir_raises(tempfile.mkdtemp())
    print("test_sen12_loader.py: all tests passed")
