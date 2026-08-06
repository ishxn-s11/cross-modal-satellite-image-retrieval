"""Unit tests for the unified dataset interface + factory."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.interface import DatasetNotFound, available_datasets, build_dataset, resolve_dataset_name
from src.utils.config import DEFAULT_CONFIG, deep_merge


def make_cfg(**ds_overrides):
    cfg = deep_merge({}, DEFAULT_CONFIG)
    cfg["dataset"].update(ds_overrides)
    return cfg


def test_registry_contains_all_datasets():
    names = set(available_datasets())
    assert {"synthetic", "eurosat", "sen12ms", "so2sat", "bigearthnet_mm"} <= names


def test_resolve_name_prefers_name_over_source():
    cfg = make_cfg(name="sen12ms", source="synthetic")
    assert resolve_dataset_name(cfg) == "sen12ms"
    cfg2 = make_cfg(source="eurosat")
    # DEFAULT_CONFIG sets name='synthetic'; explicit source should still win if name is default.
    cfg2["dataset"]["name"] = None
    assert resolve_dataset_name(cfg2) == "eurosat"


def test_synthetic_dataset_has_metadata():
    cfg = make_cfg(num_patches=120, image_size=32, seed=0)
    ds = build_dataset(cfg)
    assert ds.dataset_id == "synthetic"
    assert ds.n == 120
    assert ds.has_metadata()
    md = ds.metadata_for(0)
    assert md.dataset == "synthetic"
    assert md.latitude is not None       # synthetic geo placement
    assert md.acquisition_date is not None


def test_sen12_missing_falls_back_to_synthetic(tmp_path):
    cfg = make_cfg(name="sen12ms", root=os.path.join(str(tmp_path), "nodata"),
                   num_patches=120, image_size=32, seed=0)
    ds = build_dataset(cfg)  # allow_fallback=True by default
    assert ds.dataset_id == "synthetic"


def test_sen12_missing_no_fallback_raises(tmp_path):
    cfg = make_cfg(name="sen12ms", root=os.path.join(str(tmp_path), "nodata"), allow_fallback=False)
    try:
        build_dataset(cfg)
        raise AssertionError("expected DatasetNotFound")
    except DatasetNotFound:
        pass


def test_unknown_dataset_raises():
    cfg = make_cfg(name="does_not_exist")
    try:
        build_dataset(cfg)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile

    tmp = tempfile.mkdtemp()
    test_registry_contains_all_datasets()
    test_resolve_name_prefers_name_over_source()
    test_synthetic_dataset_has_metadata()
    test_sen12_missing_falls_back_to_synthetic(tmp)
    test_sen12_missing_no_fallback_raises(tmp)
    test_unknown_dataset_raises()
    print("test_dataset_interface.py: all tests passed")
