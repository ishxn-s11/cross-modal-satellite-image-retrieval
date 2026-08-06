"""Unit tests for the modality-aware preprocessing pipeline + transforms."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.metadata import ImageMetadata
from src.data.preprocessing import (
    _lee_filter,
    build_transforms,
    compute_normalization_stats,
    filter_images_by_cloud,
    preprocess_patches,
    scale_to_unit,
)


def test_scale_to_unit_maps_optical_uint8():
    x = np.array([[[0, 128, 255]]], dtype=np.uint8)
    u = scale_to_unit(x, "optical")
    assert u[0, 0, 0] == 0.0
    assert abs(u[0, 0, 2] - 1.0) < 1e-6


def test_identity_preprocessing_returns_same_arrays():
    patches = {
        "optical": np.random.RandomState(0).randint(0, 255, (8, 3, 16, 16)).astype(np.uint8),
        "sar": np.random.RandomState(1).rand(8, 1, 16, 16).astype(np.float32) * 2.0,
    }
    out = preprocess_patches(patches, {})  # empty config = identity
    assert np.array_equal(out["optical"], patches["optical"])
    assert np.array_equal(out["sar"], patches["sar"])


def test_transform_scales_before_standardising():
    # Regression: raw optical uint8 used to be fed straight against stats
    # computed on the [0,1] scale -> network inputs in the hundreds.
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, (50, 3, 16, 16)).astype(np.uint8)
    stats = compute_normalization_stats({"optical": arr})
    tf = build_transforms(stats)["optical"]
    out = tf(arr[0])
    assert abs(out.max()) < 15.0, f"optical input too large: {out.max():.1f}"
    assert abs(out.mean()) < 1.5


def test_sar_log_transform_and_clip():
    sar = np.random.RandomState(1).rand(4, 1, 16, 16).astype(np.float32) * 3.0
    cfg = {"sar": {"log_transform": True, "clip_min": 0.0, "clip_max": 1.5}}
    out = preprocess_patches({"sar": sar}, cfg)["sar"]
    assert out.max() <= 1.5 + 1e-6
    assert out.min() >= 0.0


def test_sar_speckle_filter_smooths():
    rng = np.random.RandomState(0)
    band = rng.rand(32, 32).astype(np.float32) * 2.0
    filtered = _lee_filter(band, 3)
    assert filtered.shape == band.shape
    assert filtered.var() < band.var()  # speckle reduced


def test_multispectral_band_selection():
    ms = np.random.RandomState(0).rand(4, 8, 16, 16).astype(np.float32)
    cfg = {"multispectral": {"band_selection": [0, 1, 2, 4, 5]}}
    out = preprocess_patches({"multispectral": ms}, cfg)["multispectral"]
    assert out.shape == (4, 5, 16, 16)
    # selected bands preserved in order
    assert np.allclose(out[0, 0], ms[0, 0])
    assert np.allclose(out[0, 3], ms[0, 4])


def test_band_selection_out_of_range_raises():
    ms = np.random.RandomState(0).rand(4, 8, 16, 16).astype(np.float32)
    cfg = {"multispectral": {"band_selection": [0, 9]}}
    try:
        preprocess_patches({"multispectral": ms}, cfg)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_value_replacement():
    sar = np.random.RandomState(0).rand(4, 1, 16, 16).astype(np.float32)
    sar[0, 0, 0, 0] = -9999.0
    cfg = {"sar": {"invalid_value": -9999.0, "invalid_fill": "zero"}}
    out = preprocess_patches({"sar": sar}, cfg)["sar"]
    assert out[0, 0, 0, 0] == 0.0


def test_resize():
    patches = {"optical": np.random.RandomState(0).randint(0, 255, (4, 3, 32, 32)).astype(np.uint8)}
    cfg = {"resize": 16}
    out = preprocess_patches(patches, cfg)["optical"]
    assert out.shape == (4, 3, 16, 16)


def test_cloud_filter_keeps_low_cloud_and_unknown():
    meta = [ImageMetadata(image_id=i, cloud_cover=cc) for i, cc in
            enumerate([0.1, 0.2, 0.3, 0.9, None])]
    patches = {"optical": np.random.RandomState(0).rand(5, 3, 8, 8).astype(np.float32)}
    labels = np.arange(5)
    p, l, m = filter_images_by_cloud(patches, labels, meta, 0.5)
    # kept: cloud <= 0.5 (0,1,2) + unknown (4) -> 4
    assert len(l) == 4
    assert set(l.tolist()) == {0, 1, 2, 4}
    assert len(p["optical"]) == 4
    assert len(m) == 4


if __name__ == "__main__":
    test_scale_to_unit_maps_optical_uint8()
    test_identity_preprocessing_returns_same_arrays()
    test_transform_scales_before_standardising()
    test_sar_log_transform_and_clip()
    test_sar_speckle_filter_smooths()
    test_multispectral_band_selection()
    test_band_selection_out_of_range_raises()
    test_invalid_value_replacement()
    test_resize()
    test_cloud_filter_keeps_low_cloud_and_unknown()
    print("test_preprocessing.py: all tests passed")
