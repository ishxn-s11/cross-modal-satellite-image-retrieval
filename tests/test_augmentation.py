"""Unit tests for the remote-sensing-safe augmentation chain."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.augmentation import build_augmentation


def test_disabled_returns_none():
    assert build_augmentation("optical", {"enabled": False}) is None
    assert build_augmentation("sar", {}) is None


def test_all_zero_settings_returns_none():
    cfg = {"enabled": True, "random_crop": 0.0, "hflip": False, "vflip": False,
           "rotation_90": False, "noise_std": 0.0, "spectral_jitter": 0.0}
    assert build_augmentation("optical", cfg) is None


def test_hflip_preserves_shape():
    aug = build_augmentation("optical", {"enabled": True, "hflip": True})
    x = np.random.RandomState(0).rand(3, 16, 16).astype(np.float32)
    np.random.seed(0)
    out = aug(x)
    assert out.shape == x.shape
    assert np.array_equal(out, x[:, :, ::-1]) or np.array_equal(out, x)


def test_random_crop_resizes_back():
    aug = build_augmentation("optical", {"enabled": True, "random_crop": 0.25})
    x = np.random.RandomState(1).rand(3, 16, 16).astype(np.float32)
    np.random.seed(1)
    out = aug(x)
    assert out.shape == (3, 16, 16)


def test_rot90_preserves_shape():
    aug = build_augmentation("optical", {"enabled": True, "rotation_90": True})
    x = np.random.RandomState(2).rand(3, 16, 16).astype(np.float32)
    np.random.seed(2)
    out = aug(x)
    assert out.shape == x.shape


def test_noise_stays_bounded():
    aug = build_augmentation("optical", {"enabled": True, "noise_std": 0.05})
    x = np.random.RandomState(3).rand(3, 16, 16).astype(np.float32)
    np.random.seed(3)
    out = aug(x)
    assert out.max() <= 2.0
    assert out.min() >= 0.0


def test_spectral_jitter_multispectral_only():
    aug = build_augmentation("multispectral", {"enabled": True, "spectral_jitter": 0.1})
    x = np.random.RandomState(4).rand(8, 16, 16).astype(np.float32)
    np.random.seed(4)
    out = aug(x)
    assert out.shape == x.shape
    # optical never gets spectral jitter (semantics) -> chain is a no-op -> None
    assert build_augmentation("optical", {"enabled": True, "spectral_jitter": 0.1}) is None


def test_full_chain_composes():
    cfg = {"enabled": True, "random_crop": 0.1, "hflip": True, "vflip": True,
           "rotation_90": True, "noise_std": 0.02, "spectral_jitter": 0.05}
    aug = build_augmentation("multispectral", cfg)
    x = np.random.RandomState(5).rand(8, 32, 32).astype(np.float32)
    np.random.seed(5)
    out = aug(x)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


if __name__ == "__main__":
    test_disabled_returns_none()
    test_all_zero_settings_returns_none()
    test_hflip_preserves_shape()
    test_random_crop_resizes_back()
    test_rot90_preserves_shape()
    test_noise_stays_bounded()
    test_spectral_jitter_multispectral_only()
    test_full_chain_composes()
    print("test_augmentation.py: all tests passed")
