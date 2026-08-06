"""Configuration: defaults, YAML loading, dot-path CLI overrides."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": "synthetic",  # synthetic | eurosat | sen12ms | so2sat | bigearthnet_mm
        "source": "synthetic",  # legacy alias, kept for backward compatibility
        "allow_fallback": True,  # fall back to synthetic when a real dataset is absent
        "root": "data/raw",
        "num_patches": 2000,
        "image_size": 64,
        "seed": 42,
        "eurosat_max_patches": 6000,  # subsample for real-data CPU runs
        # Real-dataset options (see docs/datasets.md).
        "sen12": {
            "roi_csv": None,          # path (root-relative or absolute) to an ROI CSV
            "patch_size": 64,         # crop size for the 256x256 scenes
            "max_scenes": None,       # cap the number of scenes read
            "ms_bands": [1, 2, 3, 4, 5, 7, 8, 11],  # S2 bands -> multispectral stack
            "sar_bands": ["VV", "VH"],
            "class_names": None,      # optional 17-name list override
        },
        "so2sat": {
            "h5_file": "training.h5",  # training | validation | testing
            "max_patches": None,       # subsample cap
        },
        "bigearthnet_mm": {
            "labels_csv": None,        # path to BigEarthNet_19_labels.csv
            "max_patches": None,       # subsample cap
        },
    },
    "modalities": ["optical", "multispectral", "sar"],
    "model": {
        "backbone": "resnet18",  # resnet18 | resnet34 | resnet50 | vit_b_16 | satmae | prithvi
        "pretrained": True,
        "embedding_dim": 128,
        "freeze_backbone": True,
        "unfreeze_stage": "stage4",  # "none" | "stage4" | "stage3" (ViT: "last")
        "n_classes": 10,
        "projection_heads": "shared",  # "shared" | "per_modality"
        "vit_image_size": None,        # default = dataset.image_size
        "vit_pretrained": True,
        # Foundation-model checkpoints (optional, user-supplied, NOT bundled).
        "foundation": {
            "satmae": {"path": None, "feature_dim": 384},
            "prithvi": {"path": None, "feature_dim": 768},
        },
    },
    "training": {
        "epochs": 6,
        "batch_size": 64,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "clip_weight": 1.0,
        "supcon_weight": 1.0,
        "cls_weight": 1.0,
        "num_workers": 0,
        "patience": 3,
        "device": "auto",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        # Geographic / temporal alignment (off by default; needs metadata).
        "geo_weight": 0.0,
        "geo_same_km": 5.0,       # pairs within this distance are positives
        "geo_push_distant": False,
        "geo_distant_km": 100.0,
        # Hard-negative mining (off by default).
        "hard_negatives": {
            "enabled": False,
            "n_hard": 8,
            "strategy": "embedding",
        },
    },
    "retrieval": {
        "top_k": [5, 10],
        "gallery_fraction": 0.85,  # fraction of the *test* split used as gallery
        "n_query": 400,  # cap on number of queries evaluated (sampled)
    },
    "preprocessing": {
        # Modality-aware patch preprocessing (identity by default -> legacy behaviour).
        "resize": None,        # int target spatial size, or null (keep native)
        "cloud_max": None,     # drop images with cloud_cover > this (needs metadata)
        "sar": {"log_transform": False, "clip_min": None, "clip_max": None,
                "invalid_value": None, "invalid_fill": "zero",
                "speckle_filter": "none", "speckle_window": 3},
        "optical": {"clip_min": None, "clip_max": None,
                    "invalid_value": None, "invalid_fill": "zero"},
        "multispectral": {"band_selection": None, "missing_bands": "raise",
                          "clip_min": None, "clip_max": None,
                          "invalid_value": None, "invalid_fill": "zero"},
    },
    "augmentation": {
        # Applied to the training set only, on the [0,1] scale. All augmentations
        # preserve remote-sensing semantics (no arbitrary-angle rotation).
        "enabled": False,
        "random_crop": 0.0,     # fraction of the image cropped away (then resized back)
        "hflip": False,
        "vflip": False,
        "rotation_90": False,   # random k*90 deg rotation
        "noise_std": 0.0,       # gaussian noise std (on [0,1] scale)
        "spectral_jitter": 0.0, # per-band multiplicative jitter (multispectral only)
    },
    "evaluation": {
        "same_modal_pairs": [
            ["optical", "optical"],
            ["multispectral", "multispectral"],
            ["sar", "sar"],
        ],
        "cross_modal_pairs": [
            ["optical", "multispectral"],
            ["multispectral", "optical"],
            ["optical", "sar"],
            ["sar", "optical"],
            ["multispectral", "sar"],
            ["sar", "multispectral"],
        ],
    },
    "outputs": {
        "dir": "outputs",
        "model_dir": "models",
        "log_file": "outputs/logs/pipeline.log",
    },
    "persistence": {
        "embeddings_dir": "embeddings",
        "faiss_dir": "faiss",
        "database_path": "database/metadata.db",
    },
}


def _deepcopy(value: Any) -> Any:
    """Copy a config value so merged output never aliases the input dicts.

    (Without this, ``deep_merge({}, DEFAULT_CONFIG)`` would return a dict whose
    nested ``dataset``/``model``/... dicts are *references* to the module-level
    ``DEFAULT_CONFIG``, so later mutation of the merged result silently corrupts
    the defaults.)
    """
    if isinstance(value, dict):
        return {k: _deepcopy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deepcopy(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base`` (deep-copied)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = _deepcopy(value)
    return out


def _set_dot_path(config: Dict[str, Any], path: str, value: str) -> None:
    keys = path.split(".")
    node = config
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = _coerce(value)


def _coerce(value: str):
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    path: Optional[str] = None, overrides: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Load a YAML config over defaults, then apply ``k=v`` CLI overrides."""
    config = deep_merge({}, DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            config = deep_merge(config, yaml.safe_load(fh) or {})
    for ov in overrides or []:
        if "=" in ov:
            key, value = ov.split("=", 1)
            _set_dot_path(config, key.strip(), value.strip())
    return config


def pretty_print(config: Dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
