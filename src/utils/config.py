"""Configuration: defaults, YAML loading, dot-path CLI overrides."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "source": "synthetic",  # "synthetic" | "eurosat"
        "root": "data/raw",
        "num_patches": 2000,
        "image_size": 64,
        "seed": 42,
        "eurosat_max_patches": 6000,  # subsample for real-data CPU runs
    },
    "modalities": ["optical", "multispectral", "sar"],
    "model": {
        "backbone": "resnet18",
        "pretrained": True,
        "embedding_dim": 128,
        "freeze_backbone": True,
        "unfreeze_stage": "stage4",  # "none" | "stage4" | "stage3"
        "n_classes": 10,
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
    },
    "retrieval": {
        "top_k": [5, 10],
        "gallery_fraction": 0.85,  # fraction of the *test* split used as gallery
        "n_query": 400,  # cap on number of queries evaluated (sampled)
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


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
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
