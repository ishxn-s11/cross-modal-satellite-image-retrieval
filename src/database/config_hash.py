"""Stable cache-key hashing.

Embeddings and index galleries are derived from the *combination* of a dataset
and a trained model. The caches key on this digest so that any change to the
dataset shape, the model architecture, or the saved checkpoint invalidates the
cached artefacts automatically (a config change or a retrain therefore never
serves stale embeddings).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


def _config_material(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten just the parts of the config that affect the embedding space."""
    ds = cfg.get("dataset", {})
    md = cfg.get("model", {})
    return {
        "dataset": {
            "source": ds.get("source"),
            "num_patches": ds.get("num_patches"),
            "image_size": ds.get("image_size"),
            "seed": ds.get("seed"),
            "eurosat_max_patches": ds.get("eurosat_max_patches"),
        },
        "model": {
            "backbone": md.get("backbone"),
            "pretrained": md.get("pretrained"),
            "embedding_dim": md.get("embedding_dim"),
            "freeze_backbone": md.get("freeze_backbone"),
            "unfreeze_stage": md.get("unfreeze_stage"),
            "projection_heads": md.get("projection_heads"),
            "embedding_mode": md.get("embedding_mode"),
            "vit_image_size": md.get("vit_image_size"),
            "foundation": md.get("foundation"),
        },
        "modalities": sorted(cfg.get("modalities", [])),
    }


def _checkpoint_fingerprint(checkpoint_path: Optional[str]) -> str:
    """mtime + size of the best checkpoint, or '' when it does not exist.

    Including this in the digest means retraining the model invalidates any
    cached embeddings even when the YAML config is unchanged.
    """
    if not checkpoint_path:
        return ""
    path = os.path.join(*checkpoint_path) if isinstance(checkpoint_path, (list, tuple)) else checkpoint_path
    if not os.path.exists(path):
        return ""
    st = os.stat(path)
    return f"{st.st_mtime:.3f}:{st.st_size}"


def compute_cache_key(cfg: Dict[str, Any], checkpoint_path: Optional[str] = None) -> str:
    """Return a short stable hex digest identifying the embedding space."""
    payload = {
        "config": _config_material(cfg),
        "checkpoint": _checkpoint_fingerprint(checkpoint_path),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]