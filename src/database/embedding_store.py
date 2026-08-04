"""Embedding cache store.

Stores the *full-dataset* L2-normalised embeddings for each modality on disk as
compressed ``.npz`` files. Because the cache key is derived from the dataset
+ model configuration (+ the checkpoint mtime), any run with an unchanged
trained model reuses the cached embeddings instead of recomputing them through
the network -- this is the main warm-start win for the pipeline and web app.

Layout::

    embeddings/
      {modality}_{config_hash[:16]}.npz      mapping: dataset-id -> embedding
      {modality}_{config_hash[:16]}_meta.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

Pair = Tuple[np.ndarray, np.ndarray]  # (embeddings (N,D) float32, labels (N,) int64)


class EmbeddingStore:
    """On-disk cache of full-dataset per-modality embeddings."""

    def __init__(self, root: str = "embeddings") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers --------------------------------------------------------
    def _paths(self, modality: str, config_hash: str) -> Tuple[Path, Path]:
        tag = f"{modality}_{config_hash[:16]}"
        return self.root / f"{tag}.npz", self.root / f"{tag}_meta.json"

    def key(self, modality: str, config_hash: str) -> str:
        return f"{modality}_{config_hash[:16]}"

    # -- existence -----------------------------------------------------------
    def exists(self, modality: str, config_hash: str) -> bool:
        npz, _ = self._paths(modality, config_hash)
        return npz.exists()

    def list_keys(self) -> np.ndarray:
        return np.array([p.stem for p in self.root.glob("*.npz")])

    # -- read / write --------------------------------------------------------
    def save(
        self,
        modality: str,
        config_hash: str,
        embeddings: np.ndarray,
        labels: np.ndarray,
        extra: Optional[Dict] = None,
    ) -> Dict:
        emb = np.asarray(embeddings, dtype=np.float32)
        lab = np.asarray(labels, dtype=np.int64)
        if emb.shape[0] != lab.shape[0]:
            raise ValueError("embeddings/labels length mismatch")
        npz_path, meta_path = self._paths(modality, config_hash)
        np.savez_compressed(npz_path, embeddings=emb, labels=lab)
        meta = {
            "modality": modality,
            "config_hash": config_hash[:16],
            "n": int(emb.shape[0]),
            "dim": int(emb.shape[1]),
            "dtype": str(emb.dtype),
            "created_at": time.time(),
            **({} if extra is None else extra),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return meta

    def load(self, modality: str, config_hash: str) -> Optional[Pair]:
        npz_path, meta_path = self._paths(modality, config_hash)
        if not npz_path.exists():
            return None
        data = np.load(npz_path)
        return data["embeddings"].astype(np.float32), data["labels"].astype(np.int64)

    def clear(self, modality: Optional[str] = None, config_hash: Optional[str] = None) -> int:
        """Delete cached files. Returns the number of files removed."""
        removed = 0
        pattern = f"{modality}_*" if modality else "*.npz"
        for p in self.root.glob(pattern):
            if config_hash and config_hash[:16] not in p.stem:
                continue
            for suffix in (".npz", "_meta.json"):
                q = self.root / (p.stem + suffix)
                if q.exists():
                    q.unlink()
                    removed += 1
        return removed