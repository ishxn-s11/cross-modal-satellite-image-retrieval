"""FAISS gallery persistence store.

Persists a fully-built :class:`~src.retrieval.engine.Gallery` (a FAISS index
plus the mapping from index positions back to dataset ids and their labels) to
disk. On a warm start the web app / evaluation reload the gallery directly
instead of re-embedding every gallery image.

Layout::

    faiss/
      {modality}_{config_hash[:16]}_{indices_hash[:16]}.index   # binary FAISS index
      {modality}_{config_hash[:16]}_{indices_hash[:16]}_meta.json  # ids, labels, dim, nvec

The ``indices_hash`` is derived from the exact set of gallery dataset-ids, so
two different gallery splits for the same modality never collide on disk.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from ..retrieval.index import FaissCosineIndex


def _indices_hash(indices) -> str:
    key = hashlib.sha256(
        np.asarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()[:16]
    return key


class IndexStore:
    """On-disk persistence for per-modality FAISS galleries."""

    def __init__(self, root: str = "faiss") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- path helpers --------------------------------------------------------
    def _paths(self, modality: str, config_hash: str, indices) -> tuple:
        tag = f"{modality}_{config_hash[:16]}_{_indices_hash(indices)}"
        return self.root / f"{tag}.index", self.root / f"{tag}_meta.json"

    def exists(self, modality: str, config_hash: str, indices) -> bool:
        index_path, _ = self._paths(modality, config_hash, indices)
        return index_path.exists()

    # -- write ---------------------------------------------------------------
    def save(self, gallery, modality: str, config_hash: str) -> dict:
        indices = np.asarray(gallery.indices, dtype=np.int64)
        index_path, meta_path = self._paths(modality, config_hash, indices)
        gallery.index.save(str(index_path))
        meta = {
            "modality": modality,
            "config_hash": config_hash[:16],
            "indices_hash": _indices_hash(indices),
            "n": int(indices.shape[0]),
            "dim": int(gallery.index.dim),
            "index_type": "ivf" if getattr(gallery.index, "ivf", False) else "flat",
            "nlist": getattr(gallery.index.index, "nlist", None) if getattr(gallery.index, "ivf", False) else None,
            "indices": indices.tolist(),
            "labels": np.asarray(gallery.labels, dtype=np.int64).tolist(),
            "created_at": time.time(),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return meta

    # -- read ----------------------------------------------------------------
    def load(self, modality: str, config_hash: str, indices) -> Optional["Gallery"]:
        """Reconstruct a Gallery from disk, or None if not cached."""
        from ..retrieval.engine import Gallery  # deferred to avoid import cycle

        index_path, meta_path = self._paths(modality, config_hash, indices)
        if not index_path.exists() or not meta_path.exists():
            return None
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        index = FaissCosineIndex.load(str(index_path))
        ids = np.asarray(meta["indices"], dtype=np.int64)
        labels = np.asarray(meta["labels"], dtype=np.int64)
        embeddings = self._reconstruct(index, meta)

        return Gallery(
            modality=modality,
            index=index,
            indices=ids,
            labels=labels,
            embeddings=embeddings,
        )

    @staticmethod
    def _reconstruct(index: FaissCosineIndex, meta: dict) -> np.ndarray:
        """Pull the embedding vectors back out of the FAISS index (best effort)."""
        try:
            n = int(meta["n"])
            return index.index.reconstruct_n(0, n).astype(np.float32)
        except Exception:  # pragma: no cover - reconstruct not supported by every index
            return np.zeros((int(meta["n"]), int(meta["dim"])), dtype=np.float32)

    # -- clear ---------------------------------------------------------------
    def clear(self, modality: Optional[str] = None, config_hash: Optional[str] = None) -> int:
        removed = 0
        for p in self.root.glob("*.index"):
            if modality and not p.stem.startswith(f"{modality}_"):
                continue
            if config_hash and config_hash[:16] not in p.stem:
                continue
            meta_p = self.root / (p.stem + "_meta.json")
            p.unlink()
            removed += 1
            if meta_p.exists():
                meta_p.unlink()
                removed += 1
        return removed