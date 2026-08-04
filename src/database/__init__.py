"""Persistent storage layer for the retrieval system.

This package turns the previously in-memory-only retrieval artefacts into an
on-disk, reproducible *database* split across three logical stores:

* :mod:`src.database.metadata_store` -- a SQLite DB that records dataset
  metadata, per-modality gallery records and a retrieval-query log.
* :mod:`src.database.embedding_store` -- an npz cache of the *full-dataset*
  per-modality embeddings, keyed by a stable configuration hash so re-runs
  with an unchanged model/dataset are cheap.
* :mod:`src.database.index_store` -- persisted FAISS galleries
  (``.index`` + ``meta.json``) keyed by ``(modality, gallery-subset-hash)``
  so identical galleries are reloaded instead of rebuilt.

Public convenience helpers
--------------------------
``compute_cache_key(cfg, checkpoint_path=None)`` -- a stable hex digest that
identifies a (dataset, model) embedding space so caches invalidate whenever
the configuration or the trained checkpoint changes.
"""

from .config_hash import compute_cache_key
from .embedding_store import EmbeddingStore
from .index_store import IndexStore
from .metadata_store import MetadataStore

__all__ = [
    "compute_cache_key",
    "EmbeddingStore",
    "IndexStore",
    "MetadataStore",
]