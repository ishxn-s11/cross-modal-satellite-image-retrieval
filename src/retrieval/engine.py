"""Retrieval engine: embed a dataset subset, build a gallery, run queries.

The engine now supports *persistent* indexing. When constructed with an
:class:`~src.database.embedding_store.EmbeddingStore` and
:class:`~src.database.index_store.IndexStore` (and a config hash), it:

* caches the full-dataset per-modality embeddings on disk and reuses them on
  warm starts instead of recomputing through the network, and
* reloads previously-built FAISS galleries rather than rebuilding them.

The existing in-memory L1 cache is retained as a fast path within a process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..database import EmbeddingStore, IndexStore
from ..data.dataset import MultiModalDataset
from ..models.encoder import ModalityAdaptiveEncoder
from .index import FaissCosineIndex, build_index_from_embeddings


@dataclass
class Gallery:
    """A per-modality searchable database."""

    modality: str
    index: FaissCosineIndex
    indices: np.ndarray  # dataset ids, aligned with index positions
    labels: np.ndarray
    embeddings: np.ndarray  # L2-normalised (n, D)

    @property
    def size(self) -> int:
        return int(self.indices.shape[0])


@dataclass
class RetrievalResult:
    query_ids: np.ndarray
    gallery_ids: np.ndarray  # top-k dataset ids, shape (Nq, k)
    scores: np.ndarray
    query_labels: np.ndarray
    gallery_labels: np.ndarray  # labels of the retrieved items (Nq, k)
    k: int
    search_times_ms: np.ndarray  # per-query search latency (Nq,)
    query_modality: str
    gallery_modality: str

    def relevant_mask(self) -> np.ndarray:
        """bool (Nq, k): retrieved item shares the query's semantic class."""
        return self.gallery_labels == self.query_labels[:, None]


class RetrievalEngine:
    def __init__(
        self,
        model: ModalityAdaptiveEncoder,
        dataset: MultiModalDataset,
        device: torch.device,
        batch_size: int = 128,
        embedding_store: Optional[EmbeddingStore] = None,
        index_store: Optional[IndexStore] = None,
        config_hash: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.embedding_store = embedding_store
        self.index_store = index_store
        self.config_hash = config_hash
        self._cache: Dict[Tuple[str, Tuple[int, ...]], Tuple[np.ndarray, np.ndarray]] = {}
        self._full_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # PERSISTENCE -------------------------------------------------------
    # ------------------------------------------------------------------
    def _persist_enabled(self) -> bool:
        return self.embedding_store is not None and bool(self.config_hash)

    def cache_full_embeddings(
        self, modality: str, force: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure (and return) the full-dataset embeddings for ``modality``.

        Loads from the on-disk embed store when available; otherwise computes
        them once and persists them. The result is also retained in memory so
        later per-subset ``embed`` calls become simple row-selects.
        """
        if self._persist_enabled():
            if not force and modality in self._full_cache:
                return self._full_cache[modality]
            if not force:
                cached = self.embedding_store.load(modality, self.config_hash)
                if cached is not None:
                    self._full_cache[modality] = cached
                    return cached
        emb, labels = self._compute(np.arange(len(self.dataset)), modality)
        if self._persist_enabled():
            self.embedding_store.save(modality, self.config_hash, emb, labels)
        self._full_cache[modality] = (emb, labels)
        return emb, labels

    # ------------------------------------------------------------------
    # EMBEDDING ---------------------------------------------------------
    # ------------------------------------------------------------------
    def embed(self, indices: Sequence[int], modality: str) -> Tuple[np.ndarray, np.ndarray]:
        """Embed dataset ids in `modality` -> (L2-normalised (n, D), labels (n,))."""
        key = (modality, tuple(indices))
        if key in self._cache:
            return self._cache[key]
        rowed = self._cache_from_full(indices, modality)
        if rowed is not None:
            self._cache[key] = rowed
            return rowed
        emb, labels = self._compute(list(indices), modality)
        self._cache[key] = (emb, labels)
        return emb, labels

    def _cache_from_full(
        self, indices: Sequence[int], modality: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Row-select the requested ids from cached full-dataset embeddings."""
        ids = np.asarray(list(indices), dtype=np.int64)
        if ids.size == 0 or np.any(ids < 0) or np.any(ids >= len(self.dataset)):
            return None
        full = self._full_cache.get(modality)
        if full is None and self._persist_enabled():
            cached = self.embedding_store.load(modality, self.config_hash)
            if cached is not None:
                full = cached
                self._full_cache[modality] = full
        if full is None:
            return None
        emb, labels = full
        if ids.max() >= emb.shape[0]:  # full cache predates dataset growth
            return None
        return emb[ids].astype(np.float32), labels[ids]

    def _compute(self, ids: Sequence[int], modality: str) -> Tuple[np.ndarray, np.ndarray]:
        """Run the network over ``ids`` and return (embeddings, labels)."""
        self.model.eval()
        ids = np.asarray(list(ids), dtype=np.int64)
        embs: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                xs = torch.stack([self.dataset[int(i)][modality] for i in chunk])
                emb = self.model.embed(xs.to(self.device), modality)
                embs.append(emb.detach().cpu().numpy())
                labels.append(self.dataset.labels[chunk])
        emb_all = np.concatenate(embs, axis=0).astype(np.float32)
        label_all = np.concatenate(labels, axis=0)
        return emb_all, label_all

    # ------------------------------------------------------------------
    # GALLERY -----------------------------------------------------------
    # ------------------------------------------------------------------
    def build_gallery(
        self, indices: Sequence[int], modality: str, nlist: Optional[int] = None
    ) -> Gallery:
        """Build (or reload from disk) a gallery over ``indices`` in ``modality``."""
        if self.index_store is not None and self.config_hash:
            cached = self.index_store.load(modality, self.config_hash, indices)
            if cached is not None:
                self._cache[(modality, tuple(indices))] = (
                    cached.embeddings,
                    cached.labels,
                )
                return cached

        emb, labels = self.embed(indices, modality)
        index = build_index_from_embeddings(emb, nlist=nlist)
        gallery = Gallery(
            modality=modality,
            index=index,
            indices=np.asarray(list(indices), dtype=np.int64),
            labels=labels,
            embeddings=emb,
        )
        if self.index_store is not None and self.config_hash:
            self.index_store.save(gallery, modality, self.config_hash)
        return gallery

    # ------------------------------------------------------------------
    # RETRIEVAL ---------------------------------------------------------
    # ------------------------------------------------------------------
    def retrieve(
        self,
        gallery: Gallery,
        query_indices: Sequence[int],
        query_modality: str,
        k: int,
        time_queries: bool = True,
    ) -> RetrievalResult:
        """Search the gallery for each query image and return top-k results."""
        q_emb, q_labels = self.embed(query_indices, query_modality)
        nq, k = q_emb.shape[0], min(k, gallery.size)
        scores = np.zeros((nq, k), dtype=np.float32)
        ids = np.zeros((nq, k), dtype=np.int64)
        times = np.zeros(nq, dtype=np.float64)

        if time_queries:
            for i in range(nq):
                t0 = time.perf_counter()
                s, idx = gallery.index.search(q_emb[i : i + 1], k)
                times[i] = (time.perf_counter() - t0) * 1000.0
                scores[i] = s[0]
                ids[i] = idx[0]
        else:
            t0 = time.perf_counter()
            scores, ids = gallery.index.search(q_emb, k)
            times[:] = (time.perf_counter() - t0) / max(1, nq) * 1000.0

        # Map index positions back to dataset ids.
        gallery_ids = gallery.indices[ids.astype(np.int64)]
        gallery_labels = gallery.labels[ids.astype(np.int64)]
        return RetrievalResult(
            query_ids=np.asarray(list(query_indices), dtype=np.int64),
            gallery_ids=gallery_ids,
            scores=scores,
            query_labels=q_labels,
            gallery_labels=gallery_labels,
            k=k,
            search_times_ms=times,
            query_modality=query_modality,
            gallery_modality=gallery.modality,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._full_cache.clear()