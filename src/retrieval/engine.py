"""Retrieval engine: embed a dataset subset, build a gallery, run queries.

The engine supports *persistent* indexing. When constructed with an
:class:`~src.database.embedding_store.EmbeddingStore` and
:class:`~src.database.index_store.IndexStore` (and a config hash), it caches the
full-dataset per-modality embeddings on disk and reloads previously-built FAISS
galleries on warm starts.

Two-stage retrieval: FAISS returns ``candidate_k`` candidates; an optional
re-ranker (:class:`~src.retrieval.rerank.ReRanker`) re-scores them and the top
``final_k`` are returned. With no re-ranker / candidate_k == k this reduces to
single-stage exact search.
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
from ..training.geo import haversine_km
from .index import FaissCosineIndex, build_index_from_embeddings
from .rerank import ReRanker


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
    rerank_times_ms: np.ndarray = field(default=None)  # per-query re-rank (Nq,)

    def __post_init__(self) -> None:
        if self.rerank_times_ms is None:
            self.rerank_times_ms = np.zeros_like(self.search_times_ms)

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
        """Ensure (and return) the full-dataset embeddings for ``modality``."""
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
        self,
        indices: Sequence[int],
        modality: str,
        nlist: Optional[int] = None,
        index_kwargs: Optional[Dict] = None,
    ) -> Gallery:
        """Build (or reload from disk) a gallery over ``indices`` in ``modality``.

        ``index_kwargs`` (from ``retrieval.index`` config) selects the index
        type / metric. The persisted store keeps the index type + metric in its
        meta file so reloads reconstruct the same index.
        """
        if self.index_store is not None and self.config_hash:
            cached = self.index_store.load(modality, self.config_hash, indices)
            if cached is not None:
                self._cache[(modality, tuple(indices))] = (
                    cached.embeddings,
                    cached.labels,
                )
                return cached

        emb, labels = self.embed(indices, modality)
        kwargs = dict(index_kwargs or {})
        if nlist is not None:
            kwargs.setdefault("nlist", nlist)
        index = build_index_from_embeddings(emb, **kwargs)
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
    # METADATA ----------------------------------------------------------
    # ------------------------------------------------------------------
    def _query_metadata(self, indices: Sequence[int]) -> Optional[List]:
        if not self.dataset.metadata:
            return None
        return [self.dataset.metadata_for(int(i)) for i in indices]

    def _gallery_metadata(self, gallery: Gallery) -> Optional[List]:
        if not self.dataset.metadata:
            return None
        return [self.dataset.metadata_for(int(i)) for i in gallery.indices]

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
        candidate_k: Optional[int] = None,
        reranker: Optional[ReRanker] = None,
        query_metadata: Optional[Sequence] = None,
        gallery_metadata: Optional[Sequence] = None,
    ) -> RetrievalResult:
        """Search the gallery for each query image and return top-k results.

        Two-stage when ``candidate_k > k`` and a ``reranker`` is provided:
        FAISS returns ``candidate_k`` candidates, the re-ranker re-scores them
        and the top ``k`` are returned.
        """
        q_emb, q_labels = self.embed(query_indices, query_modality)
        nq, k = q_emb.shape[0], min(k, gallery.size)
        search_k = min(candidate_k or k, gallery.size)

        if query_metadata is None:
            query_metadata = self._query_metadata(query_indices)
        if gallery_metadata is None:
            gallery_metadata = self._gallery_metadata(gallery)

        scores = np.zeros((nq, k), dtype=np.float32)
        ids = np.zeros((nq, k), dtype=np.int64)
        times = np.zeros(nq, dtype=np.float64)
        rerank_ms = np.zeros(nq, dtype=np.float64)

        t0 = time.perf_counter()
        cand_scores, cand_ids = gallery.index.search(q_emb, search_k)
        times[:] = (time.perf_counter() - t0) / max(1, nq) * 1000.0

        for i in range(nq):
            row_ids = cand_ids[i].astype(np.int64)
            row_embs = gallery.embeddings[row_ids]
            q_meta = query_metadata[i] if query_metadata is not None and i < len(query_metadata) else None
            c_metas = None
            if gallery_metadata is not None:
                c_metas = [gallery_metadata[int(j)] if int(j) < len(gallery_metadata) else None for j in row_ids]
            if reranker is not None and search_k > k:
                t_r = time.perf_counter()
                rs = reranker.score(q_emb[i], row_embs, q_meta, c_metas)
                rerank_ms[i] = (time.perf_counter() - t_r) * 1000.0
                top = np.argsort(-np.asarray(rs))[:k]
            else:
                rs = cand_scores[i]
                top = np.arange(k)
            scores[i] = np.asarray(rs)[top]
            ids[i] = row_ids[top]

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
            rerank_times_ms=rerank_ms,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        self._full_cache.clear()


# ---------------------------------------------------------------------------
# Rich result records (similarity + metadata + geographic info, see spec)
# ---------------------------------------------------------------------------


def _haversine_km_or_none(q_meta, meta) -> Optional[float]:
    if q_meta is None or meta is None:
        return None
    try:
        if q_meta.latitude is None or q_meta.longitude is None or meta.latitude is None or meta.longitude is None:
            return None
        return round(float(haversine_km(
            np.float32(q_meta.latitude), np.float32(q_meta.longitude),
            np.float32(meta.latitude), np.float32(meta.longitude),
        )), 2)
    except (TypeError, ValueError):
        return None


def result_records(
    result: RetrievalResult,
    dataset: MultiModalDataset,
    class_names: Sequence[str],
    query_metadata: Optional[Sequence] = None,
    gallery_metadata: Optional[Sequence] = None,
) -> List[Dict]:
    """Build rich per-query result records with all available metadata.

    Only fields that actually exist are included (others are omitted).
    """
    if query_metadata is None:
        query_metadata = (
            [dataset.metadata_for(int(i)) for i in result.query_ids]
            if dataset.metadata else None
        )
    if gallery_metadata is None:
        gallery_metadata = (
            [dataset.metadata_for(int(i)) for i in range(len(dataset))]
            if dataset.metadata else None
        )

    rows: List[Dict] = []
    for i in range(result.query_ids.shape[0]):
        qid = int(result.query_ids[i])
        q_meta = query_metadata[i] if query_metadata is not None and i < len(query_metadata) else None
        retrieved = []
        for j in range(result.k):
            rid = int(result.gallery_ids[i, j])
            meta = gallery_metadata[rid] if gallery_metadata is not None and rid < len(gallery_metadata) else None
            rec = {
                "rank": j + 1,
                "image_id": rid,
                "similarity_score": round(float(result.scores[i, j]), 4),
                "modality": result.gallery_modality,
                "sensor": getattr(meta, "sensor", None),
                "land_cover": (getattr(meta, "land_cover", None)
                               or class_names[int(result.gallery_labels[i, j])]),
                "latitude": getattr(meta, "latitude", None),
                "longitude": getattr(meta, "longitude", None),
                "acquisition_date": getattr(meta, "acquisition_date", None),
                "geographic_distance": _haversine_km_or_none(q_meta, meta),
                "image_path": getattr(meta, "file_path", None),
            }
            rec = {k2: v for k2, v in rec.items() if v is not None}
            retrieved.append(rec)
        rows.append(
            {
                "query": {
                    "image_id": qid,
                    "modality": result.query_modality,
                    "class": class_names[int(result.query_labels[i])],
                },
                "retrieved": retrieved,
                "search_time_ms": round(float(result.search_times_ms[i]), 4),
                "rerank_time_ms": round(float(result.rerank_times_ms[i]), 4),
            }
        )
    return rows
