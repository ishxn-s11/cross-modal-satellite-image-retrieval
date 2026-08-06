"""FAISS indexes for fast similarity retrieval.

Supports multiple index types and similarity metrics:

* **flat**    -- exact ``IndexFlatIP`` / ``IndexFlatL2`` (baseline).
* **ivf**     -- ``IndexIVFFlat`` (approximate, ``nlist``).
* **hnsw**    -- ``IndexHNSWFlat`` (graph-based, ``M``).
* **ivfpq**   -- ``IndexIVFPQ`` (compressed, ``m`` sub-quantizers).

Metrics: ``cosine`` (embeddings are L2-normalised so inner product == cosine),
``ip`` (raw inner product) and ``euclidean`` (L2 distance).

``FaissCosineIndex`` retains its original name and attributes (``dim``, ``ivf``,
``nvec``, ``index``) so existing persistence code and the retrieval engine keep
working unchanged; the extra capabilities are opt-in via the constructor.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import faiss
import numpy as np

VALID_INDEX_TYPES = ("flat", "ivf", "hnsw", "ivfpq")
VALID_METRICS = ("cosine", "ip", "euclidean")


def _faiss_metric(metric: str) -> int:
    return {"cosine": faiss.METRIC_INNER_PRODUCT,
            "ip": faiss.METRIC_INNER_PRODUCT,
            "euclidean": faiss.METRIC_L2}[metric]


class FaissCosineIndex:
    """A FAISS index with cosine/IP/Euclidean semantics and optional GPU."""

    def __init__(
        self,
        dim: int,
        nlist: Optional[int] = None,
        index_type: Optional[str] = None,
        metric: str = "cosine",
        m: int = 8,
        ef_construction: int = 64,
        ef_search: int = 32,
        use_gpu: bool = False,
    ) -> None:
        # Backward compat: ``nlist`` without an explicit index type means IVF
        # (the legacy FaissCosineIndex(dim, nlist=...) call). An explicit type
        # (e.g. "flat" from config) always wins.
        if index_type is None:
            index_type = "ivf" if nlist else "flat"
        if index_type not in VALID_INDEX_TYPES:
            raise ValueError(f"unknown index_type '{index_type}'; choose {VALID_INDEX_TYPES}")
        if metric not in VALID_METRICS:
            raise ValueError(f"unknown metric '{metric}'; choose {VALID_METRICS}")
        self.dim = int(dim)
        self.metric = metric
        self.index_type = index_type
        self.nlist = int(nlist or 32)
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.use_gpu = bool(use_gpu)

        fm = _faiss_metric(metric)
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(dim) if metric != "euclidean" else faiss.IndexFlatL2(dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim) if metric != "euclidean" else faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, fm)
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, self.m, fm)
            self.index.hnsw.efConstruction = self.ef_construction
        elif index_type == "ivfpq":
            quantizer = faiss.IndexFlatIP(dim) if metric != "euclidean" else faiss.IndexFlatL2(dim)
            self.index = faiss.IndexIVFPQ(quantizer, dim, self.nlist, self.m, 8, fm)
        self.ivf = index_type in ("ivf", "ivfpq")
        self._trained = index_type == "flat" or index_type == "hnsw"
        self.nvec = 0
        if use_gpu:
            self._to_gpu()

    # -- normalization --------------------------------------------------------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """L2-normalise (only for the cosine metric, which relies on it)."""
        if self.metric != "cosine":
            return np.ascontiguousarray(x, dtype=np.float32)
        x = np.ascontiguousarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    # -- GPU ------------------------------------------------------------------
    def _to_gpu(self) -> None:
        try:
            if not faiss.get_num_gpus():
                self.use_gpu = False
                return
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        except Exception:
            self.use_gpu = False  # fall back to CPU

    def _cpu_index(self) -> "faiss.Index":
        if self.use_gpu:
            try:
                return faiss.index_gpu_to_cpu(self.index)
            except Exception:  # pragma: no cover
                return self.index
        return self.index

    # -- lifecycle -----------------------------------------------------------
    def add(self, embeddings: np.ndarray) -> None:
        emb = self._normalize(embeddings)
        if self.index_type in ("ivf", "ivfpq") and not self._trained:
            self.index.train(emb)
            self._trained = True
        self.index.add(emb)
        self.nvec += emb.shape[0]

    def train(self, embeddings: np.ndarray) -> None:
        self.index.train(self._normalize(embeddings))
        self._trained = True

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores (Nq, k), ids (Nq, k)). Ids are index positions."""
        q = self._normalize(query)
        k = min(k, self.nvec)
        if k <= 0:
            return np.zeros((q.shape[0], 0), dtype=np.float32), np.zeros((q.shape[0], 0), dtype=np.int64)
        if self.ivf:
            self.index.nprobe = min(8, self.nlist)
        if self.index_type == "hnsw" and self.ef_search:
            self.index.hnsw.efSearch = self.ef_search
        scores, ids = self.index.search(q, k)
        return scores, ids

    def remove(self, ids: np.ndarray) -> int:
        """Remove vectors by their index positions (supported subset of types)."""
        if self.index_type not in ("ivf", "ivfpq", "hnsw"):
            raise NotImplementedError(
                f"remove() is not supported for '{self.index_type}' indexes; use rebuild()"
            )
        removed = int(self.index.remove_ids(faiss.IDSelectorBatch(ids.tolist())))
        self.nvec -= removed
        return removed

    def rebuild(self, embeddings: np.ndarray) -> None:
        """Reset the index and re-add ``embeddings`` (train if needed)."""
        self.index.reset()
        self.nvec = 0
        self._trained = False
        self.add(embeddings)

    def save(self, path: str) -> None:
        idx = self._cpu_index()
        if self.use_gpu:
            idx = self._cpu_index()
        faiss.write_index(idx, path)

    def to_config(self) -> dict:
        return {
            "dim": self.dim,
            "index_type": self.index_type,
            "metric": self.metric,
            "nlist": self.nlist,
            "m": self.m,
            "ef_search": self.ef_search,
            "use_gpu": self.use_gpu,
        }

    @classmethod
    def load(cls, path: str) -> "FaissCosineIndex":
        index = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        obj.nvec = index.ntotal
        obj.metric = "euclidean" if index.metric_type == faiss.METRIC_L2 else "cosine"
        obj.use_gpu = False
        if isinstance(index, faiss.IndexIVFFlat):
            obj.index_type, obj.ivf, obj.nlist = "ivf", True, index.nlist
        elif isinstance(index, faiss.IndexIVFPQ):
            obj.index_type, obj.ivf, obj.nlist = "ivfpq", True, index.nlist
        elif isinstance(index, faiss.IndexHNSWFlat):
            obj.index_type, obj.ivf, obj.nlist = "hnsw", False, 0
            obj.m = getattr(index.hnsw, "M", getattr(index.hnsw, "m", 16))
        else:
            obj.index_type, obj.ivf, obj.nlist = "flat", False, 0
        obj.m = getattr(obj, "m", 8)
        obj.ef_construction = getattr(index, "hnsw", None) and index.hnsw.efConstruction
        obj.ef_search = getattr(getattr(index, "hnsw", None), "efSearch", 32)
        return obj


def build_index_from_embeddings(
    embeddings: np.ndarray,
    nlist: Optional[int] = None,
    index_type: Optional[str] = None,
    metric: str = "cosine",
    m: int = 8,
    ef_search: int = 32,
    use_gpu: bool = False,
) -> FaissCosineIndex:
    """Build an index from an (n, D) embedding matrix.

    ``index_type`` of ``None`` keeps the legacy behaviour: ``nlist`` -> IVF,
    otherwise flat.
    """
    idx = FaissCosineIndex(
        embeddings.shape[1],
        nlist=nlist,
        index_type=index_type,
        metric=metric,
        m=m,
        ef_search=ef_search,
        use_gpu=use_gpu,
    )
    idx.add(embeddings)
    return idx


def index_from_config(cfg: dict) -> dict:
    """Resolve a retrieval.index config section into build kwargs."""
    return {
        "index_type": cfg.get("type", "flat"),
        "metric": cfg.get("metric", "cosine"),
        "nlist": int(cfg.get("nlist") or 32),
        "m": int(cfg.get("m") or 8),
        "ef_search": int(cfg.get("ef_search") or 32),
        "use_gpu": bool(cfg.get("use_gpu", False)),
    }
