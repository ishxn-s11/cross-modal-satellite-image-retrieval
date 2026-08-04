"""FAISS index for fast cosine-similarity retrieval.

Embeddings are L2-normalised before insertion so that the inner-product index
(``IndexFlatIP``) computes cosine similarity. For gallery sizes used here
(tens of thousands) a flat index is both exact and fast (sub-ms per query on
CPU); it is also the natural baseline the task asks to beat.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import faiss
import numpy as np


class FaissCosineIndex:
    def __init__(self, dim: int, nlist: Optional[int] = None) -> None:
        self.dim = dim
        if nlist:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.ivf = True
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.ivf = False
        self.nvec = 0

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return x / norms

    def add(self, embeddings: np.ndarray) -> None:
        emb = self._normalize(embeddings)
        if self.ivf and self.nvec == 0:
            self.index.train(emb)
        self.index.add(emb)
        self.nvec += emb.shape[0]

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores (Nq, k), ids (Nq, k)). Ids are positions in the index."""
        q = self._normalize(query)
        k = min(k, self.nvec)
        if self.ivf:
            self.index.nprobe = min(8, self.index.nlist)
        scores, ids = self.index.search(q, k)
        return scores, ids

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str) -> "FaissCosineIndex":
        index = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        obj.ivf = isinstance(index, faiss.IndexIVFFlat)
        obj.nvec = index.ntotal
        return obj


def build_index_from_embeddings(embeddings: np.ndarray, nlist: Optional[int] = None) -> FaissCosineIndex:
    idx = FaissCosineIndex(embeddings.shape[1], nlist=nlist)
    idx.add(embeddings)
    return idx
