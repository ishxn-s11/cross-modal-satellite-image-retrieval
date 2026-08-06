"""Query-latency benchmark with per-stage breakdown.

Measures, per query: preprocessing, embedding generation, FAISS search and
(optional) re-ranking, plus the total. Reports mean / P50 / P95 and throughput
so latency is reproducible (see scripts/benchmark_latency.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..retrieval.engine import Gallery, RetrievalEngine
from ..retrieval.rerank import ReRanker


@dataclass
class LatencyReport:
    n_queries: int
    k: int
    candidate_k: Optional[int]
    reranker_name: str
    preprocessing_ms: np.ndarray = field(repr=False)
    embedding_ms: np.ndarray = field(repr=False)
    search_ms: np.ndarray = field(repr=False)
    rerank_ms: np.ndarray = field(repr=False)
    total_ms: np.ndarray = field(repr=False)

    def percentiles(self, arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
        }

    def to_dict(self) -> Dict:
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "candidate_k": self.candidate_k,
            "reranker": self.reranker_name,
            "stages": {
                "preprocessing": self.percentiles(self.preprocessing_ms),
                "embedding": self.percentiles(self.embedding_ms),
                "search": self.percentiles(self.search_ms),
                "rerank": self.percentiles(self.rerank_ms),
                "total": self.percentiles(self.total_ms),
            },
            "throughput_qps": round(float(1.0 / np.maximum(np.mean(self.total_ms) / 1000.0, 1e-9)), 2),
        }


def _time_call(fn, *args, repeat: int = 3) -> float:
    """Median wall-clock (ms) of ``fn(*args)`` over ``repeat`` calls."""
    times = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def run_latency_benchmark(
    engine: RetrievalEngine,
    gallery: Gallery,
    query_indices: Sequence[int],
    query_modality: str,
    k: int,
    candidate_k: Optional[int] = None,
    reranker: Optional[ReRanker] = None,
    n_repeat: int = 3,
) -> LatencyReport:
    """Benchmark per-query latency stages over ``query_indices``."""
    query_ids = np.asarray(list(query_indices), dtype=np.int64)
    pre = np.zeros(len(query_ids), dtype=np.float64)
    emb = np.zeros(len(query_ids), dtype=np.float64)
    search = np.zeros(len(query_ids), dtype=np.float64)
    rerank_t = np.zeros(len(query_ids), dtype=np.float64)
    total = np.zeros(len(query_ids), dtype=np.float64)

    engine.model.eval()
    search_k = min(candidate_k or k, gallery.size)
    for i, qid in enumerate(query_ids):
        t0 = time.perf_counter()
        sample = engine.dataset[int(qid)]
        x = torch.stack([sample[query_modality]])  # batch of 1
        pre[i] = _time_call(
            lambda: engine.dataset[int(qid)][query_modality], repeat=n_repeat
        )
        emb[i] = _time_call(
            lambda: engine.model.embed(x.to(engine.device), query_modality),
            repeat=n_repeat,
        )
        q_emb = engine.model.embed(x.to(engine.device), query_modality).detach().cpu().numpy()
        search[i] = _time_call(
            lambda: gallery.index.search(q_emb, search_k), repeat=n_repeat
        )
        if reranker is not None and search_k > k:
            cand_s, cand_ids = gallery.index.search(q_emb, search_k)
            row_embs = gallery.embeddings[cand_ids[0]]
            q_meta = engine._query_metadata([int(qid)])
            q_meta = q_meta[0] if q_meta else None
            c_metas = [engine.dataset.metadata_for(int(j)) for j in cand_ids[0]] if engine.dataset.metadata else None
            rerank_t[i] = _time_call(
                lambda: reranker.score(q_emb[0], row_embs, q_meta, c_metas),
                repeat=n_repeat,
            )
        total[i] = pre[i] + emb[i] + search[i] + rerank_t[i]

    return LatencyReport(
        n_queries=len(query_ids),
        k=int(k),
        candidate_k=int(search_k) if search_k != k else None,
        reranker_name=reranker.name if reranker is not None else "none",
        preprocessing_ms=pre,
        embedding_ms=emb,
        search_ms=search,
        rerank_ms=rerank_t,
        total_ms=total,
    )
