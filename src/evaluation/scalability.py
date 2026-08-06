"""Scalability benchmark: latency / recall / memory across index types & sizes.

Benchmarks flat / IVF / HNSW / IVF-PQ at gallery sizes like 10K / 100K / 1M
vectors, reporting:

* build time,
* search latency (mean, P50, P95),
* recall@k against the exact (flat) result,
* estimated index memory.

The 1M vector tier is opt-in (it needs ~0.5 GB for flat + the PQ/IVF copies);
run it only where the hardware permits (scripts/benchmark_scalability.py).
All numbers are measured, never fabricated.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..retrieval.index import build_index_from_embeddings
from .metrics import recall_curve


def _memory_mb(emb: np.ndarray, index_type: str, nlist: int, m: int) -> float:
    """Documented estimate of index memory in MB."""
    n, d = emb.shape
    base = n * d * 4.0
    if index_type == "ivf":
        base += nlist * d * 4.0
    elif index_type == "hnsw":
        base += n * 2 * m * 4.0  # link lists
    elif index_type == "ivfpq":
        base = n * (m * 1.0) + nlist * d * 4.0  # compressed codes + coarse quantizer
    return round(base / (1024.0 * 1024.0), 1)


def _search_latency(idx, queries: np.ndarray, k: int) -> Dict[str, float]:
    # Warm-up + measured batch of NQ queries.
    idx.search(queries[:1], k)
    times = []
    for q in queries:
        t0 = time.perf_counter()
        idx.search(q[None], k)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
    }


def _recall_vs_flat(gt_ids: np.ndarray, ids: np.ndarray) -> float:
    """Mean recall@k of retrieved ids against the flat (exact) ground truth."""
    hits = 0
    nq = gt_ids.shape[0]
    for i in range(nq):
        hits += len(set(gt_ids[i]) & set(ids[i]))
    return round(hits / (nq * gt_ids.shape[1]), 4)


def benchmark_scalability(
    sizes: Sequence[int] = (10_000, 100_000),
    d: int = 128,
    index_types: Sequence[str] = ("flat", "ivf", "hnsw", "ivfpq"),
    k: int = 10,
    n_query: int = 200,
    max_vectors: int = 1_000_000,
    seed: int = 0,
) -> List[Dict]:
    """Run the scalability benchmark; returns a list of result rows."""
    rng = np.random.RandomState(seed)
    rows: List[Dict] = []
    for n in sizes:
        if n > max_vectors:
            continue
        print(f"[scalability] generating {n:,} x {d} vectors ...")
        emb = rng.rand(n, d).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        queries = rng.rand(min(n_query, max(1, n // 100)), d).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Exact ground truth (flat cosine).
        t0 = time.perf_counter()
        gt = build_index_from_embeddings(emb)
        gt_scores, gt_ids = gt.search(queries, k)
        gt_time = (time.perf_counter() - t0) * 1000.0

        for itype in index_types:
            if itype not in ("flat", "ivf", "hnsw", "ivfpq"):
                continue
            t0 = time.perf_counter()
            idx = build_index_from_embeddings(
                emb, index_type=itype, nlist=16, m=8, ef_search=32
            )
            build_s = time.perf_counter() - t0
            lat = _search_latency(idx, queries, k)
            _, ids = idx.search(queries, k)
            rows.append(
                {
                    "n_vectors": int(n),
                    "dim": int(d),
                    "index_type": itype,
                    "build_s": round(build_s, 3),
                    "search_mean_ms": lat["mean_ms"],
                    "search_p50_ms": lat["p50_ms"],
                    "search_p95_ms": lat["p95_ms"],
                    "recall_at_k": _recall_vs_flat(gt_ids, ids) if itype != "flat" else 1.0,
                    "memory_mb_est": _memory_mb(emb, itype, 16, 8),
                }
            )
            print(
                f"  {itype:>6} n={n:,} build={build_s:.2f}s "
                f"search={lat['mean_ms']:.4f}ms recall@{k}={rows[-1]['recall_at_k']}"
            )
    return rows
