"""Retrieval quality metrics: precision@K, recall@K and F1@K.

Ground truth is *semantic relevance*: an item is relevant to a query when it
shares the query's land-cover class. For each query and cutoff K:

    precision@K = |retrieved x relevant| / K
    recall@K    = |retrieved x relevant| / |relevant in gallery|
    F1@K        = 2 * precision * recall / (precision + recall)

Results are averaged over all queries in the evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class RetrievalMetrics:
    k: int
    precision: float
    recall: float
    f1: float
    avg_time_ms: float
    n_queries: int
    precision_std: float
    recall_std: float
    f1_std: float
    map: float = 0.0          # mean average precision @k
    ndcg: float = 0.0         # normalized discounted cumulative gain @k


def _per_query(
    relevant: np.ndarray,  # (Nq, k) bool of hits
    total_relevant: np.ndarray,  # (Nq,) relevant items in the gallery
    k: int,
) -> Dict[str, np.ndarray]:
    hits = relevant.sum(axis=1).astype(np.float64)
    precision = hits / max(1, k)
    denom = np.maximum(total_relevant.astype(np.float64), 1e-8)
    recall = hits / denom
    score = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(score > 0, 2.0 * precision * recall / score, 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision_at_k(relevant: np.ndarray, k: int) -> float:
    """Average precision @k for a single query (0 if no relevant items)."""
    row = relevant[:k]
    n_rel = int(row.sum())
    if n_rel == 0:
        return 0.0
    hits = np.cumsum(row)
    precisions = hits / np.arange(1, k + 1)
    return float((precisions * row).sum() / min(k, n_rel))


def mean_average_precision(relevant: np.ndarray, k: int) -> float:
    """mAP@k over a (Nq, k) boolean relevance matrix."""
    if relevant.shape[0] == 0:
        return 0.0
    return float(np.mean([average_precision_at_k(row, k) for row in relevant]))


def ndcg_at_k(relevant: np.ndarray, k: int) -> float:
    """NDCG@k for a single query (binary gains), 0 if no relevant items."""
    gains = relevant[:k].astype(np.float64)
    dcg = float((gains / np.log2(np.arange(2, k + 2))).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal / np.log2(np.arange(2, k + 2))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg(relevant: np.ndarray, k: int) -> float:
    """Mean NDCG@k over a (Nq, k) boolean relevance matrix."""
    if relevant.shape[0] == 0:
        return 0.0
    return float(np.mean([ndcg_at_k(row, k) for row in relevant]))


def recall_curve(
    relevant: np.ndarray, total_relevant: np.ndarray, max_k: int
) -> Dict[int, float]:
    """Recall@k for k = 1..max_k, averaged over queries."""
    out: Dict[int, float] = {}
    denom = np.maximum(total_relevant.astype(np.float64), 1e-8)
    for k in range(1, int(max_k) + 1):
        hits = relevant[:, :k].sum(axis=1).astype(np.float64)
        out[k] = float((hits / denom).mean())
    return out


def retrieval_metrics(
    relevant: np.ndarray,
    total_relevant: np.ndarray,
    k: int,
    time_ms: np.ndarray,
) -> RetrievalMetrics:
    """Aggregate P/R/F1/mAP/NDCG over all queries for a given cutoff K."""
    per = _per_query(relevant, total_relevant, k)
    return RetrievalMetrics(
        k=k,
        precision=float(per["precision"].mean()),
        recall=float(per["recall"].mean()),
        f1=float(per["f1"].mean()),
        precision_std=float(per["precision"].std()),
        recall_std=float(per["recall"].std()),
        f1_std=float(per["f1"].std()),
        avg_time_ms=float(time_ms.mean()),
        n_queries=int(relevant.shape[0]),
        map=mean_average_precision(relevant, k),
        ndcg=mean_ndcg(relevant, k),
    )


def to_dict(m: RetrievalMetrics, pair_desc: str = "", kind: str = "") -> Dict:
    return {
        "pair": pair_desc,
        "kind": kind,
        "k": m.k,
        "n_queries": m.n_queries,
        "precision@k": round(m.precision, 4),
        "recall@k": round(m.recall, 4),
        "f1@k": round(m.f1, 4),
        "f1@k_std": round(m.f1_std, 4),
        "map@k": round(m.map, 4),
        "ndcg@k": round(m.ndcg, 4),
        "avg_retrieval_time_ms": round(m.avg_time_ms, 4),
    }


_row_head = ["pair", "kind", "k", "f1@k", "precision@k", "recall@k", "map@k", "ndcg@k", "avg_time_ms"]


def format_table(rows: Sequence[Dict]) -> str:
    headers = _row_head
    lines = []
    fmt = "{:<22}{:<7}{:>5}{:>10}{:>12}{:>12}{:>10}{:>10}{:>14}"
    lines.append(fmt.format(*headers))
    lines.append("-" * 78)
    for r in rows:
        lines.append(
            fmt.format(
                r["pair"],
                r["kind"],
                r["k"],
                r["f1@k"],
                r["precision@k"],
                r["recall@k"],
                r["map@k"],
                r["ndcg@k"],
                r["avg_retrieval_time_ms"],
            )
        )
    return "\n".join(lines)