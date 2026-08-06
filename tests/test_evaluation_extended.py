"""Tests for the Phase 8 evaluation extensions (mAP/NDCG, latency, scalability)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import (
    average_precision_at_k,
    mean_average_precision,
    mean_ndcg,
    ndcg_at_k,
    recall_curve,
    retrieval_metrics,
)
from src.utils.config import set_nested


def test_average_precision_at_k():
    # relevant at positions 1 and 3 of 5: P@1=1, P@3=2/3 -> AP=(1+2/3)/min(5,2)=0.8333
    rel = np.array([1, 0, 1, 0, 0], dtype=bool)
    ap = average_precision_at_k(rel, 5)
    assert abs(ap - (1.0 + 2 / 3) / 2) < 1e-6
    # no relevant -> 0
    assert average_precision_at_k(np.zeros(5, bool), 5) == 0.0


def test_mean_average_precision():
    rel = np.array([[1, 0, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=bool)
    ap0 = average_precision_at_k(rel[0], 5)
    m = mean_average_precision(rel, 5)
    assert abs(m - ap0 / 2) < 1e-6


def test_ndcg_at_k():
    rel = np.array([1, 0, 1, 0, 0], dtype=bool)
    dcg = 1.0 / np.log2(2) + 1.0 / np.log2(4)
    idcg = 1.0 / np.log2(2) + 1.0 / np.log2(3)
    assert abs(ndcg_at_k(rel, 5) - dcg / idcg) < 1e-6
    assert ndcg_at_k(np.zeros(5, bool), 5) == 0.0


def test_mean_ndcg():
    rel = np.array([[1, 0, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=bool)
    m = mean_ndcg(rel, 5)
    assert 0.0 <= m <= 1.0


def test_recall_curve_monotonic():
    rel = np.array([[1, 0, 1, 0, 0, 1], [0, 1, 1, 0, 0, 0]], dtype=bool)
    total = np.array([10, 10])
    curve = recall_curve(rel, total, 6)
    assert list(curve.keys()) == list(range(1, 7))
    assert all(curve[k + 1] >= curve[k] - 1e-9 for k in range(1, 6))


def test_retrieval_metrics_includes_map_ndcg():
    rel = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]], dtype=bool)
    total = np.array([10, 10])
    m = retrieval_metrics(rel, total, 5, time_ms=np.zeros(2))
    assert m.map > 0.0 and m.map <= 1.0
    assert m.ndcg > 0.0 and m.ndcg <= 1.0


def test_set_nested():
    cfg = {}
    set_nested(cfg, "a.b.c", 5)
    set_nested(cfg, "a.b.d", "x")
    assert cfg["a"]["b"]["c"] == 5
    assert cfg["a"]["b"]["d"] == "x"


def test_scalability_small():
    from src.evaluation.scalability import benchmark_scalability

    rows = benchmark_scalability(
        sizes=(1000, 2000), d=16, index_types=("flat", "ivf"),
        k=5, n_query=30, max_vectors=10000, seed=0,
    )
    assert len(rows) == 4
    flat = [r for r in rows if r["index_type"] == "flat"]
    assert all(r["recall_at_k"] == 1.0 for r in flat)
    assert all(r["search_mean_ms"] >= 0 for r in rows)


def test_latency_report_percentiles():
    from src.evaluation.latency import LatencyReport

    rep = LatencyReport(
        n_queries=4, k=5, candidate_k=None, reranker_name="none",
        preprocessing_ms=np.array([1.0, 2.0, 3.0, 4.0]),
        embedding_ms=np.array([5.0, 6.0, 7.0, 8.0]),
        search_ms=np.array([0.1, 0.2, 0.3, 0.4]),
        rerank_ms=np.zeros(4), total_ms=np.array([6.1, 8.2, 10.3, 12.4]),
    )
    d = rep.to_dict()
    assert d["stages"]["preprocessing"]["mean"] == 2.5
    assert d["throughput_qps"] > 0


if __name__ == "__main__":
    test_average_precision_at_k()
    test_mean_average_precision()
    test_ndcg_at_k()
    test_mean_ndcg()
    test_recall_curve_monotonic()
    test_retrieval_metrics_includes_map_ndcg()
    test_set_nested()
    test_scalability_small()
    test_latency_report_percentiles()
    print("test_evaluation_extended.py: all tests passed")
