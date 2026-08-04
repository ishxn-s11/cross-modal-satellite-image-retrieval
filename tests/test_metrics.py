"""Unit tests for retrieval metrics and split utilities."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import retrieval_metrics
from src.evaluation.evaluate import stratified_split


def test_perfect_retrieval():
    # 4 queries, all retrieved items relevant -> F1 = 1.
    relevant = np.ones((4, 5), dtype=bool)
    total_relevant = np.array([10, 10, 10, 10], dtype=float)
    m = retrieval_metrics(relevant, total_relevant, k=5, time_ms=np.zeros(4))
    assert abs(m.precision - 1.0) < 1e-6
    assert abs(m.recall - 0.5) < 1e-6  # 5 relevant retrieved out of 10
    assert abs(m.f1 - (2 * 1.0 * 0.5) / 1.5) < 1e-6
    assert m.n_queries == 4


def test_empty_retrieval():
    relevant = np.zeros((4, 5), dtype=bool)
    total_relevant = np.array([10, 10, 10, 10], dtype=float)
    m = retrieval_metrics(relevant, total_relevant, k=5, time_ms=np.zeros(4))
    assert abs(m.precision) < 1e-6
    assert abs(m.recall) < 1e-6
    assert abs(m.f1) < 1e-6


def test_stratified_split_balance():
    labels = np.repeat(np.arange(5), 20)  # 5 classes, 20 each
    tr, va, te = stratified_split(labels, 0.6, 0.2, seed=7)
    for c in range(5):
        assert (labels[tr] == c).sum() == 12
        assert (labels[va] == c).sum() == 4
        assert (labels[te] == c).sum() == 4
    assert set(tr) | set(va) | set(te) == set(range(len(labels)))
    assert len(set(tr) & set(te)) == 0


if __name__ == "__main__":
    test_perfect_retrieval()
    test_empty_retrieval()
    test_stratified_split_balance()
    print("test_metrics.py: all tests passed")
