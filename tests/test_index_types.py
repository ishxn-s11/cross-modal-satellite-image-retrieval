"""Unit tests for the FAISS index types + metrics."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.index import FaissCosineIndex, build_index_from_embeddings


def _emb(n, d, seed=0):
    rng = np.random.RandomState(seed)
    return rng.rand(n, d).astype(np.float32)


def test_flat_cosine_search():
    emb = _emb(100, 16)
    idx = build_index_from_embeddings(emb)
    assert idx.nvec == 100
    scores, ids = idx.search(emb[:5], 3)
    assert scores.shape == (5, 3)
    assert ids.shape == (5, 3)
    # self-similarity is 1.0 for the cosine metric
    s, i = idx.search(emb[0:1], 1)
    assert i[0, 0] == 0
    assert abs(s[0, 0] - 1.0) < 1e-3


def test_euclidean_metric_returns_distance():
    emb = _emb(50, 8, 1)
    idx = build_index_from_embeddings(emb, metric="euclidean")
    s, i = idx.search(emb[0:1], 1)
    assert i[0, 0] == 0
    assert s[0, 0] < 1e-3  # distance to itself is ~0


def test_ivf_hnsw_ivfpq_build_and_search():
    emb = _emb(1000, 16, 2)
    for itype in ("ivf", "hnsw", "ivfpq"):
        idx = build_index_from_embeddings(emb, index_type=itype, nlist=8, m=4)
        s, i = idx.search(emb[:3], 3)
        assert s.shape == (3, 3), itype
        assert idx.nvec == 1000, itype


def test_save_load_roundtrip(tmp_path):
    emb = _emb(80, 16, 3)
    idx = build_index_from_embeddings(emb, index_type="hnsw", m=8)
    p = os.path.join(str(tmp_path), "i.index")
    idx.save(p)
    loaded = FaissCosineIndex.load(p)
    assert loaded.nvec == 80
    s1, i1 = idx.search(emb[:2], 3)
    s2, i2 = loaded.search(emb[:2], 3)
    assert np.array_equal(i1, i2)


def test_remove_and_rebuild():
    emb = _emb(60, 16, 4)
    idx = build_index_from_embeddings(emb, index_type="ivf", nlist=6)
    removed = idx.remove(np.array([0, 1, 2]))
    assert removed == 3
    assert idx.nvec == 57
    idx.rebuild(emb)
    assert idx.nvec == 60


def test_invalid_index_type_and_metric_raise():
    try:
        build_index_from_embeddings(np.zeros((4, 8), np.float32), index_type="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    try:
        build_index_from_embeddings(np.zeros((4, 8), np.float32), metric="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    import tempfile

    test_flat_cosine_search()
    test_euclidean_metric_returns_distance()
    test_ivf_hnsw_ivfpq_build_and_search()
    test_save_load_roundtrip(tempfile.mkdtemp())
    test_remove_and_rebuild()
    test_invalid_index_type_and_metric_raise()
    print("test_index_types.py: all tests passed")
