"""Unit tests for the re-ranking module."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.metadata import ImageMetadata
from src.retrieval.rerank import (
    GeoReranker,
    IdentityReranker,
    MLPReranker,
    build_reranker,
)


def test_identity_score_is_cosine():
    r = IdentityReranker()
    q = np.array([1.0, 0.0])
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = r.score(q, c)
    assert s[0] > s[1]


def test_geo_reranker_ranks_nearby_higher():
    r = GeoReranker(geo_weight=1.0, scale_km=10.0)
    q = np.array([1.0, 0.0])
    c = np.array([[0.99, 0.02], [0.1, 0.1]])
    q_meta = ImageMetadata(latitude=0.0, longitude=0.0)
    c_metas = [
        ImageMetadata(latitude=0.0, longitude=0.001),  # ~0.1 km away
        ImageMetadata(latitude=50.0, longitude=50.0),  # thousands of km away
    ]
    s = r.score(q, c, q_meta, c_metas)
    assert s[0] > s[1]


def test_geo_reranker_no_metadata_returns_base():
    r = GeoReranker(geo_weight=1.0)
    q = np.array([1.0, 0.0])
    c = np.array([[1.0, 0.0], [0.0, 1.0]])
    s = r.score(q, c)  # no metadata -> base cosine only
    assert s[0] > s[1]


def _structured(n_per=50, d=8, seed=0):
    """Class-structured, normalised embeddings (centers + noise)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(4, d)
    labels = np.tile(np.arange(4), n_per)
    x = centers[labels] + 0.1 * rng.randn(4 * n_per, d)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32), labels


def test_mlp_reranker_learns_relevance():
    import torch

    torch.manual_seed(0)
    gallery, gl = _structured(80)
    queries, ql = _structured(10, seed=1)
    r = MLPReranker(emb_dim=8, hidden=16)
    r.fit(gallery, gl, queries, ql, epochs=5)
    s = r.score(queries[0], gallery)
    assert s.shape == (len(gallery),)
    same = s[gl == ql[0]].mean()
    diff = s[gl != ql[0]].mean()
    assert same > diff


def test_mlp_state_roundtrip(tmp_path):
    import torch

    torch.manual_seed(0)
    gallery, gl = _structured(30, seed=2)
    queries, ql = _structured(3, seed=3)
    r = MLPReranker(emb_dim=8, hidden=8)
    r.fit(gallery, gl, queries, ql, epochs=3)
    state = r.state_dict()
    r2 = MLPReranker.from_state(state)
    s1 = r.score(queries[0], gallery)
    s2 = r2.score(queries[0], gallery)
    assert np.allclose(s1, s2, atol=1e-5)


def test_build_reranker():
    assert build_reranker(None) is None
    assert build_reranker({"enabled": False}) is None
    assert isinstance(build_reranker({"enabled": True, "method": "identity"}), IdentityReranker)
    assert isinstance(build_reranker({"enabled": True, "method": "geo"}), GeoReranker)


if __name__ == "__main__":
    import tempfile

    test_identity_score_is_cosine()
    test_geo_reranker_ranks_nearby_higher()
    test_geo_reranker_no_metadata_returns_base()
    test_mlp_reranker_learns_relevance()
    test_mlp_state_roundtrip(tempfile.mkdtemp())
    test_build_reranker()
    print("test_rerank.py: all tests passed")
