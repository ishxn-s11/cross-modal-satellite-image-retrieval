"""Lightweight re-rankers for the two-stage retrieval pipeline.

Pipeline: FAISS returns ``candidate_k`` candidates; an optional re-ranker
re-scores them and the top ``final_k`` are returned. Re-ranking is *optional*
so users can compare FAISS-only vs FAISS + re-ranking.

Implemented re-rankers:

* :class:`IdentityReranker` -- keeps the FAISS order (no-op).
* :class:`GeoReranker`      -- adds a geographic affinity bonus when the query
  and candidates have coordinates (nearby scenes rank higher).
* :class:`MLPReranker`      -- a small trainable MLP over
  ``[q ; c ; q*c ; |q-c|]`` trained to predict same-class relevance
  (:meth:`MLPReranker.fit`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.metadata import ImageMetadata
from ..training.geo import haversine_km

# ---------------------------------------------------------------------------
# Base + identity
# ---------------------------------------------------------------------------


class ReRanker(ABC):
    name: str = "identity"

    @abstractmethod
    def score(
        self,
        q_emb: np.ndarray,
        c_embs: np.ndarray,
        q_meta: Optional[ImageMetadata] = None,
        c_metas: Optional[Sequence[ImageMetadata]] = None,
    ) -> np.ndarray:
        """Return a per-candidate score (higher = better), shape (n_candidates,)."""
        raise NotImplementedError

    def state_dict(self) -> Dict:
        return {"name": self.name}

    @classmethod
    def from_state(cls, state: Dict) -> "ReRanker":
        return cls()


class IdentityReranker(ReRanker):
    name = "identity"

    def score(self, q_emb, c_embs, q_meta=None, c_metas=None) -> np.ndarray:
        return c_embs @ q_emb


# ---------------------------------------------------------------------------
# Geographic re-ranker
# ---------------------------------------------------------------------------


class GeoReranker(IdentityReranker):
    """FAISS cosine + a geographic affinity bonus.

    ``affinity = exp(-dist_km / scale_km)``; candidates near the query rank
    higher. Candidates without coordinates get a zero bonus.
    """

    name = "geo"

    def __init__(self, geo_weight: float = 0.3, scale_km: float = 50.0) -> None:
        self.geo_weight = float(geo_weight)
        self.scale_km = float(scale_km)

    def score(self, q_emb, c_embs, q_meta=None, c_metas=None) -> np.ndarray:
        base = super().score(q_emb, c_embs, q_meta, c_metas)
        if self.geo_weight <= 0 or q_meta is None or not c_metas:
            return base
        if q_meta.latitude is None or q_meta.longitude is None:
            return base
        bonus = np.zeros(len(c_embs), dtype=np.float32)
        for i, m in enumerate(c_metas):
            if m is not None and m.latitude is not None and m.longitude is not None:
                d = haversine_km(
                    np.float32(q_meta.latitude), np.float32(q_meta.longitude),
                    np.float32(m.latitude), np.float32(m.longitude),
                )
                bonus[i] = float(np.exp(-float(d) / self.scale_km))
        return base + self.geo_weight * bonus

    def state_dict(self) -> Dict:
        return {"name": self.name, "geo_weight": self.geo_weight, "scale_km": self.scale_km}

    @classmethod
    def from_state(cls, state: Dict) -> "GeoReranker":
        return cls(geo_weight=state.get("geo_weight", 0.3), scale_km=state.get("scale_km", 50.0))


# ---------------------------------------------------------------------------
# Trainable MLP re-ranker
# ---------------------------------------------------------------------------


def _features(q_emb: np.ndarray, c_embs: np.ndarray) -> np.ndarray:
    """[q ; c ; q*c ; |q-c|] feature vector per candidate."""
    q = q_emb[None, :]
    qc = q * c_embs
    qd = np.abs(q - c_embs)
    return np.concatenate([np.broadcast_to(q, c_embs.shape), c_embs, qc, qd], axis=1)


class MLPReranker(ReRanker):
    """Small trainable MLP re-ranker over concatenated query/candidate features.

    ``fit`` builds positive (same class) / negative (different class) pairs
    from a gallery and trains the MLP to output relevance scores. Requires
    ``torch`` (already a project dependency). Keeping the MLP small (one hidden
    layer) keeps re-ranking cheap.
    """

    name = "mlp"

    def __init__(self, hidden: int = 64, emb_dim: int = 128) -> None:
        import torch

        self.hidden = int(hidden)
        self.emb_dim = int(emb_dim)
        self.torch = torch
        in_dim = 4 * self.emb_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 1),
        )
        self.model.eval()
        self._fitted = False

    def fit(
        self,
        gallery_embs: np.ndarray,
        gallery_labels: np.ndarray,
        query_embs: np.ndarray,
        query_labels: np.ndarray,
        epochs: int = 3,
        lr: float = 1e-3,
        device=None,
    ) -> "MLPReranker":
        """Train the re-ranker with same-class pairs as positives.

        Uses a handful of anchors (``query_embs``) against the gallery; pairs
        sharing the query's class are targets=1, different class targets=0.
        """
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Sample a fixed set of anchors for stable, bounded training.
        rng = np.random.RandomState(0)
        n_anchors = min(64, len(query_embs))
        anchors = rng.choice(len(query_embs), n_anchors, replace=False)
        X_all, y_all = [], []
        for a in anchors:
            feats = _features(query_embs[a], gallery_embs)
            targets = (gallery_labels == query_labels[a]).astype(np.float32)
            X_all.append(feats)
            y_all.append(targets)
        X = torch.tensor(np.concatenate(X_all), dtype=torch.float32)
        y = torch.tensor(np.concatenate(y_all), dtype=torch.float32)
        n = X.shape[0]
        for _ in range(int(epochs)):
            perm = torch.randperm(n)
            for start in range(0, n, 256):
                idx = perm[start : start + 256]
                opt.zero_grad()
                loss = loss_fn(self.model(X[idx]).squeeze(1), y[idx])
                loss.backward()
                opt.step()
        self.model.eval()
        self._fitted = True
        return self

    def score(self, q_emb, c_embs, q_meta=None, c_metas=None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("MLPReranker.fit() must be called before scoring")
        feats = _features(q_emb, c_embs)
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(feats.astype(np.float32)))
        return out.numpy().ravel()

    def state_dict(self) -> Dict:
        import io

        buf = io.BytesIO()
        self.torch.save(self.model.state_dict(), buf)
        return {
            "name": self.name,
            "hidden": self.hidden,
            "emb_dim": self.emb_dim,
            "fitted": self._fitted,
            "weights": buf.getvalue(),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "MLPReranker":
        import io

        import torch

        obj = cls(hidden=state.get("hidden", 64), emb_dim=state.get("emb_dim", 128))
        obj.model.load_state_dict(torch.load(io.BytesIO(state["weights"]), map_location="cpu"))
        obj._fitted = bool(state.get("fitted", True))
        obj.model.eval()
        return obj


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

RERANKERS = {"identity": IdentityReranker, "geo": GeoReranker, "mlp": MLPReranker}


def build_reranker(cfg: Optional[Dict]) -> Optional[ReRanker]:
    """Build a re-ranker from the ``retrieval.rerank`` config section.

    Returns ``None`` when re-ranking is disabled.
    """
    if not cfg or not cfg.get("enabled", False):
        return None
    method = cfg.get("method", "identity")
    if method not in RERANKERS:
        raise ValueError(f"unknown rerank method '{method}'; choose {sorted(RERANKERS)}")
    cls = RERANKERS[method]
    if method == "geo":
        return cls(geo_weight=float(cfg.get("geo_weight", 0.3)), scale_km=float(cfg.get("scale_km", 50.0)))
    return cls()
