"""Embedding-space visualisation: PCA / t-SNE / UMAP.

Projects embeddings to 2-D and colours them by class / modality / geographic
region / dataset, plus a before-vs-after-training comparison to show whether
modalities become aligned.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - sklearn is a core dependency
    _HAS_SKLEARN = False

METHODS = ("pca", "tsne", "umap")


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    n_samples: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Project embeddings to 2-D via pca / tsne / umap."""
    if method not in METHODS:
        raise ValueError(f"unknown method '{method}'; choose {METHODS}")
    x = np.asarray(embeddings, dtype=np.float32)
    rng = np.random.RandomState(seed)
    if n_samples and len(x) > n_samples:
        idx = rng.choice(len(x), n_samples, replace=False)
        x = x[idx]
    if method == "umap":
        try:
            import umap  # optional dependency

            return umap.UMAP(random_state=seed).fit_transform(x)
        except Exception:
            method = "pca"  # graceful fallback (documented)
    if method == "tsne":
        return TSNE(n_components=n_components, random_state=seed).fit_transform(x)
    return PCA(n_components=n_components, random_state=seed).fit_transform(x)


def _scatter(ax, xy: np.ndarray, ids: np.ndarray, n_groups: int, labels: Sequence[str],
             title: str) -> None:
    from matplotlib import colormaps

    cmap = colormaps["tab10"]
    for g in range(n_groups):
        m = ids == g
        if not m.any():
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.6, color=cmap(g), label=str(labels[g]))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2)


def plot_embeddings(
    embeddings: np.ndarray,
    group_ids: np.ndarray,
    group_names: Sequence[str],
    method: str = "pca",
    n_samples: int = 3000,
    seed: int = 0,
    title: str = "",
):
    """Plot embeddings coloured by a group id (class / modality / region / dataset)."""
    import matplotlib.pyplot as plt

    xy = reduce_embeddings(embeddings, method=method, n_samples=n_samples, seed=seed)
    ids = np.asarray(group_ids)
    if len(xy) != len(ids):  # subsampling happened
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ids), len(xy), replace=False)
        ids = ids[idx]
    n_groups = max(1, int(ids.max()) + 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter(ax, xy, ids, n_groups, group_names, title or f"Embeddings ({method.upper()})")
    plt.tight_layout()
    return fig


def plot_embedding_comparison(
    before: np.ndarray,
    after: np.ndarray,
    group_ids: np.ndarray,
    group_names: Sequence[str],
    method: str = "pca",
    n_samples: int = 3000,
    seed: int = 0,
    before_title: str = "Before training",
    after_title: str = "After training",
):
    """Side-by-side before/after training embeddings (demonstrates alignment)."""
    import matplotlib.pyplot as plt

    xy_before = reduce_embeddings(before, method=method, n_samples=n_samples, seed=seed)
    xy_after = reduce_embeddings(after, method=method, n_samples=n_samples, seed=seed)
    ids = np.asarray(group_ids)
    if len(xy_before) != len(ids):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(ids), len(xy_before), replace=False)
        ids = ids[idx]
    n_groups = max(1, int(ids.max()) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    _scatter(ax1, xy_before, ids, n_groups, group_names, before_title)
    _scatter(ax2, xy_after, ids, n_groups, group_names, after_title)
    plt.tight_layout()
    return fig


def modality_color_ids(modalities: Sequence[str], modality_order: Sequence[str]) -> np.ndarray:
    """Map a per-row modality label to a colour group id."""
    return np.array([modality_order.index(m) for m in modalities], dtype=np.int64)
