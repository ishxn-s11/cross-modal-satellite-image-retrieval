"""Shared helpers for the FAISS embeddings / retrieval notebooks.

Notebooks in this folder ``import utils`` after bootstrapping the project root
onto ``sys.path``. These helpers reuse the project's ``src/`` pipeline so the
notebooks behave identically to ``run_pipeline.py`` but stay concise.
"""

from __future__ import annotations

import os
import sys

# Make the project root importable when running from Jupyter (either cwd == root
# or the kernel was launched inside notebooks/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------------

def load_pipeline(cfg_path: str = "configs/default.yaml", use_persistence: bool = True) -> dict:
    """Return the objects the pipeline shares: config, data, model, engine, stores.

    Uses the persistent stores so gallery/embedding work hits disk on warm runs.
    """
    from src.pipeline import build_loaders, load_best_model, prepare_dataset
    from src.retrieval.engine import RetrievalEngine
    from src.utils.config import load_config
    from src.utils.io import Logger, resolve_device

    cfg = load_config(cfg_path)
    device = resolve_device(cfg["training"]["device"])
    patches, labels, class_names, _stats, transforms = prepare_dataset(cfg, Logger(path=None))
    model = load_best_model(cfg, len(class_names), device).to(device)
    _, _, full_ds = build_loaders(cfg, patches, labels, transforms)

    store_kw = {}
    if use_persistence:
        from src.database import EmbeddingStore, IndexStore, compute_cache_key
        p = cfg.get("persistence", {})
        store_kw = dict(
            embedding_store=EmbeddingStore(p.get("embeddings_dir", "embeddings")),
            index_store=IndexStore(p.get("faiss_dir", "faiss")),
            config_hash=compute_cache_key(
                cfg, os.path.join(cfg["outputs"]["model_dir"], "best_model", "model.pt")
            ),
        )
    engine = RetrievalEngine(model, full_ds, device, **store_kw)

    return {
        "cfg": cfg, "patches": patches, "labels": labels, "class_names": class_names,
        "model": model, "full_ds": full_ds, "engine": engine, "device": device,
    }


def load_dataset(cfg_path: str = "configs/default.yaml") -> dict:
    """Load dataset + dataloaders only (no model) -- for exploration notebooks."""
    from src.pipeline import build_loaders, prepare_dataset
    from src.utils.config import load_config
    from src.utils.io import Logger

    cfg = load_config(cfg_path)
    patches, labels, class_names, _stats, transforms = prepare_dataset(cfg, Logger(path=None))
    _, _, full_ds = build_loaders(cfg, patches, labels, transforms)
    return {"cfg": cfg, "patches": patches, "labels": labels,
            "class_names": class_names, "full_ds": full_ds}


def gallery_split(pipeline: dict, fraction: float = 0.5, offset: int = 0):
    """Demo split: gallery = the given slice of the dataset, queries = the rest."""
    n = len(pipeline["full_ds"])
    gallery = np.arange(int(n * offset), int(n * (offset + fraction)))
    rest = np.setdiff1d(np.arange(n), gallery)
    return gallery, rest


# ---------------------------------------------------------------------------
# Plotting / display helpers
# ---------------------------------------------------------------------------

def show_modalities(pipeline: dict, indices, modality: str = "optical", cols: int = 6, size: int = 3):
    """Grid of patches for one modality, captioned with index + class."""
    from src.utils.visualize import render_patch
    labels, class_names = pipeline["labels"], pipeline["class_names"]
    n = len(indices)
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, indices):
        rgb = render_patch(pipeline["patches"], int(i), modality)
        ax.imshow(rgb)
        ax.set_title(f"#{int(i)} {class_names[int(labels[int(i)])]}", fontsize=8, color="#444")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def show_all_modalities(pipeline: dict, index: int, size: int = 3.2):
    """Optical / multispectral / SAR views of a single patch side-by-side."""
    from src.utils.visualize import render_patch
    cls = pipeline["class_names"][int(pipeline["labels"][index])]
    fig, axes = plt.subplots(1, len(pipeline["cfg"]["modalities"]),
                             figsize=(size * len(pipeline["cfg"]["modalities"]), size))
    for ax, m in zip(np.atleast_1d(axes), pipeline["cfg"]["modalities"]):
        ax.imshow(render_patch(pipeline["patches"], int(index), m))
        ax.set_title(f"{m}  ·  #{int(index)} {cls}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_embedding_2d(embeddings: np.ndarray, labels: np.ndarray, class_names,
                      method: str = "pca", n_samples: int = 3000, seed: int = 0,
                      title: str = ""):
    """Project embeddings to 2D (PCA or UMAP) and scatter by class."""
    rng = np.random.RandomState(seed)
    if len(embeddings) > n_samples:
        idx = rng.choice(len(embeddings), n_samples, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]

    if method == "umap":
        try:
            import umap  # optional dependency
            xy = umap.UMAP(random_state=seed).fit_transform(embeddings)
        except Exception as e:  # pragma: no cover
            print(f"UMAP unavailable ({e}); falling back to PCA.")
            xy = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    else:
        xy = PCA(n_components=2, random_state=seed).fit_transform(embeddings)

    from matplotlib import colormaps
    cmap = colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(len(class_names)):
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=9, alpha=0.55, color=cmap(c),
                   label=class_names[c])
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1), markerscale=2)
    ax.set_title(title or f"Embedding space ({method.upper()}, n={len(embeddings)})")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    return fig


def show_retrieval_montage(pipeline: dict, result, k: int = 5, size: int = 3):
    """Side-by-side montage: query + top-k retrieved with relevance ticks."""
    from src.utils.visualize import render_patch
    labels, class_names = pipeline["labels"], pipeline["class_names"]
    qid = int(result.query_ids[0])
    qm, gm = result.query_modality, result.gallery_modality

    cols = k + 1
    fig, axes = plt.subplots(1, cols, figsize=(cols * size, size))
    axes = np.atleast_1d(axes).ravel()
    qimg = render_patch(pipeline["patches"], qid, qm)
    axes[0].imshow(qimg)
    axes[0].set_title(f"QUERY\n#{qid} {class_names[int(labels[qid])]}\n{'-'*0}{qm}",
                      fontsize=8, color="#1f6feb")
    axes[0].axis("off")
    rel = result.relevant_mask()[0]
    for j, ax in enumerate(axes[1:], start=1):
        rid = int(result.gallery_ids[0, j - 1])
        ax.imshow(render_patch(pipeline["patches"], rid, gm))
        mark = "✓" if bool(rel[j - 1]) else "✗"
        color = "#1a7f37" if rel[j - 1] else "#d1242f"
        ax.set_title(f"{j}. #{rid} {class_names[int(labels[rid])]}  {mark}\n{score(result, j-1)} · {gm}",
                     fontsize=8, color=color)
        ax.axis("off")
    plt.tight_layout()
    return fig


def score(result, j: int) -> str:
    return f"s={result.scores[0, j]:.3f}"


def class_distribution(pipeline: dict):
    """Bar chart of per-class patch counts."""
    labels, class_names = pipeline["labels"], pipeline["class_names"]
    counts = [(c, int((labels == c).sum())) for c in range(len(class_names))]
    _, axes = plt.subplots(figsize=(9, 4))
    classes, cnts = zip(*[ (class_names[c], n) for c, n in counts ])
    axes.bar(class_names, cnts, color="#4f9cf9")
    axes.set_title("Class distribution")
    axes.tick_params(axis="x", rotation=45, labelsize=8)
    for i, v in enumerate(cnts):
        axes.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig