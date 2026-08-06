"""Visualise the embedding space (before vs after training, by class/modality).

Saves PCA/t-SNE/UMAP scatter plots under outputs/embeddings/.

Usage:
    python scripts/visualize_embeddings.py [--config configs/default.yaml]
        [--method pca,tsne,umap] [--n 3000]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from src.pipeline import (
    build_loaders,
    build_model_from_cfg,
    load_best_model,
    modality_channels_from_patches,
    prepare_dataset,
)
from src.retrieval.engine import RetrievalEngine
from src.utils.config import load_config
from src.utils.embedding_viz import plot_embedding_comparison, plot_embeddings
from src.utils.io import Logger, resolve_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--method", default="pca,tsne,umap")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="outputs/embeddings")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(path=None)
    device = resolve_device(cfg["training"]["device"])
    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    model = build_model_from_cfg(
        cfg,
        len(class_names),
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    _, _, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )

    # "Before training" = the untrained model (random projection / pretrained
    # backbone). "After" = the trained checkpoint.
    before_model = build_model_from_cfg(
        cfg,
        len(class_names),
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    after_model = load_best_model(
        cfg,
        len(class_names),
        device,
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)

    os.makedirs(args.out, exist_ok=True)
    methods = [m.strip() for m in args.method.split(",") if m.strip()]
    idx = np.arange(0, len(full_ds), max(1, len(full_ds) // args.n))

    for m in methods:
        before, after = [], []
        for mod in cfg["modalities"]:
            engine = RetrievalEngine(before_model, full_ds, device)
            be, _ = engine.embed(idx, mod)
            engine2 = RetrievalEngine(after_model, full_ds, device)
            ae, _ = engine2.embed(idx, mod)
            before.append(be)
            after.append(ae)
        before = np.concatenate(before, axis=0)
        after = np.concatenate(after, axis=0)
        rep_labels = np.tile(labels[idx], len(cfg["modalities"]))
        rep_mods = np.concatenate(
            [np.full(len(idx), i) for i in range(len(cfg["modalities"]))]
        )

        fig1 = plot_embeddings(
            before, rep_labels, class_names, method=m, title=f"By class ({m}, before)")
        fig1.savefig(os.path.join(args.out, f"class_before_{m}.png"), dpi=110)
        fig2 = plot_embeddings(
            after, rep_labels, class_names, method=m, title=f"By class ({m}, after)")
        fig2.savefig(os.path.join(args.out, f"class_after_{m}.png"), dpi=110)
        fig3 = plot_embedding_comparison(
            before, after, rep_mods,
            [f"{x}" for x in cfg["modalities"]], method=m,
            before_title=f"By modality ({m}, before)", after_title=f"By modality ({m}, after)",
        )
        fig3.savefig(os.path.join(args.out, f"modality_before_after_{m}.png"), dpi=110)
        print(f"[viz] saved class/modal plots for {m}")
        for f in (fig1, fig2, fig3):
            import matplotlib.pyplot as plt

            plt.close(f)


if __name__ == "__main__":
    main()
