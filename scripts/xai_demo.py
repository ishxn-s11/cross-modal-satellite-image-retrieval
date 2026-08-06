"""Generate an explainability overlay for a query image.

Grad-CAM for CNN backbones, attention maps for ViT. Saves the query + overlay
side by side.

Usage:
    python scripts/xai_demo.py [--config configs/default.yaml]
        [--query-id 12] [--modality optical] [--out outputs/xai/]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.pipeline import (
    build_loaders,
    load_best_model,
    modality_channels_from_patches,
    prepare_dataset,
)
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device
from src.utils.visualize import render_patch
from src.xai.attention import attention_map
from src.xai.gradcam import gradcam, overlay_saliency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--query-id", type=int, default=12)
    ap.add_argument("--modality", default="optical")
    ap.add_argument("--out", default="outputs/xai")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(path=None)
    device = resolve_device(cfg["training"]["device"])
    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    model = load_best_model(
        cfg,
        len(class_names),
        device,
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    _, _, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )

    qid = args.query_id
    modality = args.modality
    sample = full_ds[qid]
    x = sample[modality][None].to(device)
    rgb = render_patch(patches, qid, modality)

    backbone = cfg["model"].get("backbone", "resnet18")
    if backbone in ("vit_b_16", "vit"):
        sal = attention_map(model, x, modality)
        title = f"Attention (ViT) #{qid}"
    else:
        sal = gradcam(model, x, modality)
        title = f"Grad-CAM #{qid}"

    os.makedirs(args.out, exist_ok=True)
    over = overlay_saliency(rgb, sal)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Query ({modality})")
    axes[1].imshow(sal, cmap="jet")
    axes[1].set_title("Saliency")
    axes[2].imshow(over)
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    path = os.path.join(args.out, f"xai_{modality}_{qid}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[xai] saved {path} (backbone={backbone})")


if __name__ == "__main__":
    main()
