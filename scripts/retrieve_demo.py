"""Visual demo: retrieve top-K neighbours for a few queries and save montages."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.pipeline import build_loaders, load_best_model, prepare_dataset
from src.retrieval.engine import RetrievalEngine
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device
from src.utils.visualize import render_patch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--pair", default="optical,sar", help="query_modality,gallery_modality")
    ap.add_argument("--n-queries", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--best-model", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger()
    device = resolve_device(cfg["training"]["device"])
    qm, gm = args.pair.split(",")

    patches, labels, class_names, _s, transforms = prepare_dataset(cfg, logger)
    model = load_best_model(cfg, len(class_names), device).to(device)
    if args.best_model:
        from src.utils.io import load_checkpoint

        load_checkpoint(model, args.best_model, device)
    _, _, full_ds = build_loaders(cfg, patches, labels, transforms)
    engine = RetrievalEngine(model, full_ds, device)

    # Gallery: reuse a chunk of the dataset; queries: a few distinct ids.
    n_total = len(full_ds)
    gallery_ids = np.arange(int(n_total * 0.5), n_total)
    query_ids = np.arange(0, min(args.n_queries, int(n_total * 0.5)))
    gallery = engine.build_gallery(gallery_ids, gm)
    result = engine.retrieve(gallery, query_ids, qm, k=args.k)

    out_dir = os.path.join(cfg["outputs"]["dir"], "retrieved_images")
    os.makedirs(out_dir, exist_ok=True)
    for qi, qid in enumerate(query_ids):
        fig, axes = plt.subplots(1, args.k + 1, figsize=(4 * (args.k + 1), 4))
        axes[0].imshow(render_patch(patches, int(qid), qm))
        axes[0].set_title(f"QUERY ({qm})\n{class_names[int(labels[qid])]}", fontsize=9)
        axes[0].axis("off")
        for j in range(args.k):
            rid = int(result.gallery_ids[qi, j])
            rel = "relevant" if result.relevant_mask()[qi, j] else "miss"
            axes[j + 1].imshow(render_patch(patches, rid, gm))
            axes[j + 1].set_title(
                f"#{j+1} ({gm})\n{class_names[int(labels[rid])]} [{rel}]", fontsize=9
            )
            axes[j + 1].axis("off")
        safe = f"query_{qid}__{qm}_to_{gm}.png"
        path = os.path.join(out_dir, safe)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        logger.info(f"[demo] saved {path} | time={result.search_times_ms[qi]:.3f}ms")


if __name__ == "__main__":
    main()