"""End-to-end cross-modal satellite image retrieval pipeline.

Usage
-----
    python run_pipeline.py [--config configs/default.yaml]
                           [--set dataset.source=synthetic]
                           [--set training.epochs=6]

Steps: prepare dataset -> build model -> train (contrastive alignment) ->
build per-modality galleries -> evaluate same/cross-modal top-5 & top-10
retrieval -> save metrics & summary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.metrics import format_table
from src.pipeline import (
    build_loaders,
    build_model_from_cfg,
    evaluate,
    modality_channels_from_patches,
    prepare_dataset,
    train,
)
from src.utils.config import load_config, pretty_print
from src.utils.io import Logger, resolve_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--set", action="append", default=[], help="override key=value, repeatable")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    out_cfg = cfg["outputs"]
    os.makedirs(out_cfg["dir"], exist_ok=True)
    logger = Logger(out_cfg["log_file"])
    set_seed(int(cfg["dataset"]["seed"]))
    device = resolve_device(cfg["training"]["device"])
    logger.info(f"device={device} | {args.config}")
    logger.info("config:\n" + pretty_print(cfg))

    t0 = time.perf_counter()
    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)

    model = build_model_from_cfg(
        cfg,
        len(class_names),
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    logger.info(f"[model] backbone={cfg['model']['backbone']} embeddings={cfg['model']['embedding_dim']}")

    train_dl, val_dl, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )
    train(cfg, model, train_dl, val_dl, device, logger)

    rows, summary = evaluate(cfg, model, full_ds, device, logger)
    logger.info("\n" + format_table(rows))
    logger.info("[summary] " + _fmt_summary(summary))
    logger.info(f"[pipeline] total wall-clock {time.perf_counter() - t0:.1f}s")


def _fmt_summary(summary: dict) -> str:
    s, c, w = summary["same_modal_avg"], summary["cross_modal_avg"], summary["weighted_avg"]
    return (
        f"same-modal F1@5={s['f1@5']:.4f} F1@10={s['f1@10']:.4f} | "
        f"cross-modal F1@5={c['f1@5']:.4f} F1@10={c['f1@10']:.4f} | "
        f"weighted F1@5={w['f1@5']:.4f} F1@10={w['f1@10']:.4f} | "
        f"avg retrieval time {summary['avg_retrieval_time_ms']:.3f} ms/query"
    )


if __name__ == "__main__":
    main()
