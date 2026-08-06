"""Evaluate retrieval (same/cross-modal) using the trained best model."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import format_table
from src.pipeline import (
    build_loaders,
    build_model_from_cfg,
    evaluate,
    load_best_model,
    modality_channels_from_patches,
    prepare_dataset,
)
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--best-model", default=None, help="overrides best_model path")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(cfg["outputs"]["log_file"])
    set_seed(int(cfg["dataset"]["seed"]))
    device = resolve_device(cfg["training"]["device"])

    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    m_channels = modality_channels_from_patches(patches, cfg["modalities"])
    if args.best_model:
        from src.utils.io import load_checkpoint

        model = build_model_from_cfg(cfg, len(class_names), modality_channels=m_channels).to(device)
        load_checkpoint(model, args.best_model, device)
    else:
        model = load_best_model(cfg, len(class_names), device, modality_channels=m_channels).to(device)
    _, _, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )
    rows, summary = evaluate(cfg, model, full_ds, device, logger)
    print("\n" + format_table(rows))
    print("[summary]", summary)


if __name__ == "__main__":
    main()