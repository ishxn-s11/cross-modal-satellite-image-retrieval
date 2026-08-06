"""Download / generate and cache the dataset (no training)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import prepare_dataset
from src.utils.config import load_config
from src.utils.io import Logger, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default=None, help="override dataset.source")
    args = ap.parse_args()

    overrides = []
    if args.source:
        overrides.append(f"dataset.source={args.source}")
    cfg = load_config(args.config, overrides)
    logger = Logger()
    set_seed(int(cfg["dataset"]["seed"]))
    patches, labels, class_names, _stats, _t, _metadata = prepare_dataset(cfg, logger)
    n = labels.shape[0]
    per_class = [int((labels == i).sum()) for i in range(len(class_names))]
    logger.info(f"[download] done: {n} patches, {len(class_names)} classes")
    logger.info("[download] per-class counts: " + ", ".join(
        f"{c}={k}" for c, k in zip(class_names, per_class))
    )
    logger.info(
        "[download] modalities: "
        + ", ".join(f"{m}({int(p[m].shape[1])} bands)" for m in cfg["modalities"])
    )


if __name__ == "__main__":
    main()