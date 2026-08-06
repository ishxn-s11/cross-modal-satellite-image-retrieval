"""Train the shared embedding model (contrastive alignment)."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import build_loaders, build_model_from_cfg, prepare_dataset, train
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    os.makedirs(cfg["outputs"]["dir"], exist_ok=True)
    logger = Logger(cfg["outputs"]["log_file"])
    set_seed(int(cfg["dataset"]["seed"]))
    device = resolve_device(cfg["training"]["device"])

    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    model = build_model_from_cfg(cfg, len(class_names)).to(device)
    train_dl, val_dl, _full = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )
    result = train(cfg, model, train_dl, val_dl, device, logger)
    logger.info(f"[train] completed: {result}")


if __name__ == "__main__":
    main()
