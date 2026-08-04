"""End-to-end smoke test on a tiny synthetic dataset.

Runs the full pipeline (data -> model -> 1 epoch of training -> retrieval
evaluation) in ~1-2 minutes on CPU and asserts that metrics are produced and
bounded in [0, 1].
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import format_table
from src.pipeline import build_loaders, build_model_from_cfg, evaluate, prepare_dataset, train
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.io import Logger, resolve_device, set_seed


def make_mini_config() -> dict:
    cfg = deep_merge({}, DEFAULT_CONFIG)
    cfg["dataset"]["num_patches"] = 240
    cfg["dataset"]["image_size"] = 32
    cfg["dataset"]["seed"] = 0
    cfg["modalities"] = ["optical", "multispectral"]
    cfg["model"]["embedding_dim"] = 32
    cfg["model"]["freeze_backbone"] = True
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = 32
    cfg["training"]["patience"] = 1
    cfg["retrieval"]["n_query"] = 30
    cfg["retrieval"]["gallery_fraction"] = 0.6
    cfg["evaluation"]["same_modal_pairs"] = [["optical", "optical"]]
    cfg["evaluation"]["cross_modal_pairs"] = [["optical", "multispectral"]]
    cfg["outputs"]["dir"] = "outputs/_smoke"
    cfg["outputs"]["model_dir"] = "outputs/_smoke_models"
    cfg["outputs"]["log_file"] = "outputs/_smoke/log.txt"
    return cfg


def run() -> None:
    cfg = make_mini_config()
    set_seed(int(cfg["dataset"]["seed"]))
    logger = Logger(cfg["outputs"]["log_file"])
    device = resolve_device(cfg["training"]["device"])

    patches, labels, class_names, _s, transforms = prepare_dataset(cfg, logger)
    model = build_model_from_cfg(cfg, len(class_names)).to(device)
    train_dl, val_dl, full_ds = build_loaders(cfg, patches, labels, transforms)
    train(cfg, model, train_dl, val_dl, device, logger)

    rows, summary = evaluate(cfg, model, full_ds, device, logger)
    print("\n" + format_table(rows))
    print("[summary]", summary)

    assert len(rows) > 0
    assert all(0.0 <= r["f1@k"] <= 1.0 for r in rows)
    assert all(r["avg_retrieval_time_ms"] >= 0.0 for r in rows)
    assert 0.0 <= summary["same_modal_avg"]["f1@5"] <= 1.0
    assert 0.0 <= summary["cross_modal_avg"]["f1@5"] <= 1.0
    print("smoke_test.py: PASSED")


if __name__ == "__main__":
    run()