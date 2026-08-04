"""Shared pipeline orchestration: data -> model -> train -> retrieve -> evaluate."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.dataset import MultiModalDataset, collate_modalities
from .data.eurosat import load_eurosat_multimodal
from .data.preprocessing import build_transforms, compute_normalization_stats
from .data.synthetic import CLASS_NAMES, load_or_generate_synthetic
from .evaluation.evaluate import (
    evaluate_retrieval_pairs,
    save_report,
    stratified_split,
)
from .models.encoder import ModalityAdaptiveEncoder, build_model
from .retrieval.engine import RetrievalEngine
from .training.trainer import train_model
from .utils.io import Logger, resolve_device, save_checkpoint


def prepare_dataset(
    cfg: Dict, logger: Logger
) -> Tuple[Dict, np.ndarray, List[str], Dict, Dict]:
    """Load/generate the multi-modal dataset and normalisation statistics.

    Returns (patches {modality: (N,C,H,W)}, labels (N,), class_names,
             stats, transforms).
    """
    ds_cfg = cfg["dataset"]
    modalities = list(cfg["modalities"])
    source = ds_cfg["source"]
    root = ds_cfg["root"]
    image_size = int(ds_cfg["image_size"])
    seed = int(ds_cfg["seed"])

    if source == "synthetic":
        num_patches = int(ds_cfg.get("num_patches", 2000))
        data, class_names = load_or_generate_synthetic(
            root, num_patches, image_size, seed, modalities
        )
        labels = data["labels"]
        patches = {m: data[m] for m in modalities}
    elif source == "eurosat":
        max_patches = int(ds_cfg.get("eurosat_max_patches", 6000))
        patches, labels, class_names = load_eurosat_multimodal(
            root,
            image_size=image_size,
            max_patches=max_patches,
            seed=seed,
            modalities=modalities,
        )
    else:
        raise ValueError(f"Unknown dataset source '{source}'")

    labels = np.asarray(labels, dtype=np.int64)
    stats = compute_normalization_stats(patches)
    transforms = build_transforms(stats)
    logger.info(
        f"[data] source={source} N={labels.shape[0]} classes={len(class_names)} "
        f"modalities={modalities}"
    )
    return patches, labels, class_names, stats, transforms


def build_loaders(
    cfg: Dict,
    patches: Dict,
    labels,
    transforms: Dict,
) -> Tuple[DataLoader, DataLoader, MultiModalDataset]:
    modalities = list(cfg["modalities"])
    tr_cfg = cfg["training"]
    seed = int(cfg["dataset"]["seed"])

    full_ds = MultiModalDataset(patches, labels, modalities, transforms)
    train_ids, val_ids, _ = stratified_split(
        np.asarray(labels), float(tr_cfg["train_ratio"]), float(tr_cfg["val_ratio"]), seed
    )
    bs = int(tr_cfg["batch_size"])
    nw = int(tr_cfg["num_workers"])
    train_dl = DataLoader(
        Subset(full_ds, train_ids), batch_size=bs, shuffle=True,
        num_workers=nw, collate_fn=collate_modalities,
    )
    val_dl = DataLoader(
        Subset(full_ds, val_ids), batch_size=bs, shuffle=False,
        num_workers=nw, collate_fn=collate_modalities,
    )
    return train_dl, val_dl, full_ds


def build_model_from_cfg(cfg: Dict, n_classes: int) -> ModalityAdaptiveEncoder:
    m_cfg = cfg["model"]
    return build_model(
        modalities=cfg["modalities"],
        backbone=m_cfg.get("backbone", "resnet18"),
        pretrained=bool(m_cfg.get("pretrained", True)),
        embedding_dim=int(m_cfg.get("embedding_dim", 128)),
        n_classes=int(n_classes),
        freeze_backbone=bool(m_cfg.get("freeze_backbone", True)),
        unfreeze_stage=m_cfg.get("unfreeze_stage", "none"),
    )


def train(cfg: Dict, model, train_dl, val_dl, device, logger) -> Dict:
    tr_cfg = cfg["training"]
    out_cfg = cfg["outputs"]
    ckpt_dir = os.path.join(out_cfg["model_dir"], "checkpoints")
    best_path = os.path.join(out_cfg["model_dir"], "best_model", "model.pt")
    return train_model(
        model,
        train_dl,
        val_dl,
        cfg["modalities"],
        tr_cfg,
        device,
        ckpt_dir,
        best_path,
        logger,
    )


def evaluate(cfg: Dict, model, full_ds, device, logger) -> Tuple[List[Dict], Dict]:
    ev_cfg = cfg["evaluation"]
    rows, summary = evaluate_retrieval_pairs(
        model,
        full_ds,
        cfg["modalities"],
        [tuple(p) for p in ev_cfg["same_modal_pairs"]],
        [tuple(p) for p in ev_cfg["cross_modal_pairs"]],
        cfg["retrieval"],
        device,
        seed=int(cfg["dataset"]["seed"]),
        logger=logger,
    )
    out_dir = os.path.join(cfg["outputs"]["dir"], "metrics")
    csv_path, json_path = save_report(rows, summary, out_dir)
    logger.info(f"[eval] report saved: {csv_path}")
    logger.info(f"[eval] summary saved: {json_path}")
    return rows, summary


def load_best_model(cfg: Dict, n_classes: int, device) -> ModalityAdaptiveEncoder:
    model = build_model_from_cfg(cfg, n_classes)
    path = os.path.join(cfg["outputs"]["model_dir"], "best_model", "model.pt")
    if os.path.exists(path):
        from .utils.io import load_checkpoint

        load_checkpoint(model, path, device)
    return model
