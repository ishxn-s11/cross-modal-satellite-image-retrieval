"""Shared pipeline orchestration: data -> model -> train -> retrieve -> evaluate."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.dataset import MultiModalDataset, collate_modalities
from .data.interface import DatasetInterface, build_dataset
from .data.preprocessing import build_transforms, compute_normalization_stats
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

    Routes through :func:`build_dataset`, so ``cfg['dataset']['name']`` selects
    any registered backend (synthetic, eurosat, sen12ms, so2sat,
    bigearthnet_mm) with graceful fallback to synthetic when real data is
    absent.

    Returns (patches {modality: (N,C,H,W)}, labels (N,), class_names,
             stats, transforms) -- backward compatible with the original
    5-tuple signature.
    """
    ds = build_dataset(cfg, logger)
    patches, labels, class_names = ds.to_patches()
    labels = np.asarray(labels, dtype=np.int64)
    stats = compute_normalization_stats(patches)
    transforms = build_transforms(stats)
    logger.info(
        f"[data] dataset={ds.dataset_id} N={labels.shape[0]} "
        f"classes={len(class_names)} modalities={ds.modalities} "
        f"has_metadata={ds.has_metadata()}"
    )
    return patches, labels, class_names, stats, transforms


def build_loaders(
    cfg: Dict,
    patches: Dict,
    labels,
    transforms: Dict,
    metadata: Optional[List] = None,
) -> Tuple[DataLoader, DataLoader, MultiModalDataset]:
    modalities = list(cfg["modalities"])
    tr_cfg = cfg["training"]
    seed = int(cfg["dataset"]["seed"])

    full_ds = MultiModalDataset(patches, labels, modalities, transforms, metadata=metadata)
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


def build_model_from_cfg(
    cfg: Dict,
    n_classes: int,
    modality_channels: Optional[Dict[str, int]] = None,
) -> ModalityAdaptiveEncoder:
    m_cfg = cfg["model"]
    return build_model(
        modalities=cfg["modalities"],
        backbone=m_cfg.get("backbone", "resnet18"),
        pretrained=bool(m_cfg.get("pretrained", True)),
        embedding_dim=int(m_cfg.get("embedding_dim", 128)),
        n_classes=int(n_classes),
        freeze_backbone=bool(m_cfg.get("freeze_backbone", True)),
        unfreeze_stage=m_cfg.get("unfreeze_stage", "none"),
        modality_channels=modality_channels,
    )


def modality_channels_from_dataset(
    ds: DatasetInterface, modalities=None
) -> Dict[str, int]:
    """Band counts for each modality as reported by a dataset interface."""
    mods = modalities or ds.modalities
    return {m: int(ds.bands(m)) for m in mods}


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
