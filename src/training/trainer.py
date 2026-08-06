"""CPU-friendly training loop for the shared embedding model."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.encoder import ModalityAdaptiveEncoder
from ..utils.io import Logger, save_checkpoint
from .contrastive import multimodal_losses


def _model_outputs(
    model: ModalityAdaptiveEncoder,
    x: torch.Tensor,
    modality: str,
) -> Dict[str, torch.Tensor]:
    """Embedding (normalised) + optional classifier logits for one modality."""
    feat = model.encode_features(x, modality)
    emb = F.normalize(model.project(feat, modality), dim=1)
    out: Dict[str, torch.Tensor] = {"emb": emb}
    if model.classifier is not None:
        out["logits"] = model.classifier(emb)
    return out


def _losses_from_batch(
    model: ModalityAdaptiveEncoder,
    batch: Dict[str, torch.Tensor],
    modalities: Sequence[str],
    device: torch.device,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}
    logits: Dict[str, torch.Tensor] = {}
    for m in modalities:
        out = _model_outputs(model, batch[m].to(device), m)
        embeddings[m] = out["emb"]
        if "logits" in out:
            logits[m] = out["logits"]
    # Optional geographic alignment (only when the dataset provides coords).
    geo = None
    if float(loss_cfg.get("geo_weight", 0.0)) > 0 and "latitude" in batch:
        geo = {
            "lat": batch["latitude"].to(device),
            "lon": batch["longitude"].to(device),
        }
    return multimodal_losses(
        embeddings,
        logits,
        batch["label"].to(device),
        modalities,
        temperature=float(loss_cfg.get("temperature", 0.07)),
        clip_weight=float(loss_cfg.get("clip_weight", 1.0)),
        supcon_weight=float(loss_cfg.get("supcon_weight", 1.0)),
        cls_weight=float(loss_cfg.get("cls_weight", 1.0)),
        geo_weight=float(loss_cfg.get("geo_weight", 0.0)),
        geo=geo,
        geo_same_km=float(loss_cfg.get("geo_same_km", 5.0)),
        geo_push_distant=bool(loss_cfg.get("geo_push_distant", False)),
        geo_distant_km=float(loss_cfg.get("geo_distant_km", 100.0)),
        hard_negatives=loss_cfg.get("hard_negatives"),
    )


def _classification_accuracy(
    model: ModalityAdaptiveEncoder,
    batch: Dict[str, torch.Tensor],
    modalities: Sequence[str],
    device: torch.device,
) -> float:
    """Mean classification accuracy over modalities that have a classifier."""
    labels = batch["label"].to(device)
    hits, total = 0, 0
    with torch.no_grad():
        for m in modalities:
            if model.classifier is None:
                continue
            out = _model_outputs(model, batch[m].to(device), m)
            hits += (out["logits"].argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return hits / max(1, total)


def train_epoch(
    model: ModalityAdaptiveEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    modalities: Sequence[str],
    device: torch.device,
    loss_cfg: Dict,
) -> Dict[str, float]:
    model.train()
    totals = {"total": 0.0, "clip": 0.0, "supcon": 0.0, "cls": 0.0, "geo": 0.0}
    n = 0
    for batch in loader:
        losses = _losses_from_batch(model, batch, modalities, device, loss_cfg)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        bs = batch["label"].size(0)
        for k in totals:
            totals[k] += losses[k].item() * bs
        n += bs
    return {k: v / max(1, n) for k, v in totals.items()}


def evaluate(
    model: ModalityAdaptiveEncoder,
    loader: DataLoader,
    modalities: Sequence[str],
    device: torch.device,
    loss_cfg: Dict,
) -> Dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "clip": 0.0, "supcon": 0.0, "cls": 0.0, "geo": 0.0}
    n = 0
    acc, n_acc = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            losses = _losses_from_batch(model, batch, modalities, device, loss_cfg)
            bs = batch["label"].size(0)
            for k in totals:
                totals[k] += losses[k].item() * bs
            n += bs
            if model.classifier is not None:
                acc += _classification_accuracy(model, batch, modalities, device) * bs
                n_acc += bs
    val_acc = acc / max(1, n_acc)
    return {**{k: v / max(1, n) for k, v in totals.items()}, "val_acc": val_acc}


def train_model(
    model: ModalityAdaptiveEncoder,
    train_dl: DataLoader,
    val_dl: DataLoader,
    modalities: Sequence[str],
    train_cfg: Dict,
    device: torch.device,
    checkpoint_dir: str,
    best_model_path: str,
    logger: Logger,
) -> Dict[str, float]:
    """Train, checkpointing the best model by validation accuracy."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 6))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best_acc, best_epoch, bad_epochs = -1.0, -1, 0
    patience = int(train_cfg.get("patience", 3))
    for epoch in range(1, epochs + 1):
        train_l = train_epoch(model, train_dl, optimizer, modalities, device, train_cfg)
        val = evaluate(model, val_dl, modalities, device, train_cfg)
        scheduler.step()
        acc = float(val["val_acc"])
        logger.info(
            f"[train] epoch {epoch:02d} | "
            + " ".join(f"{k}={v:.4f}" for k, v in train_l.items())
            + f" | val_acc={acc:.4f} val_loss={val['total']:.4f}"
        )
        if acc > best_acc:
            best_acc, best_epoch, bad_epochs = acc, epoch, 0
            save_checkpoint(model, os.path.join(checkpoint_dir, "best.pt"))
            save_checkpoint(model, best_model_path)
            logger.info(f"[train] improved -> best_acc={best_acc:.4f} (ep {epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"[train] early stop at epoch {epoch}")
                break
    logger.info(f"[train] done. best val_acc={best_acc:.4f} @ epoch {best_epoch}")
    return {"best_acc": best_acc, "best_epoch": best_epoch}
