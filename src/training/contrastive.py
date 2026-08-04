"""Contrastive losses for multi-modal representation learning.

Three complementary objectives are combined:

* **InfoNCE (CLIP-style)** pulls the same patch across modalities together while
  pushing different patches apart (``clip_weight``).
* **Supervised contrastive (SupCon)** brings all patches of the same land-cover
  class together regardless of modality (``supcon_weight``).
* **Cross-entropy** on an auxiliary classifier supplies a direct supervised
  signal (``ce_weight``).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE between two normalised embedding batches (B, D)."""
    sim = (z1 @ z2.t()) / temperature  # positives on the diagonal
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0


def supervised_contrastive(
    z: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    """SupCon loss on L2-normalised embeddings ``z`` (N, D) with labels (N,)."""
    n = z.size(0)
    sim = (z @ z.t()) / temperature
    exp_sim = sim.exp()
    mask = (labels[:, None] == labels[None, :]).float()
    mask.fill_diagonal_(0.0)  # exclude self-pairs
    denom = exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-8)
    pos = (exp_sim * mask).sum(dim=1)
    log_prob = torch.log(pos / denom)
    valid = mask.sum(dim=1) > 0
    return -log_prob[valid].mean() if valid.any() else z.new_zeros(())


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def multimodal_losses(
    embeddings: Dict[str, torch.Tensor],
    logits: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    modalities: Sequence[str],
    temperature: float,
    clip_weight: float = 1.0,
    supcon_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Combine the three objectives across all modalities.

    ``embeddings`` and ``logits`` are keyed by modality; ``labels`` shape (B,).
    Returns a dict of scalar loss components plus ``"total"``.
    """
    mods = list(modalities)

    # 1) InfoNCE over all ordered modality pairs.
    clip_loss = embeddings[mods[0]].new_zeros(())
    pairs = 0
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            clip_loss = clip_loss + info_nce(
                embeddings[mods[i]], embeddings[mods[j]], temperature
            )
            pairs += 1
    if pairs > 0:
        clip_loss = clip_loss / pairs

    # 2) Supervised contrastive on the pooled set of all modality embeddings.
    if len(mods) == 1:
        supcon = supervised_contrastive(embeddings[mods[0]], labels, temperature)
    else:
        pooled = torch.cat([embeddings[m] for m in mods], dim=0)
        pooled_labels = labels.repeat(len(mods))
        supcon = supervised_contrastive(pooled, pooled_labels, temperature)

    # 3) Auxiliary classification.
    active = [m for m in mods if m in logits]
    if active:
        cls_loss = sum(classification_loss(logits[m], labels) for m in active) / len(active)
    else:
        cls_loss = embeddings[mods[0]].new_zeros(())

    total = clip_weight * clip_loss + supcon_weight * supcon + cls_weight * cls_loss
    return {
        "total": total,
        "clip": clip_loss,
        "supcon": supcon,
        "cls": cls_loss,
    }