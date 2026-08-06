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

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from .geo import geographic_alignment_loss


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE between two normalised embedding batches (B, D)."""
    sim = (z1 @ z2.t()) / temperature  # positives on the diagonal
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0


def info_nce_hard(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float, n_hard: int
) -> torch.Tensor:
    """Symmetric InfoNCE whose denominator uses only the ``n_hard`` hardest
    (most confusable) negatives per anchor instead of the whole batch.

    The positive (diagonal) pair is always excluded. With
    ``n_hard >= batch - 1`` this reduces exactly to :func:`info_nce`.
    """
    n = z1.size(0)
    if n_hard is None or n_hard >= n - 1:
        return info_nce(z1, z2, temperature)
    sim = (z1 @ z2.t()) / temperature
    eye = torch.eye(n, device=z1.device, dtype=torch.bool)
    neg_sim = torch.where(~eye, sim, torch.full_like(sim, -1e9))
    hard = neg_sim.topk(min(n_hard, n - 1), dim=1).values  # (n, n_hard)
    l12 = (torch.logsumexp(hard, dim=1) - sim.diag()).mean()
    l21 = l12  # symmetric for a square similarity matrix
    return (l12 + l21) / 2.0


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


def supervised_contrastive_hard(
    z: torch.Tensor, labels: torch.Tensor, temperature: float, n_hard: int
) -> torch.Tensor:
    """SupCon whose negative denominator is limited to the ``n_hard`` hardest
    (highest-similarity, different-class) negatives per anchor.

    With ``n_hard >= batch - 1`` this reduces to :func:`supervised_contrastive`.
    """
    n = z.size(0)
    if n_hard is None or n_hard >= n - 1:
        return supervised_contrastive(z, labels, temperature)
    sim = (z @ z.t()) / temperature
    exp_sim = sim.exp()
    mask = (labels[:, None] == labels[None, :]).float()
    mask.fill_diagonal_(0.0)
    pos = (exp_sim * mask).sum(dim=1)
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    neg_mask = (labels[:, None] != labels[None, :]) & ~eye
    neg_sim = torch.where(neg_mask, sim, torch.full_like(sim, -1e9))
    hard = neg_sim.topk(min(n_hard, n - 1), dim=1).values  # (n, n_hard)
    denom = torch.logsumexp(hard, dim=1).exp()

    log_prob = torch.log(pos / denom.clamp(min=1e-8))
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
    geo_weight: float = 0.0,
    geo: Optional[Dict[str, torch.Tensor]] = None,
    geo_same_km: float = 5.0,
    geo_push_distant: bool = False,
    geo_distant_km: float = 100.0,
    hard_negatives: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Combine all objectives across all modalities.

    ``embeddings`` and ``logits`` are keyed by modality; ``labels`` shape (B,).
    ``geo`` (optional) carries ``lat``/``lon`` tensors for the geographic
    alignment term; ``geo_weight`` scales it (0 disables). ``hard_negatives``
    (optional dict) enables hard-negative mining for InfoNCE/SupCon::

        {"enabled": true, "n_hard": 8, "strategy": "embedding"}

    Returns a dict of scalar loss components plus ``"total"``.
    """
    mods = list(modalities)
    hn = hard_negatives or {}
    n_hard = int(hn.get("n_hard", 8)) if hn.get("enabled", False) else None

    # 1) InfoNCE over all ordered modality pairs (optionally hard-negative).
    clip_loss = embeddings[mods[0]].new_zeros(())
    pairs = 0
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            if n_hard is not None:
                clip_loss = clip_loss + info_nce_hard(
                    embeddings[mods[i]], embeddings[mods[j]], temperature, n_hard
                )
            else:
                clip_loss = clip_loss + info_nce(
                    embeddings[mods[i]], embeddings[mods[j]], temperature
                )
            pairs += 1
    if pairs > 0:
        clip_loss = clip_loss / pairs

    # 2) Supervised contrastive on the pooled set of all modality embeddings.
    if len(mods) == 1:
        pooled, pooled_labels = embeddings[mods[0]], labels
    else:
        pooled = torch.cat([embeddings[m] for m in mods], dim=0)
        pooled_labels = labels.repeat(len(mods))
    if n_hard is not None:
        supcon = supervised_contrastive_hard(pooled, pooled_labels, temperature, n_hard)
    else:
        supcon = supervised_contrastive(pooled, pooled_labels, temperature)

    # 3) Auxiliary classification.
    active = [m for m in mods if m in logits]
    if active:
        cls_loss = sum(classification_loss(logits[m], labels) for m in active) / len(active)
    else:
        cls_loss = embeddings[mods[0]].new_zeros(())

    # 4) Geographic / temporal alignment (off when geo_weight == 0 or no geo).
    if geo_weight and geo is not None and "lat" in geo and "lon" in geo:
        geo_loss = geographic_alignment_loss(
            embeddings[mods[0]],
            geo["lat"],
            geo["lon"],
            temperature=temperature,
            same_km=float(geo_same_km),
            push_distant=bool(geo_push_distant),
            distant_km=float(geo_distant_km),
        )
    else:
        geo_loss = embeddings[mods[0]].new_zeros(())

    total = (
        clip_weight * clip_loss
        + supcon_weight * supcon
        + cls_weight * cls_loss
        + geo_weight * geo_loss
    )
    return {
        "total": total,
        "clip": clip_loss,
        "supcon": supcon,
        "cls": cls_loss,
        "geo": geo_loss,
    }