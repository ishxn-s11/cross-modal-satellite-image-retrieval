"""Unit tests for hard-negative mining in the contrastive losses."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.training.contrastive import (
    info_nce,
    info_nce_hard,
    multimodal_losses,
    supervised_contrastive,
    supervised_contrastive_hard,
)


def _norm(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=1)


def test_info_nce_hard_reduces_to_info_nce_when_n_large():
    z1, z2 = _norm(6, 16, 1), _norm(6, 16, 2)
    a = info_nce(z1, z2, 0.07)
    b = info_nce_hard(z1, z2, 0.07, n_hard=5)  # n-1 = all negatives
    assert abs(a.item() - b.item()) < 1e-5


def test_supcon_hard_reduces_to_supcon_when_n_large():
    z = _norm(12, 16, 3)
    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    a = supervised_contrastive(z, labels, 0.07)
    b = supervised_contrastive_hard(z, labels, 0.07, n_hard=11)
    assert abs(a.item() - b.item()) < 1e-5


def _hard_supcon_from_sim(sim: torch.Tensor, labels: torch.Tensor, n_hard: int) -> torch.Tensor:
    """Mirror of supervised_contrastive_hard that backprops to the sim matrix.

    Lets us inspect the *per-pair* gradient (which aggregate per-sample
    gradients cannot): an anchor only receives gradient from the negatives that
    are actually in its top-k denominator.
    """
    temperature = 0.07
    zsim = sim / temperature
    exp_sim = zsim.exp()
    n = sim.size(0)
    mask = (labels[:, None] == labels[None, :]).float()
    mask.fill_diagonal_(0.0)
    pos = (exp_sim * mask).sum(dim=1)
    eye = torch.eye(n, dtype=torch.bool)
    neg_mask = (labels[:, None] != labels[None, :]) & ~eye
    neg_sim = torch.where(neg_mask, zsim, torch.full_like(zsim, -1e9))
    hard = neg_sim.topk(n_hard, dim=1).values
    denom = torch.logsumexp(hard, dim=1).exp()
    log_prob = torch.log((pos / denom.clamp(min=1e-8)).clamp(min=1e-12))
    valid = mask.sum(dim=1) > 0
    return -log_prob[valid].mean()


def test_hard_negatives_focus_gradient_on_hard_negatives():
    # The loss gradient w.r.t. a *pair* similarity is zero for negatives that
    # are not among the anchor's top-k hardest -- that is the mining mechanism.
    z = _norm(8, 16, 9)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    sim = (z @ z.t()).clone().requires_grad_()
    loss = _hard_supcon_from_sim(sim, labels, n_hard=2)
    loss.backward()
    grad = sim.grad
    anchor = 0
    negs = [j for j in range(8) if labels[j] != labels[0]]
    hardest = max(negs, key=lambda j: sim.data[anchor, j].item())
    easiest = min(negs, key=lambda j: sim.data[anchor, j].item())
    assert grad[anchor, hardest].abs().item() > 1e-4
    assert grad[anchor, easiest].abs().item() < 1e-6


def test_hard_negatives_truncation_smaller_denominator():
    # With a small n_hard the denominator has fewer terms -> smaller loss.
    z1 = _norm(8, 16, 4)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    full = supervised_contrastive(z1, labels, 0.07)
    hard = supervised_contrastive_hard(z1, labels, 0.07, n_hard=2)
    assert hard.item() < full.item()


def test_multimodal_losses_integrate_hard_negatives_and_geo():
    g = torch.Generator().manual_seed(7)
    n = 8
    z = {m: F.normalize(torch.randn(n, 16, generator=g), dim=1) for m in ("optical", "sar")}
    logits = {m: torch.randn(n, 4, generator=g) for m in ("optical", "sar")}
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    geo = {
        "lat": torch.tensor([0.0] * 4 + [50.0] * 4),
        "lon": torch.tensor([0.0] * 4 + [5.0] * 4),
    }
    losses = multimodal_losses(
        z, logits, labels, ["optical", "sar"], temperature=0.07,
        geo_weight=1.0, geo=geo, hard_negatives={"enabled": True, "n_hard": 3},
    )
    assert set(["total", "clip", "supcon", "cls", "geo"]) <= set(losses)
    assert torch.isfinite(losses["total"])
    assert losses["geo"].item() >= 0.0
    # geo off -> zero geo term
    losses_off = multimodal_losses(z, logits, labels, ["optical", "sar"], temperature=0.07)
    assert losses_off["geo"].item() == 0.0


if __name__ == "__main__":
    test_info_nce_hard_reduces_to_info_nce_when_n_large()
    test_supcon_hard_reduces_to_supcon_when_n_large()
    test_hard_negatives_focus_gradient_on_hard_negatives()
    test_hard_negatives_truncation_smaller_denominator()
    test_multimodal_losses_integrate_hard_negatives_and_geo()
    print("test_hard_negatives.py: all tests passed")
