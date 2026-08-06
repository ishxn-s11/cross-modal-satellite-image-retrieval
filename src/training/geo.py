"""Geographic + temporal alignment losses.

When acquisition metadata (latitude / longitude / date) is available, the model
can be taught to keep *the same location* together in embedding space even
across different acquisition dates or seasons (temporal robustness), and to
separate geographically distant, semantically-different scenes.

Both losses are optional and off by default (``training.geo_weight: 0.0``) --
datasets without geo metadata simply produce a zero loss.
"""

from __future__ import annotations

import torch


def haversine_km(
    lat1: torch.Tensor, lon1: torch.Tensor, lat2: torch.Tensor, lon2: torch.Tensor
) -> torch.Tensor:
    """Great-circle distance in kilometres between two coordinate tensors.

    Inputs may be scalar or batched (broadcastable). Uses the standard
    haversine formula with Earth radius 6371 km.
    """
    R = 6371.0
    p1 = torch.deg2rad(lat1)
    p2 = torch.deg2rad(lat2)
    dphi = torch.deg2rad(lat2 - lat1)
    dlam = torch.deg2rad(lon2 - lon1)
    a = torch.sin(dphi / 2.0) ** 2 + torch.cos(p1) * torch.cos(p2) * torch.sin(dlam / 2.0) ** 2
    return 2.0 * R * torch.atan2(torch.sqrt(a.clamp(min=0.0)), torch.sqrt((1.0 - a).clamp(min=0.0)))


def pairwise_haversine_km(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    """(B, B) pairwise great-circle distances in km."""
    return haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def geographic_alignment_loss(
    z: torch.Tensor,
    lat: torch.Tensor,
    lon: torch.Tensor,
    temperature: float = 0.07,
    same_km: float = 5.0,
    push_distant: bool = False,
    distant_km: float = 100.0,
    distant_margin: float = 0.3,
) -> torch.Tensor:
    """Align same-location pairs and (optionally) push distant pairs apart.

    ``z`` are L2-normalised embeddings (B, D). ``lat``/``lon`` (B,) may contain
    NaN for images without coordinates; those rows are ignored.

    Loss components:
      * *pull*  -- SupCon-style: for each anchor, bring all pairs within
        ``same_km`` together (this is temporal robustness too: the same
        location observed on different dates is a positive pair).
      * *push*  -- when ``push_distant``, a hinge penalising positive cosine
        similarity for pairs further than ``distant_km`` apart.

    Returns a scalar tensor (zero when < 2 valid geo coordinates).
    """
    valid = torch.isfinite(lat) & torch.isfinite(lon)
    if valid.sum() < 2:
        return z.new_zeros(())
    z, lat, lon = z[valid], lat[valid], lon[valid]
    n = z.size(0)
    sim = z @ z.t()  # cosine similarity (z is normalised)
    exp_sim = (sim / temperature).exp()
    eye = torch.eye(n, device=z.device, dtype=torch.bool)
    dist_km = pairwise_haversine_km(lat, lon)

    # --- pull: same-location pairs (any acquisition date) ---
    pos = ((dist_km <= same_km) & ~eye).float()
    denom = exp_sim.sum(dim=1) - exp_sim.diag()  # all other items (exclude self)
    pos_sum = (exp_sim * pos).sum(dim=1)
    log_prob = torch.log((pos_sum / denom.clamp(min=1e-8)).clamp(min=1e-12))
    valid_rows = pos.sum(dim=1) > 0
    pull = -log_prob[valid_rows].mean() if valid_rows.any() else z.new_zeros(())

    # --- push: geographically distant pairs ---
    if push_distant and distant_km > same_km:
        neg = (dist_km >= distant_km) & ~eye
        hinge = (sim * neg.float()).clamp(min=0.0)
        push = (hinge - distant_margin).clamp(min=0.0).mean()
    else:
        push = z.new_zeros(())

    return pull + push
