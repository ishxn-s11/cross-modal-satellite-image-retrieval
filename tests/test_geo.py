"""Unit tests for the geographic / temporal alignment losses."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.training.geo import (
    geographic_alignment_loss,
    haversine_km,
    pairwise_haversine_km,
)


def test_haversine_known_distance():
    # 1 degree of longitude at the equator is ~111.2 km.
    d = haversine_km(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0))
    assert abs(d.item() - 111.19) < 2.0
    # same point -> 0
    d0 = haversine_km(torch.tensor(10.0), torch.tensor(20.0), torch.tensor(10.0), torch.tensor(20.0))
    assert d0.item() < 1e-3


def test_pairwise_distances_symmetric_zero_diag():
    lat = torch.tensor([0.0, 0.0, 10.0])
    lon = torch.tensor([0.0, 1.0, 10.0])
    D = pairwise_haversine_km(lat, lon)
    assert D.shape == (3, 3)
    assert torch.allclose(D, D.t())
    assert torch.allclose(torch.diag(D), torch.zeros(3), atol=1e-3)


def test_geo_loss_zero_without_coordinates():
    z = F.normalize(torch.randn(8, 32), dim=1)
    lat = torch.full((8,), float("nan"))
    lon = torch.full((8,), float("nan"))
    loss = geographic_alignment_loss(z, lat, lon)
    assert float(loss) == 0.0


def test_geo_loss_pulls_same_location():
    g = torch.Generator().manual_seed(0)
    z = F.normalize(torch.randn(6, 32, generator=g), dim=1)
    # first 3 share one location -> positive pairs
    lat = torch.tensor([0.0, 0.0, 0.0, 50.0, 51.0, 52.0])
    lon = torch.tensor([0.0, 0.0, 0.0, 5.0, 6.0, 7.0])
    loss = geographic_alignment_loss(z, lat, lon, same_km=5.0)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_push_distant_adds_hinge():
    g = torch.Generator().manual_seed(1)
    z = F.normalize(torch.randn(8, 32, generator=g), dim=1)
    lat = torch.tensor([0.0, 0.0, 0.0, 0.0, 50.0, 51.0, 52.0, 53.0])
    lon = torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0])
    off = geographic_alignment_loss(z, lat, lon, push_distant=False)
    on = geographic_alignment_loss(z, lat, lon, push_distant=True, distant_km=10.0)
    assert on.item() >= off.item()


if __name__ == "__main__":
    test_haversine_known_distance()
    test_pairwise_distances_symmetric_zero_diag()
    test_geo_loss_zero_without_coordinates()
    test_geo_loss_pulls_same_location()
    test_push_distant_adds_hinge()
    print("test_geo.py: all tests passed")
