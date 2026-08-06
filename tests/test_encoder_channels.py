"""Unit tests for the encoder's handling of real (non-default) band counts."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.models.encoder import ModalityAdaptiveEncoder, build_model


def test_encoder_accepts_real_band_counts():
    # e.g. SEN12MS: optical = S2 RGB (3ch), sar = S1 VV+VH (2ch).
    m = build_model(
        ["optical", "sar"],
        embedding_dim=64,
        n_classes=5,
        modality_channels={"optical": 3, "sar": 2},
    )
    assert isinstance(m, ModalityAdaptiveEncoder)
    m.eval()  # embedding generation runs in eval mode (no BN batch-size constraint)
    out = m.embed(torch.randn(2, 2, 32, 32), "sar")
    assert tuple(out.shape) == (2, 64)
    # embeddings are L2-normalised
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_encoder_default_channels_backward_compatible():
    # No modality_channels -> legacy 3/8/1 mapping still works.
    m = build_model(["optical", "multispectral", "sar"], embedding_dim=32, n_classes=5)
    assert m.modality_channels == {"optical": 3, "multispectral": 8, "sar": 1}
    m.eval()
    out = m.embed(torch.randn(1, 1, 32, 32), "sar")
    assert tuple(out.shape) == (1, 32)


def test_encoder_multispectral_channels_match_selector():
    # Configurable S2 band subset -> adapter built for exactly those channels.
    m = build_model(["multispectral"], embedding_dim=16, n_classes=3,
                    modality_channels={"multispectral": 8})
    m.eval()
    out = m.embed(torch.randn(1, 8, 32, 32), "multispectral")
    assert tuple(out.shape) == (1, 16)


if __name__ == "__main__":
    test_encoder_accepts_real_band_counts()
    test_encoder_default_channels_backward_compatible()
    test_encoder_multispectral_channels_match_selector()
    print("test_encoder_channels.py: all tests passed")
