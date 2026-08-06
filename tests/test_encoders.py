"""Unit tests for the encoder adapter hierarchy (ResNet / ViT / foundation)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F

from src.models.encoder import (
    BaseEncoder,
    ModalityAdaptiveEncoder,
    ViTEncoder,
    _ExternalEncoder,
    _interpolate_pos_embed,
    build_backbone,
    build_model,
)


def test_vit_forward_shape_at_64():
    enc = ViTEncoder(pretrained=False, image_size=64)
    assert enc.feature_dim == 768
    out = enc.forward_features(torch.randn(2, 3, 64, 64))
    assert tuple(out.shape) == (2, 768)


def test_interpolate_pos_embed_preserves_cls():
    pe = torch.randn(1, 197, 768)  # ImageNet layout: 1 cls + 196 tokens
    cls_before = pe[:, :1].clone()
    out = _interpolate_pos_embed(pe, n_tokens=16)  # 4x4 grid at 64px
    assert tuple(out.shape) == (1, 17, 768)
    assert torch.allclose(out[:, :1], cls_before)


def test_per_modality_projection_heads():
    m = build_model(["optical", "sar"], embedding_dim=64, n_classes=5,
                    projection_heads="per_modality")
    m.eval()
    assert isinstance(m.projection, torch.nn.ModuleDict)
    out_o = m.embed(torch.randn(2, 3, 32, 32), "optical")
    out_s = m.embed(torch.randn(2, 1, 32, 32), "sar")
    assert tuple(out_o.shape) == (2, 64)
    assert tuple(out_s.shape) == (2, 64)
    assert torch.allclose(out_o.norm(dim=1), torch.ones(2), atol=1e-5)


def test_shared_projection_backward_compatible():
    m = build_model(["optical", "multispectral", "sar"], embedding_dim=32, n_classes=5)
    m.eval()
    assert not isinstance(m.projection, torch.nn.ModuleDict)
    assert m.modality_channels == {"optical": 3, "multispectral": 8, "sar": 1}
    out = m.embed(torch.randn(1, 8, 32, 32), "multispectral")
    assert tuple(out.shape) == (1, 32)


def test_build_backbone_vit_no_download():
    enc = build_backbone("vit_b_16", pretrained=False, image_size=64)
    assert isinstance(enc, BaseEncoder)
    assert isinstance(enc, ViTEncoder)


def test_foundation_model_requires_checkpoint():
    with pytest.raises(ValueError, match="satmae"):
        build_backbone("satmae", foundation={"satmae": {"path": None}})
    with pytest.raises(ValueError, match="prithvi"):
        build_backbone("prithvi", foundation={"prithvi": {"path": None}})


def test_external_encoder_loads_checkpoint(tmp_path):
    import torchvision

    backbone = torchvision.models.vit_b_16(image_size=64)
    backbone.heads = torch.nn.Identity()
    ckpt = os.path.join(str(tmp_path), "model.pth")
    torch.save({"state_dict": backbone.state_dict()}, ckpt)
    loaded = torchvision.models.vit_b_16(image_size=64)
    loaded.heads = torch.nn.Identity()
    enc2 = _ExternalEncoder(loaded, 768, ckpt)
    out = enc2.forward_features(torch.randn(1, 3, 64, 64))
    assert tuple(out.shape) == (1, 768)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("resnet999")


if __name__ == "__main__":
    import tempfile

    test_vit_forward_shape_at_64()
    test_interpolate_pos_embed_preserves_cls()
    test_per_modality_projection_heads()
    test_shared_projection_backward_compatible()
    test_build_backbone_vit_no_download()
    test_foundation_model_requires_checkpoint()
    test_external_encoder_loads_checkpoint(tempfile.mkdtemp())
    test_unknown_backbone_raises()
    print("test_encoders.py: all tests passed")
