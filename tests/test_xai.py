"""Tests for explainability (Grad-CAM / ViT attention) + embedding visualisation."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.encoder import build_model
from src.utils.embedding_viz import plot_embedding_comparison, plot_embeddings, reduce_embeddings
from src.xai.attention import attention_map
from src.xai.gradcam import gradcam, last_conv_layer, overlay_saliency


def test_gradcam_resnet_shape_and_range():
    m = build_model(["optical"], embedding_dim=32, n_classes=5, freeze_backbone=True)
    m.eval()
    x = torch.randn(1, 3, 32, 32)
    sal = gradcam(m, x, "optical", class_idx=2)
    assert sal.shape == (32, 32)
    assert sal.min() >= 0.0 and sal.max() <= 1.0
    # last_conv_layer resolves to a Conv2d
    assert isinstance(last_conv_layer(m.backbone), torch.nn.Conv2d)


def test_attention_vit_shape():
    m = build_model(["optical"], backbone="vit_b_16", embedding_dim=32,
                    n_classes=5, freeze_backbone=True, pretrained=False, image_size=64)
    m.eval()
    x = torch.randn(1, 3, 64, 64)
    sal = attention_map(m, x, "optical")
    assert sal.shape == (64, 64)
    assert sal.min() >= 0.0 and sal.max() <= 1.0


def test_overlay_saliency():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    sal = np.ones((32, 32), dtype=np.float32)
    out = overlay_saliency(img, sal)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8


def test_reduce_embeddings_pca_tsne():
    rng = np.random.RandomState(0)
    emb = rng.rand(50, 8).astype(np.float32)
    pca = reduce_embeddings(emb, method="pca")
    assert pca.shape == (50, 2)
    tsne = reduce_embeddings(emb, method="tsne")
    assert tsne.shape == (50, 2)


def test_plot_embeddings_returns_figure():
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.RandomState(1)
    emb = rng.rand(60, 8).astype(np.float32)
    ids = np.arange(60) % 3
    fig = plot_embeddings(emb, ids, ["a", "b", "c"], method="pca")
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_embedding_comparison_returns_figure():
    import matplotlib

    matplotlib.use("Agg")
    rng = np.random.RandomState(2)
    before = rng.rand(60, 8).astype(np.float32)
    after = rng.rand(60, 8).astype(np.float32)
    ids = np.arange(60) % 2
    fig = plot_embedding_comparison(before, after, ids, ["x", "y"], method="pca")
    assert fig is not None
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    test_gradcam_resnet_shape_and_range()
    test_attention_vit_shape()
    test_overlay_saliency()
    test_reduce_embeddings_pca_tsne()
    test_plot_embeddings_returns_figure()
    test_plot_embedding_comparison_returns_figure()
    print("test_xai.py: all tests passed")
