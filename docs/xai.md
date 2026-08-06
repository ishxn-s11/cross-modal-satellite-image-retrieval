# Explainability & embedding visualisation

## Saliency (Grad-CAM / attention)

```bash
python scripts/xai_demo.py --query-id 12 --modality optical
```

`scripts/xai_demo.py` saves the query image, its saliency map and an overlay.

* **CNN backbones** (ResNet) use **Grad-CAM** (`src/xai/gradcam.py`): the last
  convolutional feature maps are weighted by their gradient w.r.t. the target
  class logit (from the auxiliary classifier). Works with frozen backbones.
* **Transformer backbones** (ViT) use **self-attention** (`src/xai/attention.py`):
  the CLS-token attention from the last transformer layer, averaged over heads,
  is rendered over the patch grid and upsampled to the input resolution.
* These are **not interchangeable**: Grad-CAM is only applied to CNN encoders
  and attention only to transformer encoders. The demo picks the right one from
  `model.backbone` automatically.

The UI's *Explainability* tab (Streamlit) reuses the same functions.

## Embedding-space visualisation

```bash
python scripts/visualize_embeddings.py --method pca,tsne,umap
```

`src/utils/embedding_viz.py` projects embeddings to 2-D with **PCA / t-SNE /
UMAP** (UMAP is optional; falls back to PCA when unavailable) and colours them
by class, modality, geographic region or dataset. `plot_embedding_comparison`
shows the same points **before vs after training** side by side, making it easy
to see whether modalities become aligned in the shared space.

Outputs are written to `outputs/embeddings/`.

## Tests

`tests/test_xai.py` covers Grad-CAM shape/range, ViT attention maps, saliency
overlay, and the PCA/t-SNE/plot helpers.
