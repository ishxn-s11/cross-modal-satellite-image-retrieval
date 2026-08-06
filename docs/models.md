# Model architecture

The system maps every sensor view into a **shared embedding space** with a
modality-adaptive encoder (`src/models/encoder.py`):

```
per-modality adapter (1x1 conv, spectral init)
   -> shared backbone (BaseEncoder)
      -> projection head (shared or per-modality)
         -> L2-normalised embedding
            -> auxiliary linear classifier
```

## Adapters

A per-modality 1×1 Conv + BN + ReLU maps the modality's band count onto the
3-channel space the backbone expects, initialised to a spectrally meaningful
projection (RGB bands for optical, R/G/B slots from the multispectral stack,
replicated channel(s) for SAR). Real datasets with non-default band counts
(e.g. 2-band SAR, configurable S2 band subsets) are handled via
`modality_channels` derived from the loaded data.

## Encoders (`BaseEncoder` hierarchy)

| Config `model.backbone` | Class | feature_dim | Notes |
|---|---|---|---|
| `resnet18` / `resnet34` / `resnet50` | `ResNetEncoder` | 512 / 512 / 2048 | ImageNet-pretrained; `unfreeze_stage: stage3/4` fine-tunes the deepest block(s) |
| `vit_b_16` | `ViTEncoder` | 768 | ViT-B/16 at `model.vit_image_size` (default = patch size). ImageNet pos-embed is bilinearly interpolated to the patch grid. `unfreeze_stage: last` fine-tunes the last 2 transformer blocks |
| `satmae` | `SatMAEEncoder` | configurable | Foundation model — **weights not bundled**; requires `model.foundation.satmae.path` |
| `prithvi` | `PrithviEncoder` | configurable | Foundation model — **weights not bundled**; requires `model.foundation.prithvi.path` |

```yaml
model:
  backbone: vit_b_16
  vit_image_size: 64
  vit_pretrained: true
  embedding_dim: 256
  freeze_backbone: true
  projection_heads: shared
```

### ViT at small patch sizes

`vit_b_16` accepts an `image_size` argument; at 64×64 it uses a 4×4 patch grid
(16 tokens). When `vit_pretrained: true`, the ImageNet positional embedding is
bilinearly interpolated from the 14×14 grid to the target grid before loading,
so the spatial prior is retained at small resolutions.

### Foundation models (SatMAE / Prithvi)

These are large remote-sensing foundation models. **We do not bundle or fake
them**: selecting `backbone: satmae` (or `prithvi`) without a valid
`model.foundation.<name>.path` raises a clear error. When you supply a
checkpoint, the adapter builds a ViT backbone and loads the weights
best-effort (`strict=False`):

```yaml
model:
  backbone: satmae
  foundation:
    satmae:
      path: /path/to/satmae_encoder.pth
      feature_dim: 384   # must match your SatMAE backbone
```

> Compatibility caveat: exact state-dict layouts differ across upstream repos.
> The adapter applies `strict=False` loading and reports a clear error if the
> weights cannot be applied. Validate your checkpoint once before relying on
> it.

## Projection heads

`model.projection_heads`:

* `shared` (default) — one MLP projection shared by all modalities
  (backward compatible with existing checkpoints).
* `per_modality` — an independent projection MLP per modality
  (`nn.ModuleDict`), useful when each sensor needs its own mapping before the
  shared space.

## Embedding space

The projection outputs are **L2-normalised**, so cosine similarity and FAISS
inner-product search are equivalent. `model.embedding_dim` is configurable
(128 / 256 / 512 / 768); 128 is the default and keeps CPU training fast.

## Tests

`tests/test_encoders.py` covers: ViT forward shape at 64×64, pos-embed
interpolation, per-modality vs shared heads, foundation-model error handling,
external-checkpoint loading, and unknown-backbone errors.
