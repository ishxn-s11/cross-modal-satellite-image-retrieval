# Methodology

This note describes the design choices behind the cross-modal satellite image
retrieval system.

## Problem framing

Given a **query** satellite image from one sensor modality, return a ranked list
of the most relevant images from a **gallery** that may mix modalities. Relevance
is defined by *semantic* content (land cover / land use / scene), not by pixel
similarity. We therefore learn a **common embedding space** in which
semantically-similar scenes are nearby regardless of sensor, then use fast
nearest-neighbour search.

## Data

Each patch is one geographic location rendered through three sensors sharing the
same land-cover semantics:

| Modality | Channels | Sensor model |
|---|---|---|
| optical | 3 (RGB) | true-colour composite: tone response + atmosphere + sensor noise |
| multispectral | 8 | reflectance stack (Blue…SWIR1), per-band radiance + resolution blur |
| sar | 1 (VV-like) | backscatter intensity + multiplicative Gamma speckle + incidence shading |

All patches carry a ground-truth class (10 land-cover classes), used to define
relevance for evaluation. A real-data path (EuroSAT) is provided for optical
imagery; companion bands for it are simulated and clearly flagged.

## Model: modality-adaptive encoder

```
per-modality adapter (1x1 conv, spectral init)  ->  shared ResNet18  ->  projection MLP -> L2-normalised embedding
```

* **Adapters** normalise the different band counts (3 / 8 / 1) onto the backbone's
  channel space, initialised to a sensible spectral mapping and trained.
* The **shared backbone** extracts generic spatial features; the deepest block may
  be fine-tuned (`unfreeze_stage`).
* The **projection head** maps features to a low-dim (128) **L2-normalised** space.

## Training objective

Three complementary losses (`src/training/contrastive.py`):

- **InfoNCE (CLIP-style)** — aligns the same patch across modalities.
- **Supervised contrastive (SupCon)** — clusters patches of the same class.
- **Cross-entropy** on an auxiliary classifier — direct supervised signal.

Training is fast on CPU because the backbone stays frozen except the deepest
block; only the small adapters/projection are updated.

## Retrieval

FAISS `IndexFlatIP` over L2-normalised embeddings (inner product = cosine
similarity). Flat exact search is used for gallery sizes seen here; switching to
an IVF index scales to much larger galleries. Per-query search latency is
measured at the FAISS step (<0.1 ms on the reference run).

## Evaluation

For every (query-modality, gallery-modality) pair and cutoff K ∈ {5, 10}:

```python
precision@K = |relevant ∩ topK| / K
recall@K    = |relevant ∩ topK| / |relevant in gallery|
F1@K        = 2·P·R / (P + R)
```

averaged over all queries. Both **same-modal** and **cross-modal** pairs are
reported. Because cross-modal retrieval is harder, a **cross-weighted** average
(cross-modal terms weighted 1.5×) summarises overall performance. Full per-pair
metrics are written to `outputs/metrics/`.

## Reference run

See the *Reference results* section of the README for numbers on the default
2,000-patch / 6-epoch configuration (~**0.05 ms/query**, cross-modal competitive
with same-modal for optical↔multispectral).