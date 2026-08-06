# Training objectives

The model is trained with a weighted combination of objectives
(`src/training/contrastive.py`, `src/training/geo.py`):

```
Total = λ1·InfoNCE  +  λ2·SupCon  +  λ3·Classification  +  λ4·Geographic Alignment
```

| Weight | Config key | Default | Objective |
|---|---|---|---|
| λ1 | `training.clip_weight` | 1.0 | InfoNCE (CLIP-style): same patch across modalities |
| λ2 | `training.supcon_weight` | 1.0 | Supervised contrastive: same land-cover class together |
| λ3 | `training.cls_weight` | 1.0 | Cross-entropy on the auxiliary classifier |
| λ4 | `training.geo_weight` | 0.0 | Geographic / temporal alignment (off by default) |

## InfoNCE

Symmetric CLIP-style InfoNCE between every ordered pair of modalities: for a
batch of patches, the same patch's embedding in modality *a* must be closer to
its embedding in modality *b* than to every other patch in the batch
(the other patches are the batch negatives).

## Supervised contrastive (SupCon)

All patches of the same land-cover class are pulled together (pooled across
modalities) regardless of sensor.

## Classification

An auxiliary linear classifier (on the shared embedding) supplies a direct
supervised signal.

---

## Hard-negative mining (optional, default off)

```yaml
training:
  hard_negatives:
    enabled: false
    n_hard: 8          # keep only the 8 hardest negatives per anchor
    strategy: embedding
```

Instead of a contrastive denominator over the *whole* batch (mostly easy
negatives), hard-negative mining restricts each anchor's denominator to its
``n_hard`` **most confusable** negatives -- highest embedding similarity for
InfoNCE, highest-similarity *different-class* items for SupCon. This
concentrates the gradient on the negatives that actually need to be separated.

* `strategy: embedding` (implemented) -- hardest by current embedding
  similarity, the standard, defensible method.
* The geographic/temporal term below provides the complementary
  geo-aware supervision (nearby-but-different-class scenes).

When `n_hard >= batch - 1` the loss reduces exactly to the standard
InfoNCE / SupCon (so enabling the feature with a large `n_hard` is harmless).

## Geographic + temporal alignment (optional, default off)

```yaml
training:
  geo_weight: 0.0        # λ4 -- set > 0 to enable
  geo_same_km: 5.0       # pairs within this distance are positive
  geo_push_distant: false
  geo_distant_km: 100.0
```

Requires acquisition metadata (lat/lon) -- the synthetic dataset provides
deterministic simulated coordinates, and SEN12MS provides real ones. Rows
without coordinates (NaN) are skipped, so enabling the loss on a dataset
without geo is a no-op.

* **Pull** -- SupCon-style: for each anchor, bring all pairs within
  `geo_same_km` together. Because the *same location observed on different
  acquisition dates* is a positive pair, this is also **temporal
  robustness**: the model learns semantic consistency across seasons.
* **Push** (optional) -- a hinge that penalises positive cosine similarity for
  pairs further than `geo_distant_km` apart, separating geographically distant,
  semantically different scenes.

## Notes

* Every advanced objective is **independently switchable**; the default config
  (all advanced weights zero / disabled) reproduces the reference results in
  the README exactly.
* Only objectives that are actually available for the data are applied: no geo
  metadata -> geo loss is exactly zero; no labels -> SupCon/CE degrade to
  nothing meaningful (all datasets here carry labels).
* Tests: `tests/test_hard_negatives.py` (hard-loss reduction, gradient
  concentration, integration) and `tests/test_geo.py` (haversine,
  NaN-handling, pull/push).
