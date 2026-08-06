# Preprocessing & augmentation

The pipeline applies two stages before the network sees any image:

1. **Patch preprocessing** — a configurable, *modality-aware* pass over the raw
   loaded arrays (in `src/data/preprocessing.py`). This is where SAR-specific
   handling lives; it is **not** optical preprocessing blindly applied to SAR.
2. **Input normalisation** — per-modality [0,1] scaling + mean/std
   standardisation (the `build_transforms` step). The transform *always* scales
   to [0,1] first, so the values the network sees are consistent with the
   statistics used to normalise them.

Both stages are controlled by the `preprocessing` and `augmentation` sections
of the config. With all settings at their defaults the behaviour is identical
to the original pipeline (identity preprocessing, no augmentation).

---

## Stage 1: modality-aware patch preprocessing

```yaml
preprocessing:
  resize: null            # int target spatial size, or null (keep native)
  cloud_max: null         # drop images whose cloud_cover exceeds this
  sar:
    log_transform: false  # apply log(1+x) to intensity before scaling
    clip_min: null
    clip_max: null
    invalid_value: null   # scalar value (or "nan") treated as missing
    invalid_fill: zero    # zero | median | nan_to_num
    speckle_filter: none  # none | lee
    speckle_window: 3
  optical:
    clip_min: null
    clip_max: null
    invalid_value: null
    invalid_fill: zero
  multispectral:
    band_selection: null  # e.g. [0,1,2,4,5] to subset the band stack
    missing_bands: raise  # raise | warn | drop
    clip_min: null
    clip_max: null
    invalid_value: null
    invalid_fill: zero
```

### SAR

* **Log transform** — `sar.log_transform: true` applies `log(1 + x)` to the
  intensity before scaling, compressing the wide radiometric range typical of
  SAR before statistics are computed.
* **Clipping** — `clip_min` / `clip_max` bound the intensity range
  (e.g. `clip_max: 2.0` for Sentinel-1 intensity).
* **Speckle** — `speckle_filter: lee` applies an edge-preserving **Lee filter**
  (configurable `speckle_window`, default 3) to reduce multiplicative speckle
  before training/evaluation. `none` is the default and leaves the array
  untouched.
* **Invalid values** — SAR scenes often contain `NaN` / fill values
  (e.g. `-9999`, no-data). `invalid_value` identifies them and `invalid_fill`
  replaces them with `zero`, a per-band `median`, or via `nan_to_num`.
* **Numerical stability** — `log1p` is used (never `log` on raw), clips are
  applied after log, and the [0,1] scaling clamps SAR to `[0, 2]` for stable
  statistics.

### Optical

* Clipping (`clip_min` / `clip_max`, e.g. `[0, 255]` for display-space RGB) and
  the same invalid-value handling as SAR. Optical is treated as display-space
  RGB (no calibration data is available for true reflectance conversion).
* **Cloud filtering** — `preprocessing.cloud_max` drops images whose
  metadata `cloud_cover` exceeds the threshold (only when the dataset provides
  cloud cover, e.g. SEN12MS). Images with unknown cloud cover are kept.

### Multispectral

* **Band selection** — `band_selection: [0,1,2,4,5]` keeps only the chosen
  bands (by index into the stack). This is how e.g. SEN12MS's 13-band S2 stack
  is reduced to a configurable subset.
* **Missing-band validation** — if a requested index exceeds the stack depth,
  `missing_bands: raise` (default) errors, `warn` logs and drops, `drop` drops
  silently.
* Clipping + invalid-value handling as above.

### Resizing

`preprocessing.resize: <int>` resizes every modality to `<int> × <int>` with
bilinear interpolation (per band). With `null` (default) the native patch size
is kept.

---

## Stage 2: input normalisation

Every modality is scaled to a shared `[0,1]` axis (`scale_to_unit`) and then
standardised with per-modality mean/std statistics computed on that axis:

```
input -> scale_to_unit (per-modality cap) -> standardise (per-channel mean/std)
```

> **Regression fix (Phase 3):** previously the standardising transform fed raw
> optical uint8 values (0–255) directly against statistics computed on the
> [0,1] scale, producing network inputs in the hundreds. The transform now
> scales first, so optical inputs are properly standardised. This measurably
> improved cross-modal retrieval (see README reference results).

---

## Augmentation (training set only)

```yaml
augmentation:
  enabled: false
  random_crop: 0.0        # fraction cropped away, then resized back
  hflip: false
  vflip: false
  rotation_90: false      # random k*90 degree rotation
  noise_std: 0.0          # additive gaussian noise (on the [0,1] scale)
  spectral_jitter: 0.0    # per-band multiplicative jitter (multispectral only)
```

Augmentation runs **only on the training loader**, on the [0,1]-scaled array,
between scaling and standardisation. Only augmentations that preserve
remote-sensing semantics are offered:

* `random_crop` — a random window (scale-preserving) cropped then resized back;
* `hflip` / `vflip` — mirror flips;
* `rotation_90` — rotation by multiples of 90°, never arbitrary angles (which
  would corrupt the axis-aligned geometry of satellite scenes);
* `noise_std` — small additive gaussian noise;
* `spectral_jitter` — per-band multiplicative jitter, **multispectral only**
  (per-band reflectance variation is physically plausible; it is never applied
  to optical RGB or SAR).

The validation/evaluation loader always uses the un-augmented transform, so
metrics stay comparable.

### Example: SEN12MS-style SAR + optical pipeline

```yaml
preprocessing:
  cloud_max: 0.3
  sar:
    log_transform: true
    clip_max: 1.5
    speckle_filter: lee
    invalid_value: nan
    invalid_fill: zero
  multispectral:
    band_selection: [1, 2, 3, 4, 5, 7, 8, 11]
    missing_bands: raise
augmentation:
  enabled: true
  random_crop: 0.1
  hflip: true
  rotation_90: true
  noise_std: 0.02
  spectral_jitter: 0.05
```

## Tests

`tests/test_preprocessing.py` and `tests/test_augmentation.py` cover: identity
behaviour, the optical scaling regression, SAR log/clip/speckle/invalid-value
handling, band selection + missing-band validation, resizing, cloud filtering,
and each augmentation (plus the "multispectral-only" spectral-jitter rule).
