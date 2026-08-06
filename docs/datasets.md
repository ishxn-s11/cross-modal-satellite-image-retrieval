# Datasets

The system selects its dataset entirely through configuration:

```yaml
dataset:
  name: synthetic          # synthetic | eurosat | sen12ms | so2sat | bigearthnet_mm
  root: data/raw
  allow_fallback: true     # fall back to synthetic when a real dataset is absent
```

All backends implement the same
[`DatasetInterface`](../src/data/interface.py), so the training pipeline,
retrieval engine and web UI do not depend on which dataset is selected. Every
dataset exposes the standardised [`ImageMetadata`](../src/data/metadata.py)
record (image_id, dataset, sensor, modality, lat/lon, acquisition date,
land-cover, resolution, cloud cover, orbit, file paths). Fields a source does
not provide are `None` and are simply omitted from results/UI.

## Supported datasets

| name | Data | Modalities | Size | Auto-download? |
|---|---|---|---|---|
| `synthetic` | simulated multi-sensor scenes (default, offline) | optical · multispectral · sar | generated (~tens of MB) | no (self-contained) |
| `eurosat` | real Sentinel-2 RGB; MS/SAR derived & flagged | optical (real) · multispectral · sar (derived) | ~90 MB | **yes** (small public mirror) |
| `sen12ms` | **SEN12MS (SEN1-2)** real paired Sentinel-1/2 | optical · multispectral · sar | ~110 GB | **never** |
| `so2sat` | So2Sat LCZ42 real paired Sentinel-1/2 | optical · multispectral · sar | ~55 GB | **never** |
| `bigearthnet_mm` | BigEarthNet-MM real paired Sentinel-1/2 | optical · multispectral · sar | ~20 GB | **never** |

> Real datasets are large and are **never downloaded automatically**. If the
> data directory is missing the loader raises a clear error with instructions;
> with `allow_fallback: true` (the default) the pipeline logs that hint and
> proceeds with the self-contained synthetic dataset.

---

## 1. Synthetic (default)

No download. Generated deterministically by
[`src/data/synthetic.py`](../src/data/synthetic.py) and cached under
`data/raw/synthetic/`.

- 10 land-cover classes (EuroSAT-style names).
- Every patch is one scene rendered through three sensor models: optical RGB
  composite, 8-band multispectral reflectance stack, single-channel SAR
  intensity (backscatter × speckle × incidence shading).
- Patches also carry deterministic **synthetic** scene placement (lat/lon and
  acquisition dates) so geographic/temporal evaluation and the interactive map
  work offline. These are simulation coordinates, clearly labelled synthetic.

```yaml
dataset:
  name: synthetic
  root: data/raw
  num_patches: 6000
  image_size: 64
  seed: 42
modalities: [optical, multispectral, sar]
```

## 2. EuroSAT (real optical, derived companions)

Official source: EuroSAT (Helber et al., IEEE GRSL 2019).
Download: the loader fetches a small public mirror (`nielsr/eurosat-demo`,
~90 MB) on first use — this is the **only** dataset that downloads itself, and
only when you set `name: eurosat`.

- Real Sentinel-2 RGB patches (10 classes) as the `optical` modality.
- `multispectral` / `sar` are **derived** from the real RGB patch by the same
  physical rendering models and are flagged `_sim` in the source so reports
  stay honest about observed vs simulated bands.

```yaml
dataset:
  name: eurosat
  eurosat_max_patches: 6000
```

## 3. SEN12MS / SEN1-2 (primary real dataset)

- **Official source:** SEN12MS — Schmitt, Hughes & Zhu, "The SEN12MS dataset
  for Remote Sensing Applications", IEEE TGRS 2019.
- **Download:** [mediatum](https://mediatum.ub.tum.de/1474000) (registration
  required) or Zenodo. ~110 GB for the full ROIs1868 set.
- **Modalities:** Sentinel-1 (VV, VH) and Sentinel-2 (13 bands), 256×256
  patches, 17 IGBP-style land-cover classes, across 4 seasons and 5 regions.
- **Expected folder structure** (the loader reads exactly this):

  ```
  <root>/
    ROIs/ROIs1868_summer.csv      # one CSV per season; links s1/s2/lulc scenes
    s1/{s1_scene_id}/scene.tif    # 2-band SAR
    s2/{s2_scene_id}/scene.tif    # 13-band optical
    lulc/{lulc_scene_id}/scene.tif# 1-band land-cover map
    s1_meta/{s1_scene_id}/scene_meta.json
    s2_meta/{s2_scene_id}/scene_meta.json
  ```

- **Preprocessing (as loaded):** each 256×256 scene is cropped into
  non-overlapping `patch_size` (default 64) patches; `optical` = S2 RGB
  composite (B4/B3/B2, uint8); `multispectral` = configurable S2 band subset
  (default 8 bands → float reflectance); `sar` = VV/VH scaled intensity. Labels
  are the dominant LULC class per patch. The configurable radiometric
  preprocessing (SAR log/dB transform, clipping, cloud filtering) is applied at
  the preprocessing stage — see `docs/preprocessing.md`.
- **Train/validation/test strategy:** per the paper, hold out whole scenes /
  geographic regions to avoid spatial leakage (configured through the split
  strategy — see `docs/splits.md`).

```yaml
dataset:
  name: sen12ms
  root: /path/to/sen12ms
  allow_fallback: true
  sen12:
    roi_csv: ROIs/ROIs1868_summer.csv   # optional; auto-detected otherwise
    patch_size: 64
    max_scenes: 200                     # cap for a quick run
    ms_bands: [1, 2, 3, 4, 5, 7, 8, 11]
    sar_bands: [VV, VH]
    class_names: null                   # optional 17-name legend override
modalities: [optical, sar]
```

> **Verification note.** The loader is implemented against the documented
> layout above and validated by a fixture-based test (`tests/test_sen12_loader.py`
> builds a synthetic SEN12MS-like tree). It has **not** been run against a live
> 110 GB download in this repository's CI; if your downloaded folder differs,
> report the structure and the loader will be adjusted.

## 4. So2Sat LCZ42 (secondary real dataset)

- **Official source:** Zhu et al., "So2Sat LCZ42: A Benchmark Data Set for the
  Classification of Global Local Climate Zones", IEEE GRSM 2020.
- **Download:** [technical University of Munich / Zenodo](https://mediatum.ub.tum.de/1551843).
  ~55 GB.
- **Modalities:** Sentinel-1 (VV/VH) and Sentinel-2 (8 bands), 256×256 patches,
  17 local-climate-zone labels.
- **Expected structure:** HDF5 split files under `<root>/so2sat/`:

  ```
  <root>/so2sat/training.h5      # sen1, sen2, label_idx
  <root>/so2sat/validation.h5
  <root>/so2sat/testing.h5
  ```

```yaml
dataset:
  name: so2sat
  root: /path/to/so2sat
  so2sat:
    h5_file: training.h5
    max_patches: 5000
modalities: [optical, sar]
```

## 5. BigEarthNet-MM (secondary real dataset)

- **Official source:** Sumbul et al., "BigEarthNet-MM: A Large-Scale,
  Multimodal, Multilabel Benchmark Archive for Remote Sensing Image
  Classification and Retrieval", IEEE GRSM 2021.
- **Download:** [Zenodo](https://zenodo.org/record/6160062). ~20 GB.
- **Modalities:** Sentinel-1 (2 bands) and Sentinel-2 (12 bands), 120×120
  patches, multi-label CORINE land-cover (19-class scheme used by default).
- **Expected structure:**

  ```
  <root>/BigEarthNet-S1/{patch_id}/{patch_id}_S1.tif
  <root>/BigEarthNet-S2/{patch_id}/{patch_id}_S2.tif
  <root>/BigEarthNet_19_labels.csv     # patch_id,label1,label2,...
  ```

- Labels are multi-label; the single-label pipeline uses the first present
  label in canonical order as the primary class, and the full set is preserved
  in the image metadata (`extra.labels`).

```yaml
dataset:
  name: bigearthnet_mm
  root: /path/to/bigearthnet_mm
  bigearthnet_mm:
    labels_csv: /path/to/BigEarthNet_19_labels.csv
    max_patches: 5000
modalities: [optical, sar]
```

---

## Optional dependency

Reading real multi-band GeoTIFFs (SEN12MS, BigEarthNet-MM) and HDF5 (So2Sat)
requires:

```bash
pip install -r requirements-real-data.txt   # tifffile, h5py
```

The default synthetic/EuroSAT flow does not need these.

## How metadata flows through the system

`build_dataset(cfg)` returns a `DatasetInterface` whose `metadata` list of
`ImageMetadata` records is used to:

- populate the SQLite `images` table (lat/lon/date/sensor/cloud columns),
- power geographic & temporal evaluation and losses (see `docs/learning.md`),
- render result cards and the interactive map in the UI.

Any dataset that does not provide a field (e.g. EuroSAT has no coordinates)
leaves it `None`, and that field is omitted from results and the UI.
