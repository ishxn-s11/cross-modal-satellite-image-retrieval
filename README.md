# Cross-Modal Satellite Image Retrieval

A retrieval system that accepts a **query satellite image** from one sensor
modality (optical RGB, multispectral, or SAR) and returns a **ranked list of the
most relevant images** from a gallery database containing images from the same
**or different** modalities.

The system learns a **common embedding space** where images with similar
land-cover / land-use / scene content are close together *regardless of which
sensor captured them*, then performs fast nearest-neighbour retrieval. It
reports **top-5 / top-10 (F1@K)** results and **average retrieval time per
query** for both **same-modal** and **cross-modal** retrieval.

![retrieval demo](docs/retrieval_example.png)

---

## Objectives this project addresses

| Objective | How it is met |
|---|---|
| Same-modal retrieval (opt→opt, ms→ms, sar→sar) | per-modality FAISS galleries |
| Cross-modal retrieval (opt→ms, opt→sar, ms→sar, …) | shared embedding space + cross-modality search |
| Top-5 / top-10 ranking | every evaluation reports F1@5 and F1@10 |
| Multi-sensor data (≥2 aligned modalities) | optical + multispectral + SAR per patch |
| Low retrieval time | flat cosine index → ~0.2 ms / query on CPU |
| Learn a common representation space | contrastive (InfoNCE + SupCon) alignment to a shared projector |

---

## Quick start

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
source .venv/Scripts/activate          # Windows (Git Bash)
# or: source .venv/bin/activate        # Linux/macOS

# 2. Install dependencies (CPU torch by default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Download / generate the dataset, train, and evaluate -- all in one go
python run_pipeline.py --config configs/default.yaml
```

The first run downloads the ImageNet-pretrained ResNet-18 backbone. The default
dataset is **fully self-contained** (no network): it is generated on the fly and
cached under `data/raw/synthetic/`.

### Individual steps

```bash
python scripts/download_data.py            # generate / download and cache the dataset
python scripts/train.py                    # train the shared-embedding model
python scripts/evaluate.py                 # run same/cross-modal retrieval evaluation
python scripts/retrieve_demo.py --pair optical,sar --k 5   # save retrieval montages
```

### Outputs

* `outputs/metrics/retrieval_metrics.csv` — F1/P/R @5 & @10 for every modality pair
* `outputs/metrics/retrieval_summary.json` — aggregate same/cross-modal averages
* `models/best_model/model.pt` — best checkpoint
* `outputs/retrieved_images/*.png` — visual montages (from `retrieve_demo.py`)

---

## Dataset

The task requires *"multi-sensor remote sensing image data consisting of two or
more aligned or semantically associated modalities."* Two sources are supported.

### 1. Synthetic multi-sensor dataset (default, works offline)

`src/data/synthetic.py` generates a paired dataset in which every patch is a
2-D land-cover scene rendered through three distinct sensor models with the
**same underlying semantics** (like the same location seen by different sensors):

- **optical** — true-colour RGB composite (tone response, atmosphere, sensor noise)
- **multispectral** — 8-band reflectance stack (Blue…SWIR1), per-band radiance noise
- **sar** — single-channel intensity built from class-dependent backscatter,
  multiplicative Gamma speckle, and incidence-angle shading

All patches have a ground-truth **land-cover class** (e.g. *Forest, Water,
Residential, …*), which grounds the semantic relevance used for evaluation.

```yaml
dataset:
  source: synthetic
  num_patches: 2000
  image_size: 64
```

### 2. Real EuroSAT ingestion

`src/data/eurosat.py` downloads a public EuroSAT mirror (~90 MB, HuggingFace
`nielsr/eurosat-demo`) and uses its **real** RGB patches as the optical modality.
Because EuroSAT ships RGB only, the multispectral / SAR views are **derived from
the real optical patch** with the same rendering models and are stringently
flagged (``"_sim"``) so reports stay honest about observed vs. simulated bands.

```yaml
dataset:
  source: eurosat
  eurosat_max_patches: 6000
```

---

## Model

`src/models/encoder.py` — a **modality-adaptive encoder**:

1. Per-modality **input adapters** (1×1 convs) project each sensor's band stack
   onto the 3-channel space of a shared backbone (init to a sensible spectral map).
2. A shared **ResNet-18 backbone** extracts generic spatial features.
3. A MLP **projection head** maps features into a low-dim, **L2-normalised**
   embedding space (default 128-D).
4. An auxiliary linear **classifier** provides a supervised signal.

### Training (objective)

Three objectives are combined (`src/training/contrastive.py`):

- **InfoNCE (CLIP-style)** aligns the same patch across modalities
  (`clip_weight`).
- **Supervised contrastive (SupCon)** clusters patches of the same class
  (`supcon_weight`).
- **Cross-entropy** on the classifier gives direct supervision (`cls_weight`).

The backbone is frozen by default (`model.freeze_backbone: true`) so training is
fast on CPU — for the default run only the small adapters/projection are updated.

---

## Retrieval

`src/retrieval/` builds a **FAISS `IndexFlatIP`** per gallery modality over the
L2-normalised embeddings (inner product = cosine similarity). Flat exact search
is both exact and extremely fast at the gallery sizes used here.

```yaml
retrieval:
  top_k: [5, 10]
  gallery_fraction: 0.85
  n_query: 400
```

---

## Evaluation

`src/evaluation/` computes precision@K, recall@K, and F1@K for each modality
pair, averaged over queries, where *relevance* = same land-cover class:

```
precision@K = |relevant ∩ retrieved top-K| / K
recall@K    = |relevant ∩ retrieved top-K| / |relevant in gallery|
F1@K        = 2·P·R / (P+R)
```

Both **same-modal** (opt→opt, ms→ms, sar→sar) and **cross-modal** pairs are
evaluated. Because cross-modal retrieval is harder and more valuable, the
summary also reports a **cross-weighted** average (cross-modal F1 weighted 1.5×).

Average retrieval time per query is measured at the FAISS search step and
reported for every pair.

---

## Reference results (default config)

Full pipeline: 2,000 patches, 10 land-cover classes, ResNet-18 backbone
(with the deepest block fine-tuned), 128-D embeddings, 6 epochs
(`configs/default.yaml`). Evaluated on a held-out query/gallery split.

| Metric (mean over queries) | Value |
|---|---|
| Same-modal F1@5 / F1@10  | 0.172 / 0.287 |
| Cross-modal F1@5 / F1@10 | 0.116 / 0.196 |
| Cross-weighted F1@5 / F1@10 | 0.139 / 0.232 |
| Avg. retrieval time / query | **0.05 ms** |

Per-pair detail (best pairs):

| Query → Gallery | F1@5 | F1@10 |
|---|---|---|
| multispectral → multispectral | 0.221 | 0.350 |
| optical → multispectral | 0.177 | 0.301 |
| multispectral → optical | 0.203 | 0.332 |
| optical → optical | 0.180 | 0.300 |

Cross-modal optical↔multispectral retrieval is competitive with same-modal
retrieval; SAR↔optical is lower, reflecting the limited discriminative content
of a single SAR intensity channel and the smaller gallery. Larger galleries
raise the recall ceiling and therefore F1.

---

## Configuration

All settings live in `configs/default.yaml`. Override anything from the CLI:

```bash
python run_pipeline.py --config configs/default.yaml \
    --set dataset.num_patches=4000 --set training.epochs=8
```

Key switches:

| Setting | Effect |
|---|---|
| `dataset.source`          | `synthetic` or `eurosat` |
| `model.backbone`          | `resnet18` / `resnet34` / `resnet50` |
| `model.freeze_backbone`   | keep backbone frozen (faster CPU) or fine-tune |
| `training.epochs` / `lr`  | training budget |
| `training.clip_weight` / `supcon_weight` / `cls_weight` | objective balance |

---

## Project layout

```
configs/          YAML configs (default.yaml)
src/data/         modalities, synthetic + EuroSAT generators, dataset, preprocessing
src/models/       modality-adaptive encoder + projection head
src/training/     contrastive losses + trainer
src/retrieval/    FAISS index + retrieval engine (persistence-aware)
src/database/     persistent stores: SQLite metadata, embedding cache, FAISS gallery store
src/evaluation/   P/R/F1@K metrics + evaluation harness
src/pipeline.py   end-to-end orchestration
scripts/          CLI entry points (train / evaluate / retrieve_demo / download)
tests/            unit + end-to-end smoke tests
api/  web/        FastAPI demo + web UI (Retrieve / Browse / Upload / Metrics / History)
notebooks/        5 Jupyter notebooks (exploration, embeddings, retrieval, FAISS, persistence)
database/         SQLite metadata store (runtime-generated, git-ignored)
embeddings/       cached full-dataset embeddings per modality (runtime-generated)
faiss/            persisted per-modality FAISS galleries (runtime-generated)
outputs/          metrics, reports, retrieved images, logs
```

---

## Persistent retrieval database

Embeddings, FAISS indices and metadata are **persisted** so that warm runs (and
web-app restarts) skip the expensive embedding step. Three stores in
`src/database/` write to three top-level directories:

| Directory | Holds | Keyed by |
|---|---|---|
| `database/` | SQLite metadata: images, gallery records, retrieval log (`MetadataStore`) | — |
| `embeddings/` | full-dataset embeddings per modality, `.npz` (`EmbeddingStore`) | config hash |
| `faiss/` | per-modality FAISS galleries, `.index` + `_meta.json` (`IndexStore`) | config hash + gallery-subset hash |

The cache key is a **config hash** (`src/database/config_hash.py`) derived from
the dataset + model configuration **and the trained checkpoint's mtime**, so a
retrain or a config change invalidates the cache automatically. The
`RetrievalEngine` uses these stores transparently: on a cold start it embeds
and builds; on a warm start it reloads (`build_gallery` logs
`loading cached gallery …`).

```python
from src.database import EmbeddingStore, IndexStore, MetadataStore, compute_cache_key
```

## Notebooks

`notebooks/` ships five fully-executed, self-contained tutorials that reuse the
project `src/` (run `pip install -r requirements-notebooks.txt` first):

| Notebook | What it shows |
|---|---|
| `01_dataset_exploration` | the 3 aligned sensors, class distribution, spectral signatures |
| `02_model_and_embeddings` | the shared 128-D embedding space, PCA/UMAP, cross-modal self-similarity |
| `03_interactive_retrieval` | same- & cross-modal queries with montages + batch P@K |
| `04_faiss_index_comparison` | flat vs IVF vs HNSW FAISS: latency and recall@10 |
| `05_database_and_persistence` | the 3 stores end-to-end, config-hash invalidation, warm reload |

```bash
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

## Web UI

The web app (`api/app.py` + `web/index.html`) is backed by the persisted
database and boots from cached galleries (warm start). It exposes five tabs:
**Retrieve** (with a 🎲 random-query picker), **Browse DB**, **Upload** (query by
image file), **Metrics** (F1/P/R@K from the last evaluation) and **History**
(recent retrievals logged to SQLite).

```bash
pip install -r requirements-web.txt
uvicorn api.app:app --reload
```

## Notes for scaling up

- Switch to a larger backbone (`resnet50`) or unfreeze the backbone for more
  capacity; CPU training time grows accordingly.
- For large galleries (100k+), set `retrieval.gallery_fraction` and/or use the
  IVF index (`FaissCosineIndex(..., nlist=...)`).
- Bring your own multi-sensor data by dropping aligned modality folders under
  `data/raw` and adding a loader mirroring `synthetic.py` / `eurosat.py`.