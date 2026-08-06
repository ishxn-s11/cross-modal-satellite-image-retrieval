# Phase 1 — Repository Audit & Gap Analysis

Date: 2026-08-06
Scope: full audit of `satelite-img-retrieval` (the working copy of the
Cross-Modal Satellite Image Retrieval project) before any upgrade work.

---

## 1. Baseline verification (performed, not assumed)

The following were actually run against the current repo on this machine
(Python 3.10.10 in `.venv`, torch 2.13.0+cpu, faiss 1.14.3, no CUDA):

- `python tests/test_metrics.py` → **PASSED**
- `python tests/smoke_test.py` → **PASSED** (full mini pipeline: synthetic
  240-patch → train 1 epoch → retrieve → evaluate; ~13 s on CPU)

The end-to-end path works. Reference numbers from the mini smoke run:

```
optical->optical         same  k=5 F1=0.2222  k=10 F1=0.2437
optical->multispectral   cross k=5 F1=0.1508  k=10 F1=0.1275
smoke_test.py: PASSED
```

## 2. What already exists (and is worth preserving)

The repo is already close to the target shape in several important ways:

| Area | Existing implementation | Notes |
|---|---|---|
| Architecture | `src/{data,models,training,retrieval,evaluation,database,utils}` + `api/` + `web/` + `scripts/` + `notebooks/` + `tests/` + `configs/` | Matches the target layout closely |
| Data | `synthetic.py` (paired opt/MS/SAR, 10 classes, seeded, disk-cached); `eurosat.py` (real RGB + honestly-flagged `_sim` MS/SAR) | Self-contained, offline-capable |
| Dataset class | `MultiModalDataset` + `collate_modalities` (paired cross-modal sampling) | Correct and reusable |
| Preprocessing | per-modality `normalize_scale` → mean/std standardisation | Minimal but sound |
| Model | `ModalityAdaptiveEncoder`: per-modality 1×1-conv adapters (spectral init) → shared ResNet18/34/50 → MLP projection → L2-norm + linear classifier | Clean, extensible |
| Losses | `info_nce` (symmetric CLIP-style), `supervised_contrastive`, CE; combined in `multimodal_losses` with configurable weights | Correct implementations |
| Training | AdamW + cosine LR + early stop + best-checkpoint by val acc; CPU-friendly (freeze/unfreeze blocks) | Works |
| Retrieval | FAISS `IndexFlatIP` (cosine via L2-norm), optional IVF; in-process L1 cache | Works |
| Persistence | `EmbeddingStore` (npz), `IndexStore` (faiss+meta), `MetadataStore` (SQLite: images/galleries/retrieval_logs), config-hash keying with checkpoint-mtime invalidation | Strong, matches target §33–34 |
| Evaluation | P@K / R@K / F1@K per pair; same/cross/weighted summaries; CSV+JSON reports; per-query timing | Good |
| API | FastAPI: status, retrieve, retrieve_upload, browse, history, eval, database, gallery-info, dataset-stats, random-query, cache-clear | Works, pydantic-validated |
| Web | self-contained dark SPA (Retrieve / Browse / Upload / Metrics / History) | Functional; no build step |
| Scripts | `train`, `evaluate`, `retrieve_demo`, `download_data`, `run_pipeline` | All wired to config |
| Notebooks | 5 executed notebooks reusing `src/` (exploration, embeddings+PCA/UMAP, retrieval, FAISS comparison, persistence) | Good |
| Config | `configs/default.yaml` + `DEFAULT_CONFIG` + CLI `--set k=v` deep-merge | Good |
| Docs | README + methodology + per-store READMEs | Honest (flags simulated bands) |

## 3. Gap analysis vs the master target

Legend: ✅ present · ◐ partial · ❌ missing

### Data & metadata
| # | Requirement | Status | Where |
|---|---|---|---|
| 3 | Real SEN1-2 / So2Sat LCZ42 / BigEarthNet-MM support | ❌ | only synthetic + EuroSAT(RGB) |
| 3 | Unified `DatasetInterface` (select via config) | ❌ | `prepare_dataset` uses if/elif on `source` |
| 9 | Standardised metadata (id, sensor, lat/lon, date, land-cover, cloud, orbit, paths) | ❌ | none; SQLite `images` has only id/class/split |
| 8 | Geographic split | ❌ | random stratified split only |
| 8 | Temporal split | ❌ | — |

### Preprocessing / augmentation
| # | Requirement | Status | Where |
|---|---|---|---|
| 4 | Configurable SAR pipeline (log, clip, speckle-aware, invalid values) | ◐ | hardcoded `[0,2]` clip only |
| 5 | Optical reflectance norm, cloud masking | ❌ | basic rescale only |
| 6 | MS band selection, missing-band validation | ❌ | fixed 8-band |
| 7 | Configurable augmentation (crop/flip/rotate/noise, modality-aware) | ❌ | none |

### Model
| # | Requirement | Status | Where |
|---|---|---|---|
| 10 | ViT encoder | ❌ | ResNet only |
| 10 | Foundation models (Prithvi / SatMAE) via adapter interface | ❌ | no `BaseEncoder` |
| 11 | Modality-specific encoders / projection heads | ◐ | shared backbone + per-modality adapters (acceptable design; no per-modality heads) |
| 12 | Shared embedding space, configurable dim (128–768) | ✅ | `embedding_dim` configurable |

### Losses / learning
| # | Requirement | Status | Where |
|---|---|---|---|
| 13 | InfoNCE + SupCon + classification, λ weights, temperature | ✅ | `multimodal_losses` |
| 14 | Hard-negative mining | ❌ | random batch negatives only |
| 15 | Geographic-aware loss / sampling | ❌ | — |
| 16 | Temporal learning + evaluation | ❌ | — |

### Retrieval / search
| # | Requirement | Status | Where |
|---|---|---|---|
| 17 | Same+cross modal, metric configurable (cosine/IP/Euclid) | ◐ | cosine via IP only |
| 18 | FAISS flat/IVF/HNSW/IVF-PQ, persistence, build/save/load/update/rebuild | ◐ | flat + IVF only; build/save/load present, no update/rebuild |
| 18 | Optional GPU FAISS | ❌ | — |
| 19 | Two-stage: FAISS top-100 → re-rank → top-10 | ❌ | single-stage only |
| 20 | Result objects with similarity+metadata+geo distance | ◐ | id/class/score/image/relevant only |

### Evaluation / benchmarking
| # | Requirement | Status | Where |
|---|---|---|---|
| 21 | P@K, R@K, F1@K | ✅ | — |
| 21 | mAP@K, NDCG@K, Recall@K curve | ❌ | — |
| 21 | Same-modal & cross-modal F1@5/@10, avg latency | ✅ | — |
| 22 | Latency breakdown (preprocess/embed/search/rerank/total, P50/P95/throughput) | ❌ | FAISS time only |
| 23 | Scalability benchmark (10K/100K/1M × flat/IVF/HNSW/IVF-PQ, memory) | ❌ | notebook compares flat/IVF/HNSW on toy sizes |
| 24 | Baseline comparison (5 baselines → comparison table) | ❌ | — |
| 25 | Ablation scripts (encoder/dim/loss/hard-neg/rerank/geo) → CSV/JSON | ❌ | — |

### XAI / visualisation
| # | Requirement | Status | Where |
|---|---|---|---|
| 26 | Grad-CAM / attention / saliency | ❌ | — |
| 27 | PCA / t-SNE / UMAP; before-vs-after; by modality/class/region/dataset | ◐ | PCA+UMAP in notebook utils only |

### UI / API / DB / config / engineering
| # | Requirement | Status | Where |
|---|---|---|---|
| 29 | Streamlit UI (Home/Retrieval/Results/Map/Analytics/Embeddings/Explainability/Gallery/History) | ❌ | SPA with 5 tabs instead |
| 30 | Rich result cards (similarity, sensor, land-cover, location, date, distance) | ◐ | class + relevance only |
| 31 | Interactive map | ❌ | — |
| 32 | Endpoints: `/health`, `/predict`, `/batch-retrieve`, `/metrics`, `/model-info` | ◐ | `/api/status`, `/api/retrieve`, `/api/retrieve_upload`, `/api/eval` exist; names differ |
| 33 | DB tables: datasets, embeddings, retrieval_history, evaluation_results, model_versions | ◐ | images/galleries/retrieval_logs only |
| 34 | Caching (model, embeddings, FAISS, queries) | ✅ | model+embeddings+FAISS cached; no query cache |
| 35 | Tests: dataset, preproc, modality detection, encoder shape, losses, FAISS, metrics, API, DB, E2E | ◐ | 2 files (unit metrics + E2E smoke) |
| 36 | Env-var support (paths, model, DB, API, deployment) | ❌ | YAML + CLI only, no env |

### Structure / docs / packaging
| # | Requirement | Status | Where |
|---|---|---|---|
| 37 | Target layout (`configs/`, `data/`, `models/`, `src/`, `api/`, `web/`, `database/`, `outputs/`, `docs/`, `tests/`) | ◐ | close; missing Dockerfile, LICENSE, `models/pretrained` |
| 38 | Full README (23 sections) | ◐ | strong README; no real-dataset download docs, no UI screenshots, no Docker, no API reference, no ablation/results tables, no limitations/future/citation |
| 39 | Diagrams (architecture, flow, ER, deployment, wireframes) | ❌ | none |

## 4. Non-functional findings

1. **Version control.** The Desktop folder is **not itself a git repo**. `git rev-parse
   --show-toplevel` resolves to `C:/Users/TRAGI` (home dir), whose `origin` is
   `ishxn-s11/kaggle-quest.git` and whose HEAD (`8822acd "first commit"`) tracks
   **zero files** — everything under the project is untracked. The master brief
   names the repo `ishxn-s11/Cross-Modal-Satellite-Image-Retrieval`. This must be
   resolved (init a repo in the project folder, or fix the remote) **before any
   commit/push**.
2. **Real data is big.** SEN12MS is ~100+ GB. Per the brief, we must NOT auto-download.
   Plan: build the loaders, preprocessing, config wiring and docs; make them
   data-optional (graceful fallback to synthetic when data absent).
3. **UI divergence.** The brief asks for Streamlit; the repo ships a FastAPI-served
   SPA. Decision needed: add a Streamlit app reusing the same `src/` + persistence
   (recommended), or port the SPA.
4. **API endpoint names** differ from the brief (`/api/retrieve` vs `/retrieve`).
   Keep the existing names for backward compatibility and add the brief's names as
   aliases.
5. **Minor / hygiene:**
   - `api/app.py` uses deprecated `@app.on_event("startup")` (FastAPI `lifespan` preferred).
   - `eurosat.simulate_bands_from_optical` takes an unused `label_map` arg.
   - Notebook cells currently report 0 stored outputs (need re-execution for screenshots).
   - No `pytest` runner / config / CI; tests are ad-hoc `__main__` scripts.
   - No `.env`/env-var plumbing in `src/utils/config.py`.

## 5. Proposed implementation order (adapted to this repo)

The 12 phases in the brief map cleanly onto the existing code. Preserve everything
in §2; extend in place rather than rewrite.

1. **Audit** ✅ (this document).
2. **Real dataset layer** — `DatasetInterface` + `SEN12Dataset` (+ So2Sat /
   BigEarthNet-MM stubs), config `dataset.name`, metadata dataclass, dataset docs.
3. **Preprocessing / augmentation** — modality-aware pipeline modules; SAR/optical/MS
   processors; augmentation; keep current defaults identical.
4. **Baseline verify** — ResNet → embedding → FAISS → top-5/10 already works.
5. **Contrastive upgrade** — hard-negative mining, geographic alignment loss,
   temporal sampling (all configurable, all off by default).
6. **Encoders** — `BaseEncoder` adapter, ViT, modality-specific heads; foundation
   models gated behind optional flags.
7. **Two-stage retrieval** — candidate_k → re-ranker → final_k; configurable; toggleable.
8. **Evaluation** — mAP@K, NDCG@K, Recall@K, latency breakdown (P50/P95/throughput),
   scalability + baseline + ablation harnesses.
9. **XAI + visualisation** — Grad-CAM/attention, t-SNE, before/after embedding plots.
10. **UI** — Streamlit app (or extended SPA) with Map/Analytics/Embeddings/
    Explainability/Gallery/History + result cards + interactive map.
11. **API + deployment** — endpoint aliases, `/health`, `/predict`, `/batch-retrieve`,
    `/metrics`, `/model-info`, Dockerfile, env-var config.
12. **Docs + tests + packaging** — full README, diagrams, expanded pytest suite, LICENSE.

Each phase: run existing tests → run new tests → small E2E → verify outputs →
update docs. Every advanced feature is independently switchable via config so the
synthetic default run stays exactly reproducible.
