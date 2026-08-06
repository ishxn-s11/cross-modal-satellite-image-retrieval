# Acceptance-criteria checklist

Status of the brief's final acceptance criteria against the current codebase
(as of this commit). ✅ = implemented & tested, ◐ = implemented but requires
user-supplied data/weights to fully exercise, — = intentionally not bundled.

## Data
| criterion | status | evidence |
|---|---|---|
| Real Sentinel-1/Sentinel-2 data | ◐ | `SEN12Dataset` loader (SEN12MS) + So2Sat + BigEarthNet-MM; validated on a fixture; requires a ~110 GB user download to run (docs/datasets.md) |
| At least one real paired/associated multimodal dataset | ◐ | SEN12MS (S1/S2 aligned), So2Sat LCZ42, BigEarthNet-MM loaders |
| Correct modality-specific preprocessing | ✅ | `preprocessing.py` (SAR log/clip/speckle, optical cloud-filter, MS band selection) + tests |

## Retrieval
| criterion | status | evidence |
|---|---|---|
| Optical→Optical, SAR→SAR | ✅ | default config evaluates all same-modal pairs |
| Optical→SAR, SAR→Optical | ✅ | all 6 cross-modal pairs evaluated |
| Multispectral support where dataset/model permits | ✅ | 8-band MS on synthetic/EuroSAT; S2 subset on SEN12 |
| Top-5 / Top-10 | ✅ | `retrieval.top_k: [5, 10]` |

## Model
| criterion | status | evidence |
|---|---|---|
| Baseline model | ✅ | ResNet-18/34/50 (`resnet18` default) |
| Contrastive learning | ✅ | InfoNCE + SupCon + CE (`multimodal_losses`) |
| Shared embedding space | ✅ | L2-normalised, `embedding_dim` configurable |
| Hard negatives | ✅ | `training.hard_negatives` + tests |
| Optional foundation model | ◐ | SatMAE/Prithvi adapters; require user checkpoint (not bundled) |

## Search
| criterion | status | evidence |
|---|---|---|
| FAISS | ✅ | flat/ivf/hnsw/ivfpq × cosine/ip/euclidean + GPU opt-in |
| Persistent index | ✅ | `IndexStore` + config-hash invalidation |
| Fast retrieval | ✅ | ~0.1 ms/query measured (reference run) |
| Optional re-ranking | ✅ | identity/geo/MLP re-rankers |

## Evaluation
| criterion | status | evidence |
|---|---|---|
| F1@5 / F1@10 | ✅ | every pair + same/cross summary |
| Precision@K / Recall@K | ✅ | per pair |
| Retrieval latency | ✅ | `benchmark_latency.py` (stage breakdown, P50/P95) |
| Same-modal evaluation | ✅ | summary `same_modal_avg` |
| Cross-modal evaluation | ✅ | summary `cross_modal_avg` |

## Advanced
| criterion | status | evidence |
|---|---|---|
| Geographic relevance | ✅ | geo-alignment loss + geo re-ranker + map UI |
| Temporal robustness | ✅ | same-location-across-dates positives (geo loss) |
| XAI | ✅ | Grad-CAM (CNN) + ViT attention |
| Embedding visualisation | ✅ | PCA / t-SNE / UMAP before-vs-after |
| Scalability benchmarking | ✅ | flat/ivf/hnsw/ivfpq × 10K/100K/1M |

## UI
| criterion | status | evidence |
|---|---|---|
| Upload query | ✅ | FastAPI upload + Streamlit upload |
| Select modality | ✅ | both UIs |
| Select gallery modality | ✅ | both UIs |
| Top-K results | ✅ | both UIs |
| Similarity scores | ✅ | result cards |
| Metadata | ✅ | result cards + `result_records` |
| Map | ✅ | Streamlit Map tab (plotly) |
| Retrieval time | ✅ | Results tab + latency benchmark |
| Analytics | ✅ | Streamlit Analytics tab + `/metrics` |

## Engineering
| criterion | status | evidence |
|---|---|---|
| Tests | ✅ | 87 pytest tests |
| Logging | ✅ | `src/utils/io.Logger` |
| Configuration | ✅ | YAML + CLI `--set` + `RETRIEVAL_*` env |
| Docker | ✅ | Dockerfile + .dockerignore |
| API | ✅ | clean endpoints + validation |
| Documentation | ✅ | README + docs/ (datasets, preprocessing, models, learning, retrieval, benchmarks, xai, api, diagrams, acceptance) |

## Known gaps / notes
* Real-dataset runs (SEN12MS etc.) need the user to download data; loaders are
  fixture-tested, not validated against a live download in CI.
* Foundation models (SatMAE/Prithvi) require a user-supplied checkpoint.
* UI screenshots are not bundled (run the Streamlit / FastAPI app to capture).
* The reference README numbers are measured on the default synthetic config.
