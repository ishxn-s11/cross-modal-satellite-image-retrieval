# Benchmarks & evaluation

Beyond the core P/R/F1@K report, the project ships reproducible benchmark
harnesses. **Every number is measured on this machine — none are fabricated.**

## Evaluation metrics

Each (query-modality → gallery-modality) pair is reported at every cutoff `K`
in `retrieval.top_k` (default [5, 10]):

| metric | definition |
|---|---|
| `precision@K` | relevant ∩ top-K / K |
| `recall@K` | relevant ∩ top-K / relevant in gallery |
| `f1@K` | harmonic mean of P@K and R@K |
| `map@K` | mean average precision @K |
| `ndcg@K` | normalised discounted cumulative gain @K |

`scripts/run_pipeline.py` writes `outputs/metrics/retrieval_metrics.csv` +
`retrieval_summary.json`; the summary reports same-modal / cross-modal /
cross-weighted F1@5 & F1@10 and average retrieval time per query.

## Latency benchmark

```bash
python scripts/benchmark_latency.py --k 10 --candidate-k 100 --rerank geo --n-queries 200 --train
```

Per-query latency is broken into **preprocessing / embedding / search /
re-rank / total**, each with mean / P50 / P95, plus throughput (queries/sec):

```json
"stages": {
  "preprocessing": {"mean": 0.425, "p50": 0.453, "p95": 0.627},
  "embedding":     {"mean": 28.2,  "p50": 27.9,  "p95": 37.9},
  "search":        {"mean": 0.108, "p50": 0.095, "p95": 0.187},
  "rerank":        {"mean": 11.1,  "p50": 10.0,  "p95": 17.0},
  "total":         {"mean": 39.8,  "p50": 39.2,  "p95": 52.5}
},
"throughput_qps": 25.1
```

(Example output, CPU torch, 64×64 synthetic patches.)

## Scalability benchmark

```bash
python scripts/benchmark_scalability.py --sizes 10000,100000 --max-vectors 1000000
# 1M tier is opt-in: --sizes 10000,100000,1000000
```

Compares `flat / ivf / hnsw / ivfpq` at each gallery size and reports build
time, search mean/P50/P95, **recall@K vs the exact flat result**, and an
estimated index memory footprint. Sizes above `--max-vectors` are skipped.

## Baseline comparison

```bash
python scripts/benchmark_baselines.py --epochs 2 --num-patches 2000
```

Runs six variants through the identical pipeline and writes a comparison table:

| variant | description |
|---|---|
| `baseline1_resnet_cosine` | raw ImageNet-pretrained ResNet-18 features, no contrastive training |
| `baseline2_resnet_infonce` | ResNet + InfoNCE only |
| `baseline3_resnet_infonce_supcon` | ResNet + InfoNCE + SupCon |
| `baseline4_vit_contrastive` | ViT-B/16 + full objective |
| `baseline5_foundation_contrastive` | SatMAE/Prithvi (skipped unless a checkpoint is configured) |
| `proposed_full` | the full default config |

Variants that cannot run are recorded as `skipped` (never fabricated).

## Ablation studies

```bash
python scripts/run_ablations.py --epochs 2 --num-patches 2000 --axes encoder,loss,hard_negatives
```

Sweeps one axis at a time — `encoder`, `embedding_dim`, `loss`,
`hard_negatives`, `reranking`, `geo_supervision` — writing
`outputs/benchmarks/ablation/ablation_<axis>.csv` + `ablations.json`.

> These harnesses train models end-to-end, so they are CPU-expensive; use
> `--epochs` / `--num-patches` to bound the runtime.
