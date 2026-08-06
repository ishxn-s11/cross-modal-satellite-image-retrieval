# Retrieval

The retrieval engine (`src/retrieval/`) turns a query image's embedding into a
ranked list of gallery images, optionally through a two-stage pipeline:

```
query image
   -> embed (shared space)
      -> FAISS search (candidate_k)        [stage 1: fast candidate recall]
         -> re-ranker (candidate_k -> k)   [stage 2: re-score, optional]
            -> top-k results (+ metadata)
```

## Two-stage retrieval

```yaml
retrieval:
  candidate_k: 100      # FAISS returns this many candidates (null = single stage)
  final_k: 10           # the re-ranker keeps this many (top_k / k per query)
  rerank:
    enabled: true       # identity | geo | mlp
    method: geo
    geo_weight: 0.3
    scale_km: 50.0
```

* `candidate_k` controls the FAISS candidate budget. With `candidate_k: null`
  (default) retrieval is single-stage: FAISS returns the final top-k directly.
* With a re-ranker enabled and `candidate_k > k`, the engine re-scores the
  candidates and returns the top `final_k`.
* Re-ranking is optional, so FAISS-only vs FAISS + re-ranking can be compared
  directly (see the evaluation harness and `scripts/benchmark.py`).

### Re-rankers (`src/retrieval/rerank.py`)

| method | behaviour |
|---|---|
| `identity` | keeps the FAISS order (no-op) |
| `geo` | adds `geo_weight * exp(-geo_distance_km / scale_km)` to the cosine score for candidates with coordinates (nearby scenes rank higher) |
| `mlp` | a small trainable MLP over `[q ; c ; q*c ; |q-c|]` trained to predict same-class relevance (`MLPReranker.fit`); save/restore via `state_dict`/`from_state` |

## FAISS index types and metrics

```yaml
retrieval:
  index:
    type: flat      # flat | ivf | hnsw | ivfpq
    metric: cosine  # cosine | ip | euclidean
    nlist: 32       # ivf / ivfpq
    m: 8            # hnsw (M) / ivfpq (sub-quantizers)
    ef_search: 32   # hnsw
    use_gpu: false  # optional GPU FAISS (falls back to CPU if unavailable)
```

| type | index | notes |
|---|---|---|
| `flat` | `IndexFlatIP` / `IndexFlatL2` | exact, baseline |
| `ivf` | `IndexIVFFlat` | approximate, `nlist` clusters, `nprobe=8` |
| `hnsw` | `IndexHNSWFlat` | graph-based, `m`, `ef_search` |
| `ivfpq` | `IndexIVFPQ` | compressed (low memory), `m` sub-quantizers |

* `metric: cosine` L2-normalises embeddings so inner product == cosine
  (recommended). `ip` uses raw inner product; `euclidean` uses L2 distance.
* `use_gpu: true` moves the index to the GPU when FAISS GPU support is
  installed; it silently falls back to CPU otherwise.
* **Persistence** — the gallery store (`IndexStore`) saves the index type,
  metric and settings in the gallery meta file, so warm reloads reconstruct the
  exact index. `build / save / load` are fully supported; `rebuild()` resets
  and re-adds, and `remove()` works for `ivf` / `hnsw` / `ivfpq` (use `rebuild`
  for flat).

## Result objects

`src/retrieval/engine.py::result_records` produces rich per-result records that
only include fields the dataset actually provides:

```python
{
  "rank": 1,
  "image_id": 42,
  "similarity_score": 0.94,
  "modality": "sar",
  "sensor": "Sentinel-1",
  "land_cover": "Urban and Built-up Lands",
  "latitude": 29.1, "longitude": 76.4,
  "acquisition_date": "2018-05-14",
  "geographic_distance": 12.4,   # km from the query (when both have coords)
  "image_path": ".../scene.tif",
}
```

Null fields are omitted, so a dataset without coordinates simply has no
`latitude`/`longitude`/`geographic_distance` entries.

## Query-time latency

`RetrievalResult` carries `search_times_ms` (FAISS) and `rerank_times_ms`
(re-ranker) per query; the latency benchmark (`docs/benchmarks.md`) breaks
preprocessing / embedding / search / re-rank / total down separately.
