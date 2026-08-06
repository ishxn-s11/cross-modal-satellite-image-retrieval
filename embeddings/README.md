# embeddings/

On-disk cache of the **full-dataset, per-modality** embeddings.

Embeddings are computed once per `(dataset, model)` configuration and reused
thereafter — this is the main warm-start win. Each modality stores one file
pair:

```
embeddings/
  optical_<config_hash>.npz            # {embeddings: (N, D) float32 L2-normalised,
                                       #  labels:    (N,) int64}  for ALL dataset ids
  optical_<config_hash>_meta.json      # modality, dim, n, config_hash, created_at
```

## Cache invalidation

The `<config_hash>` is derived from the dataset + model configuration **and the
trained checkpoint's mtime/size** (see `src/database/config_hash.py`), so a
change to `configs/default.yaml` or a retrain invalidates the cache
automatically — a subsequent run recomputes and rewrites the embeddings.

## Usage

```python
from src.database import EmbeddingStore
store = EmbeddingStore("embeddings")
store.save("optical", hash_, embeddings, labels)
emb, labels = store.load("optical", hash_)   # -> None on cache miss
store.exists("optical", hash_)
```

`RetrievalEngine.cache_full_embeddings()` in `src/retrieval/engine.py` uses this
store internally; the pipeline and web app populate it on first run.