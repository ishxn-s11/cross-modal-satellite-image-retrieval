# faiss/

Persisted **FAISS galleries** (the searchable per-modality databases).

On a warm start the web app / evaluation reload a gallery directly from disk
instead of re-embedding every gallery image. Each gallery is stored as:

```
faiss/
  optical_<config_hash>_<indices_hash>.index       # binary FAISS index
  optical_<config_hash>_<indices_hash>_meta.json   # modality, dim, n, index_type,
                                                   # indices (dataset ids) + labels
```

## Keying

* `<config_hash>`  — identifies the embedding space (dataset + model config).
* `<indices_hash>` — identifies the exact set of gallery dataset-ids, so two
  different gallery splits for the same modality never collide.

## Usage

```python
from src.database import IndexStore
store = IndexStore("faiss")
store.save(gallery, modality="optical", config_hash=hash_)     # persist
gallery = store.load("optical", config_hash=hash_, indices)    # reload or None
store.exists("optical", config_hash=hash_, indices)
```

`RetrievalEngine.build_gallery()` in `src/retrieval/engine.py` uses this store
automatically. Flat and IVF indices are both supported (`FaissCosineIndex.save()`
in `src/retrieval/index.py`).