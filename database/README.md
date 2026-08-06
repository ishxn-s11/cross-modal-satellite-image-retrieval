# database/

SQLite **metadata store** for the retrieval system.

This directory holds `metadata.db` (generated at runtime, git-ignored), a small
relational database recording the facts surrounding retrieval — the embedding
vectors and FAISS indices themselves live in `../embeddings/` and `../faiss/`.

## Tables

| Table | Purpose |
|---|---|
| `images` | every dataset patch: id, class label/name, and split (train/val/test). |
| `galleries` | per-modality galleries that have been built: vector count, embedding dim, on-disk index path, config hash. |
| `retrieval_logs` | a bounded history of retrieval queries (query id, query→gallery modality, K, avg time) shown in the web UI's **History** tab. |

## Usage

```python
from src.database import MetadataStore
store = MetadataStore("database/metadata.db")
store.save_images(labels, class_names, splits)
store.save_gallery(name="optical_gallery", modality="optical", num_vectors=3000, ...)
store.log_retrieval(query_id=5, query_mod="optical", gallery_mod="sar", k=5, avg_time_ms=0.05)
store.recent_retrievals(limit=20)
```

Content is written by `api/app.py` on startup and, for illustration, by
`notebooks/05_database_and_persistence.ipynb`.