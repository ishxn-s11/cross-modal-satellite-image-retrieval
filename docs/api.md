# API & deployment

## FastAPI endpoints

```bash
pip install -r requirements-web.txt
uvicorn api.app:app --reload            # http://127.0.0.1:8000
```

| endpoint | method | description |
|---|---|---|
| `/health` | GET | liveness + dataset/modalities/gallery size |
| `/predict` | POST | predict the land-cover class of a dataset image (with confidence) |
| `/retrieve` | POST | top-k retrieval (query id → ranked gallery) |
| `/batch-retrieve` | POST | retrieval for many query ids in one request |
| `/gallery` | GET | gallery + dataset metadata |
| `/metrics` | GET | evaluation results (from `outputs/metrics/`) |
| `/history` | GET | recent retrievals from SQLite |
| `/model-info` | GET | architecture, parameter counts, index config, config hash |

The original `/api/*` endpoints (`/api/retrieve`, `/api/retrieve_upload`,
`/api/browse`, `/api/eval`, `/api/status`, `/api/history`, `/api/database`,
`/api/cache/clear`) are **kept for compatibility** — the new names are aliases.

Request bodies are pydantic-validated; invalid modalities return `400` and
out-of-range image ids return `404`.

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query_id": 12, "query_modality": "optical", "gallery_modality": "sar", "k": 5}'
```

## Configuration via environment variables

`src/utils/config.py` maps `RETRIEVAL_*` env vars onto config paths (CLI
`--set` still wins):

| env var | config path |
|---|---|
| `RETRIEVAL_DATA_ROOT` | `dataset.root` |
| `RETRIEVAL_DATASET` | `dataset.name` |
| `RETRIEVAL_IMAGE_SIZE` | `dataset.image_size` |
| `RETRIEVAL_MODEL_DIR` / `RETRIEVAL_OUTPUTS_DIR` | `outputs.model_dir` / `outputs.dir` |
| `RETRIEVAL_EMBEDDINGS_DIR` / `RETRIEVAL_FAISS_DIR` | `persistence.*` |
| `RETRIEVAL_DATABASE_PATH` | `persistence.database_path` |
| `RETRIEVAL_DEVICE` | `training.device` |
| `RETRIEVAL_BACKBONE` / `RETRIEVAL_EMBEDDING_DIM` | `model.*` |
| `RETRIEVAL_CONFIG` | path to the YAML config (used by API / Streamlit) |

No secrets are read from config or env — API keys/credentials are never part of
this project.

## Docker

```bash
docker build -t cross-modal-retrieval .
docker run -p 8000:8000 \
  -v "$PWD/models:/app/models" -v "$PWD/data:/app/data" \
  -e RETRIEVAL_DATASET=synthetic \
  cross-modal-retrieval
```

Runtime artifacts (`models`, `data`, `outputs`, `embeddings`, `faiss`,
`database`) are declared as volumes so a trained model + cached galleries
persist outside the image.

## Tests

`tests/test_api.py` starts the app with `TestClient` on a small self-contained
config and exercises `/health`, `/predict`, `/retrieve`, `/batch-retrieve`,
`/model-info`, validation (400/404), `/metrics`, `/history`, `/gallery`.
