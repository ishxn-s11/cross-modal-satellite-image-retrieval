"""FastAPI demo: serve cross-modal retrieval over a persisted database.

Run:
    pip install -r requirements-web.txt
    uvicorn api.app:app --host 127.0.0.1 --port 8000
    # then open http://127.0.0.1:8000

The app is backed by the persistent stores in ``database/``, ``embeddings/``
and ``faiss/``: on a warm start the per-modality FAISS galleries are reloaded
from disk instead of being re-embedded, and every retrieval is recorded in the
SQLite metadata store (visible in the "History" tab of the web UI).
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.database import EmbeddingStore, IndexStore, MetadataStore, compute_cache_key
from src.pipeline import build_loaders, build_model_from_cfg, load_best_model, prepare_dataset
from src.retrieval.engine import RetrievalEngine
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device
from src.utils.visualize import render_patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "configs", "default.yaml")
WEB_DIR = os.path.join(ROOT, "web")

app = FastAPI(title="Cross-Modal Satellite Image Retrieval")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_state: Dict = {}


class RetrieveRequest(BaseModel):
    query_id: int
    query_modality: str = "optical"
    gallery_modality: str = "sar"
    k: int = 5


class RetrieveResponse(BaseModel):
    query: dict
    retrieved: List[dict]
    avg_time_ms: float


# ---------------------------------------------------------------------------
# Startup: load dataset + model, build/reload galleries from the persistence
# directories, and populate the SQLite metadata store.
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    cfg = load_config(CONFIG_PATH)
    device = resolve_device(cfg["training"]["device"])
    logger = Logger(os.path.join(cfg["outputs"]["dir"], "logs", "api.log"))

    # Persistent stores.
    persistence = cfg.get("persistence", {})
    embedding_store = EmbeddingStore(persistence.get("embeddings_dir", "embeddings"))
    index_store = IndexStore(persistence.get("faiss_dir", "faiss"))
    metadata_store = MetadataStore(persistence.get("database_path", "database/metadata.db"))

    ckpt = os.path.join(cfg["outputs"]["model_dir"], "best_model", "model.pt")
    config_hash = compute_cache_key(cfg, ckpt)

    patches, labels, class_names, _s, transforms = prepare_dataset(cfg, Logger(path=None))
    model = load_best_model(cfg, len(class_names), device).to(device)
    _, _, full_ds = build_loaders(cfg, patches, labels, transforms)
    engine = RetrievalEngine(
        model, full_ds, device,
        embedding_store=embedding_store,
        index_store=index_store,
        config_hash=config_hash,
    )

    # Warm the full embedding cache (hits disk on warm starts).
    emb_cache = {}
    for m in cfg["modalities"]:
        emb_cache[m] = engine.cache_full_embeddings(m)
        logger.info(f"[db] embeddings '{m}': {emb_cache[m][0].shape} cached")

    # Demo split: gallery = second half of the dataset, queries = first half.
    n = len(full_ds)
    gallery_ids = np.arange(n // 2, n)
    query_ids = np.arange(0, n // 2)
    galleries: Dict[str, object] = {}
    for m in cfg["modalities"]:
        if index_store.exists(m, config_hash, gallery_ids):
            logger.info(f"[db] loading cached gallery '{m}'")
        galleries[m] = engine.build_gallery(gallery_ids, m)
        metadata_store.save_gallery(
            name=f"{m}_gallery",
            modality=m,
            num_vectors=galleries[m].size,
            embedding_dim=galleries[m].embeddings.shape[1],
            index_path=str(index_store._paths(m, config_hash, gallery_ids)[0]),
            config_hash=config_hash,
        )
        logger.info(f"[db] gallery '{m}': {galleries[m].size} vectors ready")

    # Record image metadata (train/val/test splits) for the browse/dashboard.
    from src.evaluation.evaluate import stratified_split
    tr_ids, val_ids, te_ids = stratified_split(labels, 0.7, 0.15, int(cfg["dataset"]["seed"]))
    split_of = np.full(int(labels.shape[0]), "test", dtype=object)
    split_of[tr_ids] = "train"
    split_of[val_ids] = "val"
    metadata_store.save_images(labels, class_names, split_of.tolist())

    _state.update(
        cfg=cfg,
        device=device,
        patches=patches,
        labels=labels,
        class_names=class_names,
        full_ds=full_ds,
        engine=engine,
        galleries=galleries,
        gallery_ids=gallery_ids,
        query_ids=query_ids,
        metadata_store=metadata_store,
        embedding_store=embedding_store,
        index_store=index_store,
        config_hash=config_hash,
    )
    logger.info(f"[startup] ready (galleries={ {m: galleries[m].size for m in galleries} })")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def _index() -> str:
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Status / database introspection
# ---------------------------------------------------------------------------

@app.get("/api/status")
def _status() -> dict:
    emb = {m: bool(_state["embedding_store"].exists(m, _state["config_hash"])) for m in _state["cfg"]["modalities"]}
    idx = {m: _state["index_store"].exists(m, _state["config_hash"], _state["gallery_ids"]) for m in _state["cfg"]["modalities"]}
    return {
        "modalities": list(_state["cfg"]["modalities"]),
        "gallery_size": int(_state["gallery_ids"].shape[0]),
        "config_hash": _state["config_hash"],
        "cache": {"embeddings": emb, "faiss": idx},
        "database": _state["metadata_store"].stats(),
    }


@app.get("/api/database")
def _database() -> dict:
    """Manifest of everything persisted for this config hash."""
    mstore = _state["metadata_store"]
    return {
        "config_hash": _state["config_hash"],
        "images": mstore.image_count(),
        "class_counts": mstore.class_counts(),
        "galleries": mstore.list_galleries(),
        "recent_retrievals": mstore.recent_retrievals(5),
        "embedding_files": sorted(str(p) for p in _state["embedding_store"].root.glob("*.npz")),
        "index_files": sorted(str(p) for p in _state["index_store"].root.glob("*.index")),
    }


@app.get("/api/gallery/info")
def _gallery_info() -> dict:
    return {
        "gallery_size": int(_state["gallery_ids"].shape[0]),
        "query_size": int(_state["query_ids"].shape[0]),
        "per_modality": {
            m: {
                "n": _state["galleries"][m].size,
                "dim": int(_state["galleries"][m].embeddings.shape[1]),
            }
            for m in _state["cfg"]["modalities"]
        },
        "metadata": _state["metadata_store"].list_galleries(),
    }


@app.get("/api/dataset/stats")
def _dataset_stats() -> dict:
    patches, labels, class_names = _state["patches"], _state["labels"], _state["class_names"]
    counts = {c: int((labels == i).sum()) for i, c in enumerate(class_names)}
    mods = {
        m: {
            "bands": int(patches[m].shape[1]),
            "shape": list(patches[m].shape),
            "min": float(patches[m].min()),
            "max": float(patches[m].max()),
        }
        for m in _state["cfg"]["modalities"]
    }
    return {
        "source": _state["cfg"]["dataset"]["source"],
        "n": int(labels.shape[0]),
        "image_size": int(_state["cfg"]["dataset"]["image_size"]),
        "classes": class_names,
        "class_counts": counts,
        "modalities": mods,
        "embedding_norm_max": float(max(np.abs(p).max() for p in patches.values())),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

@app.post("/api/retrieve", response_model=RetrieveResponse)
def _retrieve(req: RetrieveRequest) -> RetrieveResponse:
    engine = _state["engine"]
    gallery = _state["galleries"][req.gallery_modality]
    result = engine.retrieve(gallery, [req.query_id], req.query_modality, k=req.k)
    patches, labels, class_names = _state["patches"], _state["labels"], _state["class_names"]
    qid = req.query_id
    query_img = _png_b64(render_patch(patches, int(qid), req.query_modality))
    retrieved = []
    for j in range(req.k):
        rid = int(result.gallery_ids[0, j])
        retrieved.append(
            {
                "rank": j + 1,
                "id": rid,
                "class": class_names[int(labels[rid])],
                "relevant": bool(result.relevant_mask()[0, j]),
                "score": round(float(result.scores[0, j]), 4),
                "image": _png_b64(render_patch(patches, rid, req.gallery_modality)),
            }
        )
    _state["metadata_store"].log_retrieval(
        int(qid), req.query_modality, req.gallery_modality, req.k,
        float(result.search_times_ms[0]), len(retrieved),
    )
    return RetrieveResponse(
        query={
            "id": qid,
            "modality": req.query_modality,
            "class": class_names[int(labels[qid])],
            "image": query_img,
        },
        retrieved=retrieved,
        avg_time_ms=float(result.search_times_ms[0]),
    )


@app.post("/api/retrieve_upload")
async def _retrieve_upload(
    file: UploadFile = File(...),
    query_modality: str = Form("optical"),
    gallery_modality: str = Form("optical"),
    k: int = Form(5),
) -> dict:
    """Retrieve against a gallery using an uploaded single/3-channel image."""
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("L" if query_modality == "sar" else "RGB")
    size = int(_state["cfg"]["dataset"]["image_size"])
    img = img.resize((size, size))
    if query_modality == "sar":
        arr = np.asarray(img, dtype=np.float32)[None] / 255.0
    else:
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    arr = np.ascontiguousarray(arr)
    transform = _state["engine"].dataset.transforms.get(query_modality)
    if transform is not None:
        arr = transform(arr)
    import torch

    x = torch.from_numpy(arr)[None]
    emb = _state["engine"].model.embed(x.to(_state["device"]), query_modality)
    gallery = _state["galleries"][gallery_modality]
    t0 = time.perf_counter()
    scores, ids = gallery.index.search(emb.detach().cpu().numpy(), k)
    ms = (time.perf_counter() - t0) * 1000.0
    class_names, labels, patches = _state["class_names"], _state["labels"], _state["patches"]

    retrieved = []
    for j in range(ids.shape[1]):
        rid = int(gallery.indices[ids[0, j]])
        retrieved.append(
            {
                "rank": j + 1,
                "id": rid,
                "class": class_names[int(labels[rid])],
                "score": round(float(scores[0, j]), 4),
                "image": _png_b64(render_patch(patches, rid, gallery_modality)),
            }
        )
    _state["metadata_store"].log_retrieval(
        -1, query_modality, gallery_modality, int(k), ms, len(retrieved),
    )
    return {
        "query_modality": query_modality,
        "gallery_modality": gallery_modality,
        "avg_time_ms": round(ms, 3),
        "retrieved": retrieved,
    }


@app.get("/api/random_query")
def _random_query(limit: int = 12) -> dict:
    """Pick a few valid query ids for the UI's 'surprise me' button."""
    rng = np.random.RandomState()
    qids = _state["query_ids"]
    picked = rng.choice(qids, size=min(int(limit), len(qids)), replace=False)
    return {"query_ids": [int(i) for i in sorted(picked)]}


# ---------------------------------------------------------------------------
# Browsing / history / metrics
# ---------------------------------------------------------------------------

@app.get("/api/browse")
def _browse(modality: str = "optical", start: int = 0, limit: int = 12) -> dict:
    patches, labels, class_names = _state["patches"], _state["labels"], _state["class_names"]
    n = int(labels.shape[0])
    ids = list(range(start, min(start + limit, n)))
    return {
        "modality": modality,
        "start": start,
        "total": n,
        "items": [
            {
                "id": i,
                "class": class_names[int(labels[i])],
                "image": _png_b64(render_patch(patches, i, modality)),
            }
            for i in ids
        ],
    }


@app.get("/api/history")
def _history(limit: int = 20) -> dict:
    rows = _state["metadata_store"].recent_retrievals(limit)
    for r in rows:
        r["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))
    return {"items": rows, "stats": _state["metadata_store"].retrieval_stats()}


@app.get("/api/eval")
def _eval() -> dict:
    import json as _json

    base = _state["cfg"]["outputs"]["dir"]
    p = os.path.join(base, "metrics", "retrieval_summary.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return _json.load(fh)
    return {"summary": None, "rows": []}


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

@app.delete("/api/cache/clear")
def _clear_cache() -> dict:
    """Clear persisted embeddings + FAISS galleries, then rebuild in memory."""
    engine = _state["engine"]
    gallery_ids = _state["gallery_ids"]
    removed_emb = _state["embedding_store"].clear()
    removed_idx = _state["index_store"].clear()
    engine.clear_cache()
    # Rebuild galleries fresh (no disk hit now).
    for m in _state["cfg"]["modalities"]:
        engine.cache_full_embeddings(m, force=True)
        _state["galleries"][m] = engine.build_gallery(gallery_ids, m)
    return {
        "removed_files": removed_emb + removed_idx,
        "rebuilt": {m: _state["galleries"][m].size for m in _state["cfg"]["modalities"]},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _png_b64(img_rgb: np.ndarray) -> str:
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# Static assets (css/...), optional.
if os.path.isdir(os.path.join(WEB_DIR, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(WEB_DIR, "assets")), name="assets")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
