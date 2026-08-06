"""API tests: health / predict / retrieve / batch-retrieve / model-info.

Uses a small, self-contained config (fresh random model in temp dirs) so it
does not depend on any on-disk checkpoint or dataset download.
"""

import os
import sys
import tempfile

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Small config must be set before importing api.app (CONFIG_PATH is read at
# import time).
_tmp = tempfile.mkdtemp()
_CONFIG = {
    "dataset": {"name": "synthetic", "num_patches": 60, "image_size": 32, "seed": 0,
                "allow_fallback": True},
    "modalities": ["optical", "sar"],
    "model": {"backbone": "resnet18", "pretrained": False, "embedding_dim": 16,
              "freeze_backbone": True},
    "training": {"epochs": 0, "batch_size": 32},
    "retrieval": {"top_k": [5], "gallery_fraction": 0.6, "n_query": 20},
    "outputs": {"dir": os.path.join(_tmp, "outputs"),
                "model_dir": os.path.join(_tmp, "models")},
    "persistence": {"embeddings_dir": os.path.join(_tmp, "embeddings"),
                    "faiss_dir": os.path.join(_tmp, "faiss"),
                    "database_path": os.path.join(_tmp, "db.sqlite")},
}
_CONFIG_PATH = os.path.join(_tmp, "api_test.yaml")
with open(_CONFIG_PATH, "w", encoding="utf-8") as fh:
    yaml.safe_dump(_CONFIG, fh)
os.environ["RETRIEVAL_CONFIG"] = _CONFIG_PATH

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402


def test_health_and_endpoints():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

        r = client.get("/model-info")
        assert r.status_code == 200
        info = r.json()
        assert info["backbone"] == "resnet18"
        assert info["embedding_dim"] == 16

        r = client.post("/predict", json={"query_id": 0, "query_modality": "optical"})
        assert r.status_code == 200
        pred = r.json()
        assert pred["image_id"] == 0
        assert "predicted_class" in pred

        r = client.post("/retrieve", json={"query_id": 0, "query_modality": "optical",
                                           "gallery_modality": "sar", "k": 3})
        assert r.status_code == 200
        assert len(r.json()["retrieved"]) == 3

        r = client.post("/batch-retrieve",
                        json={"query_ids": [0, 1, 2], "query_modality": "sar",
                              "gallery_modality": "optical", "k": 2})
        assert r.status_code == 200
        assert len(r.json()) == 3

        # validation: bad modality -> 400, out-of-range id -> 404
        assert client.post("/retrieve", json={"query_id": 0, "query_modality": "nope",
                                              "gallery_modality": "sar"}).status_code == 400
        assert client.post("/predict", json={"query_id": 9999,
                                             "query_modality": "optical"}).status_code == 404

        assert client.get("/metrics").status_code == 200
        assert client.get("/history").status_code == 200
        assert client.get("/gallery").status_code == 200


if __name__ == "__main__":
    test_health_and_endpoints()
    print("test_api.py: all tests passed")
