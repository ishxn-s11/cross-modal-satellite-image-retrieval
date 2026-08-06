"""Integration tests for two-stage retrieval + rich result records."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MultiModalDataset
from src.data.interface import build_dataset
from src.data.preprocessing import build_transforms, compute_normalization_stats
from src.models.encoder import ModalityAdaptiveEncoder
from src.pipeline import build_model_from_cfg, modality_channels_from_patches
from src.retrieval.engine import RetrievalEngine, result_records
from src.retrieval.rerank import build_reranker
from src.utils.config import DEFAULT_CONFIG, deep_merge
from src.utils.io import Logger


def _mini(model_backbone="resnet18"):
    cfg = deep_merge({}, DEFAULT_CONFIG)
    cfg["dataset"].update(num_patches=80, image_size=32, seed=0)
    cfg["modalities"] = ["optical", "sar"]
    cfg["model"]["embedding_dim"] = 16
    cfg["model"]["freeze_backbone"] = True
    ds = build_dataset(cfg, Logger(path=None))
    patches, labels, class_names = ds.to_patches()
    metadata = ds.metadata if ds.has_metadata() else None
    stats = compute_normalization_stats(patches)
    transforms = build_transforms(stats)
    full_ds = MultiModalDataset(patches, labels, cfg["modalities"], transforms, metadata=metadata)
    model = build_model_from_cfg(
        cfg,
        len(class_names),
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).eval()
    return cfg, model, full_ds, class_names


def test_twostage_geo_rerank_returns_k():
    cfg, model, ds, _cn = _mini()
    engine = RetrievalEngine(model, ds, torch.device("cpu"))
    n = len(ds)
    gallery = engine.build_gallery(np.arange(n // 2, n), "optical")
    reranker = build_reranker({"enabled": True, "method": "geo"})
    res = engine.retrieve(
        gallery, np.arange(0, 5), "optical", k=3, candidate_k=20, reranker=reranker
    )
    assert res.k == 3
    assert res.gallery_ids.shape == (5, 3)
    assert res.rerank_times_ms.shape == (5,)


def test_twostage_no_reranker_matches_singlestage_topk():
    cfg, model, ds, _cn = _mini()
    engine = RetrievalEngine(model, ds, torch.device("cpu"))
    n = len(ds)
    gallery = engine.build_gallery(np.arange(n // 2, n), "optical")
    # candidate_k == k with no reranker == single-stage top-k
    r1 = engine.retrieve(gallery, np.arange(0, 5), "optical", k=3)
    r2 = engine.retrieve(gallery, np.arange(0, 5), "optical", k=3, candidate_k=3)
    assert np.array_equal(r1.gallery_ids, r2.gallery_ids)


def test_result_records_include_metadata():
    cfg, model, ds, class_names = _mini()
    engine = RetrievalEngine(model, ds, torch.device("cpu"))
    n = len(ds)
    gallery = engine.build_gallery(np.arange(n // 2, n), "optical")
    res = engine.retrieve(gallery, np.arange(0, 2), "optical", k=3)
    records = result_records(res, ds, class_names)
    assert len(records) == 2
    rec = records[0]["retrieved"][0]
    # synthetic metadata carries latitude/longitude/date -> they must appear
    assert rec["image_id"] >= 0
    assert "latitude" in rec
    assert "acquisition_date" in rec
    assert "geographic_distance" in rec
    assert "similarity_score" in rec
    assert "modality" in rec


def test_index_config_used_in_eval():
    # Building galleries through the eval harness with a non-flat index.
    from src.evaluation.evaluate import evaluate_retrieval_pairs

    cfg, model, ds, _cn = _mini()
    cfg["retrieval"]["index"]["type"] = "hnsw"
    cfg["retrieval"]["index"]["m"] = 8
    rows, _summary = evaluate_retrieval_pairs(
        model, ds, cfg["modalities"],
        [["optical", "optical"]], [["optical", "sar"]],
        cfg["retrieval"], torch.device("cpu"), seed=0,
    )
    assert len(rows) > 0
    assert all(0.0 <= r["f1@k"] <= 1.0 for r in rows)


if __name__ == "__main__":
    test_twostage_geo_rerank_returns_k()
    test_twostage_no_reranker_matches_singlestage_topk()
    test_result_records_include_metadata()
    test_index_config_used_in_eval()
    print("test_retrieval_extended.py: all tests passed")
