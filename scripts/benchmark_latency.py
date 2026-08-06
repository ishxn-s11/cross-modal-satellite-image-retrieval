"""Benchmark per-query latency (preprocessing / embedding / search / rerank).

Usage:
    python scripts/benchmark_latency.py [--config configs/default.yaml]
        [--set key=value] [--k 10] [--candidate-k 100] [--rerank geo]
        [--n-queries 100]

Writes outputs/benchmarks/latency.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.evaluation.latency import run_latency_benchmark
from src.pipeline import (
    build_loaders,
    load_best_model,
    modality_channels_from_patches,
    prepare_dataset,
    run_experiment,
)
from src.retrieval.engine import RetrievalEngine
from src.retrieval.rerank import build_reranker
from src.utils.config import load_config
from src.utils.io import Logger, resolve_device, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--candidate-k", type=int, default=None)
    ap.add_argument("--rerank", default=None, help="identity|geo|mlp")
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--train", action="store_true",
                    help="train a fresh model for this config before benchmarking")
    ap.add_argument("--out", default="outputs/benchmarks/latency.json")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(path=None)
    device = resolve_device(cfg["training"]["device"])
    if args.train:
        run_experiment(cfg, logger)  # trains + evaluates, writes best_model
    patches, labels, class_names, stats, transforms, metadata = prepare_dataset(cfg, logger)
    model = load_best_model(
        cfg,
        len(class_names),
        device,
        modality_channels=modality_channels_from_patches(patches, cfg["modalities"]),
    ).to(device)
    _, _, full_ds = build_loaders(
        cfg, patches, labels, transforms, stats=stats, metadata=metadata
    )
    engine = RetrievalEngine(model, full_ds, device)

    # Demo split: gallery = second half, queries = first N.
    n = len(full_ds)
    gallery_ids = np.arange(n // 2, n)
    gallery = engine.build_gallery(
        gallery_ids, cfg["modalities"][0],
        index_kwargs={"index_type": cfg["retrieval"]["index"]["type"],
                      "metric": cfg["retrieval"]["index"]["metric"]},
    )
    query_ids = np.arange(0, min(args.n_queries, n // 2))

    rerank_cfg = dict(cfg["retrieval"].get("rerank") or {})
    if args.rerank:
        rerank_cfg["enabled"] = True
        rerank_cfg["method"] = args.rerank
    reranker = build_reranker(rerank_cfg)

    report = run_latency_benchmark(
        engine, gallery, query_ids, cfg["modalities"][0],
        k=args.k, candidate_k=args.candidate_k, reranker=reranker,
    )
    save_json(args.out, report.to_dict())
    print(report.to_dict())


if __name__ == "__main__":
    main()
