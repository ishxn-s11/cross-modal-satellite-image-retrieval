"""Ablation-study harness.

Sweeps one independent axis at a time (everything else fixed) through the full
pipeline and writes CSV/JSON per sweep:

* encoder            -- resnet18 / resnet50 / vit_b_16
* embedding_dim      -- 128 / 256 / 512
* loss               -- InfoNCE / SupCon / InfoNCE+SupCon
* hard_negatives     -- disabled / enabled
* reranking          -- disabled / geo
* geo_supervision    -- off / on

Every variant is a set of deep dot-path overrides on the base config; results
are measured, never fabricated.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

from ..pipeline import run_experiment
from ..utils.config import deep_merge, set_nested
from ..utils.io import Logger, save_json

SWEEPS: Dict[str, List[Dict]] = {
    "encoder": [
        {"label": "resnet18", "overrides": {"model.backbone": "resnet18"}},
        {"label": "resnet50", "overrides": {"model.backbone": "resnet50"}},
        {"label": "vit_b_16", "overrides": {"model.backbone": "vit_b_16"}},
    ],
    "embedding_dim": [
        {"label": "dim_128", "overrides": {"model.embedding_dim": 128}},
        {"label": "dim_256", "overrides": {"model.embedding_dim": 256}},
        {"label": "dim_512", "overrides": {"model.embedding_dim": 512}},
    ],
    "loss": [
        {"label": "info_nce", "overrides": {"training.clip_weight": 1.0, "training.supcon_weight": 0.0, "training.cls_weight": 0.0}},
        {"label": "supcon", "overrides": {"training.clip_weight": 0.0, "training.supcon_weight": 1.0, "training.cls_weight": 0.0}},
        {"label": "info_nce+supcon", "overrides": {"training.clip_weight": 1.0, "training.supcon_weight": 1.0, "training.cls_weight": 0.0}},
    ],
    "hard_negatives": [
        {"label": "disabled", "overrides": {"training.hard_negatives.enabled": False}},
        {"label": "enabled", "overrides": {"training.hard_negatives.enabled": True, "training.hard_negatives.n_hard": 8}},
    ],
    "reranking": [
        {"label": "disabled", "overrides": {"retrieval.rerank.enabled": False}},
        {"label": "geo", "overrides": {"retrieval.rerank.enabled": True, "retrieval.rerank.method": "geo"}},
    ],
    "geo_supervision": [
        {"label": "off", "overrides": {"training.geo_weight": 0.0}},
        {"label": "on", "overrides": {"training.geo_weight": 1.0}},
    ],
}


def _apply(cfg: Dict, overrides: Dict) -> Dict:
    out = deep_merge({}, cfg)
    for path, value in overrides.items():
        set_nested(out, path, value)
    return out


def run_ablations(
    cfg: Dict,
    sweeps: Optional[Dict[str, List[Dict]]] = None,
    out_dir: str = "outputs/benchmarks/ablation",
    budget_epochs: Optional[int] = None,
    num_patches: Optional[int] = None,
    logger=None,
) -> Dict[str, List[Dict]]:
    """Run each sweep and write per-sweep CSV/JSON result files."""
    sweeps = sweeps or SWEEPS
    os.makedirs(out_dir, exist_ok=True)
    log = logger if logger is not None else Logger(path=None)

    all_results: Dict[str, List[Dict]] = {}
    for axis, variants in sweeps.items():
        results: List[Dict] = []
        for v in variants:
            exp_cfg = _apply(cfg, v["overrides"])
            if budget_epochs is not None:
                exp_cfg["training"]["epochs"] = int(budget_epochs)
            if num_patches is not None:
                exp_cfg["dataset"]["num_patches"] = int(num_patches)
            log.info(f"[ablation] {axis}={v['label']} (epochs={exp_cfg['training']['epochs']})")
            try:
                _rows, summary = run_experiment(exp_cfg, logger)
                s, c = summary["same_modal_avg"], summary["cross_modal_avg"]
                results.append(
                    {
                        "axis": axis,
                        "variant": v["label"],
                        "same_f1@5": round(s["f1@5"], 4),
                        "same_f1@10": round(s["f1@10"], 4),
                        "cross_f1@5": round(c["f1@5"], 4),
                        "cross_f1@10": round(c["f1@10"], 4),
                        "note": "",
                    }
                )
            except Exception as exc:
                results.append({"axis": axis, "variant": v["label"],
                                "note": f"failed: {type(exc).__name__}: {exc}"})
        all_results[axis] = results
        csv_path = os.path.join(out_dir, f"ablation_{axis}.csv")
        if results:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
        log.info(f"[ablation] {axis} -> {csv_path}")
    json_path = os.path.join(out_dir, "ablations.json")
    save_json(json_path, all_results)
    log.info(f"[ablation] all -> {json_path}")
    return all_results
