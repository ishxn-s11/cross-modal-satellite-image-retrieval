"""Baseline comparison harness.

Runs several model/training variants through the exact same pipeline and writes
a comparison table (CSV + JSON) so retrieval quality can be compared fairly.

Variants (see BASELINE_VARIANTS):

1. ResNet + cosine            -- raw pretrained ResNet-18 features, no contrastive training.
2. ResNet + InfoNCE           -- contrastive alignment only.
3. ResNet + InfoNCE + SupCon  -- contrastive + supervised clustering (no CE).
4. ViT + contrastive          -- ViT backbone with the full objective.
5. Foundation + contrastive   -- SatMAE/Prithvi (requires a configured checkpoint).
6. Proposed                   -- the full default config.

Only measured numbers are reported. Variants that cannot run (e.g. no
foundation checkpoint) are recorded as "skipped".
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence

from ..pipeline import run_experiment
from ..utils.config import deep_merge, set_nested
from ..utils.io import Logger, save_json

# Each variant is a set of deep dot-path overrides on top of the base config.
BASELINE_VARIANTS: Dict[str, Dict] = {
    "baseline1_resnet_cosine": {
        "model.backbone": "resnet18",
        "model.embedding_mode": "raw",
        "model.freeze_backbone": True,
        "training.epochs": 0,
        "training.clip_weight": 0.0,
        "training.supcon_weight": 0.0,
        "training.cls_weight": 0.0,
    },
    "baseline2_resnet_infonce": {
        "training.clip_weight": 1.0,
        "training.supcon_weight": 0.0,
        "training.cls_weight": 0.0,
    },
    "baseline3_resnet_infonce_supcon": {
        "training.clip_weight": 1.0,
        "training.supcon_weight": 1.0,
        "training.cls_weight": 0.0,
    },
    "baseline4_vit_contrastive": {
        "model.backbone": "vit_b_16",
        "model.freeze_backbone": True,
    },
    "baseline5_foundation_contrastive": {
        "model.backbone": "satmae",  # requires model.foundation.satmae.path
    },
    "proposed_full": {},
}


def _apply_overrides(cfg: Dict, overrides: Dict) -> Dict:
    out = deep_merge({}, cfg)
    for path, value in overrides.items():
        set_nested(out, path, value)
    return out


def run_baselines(
    cfg: Dict,
    variants: Optional[Dict[str, Dict]] = None,
    out_dir: str = "outputs/benchmarks/baselines",
    budget_epochs: Optional[int] = None,
    num_patches: Optional[int] = None,
    logger=None,
) -> List[Dict]:
    """Run each baseline variant and write a comparison table."""
    variants = variants or BASELINE_VARIANTS
    os.makedirs(out_dir, exist_ok=True)
    log = logger if logger is not None else Logger(path=None)

    results: List[Dict] = []
    for name, overrides in variants.items():
        exp_cfg = _apply_overrides(cfg, overrides)
        if budget_epochs is not None:
            exp_cfg["training"]["epochs"] = int(budget_epochs)
        if num_patches is not None:
            exp_cfg["dataset"]["num_patches"] = int(num_patches)
        log.info(f"[baseline] running '{name}' (epochs={exp_cfg['training']['epochs']})")
        try:
            _rows, summary = run_experiment(exp_cfg, logger)
            s, c = summary["same_modal_avg"], summary["cross_modal_avg"]
            results.append(
                {
                    "variant": name,
                    "epochs": int(exp_cfg["training"]["epochs"]),
                    "same_f1@5": round(s["f1@5"], 4),
                    "same_f1@10": round(s["f1@10"], 4),
                    "cross_f1@5": round(c["f1@5"], 4),
                    "cross_f1@10": round(c["f1@10"], 4),
                    "weighted_f1@5": round(summary["weighted_avg"]["f1@5"], 4),
                    "avg_retrieval_time_ms": round(summary["avg_retrieval_time_ms"], 4),
                    "note": "",
                }
            )
        except Exception as exc:  # e.g. missing foundation weights
            log.info(f"[baseline] '{name}' skipped: {exc}")
            results.append(
                {"variant": name, "note": f"skipped: {type(exc).__name__}: {exc}"}
            )

    csv_path = os.path.join(out_dir, "baseline_comparison.csv")
    if results:
        fieldnames = [k for k in results[0].keys()]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    json_path = os.path.join(out_dir, "baseline_comparison.json")
    save_json(json_path, results)
    log.info(f"[baseline] comparison table -> {csv_path} / {json_path}")
    return results
