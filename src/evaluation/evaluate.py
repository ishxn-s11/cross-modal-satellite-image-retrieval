"""Evaluation harness: same-modal and cross-modal retrieval over query/gallery splits."""

from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..data.dataset import MultiModalDataset
from ..models.encoder import ModalityAdaptiveEncoder
from ..retrieval.engine import Gallery, RetrievalEngine
from ..utils.io import Logger, save_json
from .metrics import RetrievalMetrics, format_table, retrieval_metrics, to_dict

Pair = Tuple[str, str]


def stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split into train / val / test id arrays (balanced by class)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    train, val, test = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_tr = int(round(train_ratio * len(idx)))
        n_val = int(round(val_ratio * len(idx)))
        train.append(idx[:n_tr])
        val.append(idx[n_tr : n_tr + n_val])
        test.append(idx[n_tr + n_val :])
    return (
        np.concatenate(train).astype(np.int64),
        np.concatenate(val).astype(np.int64),
        np.concatenate(test).astype(np.int64),
    )


def _gallery_label_counts(gallery: Gallery) -> Counter:
    return Counter(int(x) for x in gallery.labels.tolist())


def evaluate_retrieval_pairs(
    model: ModalityAdaptiveEncoder,
    dataset: MultiModalDataset,
    modalities: Sequence[str],
    same_modal_pairs: Sequence[Pair],
    cross_modal_pairs: Sequence[Pair],
    retrieval_cfg: Dict,
    device,
    seed: int = 42,
    logger: Logger | None = None,
) -> Tuple[List[Dict], Dict]:
    """Run retrieval evaluation and return (report rows, summary)."""
    log = logger.info if logger else print
    top_ks = [int(k) for k in retrieval_cfg.get("top_k", [5, 10])]
    gallery_frac = float(retrieval_cfg.get("gallery_fraction", 0.85))
    n_query_cap = int(retrieval_cfg.get("n_query", 400))

    engine = RetrievalEngine(model, dataset, device)
    labels = dataset.labels
    _, _, test_ids = stratified_split(labels, 0.7, 0.15, seed)
    rng = np.random.RandomState(seed)
    rng.shuffle(test_ids)

    n_gallery = max(1, int(round(gallery_frac * len(test_ids))))
    gallery_ids = test_ids[:n_gallery]
    query_ids = test_ids[n_gallery:]
    if n_query_cap and len(query_ids) > n_query_cap:
        query_ids = query_ids[:n_query_cap]
    log(
        f"[eval] gallery={len(gallery_ids)} query={len(query_ids)} "
        f"(classes={len(np.unique(labels))})"
    )

    # Build one gallery per gallery-modality, reused across pairs.
    galleries: Dict[str, Gallery] = {}
    for m in modalities:
        galleries[m] = engine.build_gallery(gallery_ids, m)
        log(f"[eval] built gallery '{m}' ({galleries[m].size} items)")

    all_pairs: List[Pair] = [(q, g) for q, g in same_modal_pairs] + [
        (q, g) for q, g in cross_modal_pairs
    ]
    rows: List[Dict] = []
    seen_kinds: Dict[Pair, str] = {}
    for q, g in same_modal_pairs:
        seen_kinds[(q, g)] = "same"
    for q, g in cross_modal_pairs:
        seen_kinds[(q, g)] = "cross"

    for qm, gm in all_pairs:
        if (qm, gm) not in seen_kinds:
            continue
        result = engine.retrieve(galleries[gm], query_ids, qm, k=max(top_ks))
        label_counts = _gallery_label_counts(galleries[gm])
        for k in top_ks:
            rel = result.relevant_mask()[:, :k]
            total_relevant = np.array(
                [max(1, label_counts.get(int(l), 0)) for l in result.query_labels],
                dtype=np.float64,
            )
            m = retrieval_metrics(rel, total_relevant, k, result.search_times_ms)
            desc = f"{qm}->{gm}"
            rows.append(to_dict(m, desc, seen_kinds[(qm, gm)]))
            log(
                f"[eval] {desc:>18} k={k:<3} F1={m.f1:.4f} "
                f"P={m.precision:.4f} R={m.recall:.4f} "
                f"time={m.avg_time_ms:.3f}ms"
            )

    summary = _summarise(rows)
    return rows, summary


def _summarise(rows: List[Dict]) -> Dict:
    same = [r for r in rows if r["kind"] == "same"]
    cross = [r for r in rows if r["kind"] == "cross"]

    def avg(rs: List[Dict]) -> Dict:
        if not rs:
            return {"f1@5": 0.0, "f1@10": 0.0}
        f5 = [r["f1@k"] for r in rs if r["k"] == 5]
        f10 = [r["f1@k"] for r in rs if r["k"] == 10]
        return {"f1@5": float(np.mean(f5)), "f1@10": float(np.mean(f10))}

    s, c = avg(same), avg(cross)
    # Cross-modal is weighted more heavily per the task brief.
    w5 = (1.0 * s["f1@5"] + 1.5 * c["f1@5"]) / 2.5
    w10 = (1.0 * s["f1@10"] + 1.5 * c["f1@10"]) / 2.5
    times = [r["avg_retrieval_time_ms"] for r in rows]
    return {
        "same_modal_avg": s,
        "cross_modal_avg": c,
        "weighted_avg": {"f1@5": float(w5), "f1@10": float(w10)},
        "avg_retrieval_time_ms": float(np.mean(times)) if times else 0.0,
        "n_rows": len(rows),
    }


def save_report(rows: List[Dict], summary: Dict, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "retrieval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(out_dir, "retrieval_summary.json")
    save_json(json_path, {"summary": summary, "rows": rows})
    return csv_path, json_path
