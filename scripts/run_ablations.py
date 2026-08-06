"""Run ablation studies (one axis at a time) and write CSV/JSON results.

Usage:
    python scripts/run_ablations.py [--config configs/default.yaml]
        [--epochs 2] [--num-patches 2000] [--axes encoder,loss]
        [--out outputs/benchmarks/ablation]

CPU-expensive; use --epochs / --num-patches to keep runs feasible. Results are
measured, never fabricated.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.ablation import SWEEPS, run_ablations
from src.utils.config import load_config
from src.utils.io import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-patches", type=int, default=None)
    ap.add_argument("--axes", default=None, help="comma-separated subset of "
                    "encoder,embedding_dim,loss,hard_negatives,reranking,geo_supervision")
    ap.add_argument("--out", default="outputs/benchmarks/ablation")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(path=None)
    sweeps = SWEEPS
    if args.axes:
        allowed = [a.strip() for a in args.axes.split(",") if a.strip()]
        sweeps = {a: SWEEPS[a] for a in allowed if a in SWEEPS}
    results = run_ablations(
        cfg, sweeps=sweeps, out_dir=args.out,
        budget_epochs=args.epochs, num_patches=args.num_patches, logger=logger,
    )
    for axis, rows in results.items():
        for r in rows:
            print(f"  {axis:>16} {r['variant']:<18} same={r.get('same_f1@5', '-')} "
                  f"cross={r.get('cross_f1@5', '-')} {r.get('note', '')}")


if __name__ == "__main__":
    main()
