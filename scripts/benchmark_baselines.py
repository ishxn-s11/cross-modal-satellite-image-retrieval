"""Compare the five baselines + the proposed full config.

Usage:
    python scripts/benchmark_baselines.py [--config configs/default.yaml]
        [--epochs 2] [--num-patches 2000] [--out outputs/benchmarks/baselines]

Running every variant end-to-end is CPU-expensive; use --epochs to keep it
feasible. Only measured numbers are reported.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.benchmarks import run_baselines
from src.utils.config import load_config
from src.utils.io import Logger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-patches", type=int, default=None)
    ap.add_argument("--out", default="outputs/benchmarks/baselines")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    logger = Logger(path=None)
    results = run_baselines(
        cfg,
        out_dir=args.out,
        budget_epochs=args.epochs,
        num_patches=args.num_patches,
        logger=logger,
    )
    for r in results:
        if r.get("note", "").startswith("skipped"):
            print(f"  {r['variant']:>34} | {r['note']}")
        else:
            print(
                f"  {r['variant']:>34} | same F1@5={r['same_f1@5']:.4f} "
                f"cross F1@5={r['cross_f1@5']:.4f} time={r['avg_retrieval_time_ms']:.3f}ms"
            )


if __name__ == "__main__":
    main()
