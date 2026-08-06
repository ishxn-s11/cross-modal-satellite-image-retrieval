"""Benchmark FAISS index types across gallery sizes.

Usage:
    python scripts/benchmark_scalability.py [--sizes 10000,100000]
        [--index-types flat,ivf,hnsw,ivfpq] [--max-vectors 1000000]
        [--out outputs/benchmarks/scalability.json]

1M vectors (~0.5 GB for flat + copies for IVF/PQ) is opt-in; the script skips
any size above --max-vectors. All numbers are measured.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.scalability import benchmark_scalability
from src.utils.io import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10000,100000")
    ap.add_argument("--index-types", default="flat,ivf,hnsw,ivfpq")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-query", type=int, default=200)
    ap.add_argument("--max-vectors", type=int, default=1_000_000)
    ap.add_argument("--out", default="outputs/benchmarks/scalability.json")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    rows = benchmark_scalability(
        sizes=sizes,
        d=args.d,
        index_types=tuple(t.strip() for t in args.index_types.split(",")),
        k=args.k,
        n_query=args.n_query,
        max_vectors=args.max_vectors,
    )
    save_json(args.out, rows)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"[scalability] {len(rows)} rows -> {args.out} / {csv_path}")


if __name__ == "__main__":
    main()
