#!/usr/bin/env python
"""
scripts/sweep.py — parameter search on the DEV topics (assignment
Section 8, "your parameter search procedure for k1 and b").

Tuning happens here and only here, against data/full/queries_dev.tsv +
qrels_dev.txt. The held-out topics are never involved: that separation is
the entire point of the two-tier leaderboard (Section 7), and the report
has to be able to state it plainly.

The index is built once and loaded once; every configuration then re-scores
the same 50 queries in memory, so a full grid is seconds rather than
minutes.

Usage:
    python scripts/sweep.py --index-dir /tmp/idx --grid bm25
    python scripts/sweep.py --index-dir /tmp/idx --grid blend --k1 0.9 --b 0.4
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from harness.metrics import evaluate_run
from harness.trec_io import read_qrels, read_queries
from submission import bm25, boolean_vsm, custom_scorer
from submission.indexer import InvertedIndex


def evaluate(queries, qrels, score_fn, k=10):
    run = {qid: score_fn(text, k) for qid, text in queries}
    agg = evaluate_run(run, qrels, k=k)["aggregate"]
    return agg["ndcg@10"], agg["map@10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--queries", default="data/full/queries_dev.tsv")
    ap.add_argument("--qrels", default="data/full/qrels_dev.txt")
    ap.add_argument("--grid", default="bm25", choices=["bm25", "blend", "scorers"])
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--out", default=None, help="CSV of every configuration tried")
    args = ap.parse_args()

    print(f"loading index from {args.index_dir} ...", flush=True)
    index = InvertedIndex.load(args.index_dir)
    bm25.build(index)
    boolean_vsm.build(index)
    custom_scorer.build(index)
    print(f"  {index.N} docs, {len(index.terms)} terms, {int(index.offsets[-1])} postings\n", flush=True)

    queries = read_queries(args.queries)
    qrels = read_qrels(args.qrels)

    rows = []

    if args.grid == "scorers":
        print(f"{'scorer':<28}{'nDCG@10':>10}{'MAP@10':>10}")
        print("-" * 48)
        for name, fn in [
            ("BM25 (k1=1.2, b=0.75)", lambda q, k: bm25.score(q, k, k1=1.2, b=0.75)),
            (f"BM25 (k1={args.k1}, b={args.b})", lambda q, k: bm25.score(q, k, k1=args.k1, b=args.b)),
            ("TF-IDF cosine (VSM)", boolean_vsm.vsm_score),
        ]:
            nd, mp = evaluate(queries, qrels, fn)
            print(f"{name:<28}{nd:>10.4f}{mp:>10.4f}")
            rows.append({"config": name, "ndcg@10": nd, "map@10": mp})

    elif args.grid == "bm25":
        k1s = [round(x, 2) for x in np.arange(0.3, 2.41, 0.1)]
        bs = [round(x, 2) for x in np.arange(0.0, 1.01, 0.05)]
        best = None
        print(f"sweeping {len(k1s)} k1 x {len(bs)} b = {len(k1s) * len(bs)} configurations", flush=True)
        for k1 in k1s:
            for b in bs:
                nd, mp = evaluate(queries, qrels, lambda q, k, _k1=k1, _b=b: bm25.score(q, k, k1=_k1, b=_b))
                rows.append({"k1": k1, "b": b, "ndcg@10": nd, "map@10": mp})
                if best is None or nd > best[0]:
                    best = (nd, mp, k1, b)
            print(f"  k1={k1:<5} best so far nDCG@10={best[0]:.4f} at k1={best[2]}, b={best[3]}", flush=True)
        print(f"\nBEST: nDCG@10={best[0]:.4f}  MAP@10={best[1]:.4f}  at k1={best[2]}, b={best[3]}")

    elif args.grid == "blend":
        best = None
        covs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        vsms = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        print(f"sweeping {len(covs)} coverage x {len(vsms)} vsm at k1={args.k1}, b={args.b}", flush=True)
        for wc in covs:
            for wv in vsms:
                nd, mp = evaluate(
                    queries, qrels,
                    lambda q, k, _wc=wc, _wv=wv: custom_scorer.score(
                        q, k, k1=args.k1, b=args.b, w_coverage=_wc, w_vsm=_wv),
                )
                rows.append({"w_coverage": wc, "w_vsm": wv, "ndcg@10": nd, "map@10": mp})
                if best is None or nd > best[0]:
                    best = (nd, mp, wc, wv)
            print(f"  w_cov={wc:<5} best so far nDCG@10={best[0]:.4f} at w_cov={best[2]}, w_vsm={best[3]}", flush=True)
        print(f"\nBEST: nDCG@10={best[0]:.4f}  MAP@10={best[1]:.4f}  at w_coverage={best[2]}, w_vsm={best[3]}")

    if args.out and rows:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
