# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dump the FULL per-aggregator F1 table for the chunked ModernBERT pilot from wandb.

Each finished run logs ``eval/<agg>_f1`` + ``eval/<agg>_threshold`` for every aggregator plus the
overall ``eval/best_f1``/``eval/best_agg``. Training only kept the best in the headline table; this
prints ALL aggregations for every finished run so nothing is lost. Run locally (needs WANDB_API_KEY).

  WANDB_API_KEY=... uv run --no-sync python -m experiments.baseline_collection.dump_chunk_results
"""

import argparse

import wandb

AGGS = ("mean", "max", "min", "top2mean", "top3mean")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", default="modernbert-useful")
    args = p.parse_args()

    api = wandb.Api(timeout=60)
    rows = []
    runs = api.runs(args.project, filters={"display_name": {"$regex": "chunk|canary"}}, order="-created_at", per_page=50)
    for r in runs:
        n = r.name
        if not (("chunk" in n and "200k" in n) or n == "mb-clf-canary-c2048-stream"):
            continue
        s = r.summary
        if s.get("eval/best_f1") is None:
            continue
        f1 = {a: s.get(f"eval/{a}_f1") for a in AGGS}
        thr = {a: s.get(f"eval/{a}_threshold") for a in AGGS}
        rows.append((n, f1, thr, s.get("eval/best_f1"), s.get("eval/best_agg"), s.get("eval/best_threshold")))
    rows.sort(key=lambda x: -(x[3] or 0))

    label = lambda n: n.replace("mb-clf-", "").replace("-chunk", "")  # noqa: E731
    hdr = f"{'run':26s} | " + " ".join(f"{a:>8s}" for a in AGGS) + " | best (agg @ thr)"
    print(hdr)
    print("-" * len(hdr))
    for n, f1, _thr, bf, ba, bt in rows:
        cells = " ".join(f"{f1[a]:8.4f}" if f1[a] is not None else f"{'-':>8s}" for a in AGGS)
        print(f"{label(n):26s} | {cells} | {bf:.4f} ({ba} @ {bt})")
    print(f"({len(rows)} runs finished)")


if __name__ == "__main__":
    main()
