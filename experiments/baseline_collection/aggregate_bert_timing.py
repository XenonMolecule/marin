# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate the per-(WARC, rank) ``.timing.json`` files written by
``modernbert_warc_filter.py`` into a scaling report: throughput (docs/sec/chip,
tokens/sec/chip), WARCs/hr, kept-rate, and a full-10k extrapolation.

A single TPU job runs ``world`` ranks (chips) that doc-shard one WARC between them, so a
WARC's wall time is the MAX rank wall (they run concurrently); chip-seconds sum across
ranks. Independent ``--warc-shards`` jobs add throughput linearly, so the per-job rate
times N jobs is the fleet rate.

Usage::

    python -m experiments.baseline_collection.aggregate_bert_timing \
        --out-root gs://marin-us-east5/documents/bert_pipeline/bert_kept_10k \
        --total-warcs 10363
"""

import argparse
import json
from collections import defaultdict

import fsspec


def _load_timings(out_root: str) -> list[dict]:
    fs, root = fsspec.core.url_to_fs(out_root)
    files = fs.glob(f"{root}/timing/*.timing.json")
    out = []
    for f in files:
        with fs.open(f, "rt") as fh:
            out.append(json.loads(fh.read()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--total-warcs", type=int, default=10363)
    args = ap.parse_args()

    rows = _load_timings(args.out_root)
    if not rows:
        print(f"No timing files under {args.out_root}/timing/ yet.")
        return

    # Per-WARC aggregation across its ranks.
    by_warc = defaultdict(list)
    for r in rows:
        by_warc[r["warc_hash"]].append(r)

    n_warcs = len(by_warc)
    tot_docs = sum(r["n_docs"] for r in rows)
    tot_kept = sum(r["n_kept"] for r in rows)
    tot_tokens = sum(r["n_tokens"] for r in rows)
    tot_bert_s = sum(r["bert_s"] for r in rows)  # chip-seconds of BERT forward
    tot_tok_s = sum(r["tokenize_s"] for r in rows)
    world = max((r["world"] for r in rows), default=1)
    tpu_types = sorted({r.get("tpu_type", "?") for r in rows})

    # Per-WARC wall = max rank wall (ranks run concurrently within a job).
    warc_walls = [max(r["wall_s"] for r in ranks) for ranks in by_warc.values()]
    mean_warc_wall = sum(warc_walls) / len(warc_walls)

    docs_per_chip_s = tot_docs / tot_bert_s if tot_bert_s else 0.0
    tokens_per_chip_s = tot_tokens / tot_bert_s if tot_bert_s else 0.0
    warcs_per_hr_per_job = 3600.0 / mean_warc_wall if mean_warc_wall else 0.0

    print("=" * 64)
    print(f"ModernBERT WARC filter — timing/scaling  ({args.out_root})")
    print("=" * 64)
    print(f"WARCs scored (have timing) : {n_warcs}")
    print(f"TPU shape(s)               : {', '.join(tpu_types)}  (world={world} chips/job)")
    print(f"Docs scored                : {tot_docs:,}")
    print(f"Docs kept (BERT-retained)  : {tot_kept:,}  ({100.0 * tot_kept / tot_docs:.1f}% kept)")
    print(f"Tokens scored              : {tot_tokens:,}")
    print("-" * 64)
    print(f"BERT chip-seconds          : {tot_bert_s:,.0f}  (tokenize {tot_tok_s:,.0f}s)")
    print(f"Throughput  docs/chip/s    : {docs_per_chip_s:,.1f}")
    print(f"Throughput  tok/chip/s     : {tokens_per_chip_s:,.0f}")
    print(f"Mean per-WARC wall (1 job) : {mean_warc_wall:,.1f}s")
    print(f"WARCs/hr  (1 job, {world} chips): {warcs_per_hr_per_job:,.2f}")
    print("-" * 64)
    print("Extrapolation to full pool (independent jobs scale ~linearly):")
    for n_jobs in (1, 4, 8, 16):
        rate = warcs_per_hr_per_job * n_jobs
        hrs = args.total_warcs / rate if rate else float("inf")
        print(f"  {n_jobs:>2} jobs: {rate:7.1f} WARCs/hr  ->  full {args.total_warcs} in {hrs:6.1f} hr")
    print("=" * 64)


if __name__ == "__main__":
    main()
