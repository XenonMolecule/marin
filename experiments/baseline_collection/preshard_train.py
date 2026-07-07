# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pre-shard a fastText-format train file into `world` per-rank gzip shards.

The multi-host ModernBERT trainer's striding reader (read_fasttext_sharded) has every
rank re-stream the whole train prefix to pick its 1/world stride. At 1M+ rows that is
~5 GB ×world per start AND again on every preemption/resume — the scaling bottleneck.

This materializes the same round-robin partition once: valid line at global index gi
goes to shard ``gi % world`` (identical assignment to the reader's ``gi % world == rank``
keep test), taking the first ``total_rows`` valid lines. Rank r then reads only its
own ~total_rows/world-row shard via modernbert_tpu_smoke.read_presharded.

Run in-region (same bucket/region as the data and the trainer) — it reads ~5 GB and
writes ~5 GB; cross-region would be wasted egress.
"""

from __future__ import annotations

import argparse

import fsspec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Source fastText .txt.gz (label<space>text per line).")
    ap.add_argument("--out-prefix", required=True, help="Output prefix; writes {prefix}_NN.txt.gz for NN in 0..world-1.")
    ap.add_argument("--total-rows", type=int, required=True, help="Total valid rows to distribute across shards.")
    ap.add_argument("--world", type=int, default=16, help="Number of shards (must equal the trainer's world_size).")
    args = ap.parse_args()

    per_shard = args.total_rows // args.world
    target = per_shard * args.world  # drop the remainder so every shard is exactly per_shard rows
    writers = [
        fsspec.open(f"{args.out_prefix}_{r:02d}.txt.gz", "wt", compression="gzip", encoding="utf-8").open()
        for r in range(args.world)
    ]
    counts = [0] * args.world
    gi = 0
    done = 0
    try:
        with fsspec.open(args.train, "rt", compression="gzip", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                _, _, text = line.partition(" ")
                if not text:
                    continue
                r = gi % args.world
                if counts[r] < per_shard:
                    writers[r].write(line + "\n")
                    counts[r] += 1
                    done += 1
                    if done >= target:
                        break
                gi += 1
    finally:
        for w in writers:
            w.close()

    print(f"wrote {args.world} shards x {per_shard} rows = {done} total to {args.out_prefix}_NN.txt.gz", flush=True)
    if done < target:
        raise SystemExit(f"source exhausted: only {done} valid rows < requested {target}")


if __name__ == "__main__":
    main()
