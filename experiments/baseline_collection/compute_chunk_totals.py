# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute the exact per-config total training steps for the chunked pilot, from the cache offsets.

The real ``num_train_steps`` is ``num_chunks // batch`` and is recomputed worker-side, so wandb only
has the launcher placeholder. This reads the first ``--rows`` doc token-lengths from the streaming
cache's jagged offsets (no token data) and computes num_chunks (and steps) for each (ctx, tiling),
which is model-independent (base/large share lengths). Run as a small CPU job IN-REGION (us-east5).
"""

import argparse
import logging

import numpy as np

from levanter.store.tree_store import TreeStore

logger = logging.getLogger(__name__)

CACHE_DIR = "gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_random_par/_clf_token_cache_chunk32768"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", default=CACHE_DIR)
    p.add_argument("--rows", type=int, default=200_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-chunks", type=int, default=16)
    args = p.parse_args()

    exemplar = {"input_ids": np.zeros((0,), np.int32), "label": np.zeros((), np.int32)}
    store = TreeStore.open(exemplar, args.cache_dir, mode="r")
    ids = store.tree["input_ids"]
    n_total = ids.num_rows
    n = min(n_total, args.rows)
    raw = np.asarray(ids.offsets[0 : n + 1].read().result(), dtype=np.int64)
    raw[0] = 0
    lengths = np.diff(raw)[:n]
    logger.info("rows=%d (of %d); token len median=%d p90=%d max=%d", n, n_total,
                int(np.median(lengths)), int(np.percentile(lengths, 90)), int(lengths.max()))

    print(f"{'ctx':>6} {'tiling':>8} {'chunks':>12} {'steps':>8}")
    for ctx in (512, 1024, 2048, 4096, 8192):
        for tag, stride in (("no", ctx), ("ov", ctx // 2)):
            starts = np.ceil(np.maximum(1, lengths) / stride).astype(np.int64)  # len(chunk_starts)
            num_chunks = int(np.minimum(starts, args.max_chunks).sum())
            print(f"{ctx:>6} {tag:>8} {num_chunks:>12} {num_chunks // args.batch:>8}")


if __name__ == "__main__":
    main()
