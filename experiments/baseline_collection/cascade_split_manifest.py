"""Compute the frozen train/val/test WARC split for the cascade-filtered distill
dataset and write it to GCS as JSON.

Reuses the classifier's snapshot-stratified held-out (``held_out_indices``, k=1)
so the cascade distill set shares the *exact* val/test WARCs as the classifier and
the original bal350k chat set — zero leakage, identical splits. Runs in us-central2
(where the parquet lives) on CPU.

Output JSON: {"val": [...], "test": [...], "train": [...]}  (WARC indices 0..2999)
"""
import argparse
import json

import fsspec
from marin.utils import fsspec_glob

from experiments.baseline_collection.fasttext_useful_classifier import (
    USEFUL_DIR,
    held_out_indices,
)
from experiments.baseline_collection.fasttext_useful_classifier import (
    _shard_snapshots as shard_snapshots,
)

K_HOLDOUT = 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gs://marin-us-central2/datasets/high_quality_3000_distill_chat_bal350k_cascade/_split_manifest.json")
    args = ap.parse_args()

    useful_shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    n = len(useful_shards)
    snapshots = shard_snapshots(useful_shards)
    val_idx, test_idx = held_out_indices(snapshots, K_HOLDOUT)
    train_idx = [i for i in range(n) if i not in val_idx and i not in test_idx]

    manifest = {
        "n_warcs": n,
        "k_holdout": K_HOLDOUT,
        "val": sorted(val_idx),
        "test": sorted(test_idx),
        "train": train_idx,
        "n_snapshots": len(set(snapshots)),
    }
    with fsspec.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(
        f"wrote {args.out}: n_warcs={n} snapshots={len(set(snapshots))} "
        f"val={len(val_idx)} test={len(test_idx)} train={len(train_idx)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
