# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the MMLU dataset cache from one source bucket to the other regional
buckets, byte-identically, using in-cloud server-side GCS copies.

Sibling of `olmes_base/mirror_olmes_cache.py`; only the cache path and the source
bucket differ (the MMLU cache is built in us-east5, where the bulk of the 10k sweep's
checkpoints live). Every eval child reads the cache from ITS OWN region, so the cache
must exist in every bucket the manifest touches — for
`checkpoint_manifest_10k_dedup245.txt` that is all five.

Run as a CPU iris job so the copies are server-side (GCS rewrite API) and never route
through a laptop — robust to local network flakiness. Idempotent: skips objects already
present at the destination, so it is safe to re-run.

The whole cache is ~7.6MB / ~318 objects (MMLU is small: 56 subjects x 3 splits of
parquet-derived arrow), so the cross-region copy is negligible — unlike the checkpoints,
this is not a bandwidth concern.

Usage:
    iris --cluster marin job run --region us-east5 --cpu 2 --memory 8GB \\
        --enable-extra-resources \\
        -- python -m experiments.scaling_law_sweeps.mmlu.mirror_mmlu_cache
"""

from __future__ import annotations

import argparse
import logging

from rigging.filesystem import filesystem as marin_filesystem

logger = logging.getLogger(__name__)

CACHE_SUBPATH = "eval_datasets/mmlu_hf_cache"
DEFAULT_SRC_BUCKET = "marin-us-east5"  # where build_mmlu_dataset_cache wrote it
DEFAULT_DST_BUCKETS = ["marin-us-central1", "marin-eu-west4", "marin-us-central2", "marin-us-east1"]


def mirror(src_bucket: str, dst_buckets: list[str]) -> None:
    fs = marin_filesystem("gcs")
    src_prefix = f"{src_bucket}/{CACHE_SUBPATH}"
    src_files = [f for f in fs.find(f"gs://{src_prefix}") if not f.endswith("/")]
    logger.info("source %s has %d objects", src_prefix, len(src_files))

    for dst_bucket in dst_buckets:
        copied = skipped = 0
        for src in src_files:
            rel = src.split(f"{CACHE_SUBPATH}/", 1)[1]
            dst = f"{dst_bucket}/{CACHE_SUBPATH}/{rel}"
            src_uri = src if src.startswith("gs://") else f"gs://{src}"
            dst_uri = f"gs://{dst}"
            if fs.exists(dst_uri):
                skipped += 1
                continue
            fs.copy(src_uri, dst_uri)  # server-side GCS rewrite
            copied += 1
            if copied % 50 == 0:
                logger.info("  %s: copied %d, skipped %d ...", dst_bucket, copied, skipped)
        logger.info("DONE %s: copied %d, skipped %d (total %d)", dst_bucket, copied, skipped, len(src_files))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-bucket", default=DEFAULT_SRC_BUCKET)
    ap.add_argument("--dst-buckets", nargs="*", default=DEFAULT_DST_BUCKETS)
    args = ap.parse_args()
    mirror(args.src_bucket, args.dst_buckets)
    logger.info("Mirror complete.")


if __name__ == "__main__":
    main()
