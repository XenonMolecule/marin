# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Mirror a tokenized Levanter cache to another region bucket via server-side GCS rewrites.

Same mechanism as ``consolidate_kept_text.py``: ``fs.copy`` issues rewrite calls, so bytes move
inside GCS (no local download). Idempotent — an object that already exists at the destination
with the same size is skipped, so a retry never re-copies (or re-bills) finished work.

    python -m experiments.fast_curation.mirror_cache \\
        --src gs://marin-us-east5/tokenized/lpv11_fastpipe_v1_decon_10364warcs-a16e729 \\
        --dst gs://marin-us-central1/tokenized/lpv11_fastpipe_v1_decon_10364warcs-a16e729
"""
from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--num-threads", type=int, default=32)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs")
    src = args.src.removeprefix("gs://").rstrip("/")
    dst = args.dst.removeprefix("gs://").rstrip("/")

    src_files = {p.removeprefix(src + "/"): i["size"] for p, i in fs.find(src, detail=True).items()}
    dst_files = {p.removeprefix(dst + "/"): i["size"] for p, i in fs.find(dst, detail=True).items()}
    pending = [rel for rel, size in src_files.items() if dst_files.get(rel) != size]
    total = sum(src_files[rel] for rel in pending)
    logger.info(
        "mirror: %d files (%.1f GB) to copy, %d already present",
        len(pending),
        total / 1e9,
        len(src_files) - len(pending),
    )

    def _copy(rel: str) -> None:
        fs.copy(f"{src}/{rel}", f"{dst}/{rel}")

    with ThreadPoolExecutor(max_workers=args.num_threads) as pool:
        for i, _ in enumerate(pool.map(_copy, pending), start=1):
            if i % 200 == 0:
                logger.info("  %d/%d", i, len(pending))

    # Verify: every source object present at the destination with an identical size.
    dst_after = {p.removeprefix(dst + "/"): i["size"] for p, i in fs.find(dst, detail=True).items()}
    mismatched = [rel for rel, size in src_files.items() if dst_after.get(rel) != size]
    if mismatched:
        raise AssertionError(f"{len(mismatched)} objects missing/size-mismatched after mirror: {mismatched[:5]}")
    logger.info("MIRROR_VERIFIED: %d objects, sizes identical", len(src_files))


if __name__ == "__main__":
    main()
