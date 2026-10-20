# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delete source-region ``kept/`` copies AFTER re-verifying byte-equality with the target.

For each requested region and spec: a file is deleted only when the consolidated target
holds the same basename with identical (size, CRC32C) — re-checked from fresh listings at
deletion time, independent of any earlier verification run. Files that mismatch (variant
duplicates from steal-mode double-processing) or are missing from the target are kept and
reported, never deleted.

    python -m experiments.fast_curation.delete_consolidated_sources --target-region us-east1 \\
        --regions us-west4 us-central2
"""

from __future__ import annotations

import argparse
import logging

import fsspec

from experiments.fast_curation.consolidate_kept import REGION_TO_BUCKET, SPEC_IDS
from experiments.fast_curation.verify_consolidated_kept import _listing

logger = logging.getLogger(__name__)

DELETE_BATCH = 2_000


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-region", default="us-east1", choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--regions", nargs="+", required=True, choices=sorted(REGION_TO_BUCKET))
    args = ap.parse_args()
    if args.target_region in args.regions:
        raise ValueError("refusing to delete the target region")

    fs = fsspec.filesystem("gcs")
    target = _listing(fs, "lpv11_fastpipe_v2_1_fused", args.target_region)
    logger.info("target holds %d kept files", len(target))
    if len(target) < 2_000_000:
        raise RuntimeError(f"target implausibly small ({len(target)}); refusing to delete anything")

    from experiments.fast_curation.spec import get_spec  # noqa: PLC0415

    for spec_id in SPEC_IDS:
        for region in args.regions:
            src = _listing(fs, spec_id, region)
            if not src:
                logger.info("%s %s: empty, nothing to do", spec_id, region)
                continue
            prefix = get_spec(spec_id).namespace(REGION_TO_BUCKET[region]).removeprefix("gs://") + "/kept"
            deletable = [bn for bn, meta in src.items() if target.get(bn) == meta]
            kept_back = len(src) - len(deletable)
            logger.info(
                "%s %s: %d files, %d verified-deletable, %d retained (variant/missing)",
                spec_id,
                region,
                len(src),
                len(deletable),
                kept_back,
            )
            for start in range(0, len(deletable), DELETE_BATCH):
                batch = [f"{prefix}/{bn}" for bn in deletable[start : start + DELETE_BATCH]]
                fs.rm(batch)
                if (start // DELETE_BATCH) % 25 == 0:
                    logger.info("  deleted %d/%d", min(start + DELETE_BATCH, len(deletable)), len(deletable))
            remaining = _listing(fs, spec_id, region)
            logger.info("%s %s: DONE — %d remaining (expected %d)", spec_id, region, len(remaining), kept_back)


if __name__ == "__main__":
    main()
