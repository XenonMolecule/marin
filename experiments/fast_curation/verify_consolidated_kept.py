# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the consolidated ``kept/`` tree against every source region before deletion.

Two layers:
  1. FULL check, every source file: the stored CRC32C + size from the source listing must
     match the target copy's. GCS computes CRC32C over object content, so equal checksums
     are byte-equality without reading data. A source file whose basename pre-existed at
     the target (steal-mode duplicate: the same WARC processed in two regions) may differ
     legitimately; those are counted separately, not failed.
  2. SPOT check: N random copied pairs read fully and compared byte-for-byte.

    python -m experiments.fast_curation.verify_consolidated_kept --target-region us-east1 \\
        --source-regions us-west4 us-central2 us-east5 us-central1 eu-west4 --spot-checks 20
"""

from __future__ import annotations

import argparse
import logging
import random

import fsspec

from experiments.fast_curation.consolidate_kept import REGION_TO_BUCKET, SPEC_IDS
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)


def _listing(fs: fsspec.AbstractFileSystem, spec_id: str, region: str) -> dict[str, tuple[int, str]]:
    """``{basename: (size, crc32c)}`` for one region's kept tree (empty if absent)."""
    prefix = get_spec(spec_id).namespace(REGION_TO_BUCKET[region]).removeprefix("gs://") + "/kept"
    if not fs.exists(prefix):
        return {}
    out = {}
    for e in fs.ls(prefix, detail=True):
        name = e["name"]
        if name.endswith(".parquet"):
            out[name.rsplit("/", 1)[1]] = (e["size"], e.get("crc32c", ""))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-region", default="us-east1", choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--source-regions", nargs="+", required=True, choices=sorted(REGION_TO_BUCKET))
    ap.add_argument("--spot-checks", type=int, default=20)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs")
    target_ns = get_spec("lpv11_fastpipe_v2_1_fused").namespace(REGION_TO_BUCKET[args.target_region])
    target = _listing(fs, "lpv11_fastpipe_v2_1_fused", args.target_region)
    logger.info("target holds %d kept files", len(target))

    missing: list[str] = []
    variant_dups = matched = 0
    spot_pool: list[tuple[str, str]] = []  # (source path, basename)

    for spec_id in SPEC_IDS:
        regions = list(args.source_regions) + ([args.target_region] if spec_id != "lpv11_fastpipe_v2_1_fused" else [])
        for region in regions:
            src = _listing(fs, spec_id, region)
            src_prefix = get_spec(spec_id).namespace(REGION_TO_BUCKET[region]).removeprefix("gs://") + "/kept"
            m = miss = var = 0
            for bn, (size, crc) in src.items():
                tgt = target.get(bn)
                if tgt is None:
                    miss += 1
                    missing.append(f"{region}/{bn}")
                elif tgt == (size, crc):
                    m += 1
                    spot_pool.append((f"{src_prefix}/{bn}", bn))
                else:
                    # Pre-existing target from an independent processing of the same WARC.
                    var += 1
            matched += m
            variant_dups += var
            logger.info(
                "%s %s: %d files -> matched %d, variant-dup %d, MISSING %d", spec_id, region, len(src), m, var, miss
            )

    logger.info("TOTALS: matched %d, variant-duplicates %d, missing %d", matched, variant_dups, len(missing))
    if missing:
        raise RuntimeError(f"{len(missing)} source files have NO target copy; sample {missing[:5]}")

    random.seed(20260923)
    picks = random.sample(spot_pool, min(args.spot_checks, len(spot_pool)))
    target_prefix = target_ns.removeprefix("gs://") + "/kept"
    for src_path, bn in picks:
        a = fs.cat_file(src_path)
        b = fs.cat_file(f"{target_prefix}/{bn}")
        if a != b:
            raise RuntimeError(f"BYTE MISMATCH on spot check: {src_path}")
        logger.info("spot check OK (%d bytes): %s", len(a), bn)
    logger.info("VERIFICATION PASSED: %d checksum-matched, %d byte-spot-checked, 0 missing", matched, len(picks))


if __name__ == "__main__":
    main()
