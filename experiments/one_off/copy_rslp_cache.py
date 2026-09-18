# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off: server-side copy of resiliparse caches us-east5 -> us-central2.

Uses the GCS rewrite API via gcsfs (per-object server-side copy; no bytes through this
VM). Idempotent: objects whose destination exists with matching size are skipped, so
reruns resume cleanly. Used 2026-09-17 for the tokenized cache ($14) and, after user
approval of the $40 total, the 1.28 TB grid store ($26) to unlock rslp training in
us-central2.
"""

import argparse
import concurrent.futures
import logging

import fsspec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("copy")

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--src", default="marin-us-east5/tokenized/resiliparse_decon_10364warcs-beaaf5")
ap.add_argument("--dst", default="marin-us-central2/tokenized/resiliparse_decon_10364warcs-beaaf5")
args = ap.parse_args()
SRC, DST = args.src.rstrip("/"), args.dst.rstrip("/")

fs = fsspec.filesystem("gcs")
src_files = fs.find(SRC)
log.info("source objects: %d (%s -> %s)", len(src_files), SRC, DST)
dst_info = {p[len(DST) :]: i.get("size") for p, i in fs.find(DST, detail=True).items()} if fs.exists(DST) else {}


def one(p):
    rel = p[len(SRC) :]
    if dst_info.get(rel) == fs.info(p).get("size"):
        return 0
    fs.copy(p, DST + rel)
    return 1


done = skipped = 0
with concurrent.futures.ThreadPoolExecutor(16) as ex:
    for i, r in enumerate(ex.map(one, src_files), 1):
        done += r
        skipped += 1 - r
        if i % 200 == 0:
            log.info("progress %d/%d (copied %d, skipped %d)", i, len(src_files), done, skipped)
log.info("COPY COMPLETE: copied %d, skipped %d, total %d", done, skipped, len(src_files))
