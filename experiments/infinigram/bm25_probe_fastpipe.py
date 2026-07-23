# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the fastpipe_v3 300-WARC band document tiers (us-east5) for BM25.

The 5 keep-top-X% bands are written by threshold_split.py to
``documents/baseline_fastpipe_v3_{pct}_decon_deduped/300warcs/deduped/`` (100% is
the un-suffixed input). Confirm each resolves and whether ``url`` is inline (else
a provenance join is needed). Run in-region (us-east5). Raises with a summary.
"""

import itertools
import json
import logging

from marin.utils import fsspec_glob
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

_BASE = "gs://marin-us-east5/documents"
CANDIDATES: dict[str, str] = {
    "fastpipe_v3_100": f"{_BASE}/baseline_fastpipe_v3_decon_deduped/300warcs/deduped/data-*.jsonl.gz",
    "fastpipe_v3_80": f"{_BASE}/baseline_fastpipe_v3_80_decon_deduped/300warcs/deduped/data-*.jsonl.gz",
    "fastpipe_v3_60": f"{_BASE}/baseline_fastpipe_v3_60_decon_deduped/300warcs/deduped/data-*.jsonl.gz",
    "fastpipe_v3_40": f"{_BASE}/baseline_fastpipe_v3_40_decon_deduped/300warcs/deduped/data-*.jsonl.gz",
    "fastpipe_v3_20": f"{_BASE}/baseline_fastpipe_v3_20_decon_deduped/300warcs/deduped/data-*.jsonl.gz",
}


def _probe(glob: str) -> dict:
    shards = fsspec_glob(glob)
    info: dict = {"n": len(shards)}
    if shards:
        try:
            rec = next(itertools.islice(load_file(sorted(shards)[0]), 1))
            info["keys"] = sorted(rec.keys())[:18]
            info["has_url"] = "url" in rec
            info["has_text"] = bool(rec.get("text") or rec.get("generated_text"))
        except Exception as e:
            info["read_err"] = str(e)[:80]
    return info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    out = {label: _probe(glob) for label, glob in CANDIDATES.items()}
    for label, info in out.items():
        logger.info("FASTPIPE %s -> %s", label, json.dumps(info))
    compact = {k: {"n": v["n"], "url": v.get("has_url"), "text": v.get("has_text")} for k, v in out.items()}
    raise RuntimeError(f"FASTPIPE_DONE {json.dumps(compact)}")


if __name__ == "__main__":
    main()
