# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate the RAW (complete, unfiltered) resiliparse extraction for the BM25
'complete lookup' index, and confirm it carries ``url`` inline. Also probe for a
raw 300-WARC resiliparse subset. Raises with a compact summary (finelog is down).
"""

import itertools
import json
import logging

from marin.utils import fsspec_glob
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

CANDIDATES: dict[str, str] = {
    "resiliparse_raw_10k": "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz",
    "resiliparse_raw_300_extracted": "gs://marin-us-central2/extracted/dclm_400m_1x_300_resiliparse-*/*.jsonl.gz",
    "resiliparse_raw_300_subset": (
        "gs://marin-us-central2/filtered_subsets/resiliparse_random_300warcs-*/data-*.jsonl.gz"
    ),
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
        logger.info("RESI2 %s -> %s", label, json.dumps(info))
    compact = {k: {"n": v["n"], "url": v.get("has_url"), "text": v.get("has_text")} for k, v in out.items()}
    raise RuntimeError(f"RESI2_DONE {json.dumps(compact)}")


if __name__ == "__main__":
    main()
