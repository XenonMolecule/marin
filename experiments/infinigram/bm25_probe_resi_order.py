# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify resiliparse extraction shard N == pool WARC N (so the 300-WARC subset is
a positional shard selection, not a 400M-doc scan).

Resiliparse docs carry only {text,url}; the WARC-per-doc lives in the metadata
table ``metadata/dclm_400m_1x_10k_warc_metadata-79158f`` ({url,warc_file,...}). For
a few shard indices N we check: does metadata shard N cover exactly one warc_file
== pool[N], and do resiliparse shard N's urls fall inside metadata shard N? If yes
for all probes, shard order == pool order and we can select shards by position.
"""

import itertools
import json
import logging

from marin.utils import fsspec_glob
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

RESI = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"
META = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/*.jsonl.gz"
POOL_FILE = "experiments/distill/dclm_400m_1x.txt"


def _pool() -> list[str]:
    with open(POOL_FILE) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    pool = _pool()
    resi = sorted(fsspec_glob(RESI))
    meta = sorted(fsspec_glob(META))
    out = {"pool": len(pool), "resi_shards": len(resi), "meta_shards": len(meta), "probes": {}}
    for n in (0, 1, 100, 5182, len(resi) - 1):
        if n >= len(resi) or n >= len(meta):
            continue
        meta_recs = list(load_file(meta[n]))
        meta_warcs = {str(r.get("warc_file")) for r in meta_recs}
        meta_urls = {r.get("url") for r in meta_recs}
        resi_urls = [r.get("url") for r in itertools.islice(load_file(resi[n]), 50)]
        in_meta = sum(1 for u in resi_urls if u in meta_urls)
        poolwarc = pool[n] if n < len(pool) else None
        # metadata warc_file may be an s3:// path or basename; compare by basename.
        mw = next(iter(meta_warcs)) if len(meta_warcs) == 1 else f"MULTI({len(meta_warcs)})"
        match = poolwarc is not None and mw != "" and (mw.rsplit("/", 1)[-1] == poolwarc.rsplit("/", 1)[-1])
        out["probes"][n] = {
            "meta_distinct_warc": len(meta_warcs),
            "meta_warc": mw.rsplit("/", 1)[-1][:60],
            "pool_warc": poolwarc.rsplit("/", 1)[-1][:60] if poolwarc else None,
            "warc_matches_pool": match,
            "resi_urls_in_meta": f"{in_meta}/{len(resi_urls)}",
        }
        logger.info("ORDER probe n=%d: %s", n, json.dumps(out["probes"][n]))
    raise RuntimeError(f"RESI_ORDER_DONE {json.dumps(out)[:1600]}")


if __name__ == "__main__":
    main()
