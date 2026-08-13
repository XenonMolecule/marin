# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe the raw 10k resiliparse extraction to plan the 300-WARC subset: what WARC
key each doc carries, and whether each shard maps to a single WARC (so we can
select ~300 shards directly instead of scanning all ~400M docs). Raises a summary."""

import itertools
import json
import logging

from zephyr.readers import load_file

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

SRC = "gs://marin-us-central2/extracted/dclm_400m_1x_10k_resiliparse-f0887f/*.jsonl.gz"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    shards = sorted(fsspec_glob(SRC))
    out = {"n_shards": len(shards)}
    # Inspect the first 2 shards: keys, and how many distinct warc_file / warc-ish
    # values appear in the first 2000 docs of each (1 => shard maps to one WARC).
    for si in (0, 1, len(shards) // 2):
        recs = list(itertools.islice(load_file(shards[si]), 2000))
        if not recs:
            continue
        keys = sorted(recs[0].keys())
        warc_fields = [k for k in keys if "warc" in k.lower()]
        info = {"keys": keys[:20], "warc_fields": warc_fields, "n_sampled": len(recs)}
        for wf in warc_fields:
            vals = {str(r.get(wf)) for r in recs}
            info[f"distinct_{wf}"] = len(vals)
            info[f"sample_{wf}"] = next(iter(vals))[:120]
        out[f"shard_{si}"] = info
        logger.info("RESI_SCHEMA shard %d (%s): %s", si, shards[si].rsplit("/", 1)[1], json.dumps(info))
    raise RuntimeError(f"RESI_SCHEMA_DONE {json.dumps(out)[:1500]}")


if __name__ == "__main__":
    main()
