# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-cluster probe: how far along is an in-progress BM25 build? Lists the index
dir's uploaded sub-index shards and reads its _progress.json checkpoint. Raises
with a summary (finelog is down)."""

import json
import logging

import fsspec

from experiments.fsspec_paths import fsspec_exists, fsspec_glob

logger = logging.getLogger(__name__)

# Edit via args if needed; defaults to the resiliparse complete index.
INDEX_DIR = "gs://marin-us-central2/bm25_indices/full/resiliparse"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    shards = sorted({g.rsplit("/", 1)[0] for g in fsspec_glob(f"{INDEX_DIR}/shard_*/*")})
    prog = {}
    purl = f"{INDEX_DIR}/_progress.json"
    if fsspec_exists(purl):
        with fsspec.open(purl, "r") as f:
            p = json.load(f)
        prog = {"shards_done": p.get("shards_done"), "doc_id": p.get("doc_id"), "next_sub": p.get("next_sub")}
    raise RuntimeError(f"PROGRESS uploaded_sub_indices={len(shards)} checkpoint={json.dumps(prog)}")


if __name__ == "__main__":
    main()
