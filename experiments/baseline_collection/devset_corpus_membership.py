# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Look up which dev-set URLs survive into a curated 10k corpus (the deduped tree), by scanning its shards.

One Iris CPU job scans a slice of the tree's ``data-*.jsonl.gz`` shards and writes the matching rows
(url, record id, shard) to ``<output>/<tree>/part-<index>.jsonl``. Membership in the deduped tree is the
same "in the released corpus" semantics the dev set already records for DCLM, Nemotron and FineWeb
(``pipeline_labels.jsonl``: ``kept_*``).

    iris job run --region us-east5 --cpu 8 --memory 14GB --enable-extra-resources --extra cpu \\
        -- python -m experiments.baseline_collection.devset_corpus_membership --tree lpv11 \\
        --urls gs://.../devset_urls.txt --output gs://.../membership --shard-index 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from multiprocessing import Pool

import fsspec

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

TREES = {
    "lpv11": "gs://marin-us-east5/documents/baseline_llm_pipeline_v1_1_deduped/10364warcs/deduped",
    "fastpipe_v2": "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_decon_deduped/10364warcs/deduped",
}
URL_RE = re.compile(rb'"url":\s*"((?:[^"\\]|\\.)*)"')

_urls: set[str] = set()


def _init(urls: set[str]) -> None:
    global _urls
    _urls = urls


def scan_shard(path: str) -> list[dict]:
    hits = []
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rb") as f:
        for line in f:
            m = URL_RE.search(line)
            if m is None:
                continue
            url = json.loads(b'"' + m.group(1) + b'"')
            if url not in _urls:
                continue
            rec = json.loads(line)
            hits.append(
                {"url": url, "id": rec.get("warc_record_id") or rec.get("doc_id"), "shard": path.rsplit("/", 1)[1]}
            )
    return hits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", choices=sorted(TREES), required=True)
    parser.add_argument("--urls", required=True, help="gs:// text file, one URL per line")
    parser.add_argument("--output", required=True, help="gs:// directory")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--procs", type=int, default=8)
    args = parser.parse_args()

    with fsspec.open(args.urls, "rt", encoding="utf-8") as f:
        urls = {line.rstrip("\n") for line in f if line.strip()}
    shards = sorted(fsspec_glob(f"{TREES[args.tree]}/data-*.jsonl.gz"))[args.shard_index :: args.num_shards]
    logger.info("%s: %d urls, %d shards in this slice", args.tree, len(urls), len(shards))

    hits: list[dict] = []
    with Pool(args.procs, initializer=_init, initargs=(urls,)) as pool:
        for i, part in enumerate(pool.imap_unordered(scan_shard, shards), 1):
            hits.extend(part)
            if i % 100 == 0:
                logger.info("%d/%d shards, %d hits", i, len(shards), len(hits))
    out = f"{args.output.rstrip('/')}/{args.tree}/part-{args.shard_index:02d}.jsonl"
    with fsspec.open(out, "wt", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    logger.info("wrote %d hits to %s", len(hits), out)


if __name__ == "__main__":
    main()
