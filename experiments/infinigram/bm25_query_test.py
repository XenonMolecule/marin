# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end at-rest query test for a built BM25 index (run as an in-region job).

Opens the uploaded index from GCS (mirror -> mmap), runs several varied queries,
and asserts each returns ranked hits carrying usable metadata (url when inline/
joined, else doc_id). Prints the top hits and raises with a compact summary so the
result is visible via ``iris job bug-report`` while finelog is down.
"""

import argparse
import json
import logging

import fsspec

from experiments.infinigram.bm25_query import open_bm25_index
from experiments.infinigram.bm25_sources import get_bm25_target
from experiments.infinigram.targets import REGION_BUCKET, Collection

logger = logging.getLogger(__name__)

# Varied topical probes -- generic enough to hit any web corpus.
_QUERIES = [
    "climate change policy",
    "machine learning neural network",
    "how to bake sourdough bread",
    "supreme court constitutional ruling",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--collection", choices=[c.value for c in Collection], default=Collection.SMALL.value)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--count-docs", action="store_true", help="Also compute total num_docs (loads all sub-indices).")
    args = p.parse_args()

    target = get_bm25_target(args.dataset, Collection(args.collection))
    bucket = REGION_BUCKET[target.region]
    base = f"{bucket}/bm25_qtest_results/{args.dataset}-{args.collection}"

    def crumb(step: str) -> None:
        # Overwrite a tiny progress file each step, so an OOM's last-known step is visible.
        with fsspec.open(f"{base}.progress.txt", "w") as f:
            f.write(step)
        logger.info("STEP %s", step)

    crumb("start")
    index = open_bm25_index(args.dataset, Collection(args.collection), batch_size=args.batch_size)
    crumb(f"localized batch_size={args.batch_size}")

    results = {}
    for q in _QUERIES:
        hits = index.search(q, k=3)
        top = hits[0] if hits else None
        results[q] = {
            "hits": len(hits),
            "top_score": round(top.score, 3) if top else None,
            "top_url": top.metadata.get("url") if top else None,
        }
        crumb(f"queried {q!r} -> {len(hits)} hits, url={results[q]['top_url']}")

    if not any(r["hits"] > 0 for r in results.values()):
        raise AssertionError(f"NO hits for any query on {args.dataset}-{args.collection}")

    num_docs = index.num_docs if args.count_docs else None
    # Write results to GCS (in-region) so they survive bm25s's atexit stderr, which
    # otherwise clobbers the last line captured by bug-report.
    payload = {"dataset": args.dataset, "collection": args.collection, "num_docs": num_docs, "queries": results}
    with fsspec.open(f"{base}.json", "w") as f:
        json.dump(payload, f, indent=2)
    crumb("done")


if __name__ == "__main__":
    main()
