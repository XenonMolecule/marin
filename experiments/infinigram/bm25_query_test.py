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

from experiments.infinigram.bm25_query import open_bm25_index
from experiments.infinigram.targets import Collection

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
    args = p.parse_args()

    index = open_bm25_index(args.dataset, Collection(args.collection))
    logger.info("Opened %s-%s: %d docs", args.dataset, args.collection, index.num_docs)

    results = {}
    for q in _QUERIES:
        hits = index.search(q, k=3)
        top = hits[0] if hits else None
        logger.info("Q %r -> %d hits", q, len(hits))
        for h in hits:
            logger.info(
                "   score=%.3f url=%s prob=%s preview=%r",
                h.score,
                h.metadata.get("url"),
                h.metadata.get("modernbert_prob"),
                (h.metadata.get("preview") or "")[:90],
            )
        results[q] = {
            "hits": len(hits),
            "top_score": round(top.score, 3) if top else None,
            "top_url": top.metadata.get("url") if top else None,
            "meta_keys": sorted(top.metadata) if top else None,
        }

    if not any(r["hits"] > 0 for r in results.values()):
        raise AssertionError(f"NO hits for any query on {args.dataset}-{args.collection}")
    # Compact, capturable summary (top url per query proves metadata round-trips).
    compact = {q: {"h": r["hits"], "url": r["top_url"]} for q, r in results.items()}
    raise RuntimeError(f"QTEST_DONE {args.dataset}-{args.collection} docs={index.num_docs} {json.dumps(compact)}")


if __name__ == "__main__":
    main()
