# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count docs and text chars across a deduped JSONL output (run in-region).

One process pool over the shard list; per-shard counts stream to a JSON summary on GCS so
the result survives the job. Token estimates apply a measured chars-per-token ratio.

    python -m experiments.fast_curation.count_deduped_docs \
        --deduped-glob "gs://<bucket>/documents/baseline_<spec>_deduped/<n>warcs/deduped/*.jsonl.gz" \
        --output gs://<bucket>/users/<user>/analysis/deduped_token_count.json
"""

import argparse
import gzip
import json
import logging
from concurrent.futures import ProcessPoolExecutor

import fsspec

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# Measured on this corpus by the pre-dedup baseline (4,800 llama3-tokenized docs).
LLAMA3_CHARS_PER_TOKEN = 4.302


def _count_shard(path: str) -> tuple[int, int]:
    docs = chars = 0
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            docs += 1
            chars += len(json.loads(line)["text"])
    return docs, chars


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deduped-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=90)
    args = ap.parse_args()

    paths = sorted(fsspec_glob(args.deduped_glob))
    if not paths:
        raise FileNotFoundError(f"no shards match {args.deduped_glob}")
    logger.info("Counting %d shards with %d workers", len(paths), args.workers)

    docs = chars = done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for d, c in pool.map(_count_shard, paths, chunksize=4):
            docs += d
            chars += c
            done += 1
            if done % 500 == 0:
                logger.info("%d/%d shards: %d docs, %d chars", done, len(paths), docs, chars)

    tokens = chars / LLAMA3_CHARS_PER_TOKEN
    payload = {
        "deduped_glob": args.deduped_glob,
        "n_shards": len(paths),
        "docs": docs,
        "chars": chars,
        "chars_per_doc": chars / docs,
        "llama3_chars_per_token": LLAMA3_CHARS_PER_TOKEN,
        "llama3_tokens_est": tokens,
    }
    with fsspec.open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("RESULT %s", json.dumps(payload))


if __name__ == "__main__":
    main()
