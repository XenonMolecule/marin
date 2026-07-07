# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prebuild the shared tokenized-doc cache for chunked ModernBERT training.

Tokenizing 200k+ docs on a TPU worker is single-threaded (HF disables tokenizer parallelism after the
JAX fork) and takes ~an hour PER run + on every preemption. This script tokenizes the docs ONCE in a
plain CPU job (no JAX → parallelism stays on → minutes) and writes the flat cache that every chunked
training run loads in seconds. The cache key excludes ctx/stride/seed, so a single cache serves the
entire base+large × ctx{512..8192} × {non-overlap,50%-overlap} sweep.

Run as a CPU Iris job IN-REGION (us-east5; data + cache must be local — never cross-region):

  uv run iris --cluster marin job run --region us-east5 \\
      --cpu 16 --memory 48GB --disk 50GB --enable-extra-resources --priority interactive --no-wait \\
      --job-name build-chunk-token-cache \\
      -e HF_TOKEN <token> -- \\
      python -m experiments.baseline_collection.build_chunk_token_cache
"""

import argparse
import logging

from levanter.main.train_classifier import _expand_globs, _token_cache_root, build_or_load_token_cache

from experiments.baseline_collection.launch_modernbert_levanter import MODEL_ID, TRAIN_GLOB

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", default=TRAIN_GLOB)
    p.add_argument("--train-rows", type=int, default=200_000)
    p.add_argument("--max-doc-tokens", type=int, default=32768)
    p.add_argument("--tokenizer", default=MODEL_ID)
    p.add_argument("--useful-label", default="__label__useful")
    args = p.parse_args()

    paths = _expand_globs([args.train_glob])
    root = _token_cache_root(None, paths, args.useful_label, args.train_rows, args.tokenizer, args.max_doc_tokens)
    logger.info(
        "cache root: %s (%d shards, %d rows, max_doc_tokens=%d)", root, len(paths), args.train_rows, args.max_doc_tokens
    )
    doc_ids, labels = build_or_load_token_cache(
        paths, args.useful_label, args.train_rows, args.tokenizer, args.max_doc_tokens, None
    )
    logger.info("DONE: %d docs cached; useful=%d", len(doc_ids), sum(labels))


if __name__ == "__main__":
    main()
