# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the streaming, 10M-scale tokenized cache for CHUNKED ModernBERT training.

Unlike the flat in-RAM cache (``build_chunk_token_cache.py``, fine to ~1M), this builds a Levanter
``TreeCache`` — tokens stored on disk, read BY INDEX at train time, so host RAM stays bounded at any
scale. It is the same cache type the 10M *truncated* runs used, but with the token cap raised from
8192 → 32768 so chunks can see PAST the first 8192 tokens (the whole point of chunking). One cache
serves every ctx/overlap/seed and both base+large (shared tokenizer). The build is parallel +
resumable (commits shards), so a preemption resumes instead of restarting.

Run as a big CPU Iris job IN-REGION (us-east5; data + cache local — never cross-region):

  uv run iris --cluster marin job run --region us-east5 \\
      --cpu 64 --memory 128GB --disk 100GB --enable-extra-resources --priority interactive --no-wait \\
      --job-name build-chunk-treecache-10m -e HF_TOKEN <token> -- \\
      python -m experiments.baseline_collection.build_chunk_treecache
"""

import argparse
import logging

from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.main.train_classifier import ClassificationLineProcessor, _expand_globs
from levanter.store.cache import CacheOptions, build_or_load_cache
from transformers import AutoTokenizer

from experiments.baseline_collection.launch_modernbert_levanter import MODEL_ID

logger = logging.getLogger(__name__)

# The ~10M random-survivor preshard (one set serves all sizes via --train-rows), in us-east5.
TRAIN_GLOB_10M = "gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_random_par/train_shard_*.txt.gz"
# Distinct dir from the existing 8192-cap cache (_clf_token_cache) so they never collide.
CACHE_DIR_32768 = (
    "gs://marin-us-east5/classifiers/useful_fasttext/presharded_survivor_random_par/_clf_token_cache_chunk32768"
)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", default=TRAIN_GLOB_10M)
    p.add_argument("--cache-dir", default=CACHE_DIR_32768)
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--tokenizer", default=MODEL_ID)
    p.add_argument("--useful-label", default="__label__useful")
    args = p.parse_args()

    paths = _expand_globs([args.train_glob])
    if not paths:
        raise ValueError(f"no shards matched {args.train_glob}")
    logger.info("building %d-token TreeCache from %d shards -> %s", args.max_length, len(paths), args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source = TextUrlDataSource(paths)
    processor = ClassificationLineProcessor(tokenizer, args.useful_label, max_length=args.max_length)
    cache = build_or_load_cache(args.cache_dir, source, processor, options=CacheOptions.default())
    logger.info("DONE: TreeCache built at %s (%d rows)", args.cache_dir, len(cache))


if __name__ == "__main__":
    main()
