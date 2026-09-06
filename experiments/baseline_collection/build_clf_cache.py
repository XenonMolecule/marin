# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""One-time builder for the classifier token cache (so the sweep cells load, not race-build it).

Tokenizes the fastText shards into a streaming TreeCache via the same Zephyr pipeline the LM path
uses. Run as a CPU iris job in the data's region; all train_classifier runs then load this cache.
"""
import argparse
import logging

from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.main.train_classifier import ClassificationLineProcessor, _expand_globs
from levanter.store.cache import CacheOptions, build_or_load_cache
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-glob", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--useful-label", default="__label__useful")
    a = ap.parse_args()

    paths = _expand_globs([a.train_glob])
    print(f"building cache from {len(paths)} shards -> {a.cache_dir}", flush=True)
    source = TextUrlDataSource(paths)
    processor = ClassificationLineProcessor(AutoTokenizer.from_pretrained(a.tokenizer), a.useful_label, a.max_length)
    cache = build_or_load_cache(a.cache_dir, source, processor, options=CacheOptions.default())
    print(f"CACHE DONE finished={cache.is_finished} -> {a.cache_dir}", flush=True)


if __name__ == "__main__":
    main()
