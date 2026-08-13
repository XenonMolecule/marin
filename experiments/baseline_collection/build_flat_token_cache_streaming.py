# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the flat tokenized-doc cache for chunked training WITHOUT holding the corpus in RAM.

``build_chunk_token_cache.py`` (via ``build_or_load_token_cache``) reads EVERY doc's text into a
list, then hands 1/N-sized slices to a ProcessPool — which pickles the text to each worker. On the
lpv11 survivor corpus (~33 KB/doc untruncated) 1M docs is ~33 GB of text, and parent + pickle
buffers + worker copies peak well past 100 GB: both 1M builds were OOM-killed at 160 GB.

Tokens are ~10x smaller than the text they came from, so this builder streams SHARD BY SHARD:
read one shard's texts (~25k docs), tokenize them in parallel, keep only the int32 token arrays,
free the texts, repeat. Peak RSS is (all tokens so far) + (one shard of text), not (all text) x3.

Output is byte-identical to the non-streaming builder — ``ids.npy`` / ``offsets.npy`` /
``labels.npy`` + a ``_DONE`` marker at the SAME ``_token_cache_root``-derived path — so
``build_or_load_token_cache`` (and therefore every chunked training run) loads it transparently.

Run as a CPU Iris job IN-REGION (us-east5):

  uv run iris --cluster marin job run --region us-east5 \\
      --cpu 20 --memory 160GB --disk 50GB --enable-extra-resources --priority interactive --no-wait \\
      --job-name build-flat-cache-mb-1M -e HF_TOKEN <token> -- \\
      python -m experiments.baseline_collection.build_flat_token_cache_streaming \\
        --train-glob 'gs://.../train_shard_*.txt.gz' --train-rows 1000000
"""

import argparse
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor

import fsspec
import numpy as np
from levanter.main.train_classifier import _expand_globs, _parse_line, _save_token_cache, _token_cache_root

logger = logging.getLogger(__name__)

# Chars per doc handed to the tokenizer. Matches ClassificationLineProcessor: 100k chars is ~12x the
# chars needed for the 32768-token cap, so the stored ids are identical to no-cap.
MAX_TEXT_CHARS = 100_000


def _tokenize_batch(payload: tuple[str, list[str], int]) -> list[np.ndarray]:
    """Worker entrypoint: tokenize one batch of texts in a fresh process (picklable, top-level)."""
    tokenizer_name, texts, max_doc_tokens = payload
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tok(texts, truncation=True, max_length=max_doc_tokens)["input_ids"]
    return [np.asarray(e, dtype=np.int32) for e in enc]


def _read_shard(path: str, useful_label: str, limit: int | None) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = _parse_line(line, useful_label)
            if parsed is None:
                continue
            labels.append(parsed[0])
            texts.append(parsed[1][:MAX_TEXT_CHARS])
            if limit is not None and len(texts) >= limit:
                break
    return texts, labels


def build_streaming(
    paths: list[str], useful_label: str, limit: int | None, tokenizer_name: str, max_doc_tokens: int, workers: int
) -> tuple[list[np.ndarray], list[int]]:
    doc_ids: list[np.ndarray] = []
    labels: list[int] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(paths):
            remaining = None if limit is None else limit - len(doc_ids)
            if remaining is not None and remaining <= 0:
                break
            texts, shard_labels = _read_shard(path, useful_label, remaining)
            if not texts:
                continue
            # Submit in worker-sized batches; each pickled payload is one batch of text, not 1/N of
            # the corpus, so in-flight memory stays bounded.
            batch = max(1, math.ceil(len(texts) / workers))
            payloads = [(tokenizer_name, texts[j : j + batch], max_doc_tokens) for j in range(0, len(texts), batch)]
            for chunk in pool.map(_tokenize_batch, payloads):  # map preserves order
                doc_ids.extend(chunk)
            labels.extend(shard_labels)
            del texts, payloads
            logger.info("shard %d/%d (%s): %d docs cumulative", i + 1, len(paths), path.rsplit("/", 1)[-1], len(doc_ids))
    return doc_ids, labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-glob", required=True)
    p.add_argument("--train-rows", type=int, default=1_000_000)
    p.add_argument("--max-doc-tokens", type=int, default=32768)
    p.add_argument("--tokenizer", default="answerdotai/ModernBERT-base")
    p.add_argument("--useful-label", default="__label__useful")
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Tokenizer processes. Host cpu_count is the TPU host's (~180), not the request — cap it.",
    )
    args = p.parse_args()

    paths = _expand_globs([args.train_glob])
    if not paths:
        raise ValueError(f"no shards matched {args.train_glob}")
    root = _token_cache_root(None, paths, args.useful_label, args.train_rows, args.tokenizer, args.max_doc_tokens)
    fs = fsspec.core.url_to_fs(f"{root}/_DONE")[0]
    if fs.exists(f"{root}/_DONE"):
        logger.info("cache already complete: %s", root)
        return

    logger.info(
        "streaming build: %d shards -> %s (rows=%s, workers=%d)", len(paths), root, args.train_rows, args.workers
    )
    doc_ids, labels = build_streaming(
        paths, args.useful_label, args.train_rows, args.tokenizer, args.max_doc_tokens, args.workers
    )
    total_tokens = sum(len(d) for d in doc_ids)
    logger.info(
        "tokenized %d docs (%d tokens, ~%.1f GB) -> writing cache", len(doc_ids), total_tokens, total_tokens * 4 / 1e9
    )
    _save_token_cache(root, doc_ids, labels)
    logger.info("DONE: %d docs cached at %s; useful=%d", len(doc_ids), root, sum(labels))


if __name__ == "__main__":
    main()
