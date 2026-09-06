# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tokenize the deduped+deconned lpv11_fastpipe_v1 corpus into a Levanter cache (llama3).

Post-merge replacement for ``tokenize_deduped_extracted.py``, which was built on the removed
``marin.execution.executor`` framework (``default_tokenize``/``executor_main``). This calls the
current engine — ``marin.processing.tokenize.tokenize.tokenize`` — directly: it expands the input
glob, fans out its own Zephyr fleet, writes the Levanter cache, and records token totals in
``train/stats.json``.

The cache directory name is EXPLICIT (no executor fingerprint anymore) and is exactly what gets
registered in ``curation_plan._D_OBS_DEFAULTS`` / ``_method()``. Convention matches the other
curation methods: ``tokenized/{method}_decon_{n}warcs-{tag}``.

Run as an in-region iris CPU job (the Zephyr fleet does the work; this process coordinates)::

    uv run iris --cluster marin job run --no-wait --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources --region us-east5 \\
        --job-name fastcur-tokenize-lpv11 -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -e MARIN_PREFIX gs://marin-us-east5 \\
        -- python -m experiments.fast_curation.tokenize_lpv11
"""
from __future__ import annotations

import argparse
import json
import logging

import fsspec
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"  # matches every other curation baseline
DECON_GLOB = "gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v1_decon_deduped/10364warcs/deduped/data-*.jsonl.gz"
# Explicit cache dir (the executor fingerprint no longer exists); the -a16e729 tag is the upstream
# parity-merge commit this was built under, recorded for provenance.
CACHE_PATH = "gs://marin-us-east5/tokenized/lpv11_fastpipe_v1_decon_10364warcs-a16e729"


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-glob", default=DECON_GLOB)
    ap.add_argument("--cache-path", default=CACHE_PATH)
    ap.add_argument("--tokenizer", default=LLAMA3_TOKENIZER)
    args = ap.parse_args()

    config = TokenizeConfig(
        train_paths=[args.input_glob],
        validation_paths=[],
        cache_path=args.cache_path,
        tokenizer=args.tokenizer,
    )
    tokenize(config)

    stats_path = f"{args.cache_path.rstrip('/')}/train/.stats.json"  # dot-prefixed by write_stats_json
    with fsspec.open(stats_path, "r") as f:
        stats = json.load(f)
    logger.info("TOKENIZE COMPLETE: cache=%s stats=%s", args.cache_path, json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
