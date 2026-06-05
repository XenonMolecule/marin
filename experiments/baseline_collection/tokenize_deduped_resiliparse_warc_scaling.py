# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the Marin-deduped tree produced by ``dedup_resiliparse_warc_scaling.py``.

Reads
    gs://marin-{region}/documents/baseline_resiliparse_deduped/{n}warcs/deduped/data-*.jsonl.gz
Writes
    gs://marin-{region}/tokenized/resiliparse_dedup_{n}warcs-{cache_hash}/

Uses llama3 tokenizer (matches every other curation baseline in
``experiments/scaling_law_sweeps/curation_plan.py``). The dataset name
``resiliparse_dedup_{n}warcs`` is the canonical entry to paste into
``_D_OBS_DEFAULTS`` and ``_method()`` after capturing the resolved cache hash.

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 2 --memory 8GB --enable-extra-resources \\
        --priority interactive \\
        --extra cpu \\
        --region us-east5 \\
        --job-name tokenize-resiliparse-dedup-{n}warcs \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/tokenize_deduped_resiliparse_warc_scaling.py \\
           --n {n} --region us-east5
"""

from __future__ import annotations

import argparse
import sys

from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

DEFAULT_DEDUPED_NAME = "baseline_resiliparse_deduped"


def _build_step(n: int, region: str, deduped_name: str, dataset_name: str):
    deduped_glob = f"gs://marin-{region}/documents/{deduped_name}/{n}warcs/deduped/data-*.jsonl.gz"
    return default_tokenize(
        name=dataset_name,
        dataset=InputName.hardcoded(deduped_glob),
        tokenizer=llama3_tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--region",
        default="us-east5",
        help="Region bucket suffix (e.g. 'us-east5'). Must match where dedup ran.",
    )
    parser.add_argument(
        "--deduped-name",
        default=DEFAULT_DEDUPED_NAME,
        help="Deduped output dir under <bucket>/documents/ "
        f"(default: {DEFAULT_DEDUPED_NAME}). Must match dedup's --output-name.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Tokenized cache / curation_plan registration name " "(default: resiliparse_dedup_{n}warcs).",
    )
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0], *unknown]

    dataset_name = args.dataset_name or f"resiliparse_dedup_{args.n}warcs"
    step = _build_step(args.n, args.region, args.deduped_name, dataset_name)
    executor_main(
        steps=[step],
        description=(
            f"Tokenize Marin-deduped {args.n}-WARC resiliparse ({args.deduped_name}) "
            f"into Levanter cache under tokenized/{dataset_name}-*."
        ),
    )


if __name__ == "__main__":
    main()
