# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the Marin-deduped tree produced by ``dedup_extracted.py``.

Reads
    gs://marin-us-central1/documents/baseline_{spec}_deduped/{n}warcs/deduped/data-*.jsonl.gz
Writes
    gs://marin-us-central1/tokenized/{spec}_{n}warcs-{cache_hash}/

Uses llama3 tokenizer (matches every other curation baseline in
``experiments/scaling_law_sweeps/curation_plan.py``). The dataset name
``{spec}_{n}warcs`` is the canonical entry to paste into ``_D_OBS_DEFAULTS``
and ``_method()``.

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 2 --memory 8GB --enable-extra-resources \\
        --priority interactive \\
        --extra cpu \\
        --job-name tokenize-{spec}-{n}warcs \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/tokenize_deduped_extracted.py \\
           --spec {spec} --n {N}
"""

from __future__ import annotations

import argparse
import sys

from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer


def _build_step(spec: str, n: int, region: str):
    deduped_glob = f"gs://marin-{region}/documents/baseline_{spec}_deduped/{n}warcs/deduped/data-*.jsonl.gz"
    return default_tokenize(
        name=f"{spec}_{n}warcs",
        dataset=InputName.hardcoded(deduped_glob),
        tokenizer=llama3_tokenizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument(
        "--region",
        default="us-east5",
        help="Region bucket suffix (e.g. 'us-east5', 'eu-west4'). Default us-east5.",
    )
    args, unknown = parser.parse_known_args()
    # Hand the unknowns to draccus (executor_main is wrapped in @draccus.wrap()).
    sys.argv = [sys.argv[0]] + unknown

    step = _build_step(args.spec, args.n, args.region)
    executor_main(
        steps=[step],
        description=(
            f"Tokenize Marin-deduped first-{args.n} WARCs of spec={args.spec!r} "
            f"into Levanter cache under tokenized/{args.spec}_{args.n}warcs-*."
        ),
    )


if __name__ == "__main__":
    main()
