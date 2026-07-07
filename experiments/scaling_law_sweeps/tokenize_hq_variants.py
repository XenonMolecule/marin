# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the high_quality corpus VARIANTS (built by build_hq_variants.py) into
llama3 Levanter caches for the knowledge-gap dilution ablation.

Two variants, both built from the hq HF-export docs in us-central1:

  ``dense``  gs://marin-us-central1/documents/hq_variants/dense/*.jsonl.gz
             → tokenized/hq_dense-{hash}/  (the UPWEIGHT component, ~2.6B tokens)

  ``epoch``  gs://marin-us-central1/documents/hq_variants/epoch_sub344/*.jsonl.gz
             → tokenized/hq_epoch_sub344-{hash}/  (~7.33B tokens ≈ DCLM count)

llama3 tokenizer to match every curation baseline; the ``text`` field is tokenized.
Output region is set by MARIN_PREFIX (pass ``-e MARIN_PREFIX=gs://marin-us-central1``
and ``--region us-central1`` so the cache is co-located with the hq cache it mixes
with). After completion read ``{cache}/train/.stats.json:total_tokens`` and register
in ``curation_plan`` (see .agents/projects/pipeline_provenance_devset.md).

The variant is selected by the ``HQ_VARIANT`` env var (``dense`` or ``epoch``) —
NOT a CLI flag, because ``executor_main`` owns argv and rejects unknown flags.

Usage (CPU coordinator on Iris, us-central1)::

    uv run iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 8 --memory 32GB --enable-extra-resources \\
        --priority interactive --extra cpu \\
        -e MARIN_PREFIX gs://marin-us-central1 -e HQ_VARIANT dense \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/tokenize_hq_variants.py
"""

from __future__ import annotations

import os

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

VARIANTS = {
    "dense": ("hq_dense", "gs://marin-us-central1/documents/hq_variants/dense/*.jsonl.gz"),
    "epoch": ("hq_epoch_sub344", "gs://marin-us-central1/documents/hq_variants/epoch_sub344/*.jsonl.gz"),
}


def main() -> None:
    variant = os.environ.get("HQ_VARIANT")
    if variant not in VARIANTS:
        raise SystemExit(f"set HQ_VARIANT to one of {list(VARIANTS)} (got {variant!r})")
    name, glob = VARIANTS[variant]
    step = default_tokenize(
        name=name,
        dataset=InputName.hardcoded(glob),
        tokenizer=llama3_tokenizer,
        format=TextLmDatasetFormat(),
    )
    executor_main(
        steps=[step],
        description=f"Tokenize hq variant '{variant}' ({name}) into a llama3 Levanter cache for the dilution ablation.",
    )


if __name__ == "__main__":
    main()
