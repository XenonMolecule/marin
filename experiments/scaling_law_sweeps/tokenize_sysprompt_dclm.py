# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the system-prompt-conditioned DCLM dataset into a llama3 Levanter cache.

Source: the [S][D] conditional-pretraining dataset produced by the system-prompt
labeling run (Qwen3-30B-A3B over the full DCLM 400m-1x corpus, 5,918,974 records):

    gs://marin-us-east5/sysprompt_pretrain/dclm_400m_1x/qfull1/training/train-*.jsonl.gz

Each record has BOTH a bare ``text`` field (the document D) and a
``conditioned_text`` field (= ``<|start_header_id|>system<|end_header_id|>\\n\\n{S}<|eot_id|>{D}``).
We tokenize ``conditioned_text`` so the model trains on the [S][D] sequence, not
the bare document — hence ``TextLmDatasetFormat(text_key="conditioned_text")``.

Uses the llama3 tokenizer to match every curation baseline (the special-token
wrapper around S is exactly the llama3 header markers, which the tokenizer maps to
their reserved IDs). Output:

    gs://marin-us-east5/tokenized/sysprompt_dclm_qfull1-{cache_hash}/

After it completes, read ``{cache}/train/.stats.json:total_tokens`` and paste it
into ``curation_plan._D_OBS_DEFAULTS`` + register a ``sysprompt_dclm`` method, then
launch one d1024 standalone child for a single natural epoch (see the session plan).

Usage (CPU coordinator on Iris, us-east5)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-east5 --cpu 2 --memory 8GB --enable-extra-resources \\
        --priority interactive --extra cpu \\
        --job-name tokenize-sysprompt-dclm-qfull1 \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/tokenize_sysprompt_dclm.py
"""

from __future__ import annotations

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

SOURCE_GLOB = "gs://marin-us-east5/sysprompt_pretrain/dclm_400m_1x/qfull1/training/train-*.jsonl.gz"


def main() -> None:
    step = default_tokenize(
        name="sysprompt_dclm_qfull1",
        dataset=InputName.hardcoded(SOURCE_GLOB),
        tokenizer=llama3_tokenizer,
        # Tokenize the [S][D] sequence, not the bare document.
        format=TextLmDatasetFormat(text_key="conditioned_text"),
    )
    executor_main(
        steps=[step],
        description=(
            "Tokenize the system-prompt-conditioned DCLM 400m-1x [S][D] dataset "
            "(conditioned_text field) into a llama3 Levanter cache under "
            "tokenized/sysprompt_dclm_qfull1-*."
        ),
    )


if __name__ == "__main__":
    main()
