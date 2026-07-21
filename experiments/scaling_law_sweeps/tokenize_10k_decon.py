# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the CORE-v2-decontaminated dclm / nemotron 10k corpora into llama3
Levanter caches, so we can run the 10k-natural sweep on decontaminated variants
and compare against the non-decon baselines.

The survivors are produced by ``decon_apply.py`` (n=15 / DF<=10, the same
treatment high_quality / resiliparse received) from the filtered 10k corpora:

  ``dclm``      gs://marin-us-central2/documents/baseline_dclm_decon/10364warcs_core_v2/survivors/*.jsonl.gz
                -> tokenized/dclm_400m_1x_10k_dclm_decon-{hash}/
  ``nemotron``  gs://marin-us-central2/documents/baseline_nemotron_decon/10364warcs_core_v2/survivors/*.jsonl.gz
                -> tokenized/dclm_400m_1x_10k_nemotron_full_decon-{hash}/

llama3 tokenizer + default_tokenize with the SAME signature as ``pipeline_10k.py``
/ ``pipeline_10k_dclm.py`` so the only difference from the non-decon caches is the
dropped contaminated docs (a fair decon-vs-non-decon comparison).

The corpus is selected by the ``DECON_METHOD`` env var (``dclm`` or ``nemotron``)
— NOT a CLI flag, because ``executor_main`` owns argv and rejects unknown flags.

After completion read ``{cache}/train/.stats.json:total_tokens`` and register the
hash + token count in ``curation_plan`` (see
``.agents/projects/decontam_dclm_nemo_fineweb_scope.md``).

Usage (CPU coordinator on Iris, us-central2)::

    uv run iris --cluster marin job run --no-wait \\
        --region us-central2 --cpu 8 --memory 32GB --enable-extra-resources \\
        --priority interactive --extra cpu \\
        -e MARIN_PREFIX gs://marin-us-central2 -e DECON_METHOD dclm \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/tokenize_10k_decon.py
"""

from __future__ import annotations

import os

from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

CORPORA = {
    "dclm": (
        "dclm_400m_1x_10k_dclm_decon",
        "gs://marin-us-central2/documents/baseline_dclm_decon/10364warcs_core_v2/survivors/*.jsonl.gz",
    ),
    "nemotron": (
        "dclm_400m_1x_10k_nemotron_full_decon",
        "gs://marin-us-central2/documents/baseline_nemotron_decon/10364warcs_core_v2/survivors/*.jsonl.gz",
    ),
}


def main() -> None:
    method = os.environ.get("DECON_METHOD")
    if method not in CORPORA:
        raise SystemExit(f"set DECON_METHOD to one of {list(CORPORA)} (got {method!r})")
    name, glob = CORPORA[method]
    step = default_tokenize(
        name=name,
        dataset=InputName.hardcoded(glob),
        tokenizer=TOKENIZER,
    )
    executor_main(
        steps=[step],
        description=f"Tokenize CORE-v2-decontaminated 10k '{method}' ({name}) into a llama3 Levanter cache.",
    )


if __name__ == "__main__":
    main()
