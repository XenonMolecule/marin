# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild nemotron_full and llm_curated tokenized caches with BOS prepended.

Background
----------
A Levanter regression in commit f03aa9ecb (2026-04-10) changed
`MarinTokenizer.encode_batch` to default to `add_special_tokens=False`.
`BatchTokenizer.__call__` was unintentionally calling encode_batch without
`add_special_tokens=True`, so caches built after 2026-04-10 have ZERO BOS
tokens per doc while caches built before have BOS at every doc start.

The eval caches (paloma, uncheatable_eval — all built 2025) have BOS. So
models trained on the BOS-missing caches hit OOD on every single eval doc's
first token.

This module defines the two rebuild steps. The Levanter patch in
`lib/levanter/src/levanter/data/text/_batch_tokenizer.py` (passing
`add_special_tokens=not self._need_to_add_bos`) is what makes the rebuild
produce BOS-correct tokens.

Running
-------
Nemotron_full (source on us-central2, all in-region):

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -e HF_TOKEN <YOUR_HF_TOKEN> \\
        -- python experiments/baseline_collection/rebuild_bos_fixed.py --only nemotron_full

LLM-curated (source on us-central1; requires a us-central1 Ray cluster —
currently down — to stay in-region). Launch when us-central1 Ray is up::

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/rebuild_bos_fixed.py --only llm_curated
"""

from __future__ import annotations


from experiments.defaults import default_tokenize
from marin.execution.executor import InputName, executor_main

# --- Source locations (existing filter / document outputs) ---
# Both use us-central1 paths so the Iris us-central1 tokenization job is fully in-region.
# ``baseline_nemotron_full-347dfe`` is mirrored to central1; ``baseline_llm_curated-050243``
# is primary-on-central1.
NEMOTRON_FULL_FILTERED = "gs://marin-us-central1/filtered/baseline_nemotron_full-347dfe"
LLM_CURATED_DOCUMENTS = "gs://marin-us-central1/documents/baseline_llm_curated-050243"

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"


# --- Tokenize steps ---

# The only thing that differentiates this from the original `tokenize_nemotron_full`
# in pipeline.py is the `_bos_fixed` suffix in the name (which changes the cache
# hash) and the patched Levanter it runs under. The `default_tokenize` function
# shells out to BatchTokenizer, which now calls
# `encode_batch(batch_text, add_special_tokens=not self._need_to_add_bos)` — so
# every doc's input_ids will start with BOS (128000) and end with EOS (128001).
tokenize_nemotron_full_bos_fixed = default_tokenize(
    name="baseline_nemotron_full_bos_fixed",
    dataset=InputName.hardcoded(f"{NEMOTRON_FULL_FILTERED}/*.jsonl.gz"),
    tokenizer=TOKENIZER,
)

tokenize_llm_curated_bos_fixed = default_tokenize(
    name="baseline_llm_curated_bos_fixed",
    dataset=InputName.hardcoded(f"{LLM_CURATED_DOCUMENTS}/data-*.jsonl.gz"),
    tokenizer=TOKENIZER,
)


def main() -> int:
    import sys

    # Peel --only out of argv so that executor_main's draccus parser doesn't
    # complain about the unknown flag. Default to 'both'.
    only = "both"
    new_argv = []
    it = iter(sys.argv)
    for arg in it:
        if arg == "--only":
            only = next(it)
        elif arg.startswith("--only="):
            only = arg.split("=", 1)[1]
        else:
            new_argv.append(arg)
    sys.argv = new_argv

    steps = []
    if only in ("nemotron_full", "both"):
        steps.append(tokenize_nemotron_full_bos_fixed)
    if only in ("llm_curated", "both"):
        steps.append(tokenize_llm_curated_bos_fixed)

    executor_main(
        steps=steps,
        description=(
            "Rebuild BOS-broken tokenized caches using the patched Levanter "
            "(BatchTokenizer now passes add_special_tokens=not _need_to_add_bos). "
            "Must be launched in-region for the source data to avoid cross-region reads."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
