# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the three arms of a system-prompt ablation into llama3 Levanter caches.

One scale tag -> three caches, from the shards written by
``build_sysprompt30b_dataset.py emit --tag {tag}``:

    A  sysprompt      ab-*.jsonl.gz   text_key=conditioned_text
    C  doc-matched    ab-*.jsonl.gz   text_key=text
    B  token-matched  *.jsonl.gz      text_key=text   (ab-* plus extra-*)

A and C read the SAME shards and differ only in which field is tokenized, which
is what makes "same documents" true by construction rather than by a matching
step that could drift.

llama3 tokenizer throughout, matching every curation baseline — the ``[S]``
wrapper is exactly the llama3 header markers, which map to their reserved ids.

Scale-generic: ``--tag`` is the only required argument, so a future budget is
``--tag 998m_1p8e20`` once its shards exist. Cache names embed the tag, so caches
for different scales never collide.

After each cache completes, read ``{cache}/train/.stats.json:total_tokens`` and
use it — NOT the build-time estimate — to set that arm's ``--train-steps``
(``floor(total_tokens / (batch * 4096))`` for exactly one epoch). Then add the
hash to ``curation_plan._D_OBS_DEFAULTS`` and register the method.

Usage (CPU coordinator on Iris, us-central1)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 8 --memory 32GB --enable-extra-resources \\
        --priority interactive --extra cpu \\
        --job-name tokenize-sp30b-998m-9e19 \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/tokenize_sysprompt30b.py --tag 998m_9e19
"""

from __future__ import annotations

import argparse
import sys

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import InputName, executor_main

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

BUILD = "gs://marin-us-central1/sysprompt_pretrain/dclm30b/build"


def steps_for(tag: str, arms: tuple[str, ...]):
    train = f"{BUILD}/{tag}/train"
    specs = {
        # arm: (glob, text_key)
        "A": (f"{train}/ab-*.jsonl.gz", "conditioned_text"),
        "C": (f"{train}/ab-*.jsonl.gz", "text"),
        "B": (f"{train}/*.jsonl.gz", "text"),
    }
    out = []
    for arm in arms:
        glob, key = specs[arm]
        out.append(
            default_tokenize(
                name=f"sysprompt30b_{tag}_{arm}",
                dataset=InputName.hardcoded(glob),
                tokenizer=llama3_tokenizer,
                format=TextLmDatasetFormat(text_key=key),
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True, help="Scale tag, e.g. 998m_9e19.")
    ap.add_argument("--arms", default="A,B,C", help="Subset of arms to tokenize.")
    a, rest = ap.parse_known_args()

    arms = tuple(x.strip().upper() for x in a.arms.split(",") if x.strip())
    bad = set(arms) - {"A", "B", "C"}
    if bad:
        raise SystemExit(f"unknown arms: {sorted(bad)}")

    # executor_main parses sys.argv itself and rejects anything it does not know,
    # so our own flags must be removed before handing control over — leave only
    # the args it should see.
    sys.argv = [sys.argv[0], *rest]

    executor_main(
        steps=steps_for(a.tag, arms),
        description=(
            f"Tokenize system-prompt ablation arms {','.join(arms)} for scale {a.tag} "
            "into llama3 Levanter caches under tokenized/sysprompt30b_*."
        ),
    )


if __name__ == "__main__":
    main()
