# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the arms of a system-prompt ablation into llama3 Levanter caches.

One scale tag -> one cache per arm, from the shards written by
``build_sysprompt30b_dataset.py emit --tag {tag}``:

    A  sysprompt      ab-*.jsonl.gz   text_key=conditioned_text
    C  doc-matched    ab-*.jsonl.gz   text_key=text
    B  token-matched  *.jsonl.gz      text_key=text   (ab-* plus extra-*)
    D  mix50          ab-*.jsonl.gz   text_key=mixed_text  ([S] on a doc_id-keyed half)

A, C and D read the SAME shards and differ only in which field is tokenized,
which is what makes "same documents" true by construction rather than by a
matching step that could drift.

llama3 tokenizer throughout, matching every curation baseline — the ``[S]``
wrapper is exactly the llama3 header markers, which map to their reserved ids.

Calls ``marin.processing.tokenize.tokenize.tokenize`` directly (the executor
framework that used to fingerprint cache names is gone), so the cache directory
is EXPLICIT: ``tokenized/sysprompt30b_{tag}_{arm}-{cache_tag}``. A/B/C for
998m_9e19 predate this and keep their executor-hash names (see
``curation_plan._D_OBS_DEFAULTS``); D and any future arm use this convention.

After each cache completes, read ``{cache}/train/.stats.json:total_tokens`` and
use it — NOT the build-time estimate — to set that arm's ``--train-steps``
(``floor(total_tokens / (batch * 4096))`` for exactly one epoch). Then add the
cache name to ``curation_plan._D_OBS_DEFAULTS`` and register the method.

Usage (CPU coordinator on Iris, us-central1; the Zephyr fleet does the work)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 8 --memory 32GB --enable-extra-resources \\
        --priority interactive --extra cpu \\
        --job-name tokenize-sp30b-998m-9e19-D \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> -e MARIN_PREFIX gs://marin-us-central1 \\
        -- python -m experiments.scaling_law_sweeps.tokenize_sysprompt30b --tag 998m_9e19 --arms D
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec
from levanter.data.text.formats import LmDatasetFormatBase, PrebuiltLmDatasetFormat, TextLmDatasetFormat
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

BUILD = "gs://marin-us-central1/sysprompt_pretrain/dclm30b/build"
TOKENIZED = "gs://marin-us-central1/tokenized"
LLAMA3_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"  # matches every curation baseline
ARMS = ("A", "B", "C", "D", "AH", "CT", "AM")
# Cooldown split: the first HEAD_FRACTION of the ab shards (block order) are the
# conditioned head, the rest the bare tail. Shards are 2000-doc blocks, so this
# is a doc split to within one block.
HEAD_FRACTION = 0.8


def ab_shards(tag: str) -> list[str]:
    """Every ab-* shard of a scale, in block order."""
    fs = fsspec.filesystem("gcs")
    paths = sorted(p for p in fs.ls(f"{BUILD}/{tag}/train") if "/ab-" in p)
    if not paths:
        raise FileNotFoundError(f"no ab-* shards under {BUILD}/{tag}/train")
    return [f"gs://{p}" for p in paths]


def arm_specs(tag: str) -> dict[str, tuple[list[str], LmDatasetFormatBase]]:
    """arm -> (input paths, dataset format)."""
    train = f"{BUILD}/{tag}/train"
    specs: dict[str, tuple[list[str], LmDatasetFormatBase]] = {
        "A": ([f"{train}/ab-*.jsonl.gz"], TextLmDatasetFormat(text_key="conditioned_text")),
        "C": ([f"{train}/ab-*.jsonl.gz"], TextLmDatasetFormat(text_key="text")),
        "B": ([f"{train}/*.jsonl.gz"], TextLmDatasetFormat(text_key="text")),
        "D": ([f"{train}/ab-*.jsonl.gz"], TextLmDatasetFormat(text_key="mixed_text")),
        # AM: written by build_sysprompt30b_masked.py — token ids identical to A,
        # loss weights 0 on the [S] prefix.
        "AM": (
            [f"{BUILD}/{tag}/masked/prebuilt-*.jsonl.gz"],
            PrebuiltLmDatasetFormat(input_ids_key="input_ids", loss_weights_key="loss_weights"),
        ),
    }
    return specs


def cooldown_specs(tag: str) -> dict[str, tuple[list[str], LmDatasetFormatBase]]:
    """AH (conditioned head) and CT (bare tail) for the cooldown arm; lists the bucket."""
    shards = ab_shards(tag)
    split = round(HEAD_FRACTION * len(shards))
    return {
        "AH": (shards[:split], TextLmDatasetFormat(text_key="conditioned_text")),
        "CT": (shards[split:], TextLmDatasetFormat(text_key="text")),
    }


def cache_path(tag: str, arm: str, cache_tag: str) -> str:
    return f"{TOKENIZED}/sysprompt30b_{tag}_{arm}-{cache_tag}"


def main() -> None:
    configure_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="Scale tag, e.g. 998m_9e19.")
    ap.add_argument("--arms", default="D", help="Comma-separated subset of arms to tokenize.")
    ap.add_argument(
        "--cache-tag",
        default="mix50v1",
        help="Suffix of the cache dir name; bump it whenever the emitted shards change meaning.",
    )
    a = ap.parse_args()

    arms = tuple(x.strip().upper() for x in a.arms.split(",") if x.strip())
    bad = set(arms) - set(ARMS)
    if bad:
        raise SystemExit(f"unknown arms: {sorted(bad)}")

    specs = arm_specs(a.tag)
    if {"AH", "CT"} & set(arms):
        specs.update(cooldown_specs(a.tag))
    for arm in arms:
        paths, fmt = specs[arm]
        path = cache_path(a.tag, arm, a.cache_tag)
        logger.info("ARM %s: %d input path(s), first=%s format=%s -> %s", arm, len(paths), paths[0], fmt, path)
        tokenize(
            TokenizeConfig(
                train_paths=paths,
                validation_paths=[],
                cache_path=path,
                tokenizer=LLAMA3_TOKENIZER,
                format=fmt,
            )
        )
        with fsspec.open(f"{path}/train/.stats.json", "r") as f:  # dot-prefixed by write_stats_json
            stats = json.load(f)
        logger.info("ARM %s TOKENIZE COMPLETE: cache=%s stats=%s", arm, path, json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
