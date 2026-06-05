# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download, flatten, and tokenize the LIMA dataset for validation.

https://huggingface.co/datasets/GAIR/lima

LIMA ("Less Is More for Alignment", Zhou et al. 2023) is ~1k high-quality
conversations originally curated as an instruction-tuning fine-tuning set.
We use it as a *validation* dataset for the data-curation scaling-law sweep
because it gives a clean high-quality signal that's orthogonal to the
Paloma / uncheatable_eval benchmarks (those mostly measure distributional
fit; LIMA measures how well the model models high-quality human writing).

Pipeline:
  1. Download the raw HF dataset (`train.jsonl` + `test.jsonl`) to GCS.
  2. Flatten each record's `conversations` list into plain text with
     alternating "User:"/"Assistant:" prefixes (matches the original
     `experiments/lima.py` that was cleaned up in commit c8f69e1d8).
  3. Tokenize the flattened text with `default_tokenize(..., is_validation=True)`.

Usage (run in-region so tokenized cache lands locally; no cross-region egress):

    MARIN_PREFIX=gs://marin-us-central1 uv run python experiments/lima.py

The tokenized cache ends up at `gs://marin-us-central1/tokenized/lima_text-<hash>/`.
To plumb it into the data-curation sweep's validation mix, register the hash
in `experiments/scaling_law_sweeps/data_curation_math.py` under `_LIMA_CACHE_HASH`
and add the `with_lima` branch to `as_lm_mixture_config`.

Cross-region determinism
------------------------
This pipeline is written to be byte-for-byte reproducible across regions so
that two independent runs (e.g. in us-central1 and us-east5) produce
identical tokenized caches. The recipe:

- HF revision is pinned (`68958e9`) via `versioned(...)`.
- Input files are sorted lexicographically before reading.
- Row order within each file is preserved (no shuffle, no sort of records).
- JSON output is serialized with `sort_keys=True` so key order is canonical.
- Tokenizer is pinned (`llama3_tokenizer`), and Levanter's tokenize step is
  deterministic given byte-identical inputs.

Verification: diff `tokenized/lima_text-<hash>/train/.stats.json` between
two regions' buckets — total_tokens and total_elements must match.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
from dataclasses import dataclass

import fsspec
from marin.datakit.download.huggingface import DownloadConfig as HfDownloadConfig
from marin.datakit.download.huggingface import download_hf
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.tokenize.data_configs import TokenizerStep

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

# The raw HF dataset step.
lima = ExecutorStep(
    name="raw/lima",
    fn=download_hf,
    config=HfDownloadConfig(
        hf_dataset_id=versioned("GAIR/lima"),
        revision=versioned("68958e9"),
        gcs_output_path=this_output_path(),
        hf_urls_glob=["*.jsonl"],
        wait_for_completion=True,
    ),
).with_output_path("raw/lima-68958e9")


@dataclass(frozen=True)
class LimaConversationsToTextConfig:
    """Config for the LIMA conversation-to-plain-text conversion."""

    raw_lima: str
    """Input directory containing the LIMA HF jsonl files (train.jsonl, test.jsonl)."""

    output_path: str
    """Output directory for the flattened text jsonl."""


def _convert_record(record: dict) -> dict | None:
    """Flatten a LIMA record's `conversations` list to a single `text` field.

    LIMA's format: `{"conversations": [user_turn, assistant_turn, ...], "source": ...}`.
    Turns alternate strictly (User, Assistant, User, Assistant, ...).
    """
    turns = record.get("conversations", [])
    if not turns:
        return None
    parts = []
    for i, content in enumerate(turns):
        role = "User" if i % 2 == 0 else "Assistant"
        parts.append(f"{role}: {content}")
    return {"text": "\n\n".join(parts)}


def convert_lima_conversations(config: LimaConversationsToTextConfig) -> None:
    """Read every *.jsonl in `config.raw_lima`, flatten each record, write to `config.output_path`.

    Uses plain fsspec + json — LIMA is ~1k records so this completes in
    seconds without needing zephyr parallelism. Output is a single
    `train.jsonl` file (LIMA has no separate validation split we care about
    for loss computation; we concatenate train + test).
    """
    # Recursive glob so we tolerate both flat (./train.jsonl) and sha-nested
    # (./<sha>/train.jsonl) layouts. The HF download helper has used both over
    # time depending on append_sha_to_path; recursive is robust to either.
    input_glob = os.path.join(config.raw_lima, "**", "*.jsonl")
    fs, _ = fsspec.core.url_to_fs(config.raw_lima)
    input_files = sorted(fs.glob(input_glob))
    if not input_files:
        raise FileNotFoundError(f"No *.jsonl files under {input_glob}")
    # fsspec.glob returns paths without the scheme; re-prefix.
    scheme = config.raw_lima.split("://", 1)[0] if "://" in config.raw_lima else ""
    if scheme:
        input_files = [f if f.startswith(scheme + "://") else f"{scheme}://{f}" for f in input_files]

    # Write .jsonl.gz (marin's tokenize helper only globs compressed formats).
    # mtime=0 makes the gzip bytestream reproducible across runs/regions.
    output_file = os.path.join(config.output_path, "train.jsonl.gz")
    rows_in = 0
    rows_out = 0
    with fsspec.open(output_file, "wb") as dst_fh:
        with gzip.GzipFile(fileobj=dst_fh, mode="wb", mtime=0) as dst_gz:
            for src_path in input_files:
                logger.info("Reading %s", src_path)
                with fsspec.open(src_path, "r") as src:
                    for line in src:
                        line = line.strip()
                        if not line:
                            continue
                        rows_in += 1
                        record = json.loads(line)
                        flat = _convert_record(record)
                        if flat is None:
                            continue
                        # sort_keys for canonical byte layout across regions.
                        dst_gz.write((json.dumps(flat, sort_keys=True) + "\n").encode("utf-8"))
                        rows_out += 1
    logger.info("LIMA convert complete: %d rows in → %d rows out → %s", rows_in, rows_out, output_file)


lima_text = ExecutorStep(
    name="raw/lima_text",
    fn=convert_lima_conversations,
    config=LimaConversationsToTextConfig(
        raw_lima=lima,
        output_path=this_output_path(),
    ),
).with_output_path("raw/lima_text-68958e9/68958e9")


def lima_tokenized(
    tokenizer: str = llama3_tokenizer,
    is_validation: bool = True,
) -> TokenizerStep:
    """Tokenized LIMA step. Single dataset — unlike paloma which has sub-buckets."""
    # Pass the directory containing train.jsonl (not the file itself);
    # default_tokenize expects a dir and globs inside for jsonl/parquet.
    return default_tokenize(
        name="lima_text",
        dataset=lima_text,
        tokenizer=tokenizer,
        is_validation=is_validation,
    )


if __name__ == "__main__":
    executor_main(steps=[lima, lima_text, lima_tokenized()])
