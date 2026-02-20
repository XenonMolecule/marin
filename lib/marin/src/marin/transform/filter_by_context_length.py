# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter chat datasets to remove examples with insufficient loss tokens for a given context length.

When SFT datasets contain examples whose user prompts exceed the model's context window,
the greedy packer produces training sequences with zero assistant tokens (and thus zero loss).
These examples permanently consume training slots without contributing any gradient signal.

This filter tokenizes each example using the chat template, checks how many assistant tokens
fall within the first ``seq_len`` positions, and drops examples below ``min_completion_tokens``.

Because the filter is parameterized by (tokenizer, seq_len), different context lengths produce
different filtered datasets, while models sharing the same context length share one filter step.

Example usage as an executor step::

    filtered_train = ExecutorStep(
        name="filtered/rephraser_qwen3_32k",
        fn=filter_by_context_length,
        config=FilterByContextLengthConfig(
            input_path=train_dataset / "**/*.jsonl.gz",
            output_path=this_output_path(),
            tokenizer="Qwen/Qwen3-0.6B",
            seq_len=32_768,
            chat_template=QWEN_3_CHAT_TEMPLATE,
            min_completion_tokens=64,
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )
"""

import dataclasses
import gzip
import json
import logging
import math
from collections.abc import Iterator

import draccus
import fsspec
import transformers
from zephyr import Dataset, ZephyrContext, load_jsonl, zephyr_worker_ctx

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FilterByContextLengthConfig:
    """Configuration for filtering chat examples by context-length-aware completion token count.

    Attributes:
        input_path: Glob pattern for input JSONL files (e.g. from a transform step).
        output_path: Directory to write filtered JSONL output.
        tokenizer: HuggingFace tokenizer name or path used for tokenizing messages.
        seq_len: Maximum sequence length (context window) in tokens.
        min_completion_tokens: Minimum number of assistant tokens that must fit within
            the first ``seq_len`` positions for the example to be kept.
        chat_template: Jinja2 chat template string. Must contain ``{%% generation %%}``
            markers for the tokenizer to produce correct assistant masks.
            If None, uses the tokenizer's default template.
        messages_field: Name of the field in each JSONL record that holds the messages list.
    """

    input_path: str
    output_path: str
    tokenizer: str
    seq_len: int
    min_completion_tokens: int = 64
    chat_template: str | None = None
    messages_field: str = "messages"


def _filter_records(*, config: FilterByContextLengthConfig, records: Iterator[dict]) -> Iterator[dict]:
    """Worker function: tokenizes each record and yields those with enough assistant tokens."""
    tokenizer = zephyr_worker_ctx().get_shared("tokenizer")

    for record in records:
        messages = record.get(config.messages_field)
        if messages is None:
            continue

        kwargs: dict = {}
        if config.chat_template:
            kwargs["chat_template"] = config.chat_template
        # Honour per-example chat_template_kwargs if the record provides them.
        if "chat_template_kwargs" in record:
            per_example = dict(record["chat_template_kwargs"])
            if "chat_template" in per_example:
                kwargs["chat_template"] = per_example.pop("chat_template")
            kwargs.update(per_example)

        result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_assistant_tokens_mask=True,
            **kwargs,
        )

        assistant_mask = result.get("assistant_masks", [])
        truncated_mask = assistant_mask[: config.seq_len]
        assistant_token_count = sum(truncated_mask)

        if assistant_token_count >= config.min_completion_tokens:
            yield record


def filter_by_context_length(config: FilterByContextLengthConfig):
    """Filter a chat JSONL dataset, keeping only examples with enough assistant tokens."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(lambda records: _filter_records(config=config, records=records))
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl.gz")
    )

    with ZephyrContext(name="filter-by-context-length") as ctx:
        ctx.put("tokenizer", tokenizer)
        ctx.execute(pipeline)

    # Count output records per shard and remove empty shards.  The downstream
    # tokenize step picks the first file for its exemplar computation and crashes
    # with IndexError if that file has 0 records.
    output_files = fsspec_glob(f"{config.output_path}/*.jsonl.gz")
    kept = 0
    empty_files: list[str] = []
    for path in output_files:
        shard_count = 0
        with fsspec.open(path, "rb") as f:
            with gzip.open(f, "rt", encoding="utf-8") as gz:
                for _ in gz:
                    shard_count += 1
        if shard_count == 0:
            empty_files.append(path)
        kept += shard_count

    for path in empty_files:
        fs, _ = fsspec.core.url_to_fs(path)
        fs.rm(path)
        logger.info(f"Removed empty shard: {path}")
    if empty_files:
        logger.info(f"Removed {len(empty_files)} empty shard(s) out of {len(output_files)} total.")

    # Log prominently so the user can find the new dataset size in Ray logs.
    logger.info("=" * 70)
    logger.info(
        f"FILTER COMPLETE: kept {kept} examples "
        f"(seq_len={config.seq_len}, min_completion_tokens={config.min_completion_tokens})"
    )
    logger.info(f"  With batch_size=64, 1 epoch = {math.ceil(kept / 64)} steps")
    logger.info("=" * 70)

    # Write a small stats file so downstream steps can read the count if needed.
    stats = {
        "kept": kept,
        "seq_len": config.seq_len,
        "min_completion_tokens": config.min_completion_tokens,
        "tokenizer": config.tokenizer,
    }
    stats_path = f"{config.output_path}/filter_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Wrote filter stats to {stats_path}")


@draccus.wrap()
def main(config: FilterByContextLengthConfig):
    filter_by_context_length(config)


if __name__ == "__main__":
    main()
