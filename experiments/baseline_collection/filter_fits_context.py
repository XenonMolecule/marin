# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter a chat dataset to examples whose **full** rendered sequence fits in the
context window — keep iff nothing is truncated.

This differs from ``marin.transform.filter_by_context_length``, which keeps an
example when at least ``min_completion_tokens`` assistant tokens fall within the
window. That criterion can't serve a dataset with both long and *deliberately
short* targets: a high threshold drops the short ones (here, the 22-token
``[NO_USEFUL_CONTENT]`` abstentions), while a low threshold keeps long examples
whose target is mostly truncated (the trailing ``[[ ## completed ## ]]`` marker
cut off, teaching the model not to terminate).

Keep-iff-full-fit resolves both: every kept example has its entire target —
short abstentions survive, and over-length pages are dropped (not truncated),
honoring the "filter, never truncate" rule for the distillation set.

Structure mirrors ``filter_by_context_length`` (shared tokenizer, per-shard
Zephyr map, empty-shard cleanup, stats file) so the two are interchangeable as
executor steps.
"""

import dataclasses
import gzip
import json
import logging
from collections.abc import Iterator

import draccus
import fsspec
import transformers
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl
from zephyr.worker_context import zephyr_worker_ctx

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class FilterFitsContextConfig:
    """Keep chat examples whose full rendered sequence is ``<= seq_len`` tokens.

    Attributes:
        input_path: Glob for input JSONL files.
        output_path: Directory for filtered JSONL output.
        tokenizer: HuggingFace tokenizer name/path.
        seq_len: Context window; examples rendering to more tokens are dropped.
        chat_template: Jinja2 chat template; use the one the training tokenize
            step uses so the measured length matches training exactly.
        messages_field: Field holding the messages list.
    """

    input_path: str
    output_path: str
    tokenizer: str
    seq_len: int
    chat_template: str | None = None
    messages_field: str = "messages"


def fits_context(messages: list[dict], tokenizer, seq_len: int, chat_template: str | None) -> bool:
    """True iff the full rendered chat sequence is ``<= seq_len`` tokens (nothing
    truncated). Use the same chat_template the training tokenize step uses so the
    measured length matches training exactly."""
    kwargs: dict = {}
    if chat_template:
        kwargs["chat_template"] = chat_template
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    return len(ids) <= seq_len


def _filter_records(*, config: FilterFitsContextConfig, records: Iterator[dict]) -> Iterator[dict]:
    tokenizer = zephyr_worker_ctx().get_shared("tokenizer")
    for record in records:
        messages = record.get(config.messages_field)
        if messages is None:
            continue
        if fits_context(messages, tokenizer, config.seq_len, config.chat_template):
            yield record


def filter_fits_context(config: FilterFitsContextConfig) -> None:
    """Keep only chat examples whose full rendered sequence fits ``seq_len``."""
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.tokenizer, trust_remote_code=True)

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map_shard(lambda records, _shard_info: _filter_records(config=config, records=records))
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name="filter-fits-context")
    ctx.put("tokenizer", tokenizer)
    ctx.execute(pipeline)

    # Drop empty shards: the tokenize step reads the first file for its exemplar
    # and crashes on a 0-row file.
    output_files = fsspec_glob(f"{config.output_path}/*.jsonl.gz")
    kept = 0
    empty_files: list[str] = []
    for path in output_files:
        shard_count = 0
        with fsspec.open(path, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as gz:
            for _ in gz:
                shard_count += 1
        if shard_count == 0:
            empty_files.append(path)
        kept += shard_count
    for path in empty_files:
        fs, _ = fsspec.core.url_to_fs(path)
        fs.rm(path)
    if empty_files:
        logger.info("Removed %d empty shard(s) of %d total.", len(empty_files), len(output_files))

    logger.info("=" * 70)
    logger.info("FILTER COMPLETE: kept %d examples (seq_len=%d, keep-iff-full-fit)", kept, config.seq_len)
    logger.info("=" * 70)

    stats = {"kept": kept, "seq_len": config.seq_len, "tokenizer": config.tokenizer, "policy": "full_fit"}
    with fsspec.open(f"{config.output_path}/filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


@draccus.wrap()
def main(config: FilterFitsContextConfig):
    filter_fits_context(config)


if __name__ == "__main__":
    main()
