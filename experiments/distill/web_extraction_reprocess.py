# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reprocess web extraction distillation dataset with a new model.

Downloads MichaelR207/rephraser_late_check_0225 from HuggingFace, extracts
the system + user messages (dropping the original assistant response), and
generates new responses with a different model.

Supports two inference backends:
  - ``vllm``: Local vLLM on TPU via Ray (default)
  - ``api``: Remote API via litellm (Together AI, OpenAI, etc.)

Supports multi-cluster sharding via START_ROW/END_ROW environment variables.
Each cluster processes a disjoint row range and writes to its own regional
GCS bucket. A separate merge script combines outputs afterward.

Environment variables:
    START_ROW:          First row index (inclusive). Default: 0.
    END_ROW:            Last row index (exclusive). Default: 818880.
    MODEL_NAME:         Model ID. For vllm: HF model ID. For api: litellm model
                        string (e.g. ``together_ai/Qwen/Qwen3-8B``). Default: Qwen/Qwen3-8B.
    INFERENCE_BACKEND:  ``vllm`` or ``api``. Default: vllm.
    TEMPERATURE:        Sampling temperature. Default: 0.6.
    MAX_MODEL_LEN:      Max model context length. Default: 32768.
    MAX_TOKENS:         Max generation tokens. Default: 4096.
    NUM_WORKERS:        Number of Zephyr workers (vllm) or max concurrent API calls (api). Default: 16.
    TPU_TYPE:           TPU type (vllm only). Default: v5p-8.
    SAMPLE_FROM:        Start of per-spec sampling slice (fraction, 0.0-1.0). Default: 0.0.
    SAMPLE_TO:          End of per-spec sampling slice (fraction, 0.0-1.0). Default: 1.0.

Spec-aware sampling (budget control):
    Each extraction spec has ~818 rows. SAMPLE_FROM/SAMPLE_TO select a slice
    of each spec's rows. Slices are non-overlapping, so you can process data
    incrementally without repeating work.

    # Process first 10% of each spec (~82K rows, ~$1K with Kimi K2.5)
    SAMPLE_FROM=0.0 SAMPLE_TO=0.1 ...

    # Later, add the next 40% (~328K rows, no overlap with first 10%)
    SAMPLE_FROM=0.1 SAMPLE_TO=0.5 ...

    # Finish the remaining 50%
    SAMPLE_FROM=0.5 SAMPLE_TO=1.0 ...

    Each slice gets its own executor step hash, so outputs coexist on GCS.
    The merge script combines all slices. Shard-level checkpointing ensures
    no repeated API calls on crash/restart within a slice.

vLLM test (100 rows, Qwen3-8B on TPU):
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \\
        -e START_ROW 0 -e END_ROW 100 \\
        -e MODEL_NAME Qwen/Qwen3-8B \\
        -e NUM_WORKERS 1 \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/distill/web_extraction_reprocess.py

API test (100 rows, gpt-oss-20b on Together AI):
    START_ROW=0 END_ROW=100 \\
    INFERENCE_BACKEND=api \\
    MODEL_NAME=together_ai/Qwen/Qwen3-8B \\
    TOGETHER_API_KEY=your_key_here \\
    NUM_WORKERS=10 \\
    python experiments/distill/web_extraction_reprocess.py

Dry run (verify DAG, no execution):
    START_ROW=0 END_ROW=100 python experiments/distill/web_extraction_reprocess.py --dry_run
"""

import asyncio
import gzip
import json
import logging
import os
from dataclasses import dataclass

import fsspec

from fray.cluster import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.execution.remote import remote
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
HF_DATASET_ID = "MichaelR207/rephraser_late_check_0225"
HF_REVISION = "2194850"
TOTAL_TRAIN_ROWS = 818_780  # Train split: rows 0-818779
TOTAL_VAL_ROWS = 100  # Validation split: rows 818780-818879
SPLIT = os.environ.get("SPLIT", "train")  # "train" or "val"

# Row ranges are locked to the split — no accidental train/val mixing
if SPLIT == "val":
    _SPLIT_START, _SPLIT_END = TOTAL_TRAIN_ROWS, TOTAL_TRAIN_ROWS + TOTAL_VAL_ROWS
else:
    _SPLIT_START, _SPLIT_END = 0, TOTAL_TRAIN_ROWS
START_ROW = int(os.environ.get("START_ROW", str(_SPLIT_START)))
END_ROW = int(os.environ.get("END_ROW", str(_SPLIT_END)))
# Hard guardrail: never cross the train/val boundary
if SPLIT == "train" and END_ROW > TOTAL_TRAIN_ROWS:
    raise ValueError(
        f"END_ROW={END_ROW} exceeds train split boundary ({TOTAL_TRAIN_ROWS}). Use SPLIT=val for validation."
    )
if SPLIT == "val" and START_ROW < TOTAL_TRAIN_ROWS:
    raise ValueError(f"START_ROW={START_ROW} is in train split. Use SPLIT=train for training data.")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "vllm")  # "vllm" or "api"
TEMPERATURE = float(os.environ["TEMPERATURE"]) if "TEMPERATURE" in os.environ else None
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))
TPU_TYPE = os.environ.get("TPU_TYPE", "v5p-8")
# Tiktoken encoding for approximate token counting in the prepare step.
# cl100k_base works well for GPT/Qwen/most modern models.
TIKTOKEN_ENCODING = os.environ.get("TIKTOKEN_ENCODING", "cl100k_base")
RECORDS_PER_SHARD = int(os.environ.get("RECORDS_PER_SHARD", "50" if INFERENCE_BACKEND == "api" else "500"))
# Spec-aware sampling: process a slice of each spec's rows.
# SAMPLE_FROM and SAMPLE_TO define the slice as a fraction of each spec's rows.
# Defaults (0.0, 1.0) = all rows. Examples:
#   First 10%:           SAMPLE_FROM=0.0  SAMPLE_TO=0.1
#   Next 40% (no overlap): SAMPLE_FROM=0.1  SAMPLE_TO=0.5
#   Remaining 50%:       SAMPLE_FROM=0.5  SAMPLE_TO=1.0
SAMPLE_FROM = float(os.environ.get("SAMPLE_FROM", "0.0"))
SAMPLE_TO = float(os.environ.get("SAMPLE_TO", "1.0"))

# Derived
MODEL_SHORT = MODEL_NAME.split("/")[-1].lower()
MAX_DOC_TOKENS = MAX_MODEL_LEN - MAX_TOKENS

# DSPy system message — identical across all rows in the dataset
SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)


# ---------------------------------------------------------------------------
# Step 0b: Build spec index — map spec_id to (start_row, count)
# ---------------------------------------------------------------------------
@dataclass
class SpecIndexConfig:
    """Scan parquet files and build a spec_id → row range manifest."""

    input_path: str
    output_path: str


def build_spec_index(config: SpecIndexConfig) -> None:
    """Scan the spec_id column from parquet files and write a JSON manifest.

    Only reads the ``spec_id`` column (lightweight — no HTML/prompt data).
    Output: ``spec_index.json`` mapping each spec_id to its start row and count.
    """
    import pyarrow.parquet as pq

    parquet_pattern = os.path.join(config.input_path, "data", "*.parquet")
    parquet_files = sorted(fsspec_glob(parquet_pattern))
    if not parquet_files:
        parquet_pattern = os.path.join(config.input_path, "*.parquet")
        parquet_files = sorted(fsspec_glob(parquet_pattern))
    if not parquet_files:
        raise ValueError(f"No parquet files found at {config.input_path}")

    logger.info("Building spec index from %d parquet files", len(parquet_files))

    # Scan spec_id column only — fast, no large data loaded
    spec_index: dict[str, dict] = {}
    cumulative_rows = 0

    for pq_path in parquet_files:
        with fsspec.open(pq_path, "rb") as f:
            table = pq.read_table(f, columns=["spec_id"])

        spec_ids = table.column("spec_id").to_pylist()
        for i, spec_id in enumerate(spec_ids):
            global_idx = cumulative_rows + i
            if spec_id not in spec_index:
                spec_index[spec_id] = {"start_row": global_idx, "count": 0}
            spec_index[spec_id]["count"] += 1

        cumulative_rows += table.num_rows

    # Write manifest
    manifest_path = os.path.join(config.output_path, "spec_index.json")
    with fsspec.open(manifest_path, "w") as f:
        json.dump(spec_index, f, indent=2)

    logger.info(
        "Spec index: %d specs, %d total rows -> %s",
        len(spec_index),
        cumulative_rows,
        manifest_path,
    )


def _compute_sampled_row_set(
    spec_index_path: str,
    sample_from: float,
    sample_to: float,
    start_row: int,
    end_row: int,
) -> set[int]:
    """Compute the set of global_row_idx values to include based on spec-aware sampling.

    For each spec with N rows, selects rows from index ``ceil(from * N)`` to
    ``ceil(to * N)``. Slices are non-overlapping: ``[0.0, 0.1)`` and ``[0.1, 0.5)``
    never share a row. Then intersects with ``[start_row, end_row)``.
    """
    import math

    with fsspec.open(spec_index_path, "r") as f:
        spec_index = json.load(f)

    selected: set[int] = set()
    for _spec_id, info in spec_index.items():
        spec_start = info["start_row"]
        spec_count = info["count"]
        slice_start = spec_start + math.ceil(sample_from * spec_count)
        slice_end = spec_start + math.ceil(sample_to * spec_count)

        for idx in range(slice_start, slice_end):
            if start_row <= idx < end_row:
                selected.add(idx)

    return selected


# ---------------------------------------------------------------------------
# Step 1: Prepare — read cached parquet and extract prompts for a row range
# ---------------------------------------------------------------------------
@dataclass
class PrepareConfig:
    """Read cached parquet files and extract user prompts for reprocessing."""

    input_path: str
    start_row: int
    end_row: int
    output_path: str
    tiktoken_encoding: str
    max_doc_tokens: int
    spec_index_path: str | None = None
    sample_from: float = 0.0
    sample_to: float = 1.0
    records_per_shard: int = 500


def prepare_prompts(config: PrepareConfig) -> None:
    """Read parquet files and extract user messages as prompts for a row range.

    Reads only the parquet files that overlap ``[start_row, end_row)``,
    avoiding loading the entire dataset. For each row:

    - ``messages[1]["content"]`` → ``prompt`` column (the user message, as-is)
    - Rows whose prompt exceeds ``max_doc_tokens`` are dropped (not truncated)
    - Preserves metadata: warc_file, doc_id, spec_id, spec
    - Adds ``global_row_idx`` for tracking and merge deduplication

    Writes sharded JSONL.gz files plus a stats summary.
    """
    import pyarrow.parquet as pq
    import tiktoken

    # Find all parquet files from the download step
    parquet_pattern = os.path.join(config.input_path, "data", "*.parquet")
    parquet_files = sorted(fsspec_glob(parquet_pattern))
    if not parquet_files:
        # Try without data/ subdirectory
        parquet_pattern = os.path.join(config.input_path, "*.parquet")
        parquet_files = sorted(fsspec_glob(parquet_pattern))
    if not parquet_files:
        raise ValueError(f"No parquet files found at {config.input_path}")

    logger.info("Found %d parquet files at %s", len(parquet_files), config.input_path)

    enc = tiktoken.get_encoding(config.tiktoken_encoding)

    # Build sampled row set if spec-aware sampling is enabled
    sampled_rows: set[int] | None = None
    is_sampling = not (config.sample_from == 0.0 and config.sample_to == 1.0)
    if is_sampling and config.spec_index_path:
        sampled_rows = _compute_sampled_row_set(
            spec_index_path=config.spec_index_path,
            sample_from=config.sample_from,
            sample_to=config.sample_to,
            start_row=config.start_row,
            end_row=config.end_row,
        )
        logger.info(
            "Spec-aware sampling: [%.3f, %.3f) -> %d rows selected",
            config.sample_from,
            config.sample_to,
            len(sampled_rows),
        )

    num_requested = config.end_row - config.start_row

    shard_idx = 0
    shard_records: list[dict] = []
    total_written = 0
    total_dropped = 0
    total_shards_estimate = max(1, (num_requested + config.records_per_shard - 1) // config.records_per_shard)

    # Scan parquet files, only reading those that overlap our row range
    cumulative_rows = 0
    for pq_path in parquet_files:
        with fsspec.open(pq_path, "rb") as f:
            pf = pq.ParquetFile(f)
            file_rows = pf.metadata.num_rows

        # Skip files entirely before our range
        if cumulative_rows + file_rows <= config.start_row:
            cumulative_rows += file_rows
            continue
        # Stop once past our range
        if cumulative_rows >= config.end_row:
            break

        # This file overlaps — read it
        with fsspec.open(pq_path, "rb") as f:
            table = pq.read_table(f)

        local_start = max(0, config.start_row - cumulative_rows)
        local_end = min(file_rows, config.end_row - cumulative_rows)
        selected = table.slice(local_start, local_end - local_start)

        logger.info(
            "Reading %s: rows [%d:%d) of %d (global [%d:%d))",
            os.path.basename(pq_path),
            local_start,
            local_end,
            file_rows,
            cumulative_rows + local_start,
            cumulative_rows + local_end,
        )

        for row_offset in range(selected.num_rows):
            global_idx = cumulative_rows + local_start + row_offset

            # Spec-aware sampling: skip rows not in the sampled set
            if sampled_rows is not None and global_idx not in sampled_rows:
                continue

            messages = selected.column("messages")[row_offset].as_py()
            prompt = messages[1]["content"]

            # Drop rows exceeding context limit (approximate via tiktoken)
            prompt_tokens = enc.encode(prompt)
            if len(prompt_tokens) > config.max_doc_tokens:
                total_dropped += 1
                continue

            record = {
                "prompt": prompt,
                "global_row_idx": global_idx,
                "warc_file": selected.column("warc_file")[row_offset].as_py(),
                "doc_id": selected.column("doc_id")[row_offset].as_py(),
                "spec_id": selected.column("spec_id")[row_offset].as_py(),
                "spec": selected.column("spec")[row_offset].as_py(),
            }
            shard_records.append(record)

            if len(shard_records) >= config.records_per_shard:
                shard_path = f"{config.output_path}/data-{shard_idx:05d}-of-{total_shards_estimate:05d}.jsonl.gz"
                _write_jsonl_gz(shard_path, shard_records)
                total_written += len(shard_records)
                shard_records = []
                shard_idx += 1

        cumulative_rows += file_rows

    # Flush remaining records
    if shard_records:
        shard_path = f"{config.output_path}/data-{shard_idx:05d}-of-{total_shards_estimate:05d}.jsonl.gz"
        _write_jsonl_gz(shard_path, shard_records)
        total_written += len(shard_records)
        shard_idx += 1

    stats = {
        "input_path": config.input_path,
        "start_row": config.start_row,
        "end_row": config.end_row,
        "total_requested": num_requested,
        "total_written": total_written,
        "total_dropped": total_dropped,
        "total_sampled": len(sampled_rows) if sampled_rows is not None else num_requested,
        "sample_from": config.sample_from,
        "sample_to": config.sample_to,
        "num_shards": shard_idx,
        "tiktoken_encoding": config.tiktoken_encoding,
        "max_doc_tokens": config.max_doc_tokens,
    }
    with fsspec.open(f"{config.output_path}/prepare_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        "Prepared %d records (%d dropped, %d shards) -> %s",
        total_written,
        total_dropped,
        shard_idx,
        config.output_path,
    )


# ---------------------------------------------------------------------------
# Step 3: Reassemble — reconstruct chat messages format
# ---------------------------------------------------------------------------
@dataclass
class ReassembleConfig:
    """Reconstruct chat messages format from inference output."""

    input_path: str
    output_path: str
    system_message: str
    source_model: str


def reassemble_messages(config: ReassembleConfig) -> None:
    """Reconstruct chat messages format from inference output.

    Reads JSONL records with ``prompt`` + ``generated_text`` columns, builds
    the standard 3-message chat format matching the original HF dataset schema.
    Drops intermediate columns (prompt, generated_text) from the output.
    """
    from zephyr import Dataset, ZephyrContext, load_jsonl

    def _reassemble_record(record: dict) -> dict:
        from zephyr import zephyr_worker_ctx

        ctx = zephyr_worker_ctx()
        cfg: ReassembleConfig = ctx.get_shared("reassemble_config")
        messages = [
            {"role": "system", "content": cfg.system_message},
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record.get("generated_text", "")},
        ]
        return {
            "messages": messages,
            "warc_file": record.get("warc_file", ""),
            "doc_id": record.get("doc_id", ""),
            "spec_id": record.get("spec_id", ""),
            "spec": record.get("spec", ""),
            "model": cfg.source_model,
            "global_row_idx": record.get("global_row_idx"),
        }

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_jsonl)
        .map(_reassemble_record)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    ctx = ZephyrContext(name="reassemble-messages")
    ctx.put("reassemble_config", config)
    output_files = ctx.execute(pipeline)

    stats = {
        "output_files": len(output_files),
        "input_path": config.input_path,
        "source_model": config.source_model,
    }
    with fsspec.open(f"{config.output_path}/reassemble_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Reassembled %d output files -> %s", len(output_files), config.output_path)


# ---------------------------------------------------------------------------
# Step 2b: API Inference — call remote model via litellm
# ---------------------------------------------------------------------------
@dataclass
class ApiInferenceConfig:
    """Run inference via litellm API with shard-level checkpointing."""

    input_path: str
    output_path: str
    model_name: str
    system_message: str
    prompt_column: str = "prompt"
    generated_text_column: str = "generated_text"
    temperature: float | None = None  # None = use API default
    max_tokens: int = 4096
    max_concurrent: int = 10


def run_api_inference(config: ApiInferenceConfig) -> None:
    """Run inference via litellm with shard-level checkpointing.

    Each input shard produces one output file. Completed shards are skipped
    on restart, so you never pay for the same row twice. Within a shard,
    rows are processed with async concurrency for throughput.
    """
    input_files = sorted(fsspec_glob(config.input_path))
    if not input_files:
        raise ValueError(f"No input files found at {config.input_path}")

    logger.info(
        "API inference: %d input shards, model=%s, max_concurrent=%d",
        len(input_files),
        config.model_name,
        config.max_concurrent,
    )

    total_generated = 0
    total_skipped = 0
    total_failed = 0
    all_failed_indices: list[int] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0

    for input_file in input_files:
        basename = os.path.basename(input_file)
        output_file = os.path.join(config.output_path, basename)

        # Shard-level checkpoint: skip if output already exists
        try:
            with fsspec.open(output_file, "rb"):
                logger.info("Skipping %s (already exists)", basename)
                total_skipped += 1
                continue
        except FileNotFoundError:
            pass

        # Read input records
        records = _read_jsonl_gz(input_file)
        logger.info("Processing shard %s (%d records)", basename, len(records))

        # Process with async concurrency
        results, failed, failed_indices, shard_in, shard_out, shard_reason = asyncio.run(
            _process_shard_api(records, config)
        )
        total_generated += len(results)
        total_failed += failed
        all_failed_indices.extend(failed_indices)
        total_input_tokens += shard_in
        total_output_tokens += shard_out
        total_reasoning_tokens += shard_reason

        # Write output shard
        _write_jsonl_gz(output_file, results)
        logger.info(
            "Shard %s: %d records, %d failed | tokens: %d in, %d out, %d reasoning",
            basename,
            len(results),
            failed,
            shard_in,
            shard_out,
            shard_reason,
        )

    stats = {
        "model_name": config.model_name,
        "total_generated": total_generated,
        "total_skipped_shards": total_skipped,
        "total_failed_rows": total_failed,
        "failed_global_row_indices": sorted(all_failed_indices),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    with fsspec.open(os.path.join(config.output_path, "api_inference_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    if all_failed_indices:
        with fsspec.open(os.path.join(config.output_path, "failed_rows.json"), "w") as f:
            json.dump(sorted(all_failed_indices), f)
        logger.info("Wrote %d failed row indices to failed_rows.json", len(all_failed_indices))

    logger.info(
        "API inference complete: %d generated, %d failed | tokens: %d in, %d out, %d reasoning -> %s",
        total_generated,
        total_failed,
        total_input_tokens,
        total_output_tokens,
        total_reasoning_tokens,
        config.output_path,
    )


def _ensure_litellm():
    """Import litellm, installing it first if necessary."""
    try:
        import litellm

        return litellm
    except ImportError:
        import subprocess
        import sys

        logger.info("litellm not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "litellm"])
        import litellm

        return litellm


async def _process_shard_api(records: list[dict], config: ApiInferenceConfig) -> tuple[list[dict], int, int, int]:
    """Process a shard of records with concurrent API calls.

    Returns (results, fail_count, failed_indices, input_tokens, output_tokens, reasoning_tokens).
    """
    litellm = _ensure_litellm()

    semaphore = asyncio.Semaphore(config.max_concurrent)
    results: list[dict] = []
    failed = 0
    failed_indices: list[int] = []
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0

    max_retries = 5

    async def process_one(record: dict) -> tuple[dict | None, int, int, int]:
        async with semaphore:
            messages = [
                {"role": "system", "content": config.system_message},
                {"role": "user", "content": record[config.prompt_column]},
            ]
            for attempt in range(max_retries):
                try:
                    kwargs = {
                        "model": config.model_name,
                        "messages": messages,
                        "max_tokens": config.max_tokens,
                    }
                    if config.temperature is not None:
                        kwargs["temperature"] = config.temperature
                    response = await litellm.acompletion(**kwargs)
                    msg = response.choices[0].message
                    content = msg.content or ""
                    # APIs like Together AI return reasoning in a separate field
                    # (litellm exposes it as reasoning_content). Reconstruct
                    # <think>...</think> format to match local vLLM output.
                    reasoning = msg.reasoning_content
                    if reasoning:
                        record[config.generated_text_column] = f"<think>\n{reasoning}\n</think>{content}"
                    else:
                        record[config.generated_text_column] = content
                    usage = response.usage
                    reason_tok = getattr(usage, "reasoning_tokens", 0) or 0
                    return record, usage.prompt_tokens, usage.completion_tokens, reason_tok
                except Exception:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(
                            "API call failed for global_row_idx=%s (attempt %d/%d), retrying in %ds",
                            record.get("global_row_idx", "?"),
                            attempt + 1,
                            max_retries,
                            wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.exception(
                            "API call failed for global_row_idx=%s after %d attempts",
                            record.get("global_row_idx", "?"),
                            max_retries,
                        )
                        return None, 0, 0, 0
            return None, 0, 0, 0  # unreachable but satisfies type checker

    tasks = [(r, process_one(r)) for r in records]
    gathered = await asyncio.gather(*(t for _, t in tasks))
    for record, (result, in_tok, out_tok, reason_tok) in zip(records, gathered, strict=True):
        input_tokens += in_tok
        output_tokens += out_tok
        reasoning_tokens += reason_tok
        if result is not None:
            results.append(result)
        else:
            failed += 1
            idx = record.get("global_row_idx")
            if idx is not None:
                failed_indices.append(idx)

    return results, failed, failed_indices, input_tokens, output_tokens, reasoning_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_jsonl_gz(path: str) -> list[dict]:
    """Read a gzipped JSONL file and return a list of records."""
    records = []
    with fsspec.open(path, "rb") as f:
        with gzip.open(f, "rt", encoding="utf-8") as gz:
            for line in gz:
                records.append(json.loads(line))
    return records


def _write_jsonl_gz(path: str, records: list[dict]) -> None:
    """Write a list of records to a gzipped JSONL file."""
    with fsspec.open(path, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Executor Steps
# ---------------------------------------------------------------------------

# Step 0a: Download HF dataset to GCS (cached — only runs once per cluster)
download_step = ExecutorStep(
    name="raw/rephraser_late_check_0225",
    description="Download rephraser dataset from HuggingFace to regional GCS.",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id=HF_DATASET_ID,
        revision=versioned(HF_REVISION),
        gcs_output_path=this_output_path(),
    ),
)

# Step 0b: Build spec index (cached — only runs once per cluster)
spec_index_step = ExecutorStep(
    name="distill/rephraser_late_check_0225_spec_index",
    description="Build spec_id -> row range manifest from parquet files.",
    fn=build_spec_index,
    config=SpecIndexConfig(
        input_path=download_step / "data",
        output_path=this_output_path(),
    ),
)

# Encode sampling in step name so different slices get different output paths
_is_sampling = not (SAMPLE_FROM == 0.0 and SAMPLE_TO == 1.0)
_sample_tag = f"_s{SAMPLE_FROM}-{SAMPLE_TO}" if _is_sampling else ""

prepare_step = ExecutorStep(
    name=f"distill/{MODEL_SHORT}_reprocess/prepared_rows_{START_ROW}_{END_ROW}{_sample_tag}",
    description=f"Extract user prompts from cached parquet, rows [{START_ROW}:{END_ROW}){_sample_tag}.",
    fn=prepare_prompts,
    config=PrepareConfig(
        input_path=download_step / "data",
        start_row=START_ROW,
        end_row=END_ROW,
        output_path=this_output_path(),
        tiktoken_encoding=TIKTOKEN_ENCODING,
        max_doc_tokens=MAX_DOC_TOKENS,
        spec_index_path=spec_index_step / "spec_index.json" if _is_sampling else None,
        sample_from=SAMPLE_FROM,
        sample_to=SAMPLE_TO,
        records_per_shard=RECORDS_PER_SHARD,
    ),
)

if INFERENCE_BACKEND == "api":
    inference_step = ExecutorStep(
        name=f"distill/{MODEL_SHORT}_reprocess/api_inference_rows_{START_ROW}_{END_ROW}{_sample_tag}",
        description=f"Run {MODEL_SHORT} API inference on rows [{START_ROW}:{END_ROW}).",
        fn=run_api_inference,
        config=ApiInferenceConfig(
            input_path=prepare_step / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name=MODEL_NAME,
            system_message=SYSTEM_MESSAGE,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            max_concurrent=NUM_WORKERS,
        ),
    )
else:
    inference_step = ExecutorStep(
        name=f"distill/{MODEL_SHORT}_reprocess/inference_rows_{START_ROW}_{END_ROW}{_sample_tag}",
        description=f"Run {MODEL_SHORT} vLLM inference on rows [{START_ROW}:{END_ROW}).",
        fn=remote(run_inference, pip_dependency_groups=["vllm"]),
        config=TextGenerationInferenceConfig(
            input_path=prepare_step / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name=MODEL_NAME,
            engine_kwargs={
                "tensor_parallel_size": 4,
                "max_model_len": MAX_MODEL_LEN,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                **({"temperature": TEMPERATURE} if TEMPERATURE is not None else {}),
                "max_tokens": MAX_TOKENS,
            },
            system_message=SYSTEM_MESSAGE,
            template="{example}",  # Pass-through: prompt column is already fully formatted
            prompt_column="prompt",
            apply_chat_template=True,
            save_templated_prompt=False,
            max_doc_tokens=131072,  # Filtering already done in prepare step
            num_instances=(1, NUM_WORKERS),
            batch_size=256,
            tensor_parallel_size=4,
            resource_config=ResourceConfig.with_tpu(TPU_TYPE),
            generated_text_column_name="generated_text",
            filetype="jsonl.gz",
            output_filetype_override="jsonl.gz",
        ),
    )

reassemble_step = ExecutorStep(
    name=f"distill/{MODEL_SHORT}_reprocess/reassembled_rows_{START_ROW}_{END_ROW}{_sample_tag}",
    description=f"Reassemble chat messages for rows [{START_ROW}:{END_ROW}).",
    fn=reassemble_messages,
    config=ReassembleConfig(
        input_path=inference_step / "*.jsonl.gz",
        output_path=this_output_path(),
        system_message=SYSTEM_MESSAGE,
        source_model=MODEL_NAME,
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[reassemble_step],
        description=(
            f"Reprocess {HF_DATASET_ID} rows [{START_ROW}:{END_ROW}) " f"with {MODEL_NAME} (temp={TEMPERATURE})."
        ),
    )
