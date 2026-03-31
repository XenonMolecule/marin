# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr-based inference with persistent vLLM engines on TPU workers.

Replaces Ray Data ``map_batches`` with Zephyr's ``map_shard`` pattern, matching
how every other pipeline step already works (download, filter, postprocess,
tokenize).  Each Zephyr worker gets a TPU node, initializes a vLLM engine once
(cached across shards), and processes whole shards via a single ``LLM.generate()``
call.  vLLM does continuous batching internally, so there are no artificial batch
boundaries.

Key improvements over ``inference.py``:

1. **TPU scaling** — Zephyr/fray handles TPU scheduling via ``ResourceConfig.with_tpu()``.
2. **No batch boundary waste** — single ``generate()`` per shard.
3. **Clean format API** — separate ``input_format`` / ``output_format`` fields.
4. **XLA compilation caching** — sets JAX cache env vars before engine init.
5. **Shard-level checkpointing** — Zephyr ``skip_existing=True`` is O(shards).

Example usage as an executor step::

    inference_step = ExecutorStep(
        name="documents/my_inference",
        fn=run_inference_v2,
        config=InferenceV2Config(
            input_path=filter_step / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name="Qwen/Qwen3-8B",
            template="Extract the main content from this HTML:\\n\\n{example}",
            system_message="You are a helpful assistant.",
            prompt_column="html",
            engine_kwargs={"max_model_len": 32768, "enable_prefix_caching": True},
            generation_kwargs={"temperature": 0.0, "max_tokens": 4096},
        ),
        pip_dependency_groups=["vllm"],
    )
"""

import gzip
import io
import json
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import fsspec

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclass
class InferenceV2Config:
    """Configuration for Zephyr-based vLLM inference."""

    input_path: str
    """Glob pattern for input files (e.g. ``gs://bucket/data/*.jsonl.gz``)."""

    output_path: str
    """Directory to write inference output."""

    model_name: str
    """HuggingFace model name or GCS path to model weights."""

    # Format — explicit, no confusion
    input_format: str = "jsonl.gz"
    """Input file format: ``"jsonl.gz"``, ``"jsonl.zst"``, ``"jsonl"``, or ``"parquet"``."""

    output_format: str = "jsonl.gz"
    """Output file format: ``"jsonl.gz"`` or ``"parquet"``."""

    # Engine
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs passed to ``vllm.LLM()``."""

    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    """Kwargs passed to ``vllm.SamplingParams()``."""

    tensor_parallel_size: int = 4
    """Number of TPU chips for tensor parallelism."""

    # Prompting
    template: str | None = None
    """Prompt template with ``{example}`` placeholder. If None, raw text is used."""

    system_message: str | None = None
    """System message prepended to chat template."""

    prompt_column: str = "text"
    """Column name containing the input text."""

    generated_text_column: str = "generated_text"
    """Column name for the generated output text."""

    apply_chat_template: bool = True
    """Whether to apply the tokenizer's chat template."""

    max_doc_tokens: int = 7000
    """Maximum number of tokens for input document truncation."""

    # Scaling
    num_workers: int = 16
    """Number of Zephyr workers (each gets a TPU node)."""

    tpu_type: str = "v5p-8"
    """TPU type for workers (e.g. ``"v5p-8"``, ``"v6e-8"``)."""

    # Sharding — controls checkpoint granularity and parallelism.
    # Each shard becomes one generate() call + one output file.
    # Smaller = less data lost on preemption, better load balancing.
    # Larger = less overhead, more efficient vLLM batching.
    records_per_shard: int = 500
    """Target number of records per shard. Auto-computes total shards."""


# ---------------------------------------------------------------------------
# XLA compilation caching
# ---------------------------------------------------------------------------


def _get_jax_cache_env() -> dict[str, str]:
    """JAX/XLA compilation cache env vars.

    Reuses the pattern from ``marin.inference.vllm_server._vllm_jax_env()``
    to ensure compilation artifacts are cached across shards.
    """
    marin_prefix = os.environ.get("MARIN_PREFIX")
    if marin_prefix:
        cache_dir = os.path.join(marin_prefix, "compilation-cache")
    else:
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/marin-jax-compilation-cache")

    return {
        "JAX_ENABLE_COMPILATION_CACHE": "1",
        "JAX_COMPILATION_CACHE_DIR": cache_dir,
        "VLLM_XLA_CACHE_PATH": cache_dir,
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "2",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "JAX_PLATFORMS": "tpu,cpu",
    }


# ---------------------------------------------------------------------------
# Persistent engine cache (lives for the lifetime of a Zephyr worker)
# ---------------------------------------------------------------------------

_ENGINE_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_or_create_engine(config: InferenceV2Config) -> tuple[Any, Any]:
    """Get or create a persistent vLLM engine, cached per model_name.

    Returns:
        Tuple of ``(LLM, SamplingParams)``.
    """
    if config.model_name not in _ENGINE_CACHE:
        # Set env vars BEFORE importing vLLM — importing vllm triggers JAX
        # initialization, which reads JAX_PLATFORMS exactly once. The parent
        # fray job sets JAX_PLATFORMS="" (empty) for TPU configs, so we must
        # hard-set (not setdefault) to ensure JAX finds TPU devices.
        for k, v in _get_jax_cache_env().items():
            os.environ[k] = v

        from vllm import LLM, SamplingParams

        t0 = time.monotonic()

        engine_kwargs = {
            "tensor_parallel_size": config.tensor_parallel_size,
            **config.engine_kwargs,
        }
        llm = LLM(model=config.model_name, **engine_kwargs)
        sampling_params = SamplingParams(**config.generation_kwargs)

        elapsed = time.monotonic() - t0
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "unknown")
        logger.info(
            "Engine initialized: model=%s, tp=%d, load_time=%.1fs, cache_dir=%s",
            config.model_name,
            config.tensor_parallel_size,
            elapsed,
            cache_dir,
        )
        _ENGINE_CACHE[config.model_name] = (llm, sampling_params)

    return _ENGINE_CACHE[config.model_name]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_prompts(
    records: list[dict],
    config: InferenceV2Config,
    tokenizer: Any,
) -> list[Any]:
    """Format records into prompts, applying chat template if configured.

    For each record:
      1. Extract ``record[prompt_column]`` and truncate to ``max_doc_tokens``.
      2. Substitute into ``template`` (if provided).
      3. If ``apply_chat_template``: wrap with system_message and apply
         ``tokenizer.apply_chat_template``.
      4. Return as ``TokensPrompt`` (pre-tokenized) for efficiency.

    Returns a list of prompts (strings or TokensPrompt objects).
    """
    from vllm.inputs.data import TokensPrompt

    prompts: list[Any] = []

    for record in records:
        text = record.get(config.prompt_column, "")

        # Truncate to max_doc_tokens (only if non-empty)
        if text:
            tokens = tokenizer.encode(text)
            if len(tokens) > config.max_doc_tokens:
                tokens = tokens[: config.max_doc_tokens]
                text = tokenizer.decode(tokens)

        # Apply template
        if config.template is not None:
            text = config.template.format(example=text)

        # Apply chat template
        if config.apply_chat_template:
            messages: list[dict[str, str]] = []
            if config.system_message:
                messages.append({"role": "system", "content": config.system_message})
            messages.append({"role": "user", "content": text})

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Pre-tokenize for efficiency — avoids double tokenization in vLLM
            prompt_token_ids = tokenizer.encode(prompt_text)
            prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
        else:
            prompts.append(text)

    return prompts


# ---------------------------------------------------------------------------
# Record counting for shard sizing
# ---------------------------------------------------------------------------


def _count_records(input_path: str, input_format: str) -> int:
    """Count total records across all input files to compute shard count.

    For JSONL variants: counts lines (fast — no JSON parsing).
    For parquet: reads parquet metadata to get row counts (zero data loading).
    """
    files = fsspec_glob(input_path)
    if not files:
        raise ValueError(f"No files matched input_path: {input_path}")

    total = 0

    if input_format == "parquet":
        import pyarrow.parquet as pq

        for path in files:
            with fsspec.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                total += pf.metadata.num_rows
    elif input_format in ("jsonl.gz", "jsonl.zst"):
        for path in files:
            with fsspec.open(path, "rb") as f:
                if input_format == "jsonl.gz":
                    with gzip.open(f, "rt", encoding="utf-8") as gz:
                        total += sum(1 for _ in gz)
                else:
                    import zstandard as zstd

                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as reader:
                        text_reader = io.TextIOWrapper(reader, encoding="utf-8")
                        total += sum(1 for _ in text_reader)
    elif input_format == "jsonl":
        for path in files:
            with fsspec.open(path, "r") as f:
                total += sum(1 for _ in f)
    else:
        raise ValueError(f"Unsupported input_format for counting: {input_format}")

    return total


# ---------------------------------------------------------------------------
# Core shard processor
# ---------------------------------------------------------------------------


def _process_shard(records: Iterator[dict], _shard_info: Any = None) -> Iterator[dict]:
    """Process an entire shard through vLLM.

    Called by Zephyr ``map_shard``. Each invocation:
    1. Materializes the shard into a list.
    2. Gets (or creates) the persistent vLLM engine.
    3. Formats all prompts.
    4. Submits a single ``generate()`` call — vLLM handles continuous batching.
    5. Yields records with the generated text column added.
    """
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: InferenceV2Config = ctx.get_shared("config")

    record_list = list(records)
    if not record_list:
        return

    llm, sampling_params = _get_or_create_engine(config)
    tokenizer = llm.get_tokenizer()

    # Format prompts, tracking which records have non-empty text.
    # Records with empty prompt_column get an empty generated_text
    # and are excluded from the generate() call to avoid vLLM's
    # "decoder prompt cannot be empty" error.
    all_prompts = _format_prompts(record_list, config, tokenizer)
    valid_indices: list[int] = []
    valid_prompts: list[Any] = []
    for i, prompt in enumerate(all_prompts):
        is_empty = (isinstance(prompt, str) and not prompt) or (
            hasattr(prompt, "prompt_token_ids") and not prompt.prompt_token_ids
        )
        if not is_empty:
            valid_indices.append(i)
            valid_prompts.append(prompt)

    skipped = len(record_list) - len(valid_prompts)
    if skipped:
        logger.info("Shard: skipping %d records with empty prompts", skipped)

    # Single generate() call — vLLM does continuous batching internally.
    if valid_prompts:
        logger.info("Shard: generating for %d prompts", len(valid_prompts))
        t0 = time.monotonic()
        outputs = llm.generate(valid_prompts, sampling_params)
        elapsed = time.monotonic() - t0

        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        logger.info(
            "Shard: %d prompts, %.1fs, %.1f tok/s output",
            len(valid_prompts),
            elapsed,
            total_output_tokens / max(elapsed, 0.01),
        )
    else:
        outputs = []

    # Map outputs back to records. Records with empty prompts get empty text.
    output_map: dict[int, str] = {}
    for idx, output in zip(valid_indices, outputs, strict=True):
        output_map[idx] = " ".join([o.text for o in output.outputs])

    for i, record in enumerate(record_list):
        record[config.generated_text_column] = output_map.get(i, "")
        yield record


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_inference_v2(config: InferenceV2Config) -> None:
    """Zephyr-based inference with persistent vLLM engines on TPU workers.

    Resolves input files, pre-counts records for shard sizing, builds a Zephyr
    pipeline with resharding, and executes with TPU-provisioned workers.
    """
    from fray.v2 import ResourceConfig, TpuConfig
    from zephyr import Dataset, ZephyrContext, load_file, load_jsonl

    # Pick the right file loader based on input_format
    if config.input_format == "parquet":
        loader = load_file
    else:
        loader = load_jsonl

    # Pre-count records and compute shard count
    total_records = _count_records(config.input_path, config.input_format)
    num_shards = max(config.num_workers, math.ceil(total_records / config.records_per_shard))
    logger.info(
        "Inference V2: input=%s, %d records -> %d shards (%d records/shard), " "model=%s, tpu=%s, workers=%d",
        config.input_path,
        total_records,
        num_shards,
        config.records_per_shard,
        config.model_name,
        config.tpu_type,
        config.num_workers,
    )

    # Build pipeline with resharding for right-sized shards
    output_pattern = f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}"
    ds = Dataset.from_files(config.input_path).flat_map(loader).reshard(num_shards).map_shard(_process_shard)

    # Write in the requested output format with shard-level checkpointing
    if config.output_format == "parquet":
        ds = ds.write_parquet(f"{output_pattern}.parquet", skip_existing=True)
    else:
        ds = ds.write_jsonl(f"{output_pattern}.{config.output_format}", skip_existing=True)

    # Retry loop: when the coordinator actor gets preempted/OOM'd, the
    # ZephyrContext.execute() call raises ActorUnavailableError. Since we use
    # skip_existing=True, completed shards survive across retries. We just
    # re-create the context and re-execute — only unfinished shards run.
    max_retries = 10
    output_files: list[str] = []
    for attempt in range(1, max_retries + 1):
        try:
            # Use lightweight CPU/RAM with explicit TpuConfig device. The TpuConfig
            # triggers a TPU-{variant}-head resource request in fray's actor options,
            # ensuring each actor gets exclusive access to a TPU node.
            ctx = ZephyrContext(
                name="inference-v2",
                max_workers=config.num_workers,
                resources=ResourceConfig(cpu=8, ram="16g", device=TpuConfig(variant=config.tpu_type)),
                chunk_size=config.records_per_shard,
            )
            ctx.put("config", config)
            output_files = list(ctx.execute(ds))
            break  # Success — exit retry loop
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed with %s: %s. Retrying (completed shards will be skipped)...",
                    attempt,
                    max_retries,
                    type(e).__name__,
                    str(e)[:200],
                )
                # Brief pause before retry to let the cluster stabilize
                time.sleep(30)
                continue
            logger.error("All %d attempts exhausted. Last error: %s", max_retries, e)
            raise

    # Write stats summary
    stats = {
        "model": config.model_name,
        "tpu_type": config.tpu_type,
        "num_workers": config.num_workers,
        "total_records": total_records,
        "records_per_shard": config.records_per_shard,
        "num_shards": num_shards,
        "output_files": len(output_files),
    }
    stats_path = f"{config.output_path}/inference_v2_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        "Inference V2 complete: %d output files -> %s",
        len(output_files),
        config.output_path,
    )
