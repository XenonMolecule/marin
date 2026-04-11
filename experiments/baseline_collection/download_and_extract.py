# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combined WARC download + LLM extraction in a single step.

Downloads WARC files from CommonCrawl HTTPS (free ingress to any GCP region),
runs LLM-based text extraction via vLLM on TPU, post-processes inline, and
writes output to the worker's local region bucket. Designed for multi-region
orchestration where Iris places jobs wherever compute is available.

Key properties:
- **No region pinning**: Output path determined at runtime via ``marin_prefix()``.
- **TPU-type agnostic**: ``tensor_parallel_size`` derived at runtime from TPU variant.
- **Shard-level checkpointing**: One WARC = one shard. ``skip_existing=True`` with
  deterministic filenames means completed WARCs survive restarts.
- **Inline post-processing**: Strips thinking tokens, DSPy markers, filters
  ``[NO_USEFUL_CONTENT]`` records. Output is ready for tokenization.

Usage::

    python download_and_extract.py --config_path gs://bucket/config.json
"""

import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import fsspec

from experiments.baseline_collection.download_warcs import (
    _download_one_warc,
    _load_manifest,
    _warc_path_hash,
)

# NOTE: Do NOT import from marin.generation.inference_v2 or
# marin.transform.postprocess_extraction at the top level.
# Those modules trigger heavyweight import chains (transformers → torch → CUDA)
# that fail on the Iris entrypoint container (no GPU/TPU libs).
# All heavy imports are deferred to functions that run on TPU workers.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TPU → tensor_parallel_size mapping
# ---------------------------------------------------------------------------

TPU_TO_TP_SIZE: dict[str, int] = {
    "v5p-8": 4,
    "v6e-4": 4,
    "v6e-8": 8,
    "v5e-4": 4,
    "v5e-8": 8,
    "v5litepod-4": 4,
    "v5litepod-8": 8,
}


# ---------------------------------------------------------------------------
# Inline helpers (avoid heavyweight top-level imports)
# ---------------------------------------------------------------------------

# Regex patterns from postprocess_extraction.py — inlined to avoid
# importing zephyr/transformers/torch at module level.
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(
    r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]",
    flags=re.IGNORECASE,
)


def _clean_text(raw_text: str, strip_thinking: bool) -> str:
    """Strip thinking tokens and DSPy field markers from raw model output."""
    text = raw_text
    if strip_thinking:
        text = _THINK_RE.sub("", text)
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def _get_jax_cache_env() -> dict[str, str]:
    """JAX/vLLM compilation cache + TPU env vars. Inlined from inference_v2.py."""
    marin_pfx = os.environ.get("MARIN_PREFIX")
    if marin_pfx:
        cache_dir = os.path.join(marin_pfx, "compilation-cache")
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
        # Force PJRT TPU plugin — needed when JAX auto-detect misses TPU
        # (e.g. in Zephyr workers where fray imports JAX before libtpu is loaded)
        "PJRT_DEVICE": "TPU",
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DownloadAndExtractConfig:
    """Configuration for combined WARC download + LLM extraction."""

    # --- Input ---
    warc_manifest_path: str
    """Text file with one WARC S3 path per line (for this batch)."""

    output_subdir: str
    """Subdirectory under the regional bucket, e.g. ``documents/baseline_llm_extraction``.
    Full output path is ``marin_prefix() / output_subdir`` (resolved at runtime)."""

    # --- Model ---
    model_name_by_region: dict[str, str]
    """Mapping of GCP region → local model weights path. Worker picks the one
    matching its runtime region. Example::

        {"us-central1": "gs://marin-us-central1/models/qwen3-8b-extraction/hf/step-1318",
         "europe-west4": "gs://marin-eu-west4/models/qwen3-8b-extraction/hf/step-1318"}
    """

    # --- Prompting ---
    template: str
    """Prompt template with ``{example}`` placeholder for the HTML content."""

    system_message: str
    """System message for chat template."""

    prompt_column: str = "html"
    generated_text_column: str = "generated_text"
    apply_chat_template: bool = True
    max_doc_tokens: int = 28672
    """Max input tokens. Documents exceeding this (after char-length fast path) are dropped."""

    # --- Engine ---
    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs passed to ``vllm.LLM()``."""

    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    """Kwargs passed to ``vllm.SamplingParams()``."""

    # NOTE: tensor_parallel_size is derived at runtime from the TPU variant
    # the worker lands on (see TPU_TO_TP_SIZE). Not a config field.

    # --- Post-processing (inline) ---
    strip_thinking: bool = True
    filter_patterns: list[str] = field(default_factory=lambda: [r"\[NO_USEFUL_CONTENT\]"])
    min_output_chars: int = 50

    # --- Scaling ---
    num_workers: int = 16
    max_records_per_generate: int = 500
    """Max records per vLLM ``generate()`` call. Large WARCs are subdivided."""

    # NOTE: tpu_type is specified at the Iris job level, not here.
    # The worker discovers its TPU variant via IRIS_WORKER_DEVICE_VARIANT.

    # --- Download ---
    http_timeout: int = 600
    max_retries: int = 5


# ---------------------------------------------------------------------------
# Runtime discovery
# ---------------------------------------------------------------------------


def _resolve_output_path(config: DownloadAndExtractConfig) -> str:
    """Determine output path based on where the worker is running."""
    from rigging.filesystem import marin_prefix

    prefix = marin_prefix()
    return f"{prefix}/{config.output_subdir}"


def _resolve_model_name(config: DownloadAndExtractConfig) -> str:
    """Pick the model weights copy local to this worker's region."""
    from rigging.filesystem import marin_region

    region = marin_region()
    if region and region in config.model_name_by_region:
        return config.model_name_by_region[region]
    # Fallback: use first available (may incur cross-region read on first load)
    logger.warning("Region %r not in model_name_by_region; falling back to first entry", region)
    return next(iter(config.model_name_by_region.values()))


def _resolve_tp_size() -> int:
    """Derive tensor_parallel_size from the TPU this worker is running on."""
    variant = os.environ.get("IRIS_WORKER_DEVICE_VARIANT", "")
    if variant and variant in TPU_TO_TP_SIZE:
        return TPU_TO_TP_SIZE[variant]
    # Fallback: count JAX devices (works on any TPU)
    try:
        import jax

        return jax.device_count()
    except Exception:
        logger.warning("Could not detect TPU variant or JAX devices; defaulting tp=4")
        return 4


# ---------------------------------------------------------------------------
# Persistent vLLM engine (per-worker cache)
# ---------------------------------------------------------------------------

_ENGINE_CACHE: dict[str, tuple[Any, Any]] = {}


def _get_or_create_engine(
    model_name: str,
    tensor_parallel_size: int,
    engine_kwargs: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Get or create a persistent vLLM engine, cached per ``model_name``."""
    if model_name not in _ENGINE_CACHE:
        for k, v in _get_jax_cache_env().items():
            os.environ[k] = v

        # Force JAX to re-discover TPU. The Zephyr worker startup may have
        # initialized JAX with CPU before libtpu was loaded. Re-initializing
        # isn't straightforward, but we can at least log the state.
        # The actual fix is PJRT_DEVICE=TPU set at the Iris env level.
        logger.info(
            "JAX_PLATFORMS=%s, PJRT_DEVICE=%s",
            os.environ.get("JAX_PLATFORMS", "<unset>"),
            os.environ.get("PJRT_DEVICE", "<unset>"),
        )

        from vllm import LLM, SamplingParams

        t0 = time.monotonic()
        kwargs = {"tensor_parallel_size": tensor_parallel_size, **engine_kwargs}
        llm = LLM(model=model_name, **kwargs)
        sampling_params = SamplingParams(**generation_kwargs)

        logger.info(
            "Engine initialized: model=%s, tp=%d, load_time=%.1fs",
            model_name,
            tensor_parallel_size,
            time.monotonic() - t0,
        )
        _ENGINE_CACHE[model_name] = (llm, sampling_params)

    return _ENGINE_CACHE[model_name]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def _format_prompts(
    records: list[dict],
    config: DownloadAndExtractConfig,
    tokenizer: Any,
) -> list[Any]:
    """Format records into prompts with chat template applied."""
    from vllm.inputs.data import TokensPrompt

    prompts: list[Any] = []
    for record in records:
        text = record.get(config.prompt_column, "")

        # Truncate to max_doc_tokens
        if text:
            tokens = tokenizer.encode(text)
            if len(tokens) > config.max_doc_tokens:
                tokens = tokens[: config.max_doc_tokens]
                text = tokenizer.decode(tokens)

        # Apply template
        if config.template:
            text = config.template.format(example=text)

        # Apply chat template
        if config.apply_chat_template:
            messages: list[dict[str, str]] = []
            if config.system_message:
                messages.append({"role": "system", "content": config.system_message})
            messages.append({"role": "user", "content": text})

            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_token_ids = tokenizer.encode(prompt_text)
            prompts.append(TokensPrompt(prompt_token_ids=prompt_token_ids))
        else:
            prompts.append(text)

    return prompts


# ---------------------------------------------------------------------------
# Token-length filter (character-length fast path)
# ---------------------------------------------------------------------------


def _filter_by_token_length(
    records: list[dict],
    max_doc_tokens: int,
    prompt_column: str,
    tokenizer: Any,
) -> list[dict]:
    """Filter records that exceed max_doc_tokens using a character fast path.

    Char heuristic from ``filter_by_token_length.py``:
    - ``len <= max_doc_tokens``: definitely fits (1 char >= 1 token)
    - ``len > max_doc_tokens * 6``: definitely too long
    - Otherwise: tokenize to check
    """
    kept: list[dict] = []
    for record in records:
        text = record.get(prompt_column, "")
        char_len = len(text)

        if char_len <= max_doc_tokens:
            kept.append(record)
        elif char_len > max_doc_tokens * 6:
            continue
        else:
            token_len = len(tokenizer.encode(text))
            if token_len <= max_doc_tokens:
                kept.append(record)

    return kept


# ---------------------------------------------------------------------------
# Core shard processor
# ---------------------------------------------------------------------------


def _chunked(lst: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _process_warc_shard(warc_paths_iter: Iterator[str], _shard_info: Any = None) -> Iterator[dict]:
    """Process one WARC: download → filter → extract → postprocess.

    Called by Zephyr ``map_shard``. Each shard is a single WARC URL string.
    """
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: DownloadAndExtractConfig = ctx.get_shared("config")

    # Get the single WARC path for this shard
    warc_path = next(warc_paths_iter)
    logger.info("Processing WARC: %s", warc_path)

    # 1. Download and parse HTML records
    records = _download_one_warc(warc_path)
    if not records:
        logger.warning("No HTML records from WARC: %s", warc_path)
        return

    logger.info("Downloaded %d HTML records from %s", len(records), warc_path)

    # 2. Get vLLM engine (persistent per worker)
    model_name = _resolve_model_name(config)
    tp_size = _resolve_tp_size()
    llm, sampling_params = _get_or_create_engine(model_name, tp_size, config.engine_kwargs, config.generation_kwargs)
    tokenizer = llm.get_tokenizer()

    # 3. Filter by token length
    records = _filter_by_token_length(records, config.max_doc_tokens, config.prompt_column, tokenizer)
    if not records:
        logger.info("All records filtered by token length for %s", warc_path)
        return

    logger.info("%d records after token-length filter for %s", len(records), warc_path)

    # 4. Compile exclusion patterns for post-processing
    compiled_patterns = [re.compile(p) for p in config.filter_patterns]

    # 5. Process in sub-batches to avoid OOM
    total_kept = 0
    total_filtered = 0

    for batch in _chunked(records, config.max_records_per_generate):
        # Format prompts
        prompts = _format_prompts(batch, config, tokenizer)
        valid_indices: list[int] = []
        valid_prompts: list[Any] = []

        for i, prompt in enumerate(prompts):
            is_empty = (isinstance(prompt, str) and not prompt) or (
                hasattr(prompt, "prompt_token_ids") and not prompt.prompt_token_ids
            )
            if not is_empty:
                valid_indices.append(i)
                valid_prompts.append(prompt)

        # Generate
        output_map: dict[int, str] = {}
        if valid_prompts:
            t0 = time.monotonic()
            outputs = llm.generate(valid_prompts, sampling_params)
            elapsed = time.monotonic() - t0

            total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
            logger.info(
                "Batch: %d prompts, %.1fs, %.1f tok/s",
                len(valid_prompts),
                elapsed,
                total_output_tokens / max(elapsed, 0.01),
            )

            for idx, output in zip(valid_indices, outputs, strict=True):
                output_map[idx] = " ".join([o.text for o in output.outputs])

        # Post-process and yield
        for i, record in enumerate(batch):
            raw_text = output_map.get(i, "")
            record[config.generated_text_column] = raw_text

            # Inline post-processing
            cleaned = _clean_text(raw_text, config.strip_thinking)
            record["text"] = cleaned

            # Filter: skip empty, short, or excluded records
            if len(cleaned) < config.min_output_chars:
                total_filtered += 1
                continue
            if any(p.search(cleaned) for p in compiled_patterns):
                total_filtered += 1
                continue

            total_kept += 1
            yield record

    logger.info("WARC %s: kept=%d, filtered=%d", warc_path, total_kept, total_filtered)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_download_and_extract(config: DownloadAndExtractConfig) -> None:
    """Download WARCs from CommonCrawl and extract text via LLM.

    Output path is determined at runtime from the worker's GCP region.
    Shard-level checkpointing via ``skip_existing=True`` — safe to restart.
    """
    from fray.v2 import ResourceConfig, TpuConfig
    from zephyr import Dataset, ZephyrContext

    warc_paths = _load_manifest(config.warc_manifest_path)
    logger.info("Loaded %d WARC paths from %s", len(warc_paths), config.warc_manifest_path)

    output_path = _resolve_output_path(config)
    logger.info("Output path (runtime-resolved): %s", output_path)

    def _output_path_fn(shard_idx: int, total_shards: int) -> str:
        h = _warc_path_hash(warc_paths[shard_idx])
        return f"{output_path}/data-{h}.jsonl.gz"

    pipeline = (
        Dataset.from_list(warc_paths)
        .reshard(len(warc_paths))
        .map_shard(_process_warc_shard)
        .write_jsonl(_output_path_fn, skip_existing=True)
    )

    # Resolve TPU type from Iris environment
    tpu_type = os.environ.get("IRIS_WORKER_DEVICE_VARIANT", "v5p-8")
    logger.info("TPU type (from env): %s", tpu_type)

    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            ctx = ZephyrContext(
                name="download-and-extract",
                max_workers=config.num_workers,
                resources=ResourceConfig(
                    cpu=8,
                    ram="32g",
                    device=TpuConfig(variant=tpu_type),
                ),
                chunk_size=config.max_records_per_generate,
            )
            ctx.put("config", config)
            output_files = list(ctx.execute(pipeline))
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed with %s: %s. Retrying " "(completed shards will be skipped)...",
                    attempt,
                    max_retries,
                    type(e).__name__,
                    str(e)[:200],
                )
                time.sleep(30)
                continue
            logger.error("All %d attempts exhausted. Last error: %s", max_retries, e)
            raise

    logger.info(
        "Download+extract complete: %d WARCs → %d output files → %s",
        len(warc_paths),
        len(output_files),
        output_path,
    )


# ---------------------------------------------------------------------------
# CLI entry point (invoked by iris job run)
# ---------------------------------------------------------------------------


def _load_config_from_json(config_path: str) -> DownloadAndExtractConfig:
    """Load config from a JSON file (local or GCS)."""
    with fsspec.open(config_path, "r") as f:
        data = json.load(f)
    return DownloadAndExtractConfig(**data)


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Download and extract WARCs via LLM")
    parser.add_argument("--config_path", required=True, help="GCS or local path to config JSON")
    args = parser.parse_args()

    config = _load_config_from_json(args.config_path)
    run_download_and_extract(config)


if __name__ == "__main__":
    main()
