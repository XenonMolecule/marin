# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr-based inference with llama.cpp CPU servers.

Drop-in replacement for ``inference_v2.py`` that uses llama.cpp's ``llama-server``
on CPU instead of vLLM on TPU. Each Zephyr worker starts a local llama-server
subprocess, sends requests to ``/v1/chat/completions``, and yields records with
a ``generated_text`` column — identical output format to inference_v2.

Designed for horizontal scaling across many CPU workers on TPU nodes where the
CPUs are otherwise idle. The model (GGUF format, ~1 GiB for Q4_K_M) and the
llama-server binary are downloaded from GCS on first use and cached for the
worker's lifetime.

Example usage as an executor step::

    inference_step = ExecutorStep(
        name="documents/my_inference_cpu",
        fn=run_inference_llamacpp,
        config=LlamaCppInferenceConfig(
            input_path=filter_step / "*.jsonl.gz",
            output_path=this_output_path(),
            gguf_model_path="gs://bucket/models/model.gguf",
            llamacpp_binary_path=build_step,
            system_message="You are a helpful assistant.",
            template="Extract content from this HTML:\\n\\n{example}",
            prompt_column="html",
        ),
        resources=ResourceConfig.with_cpu(cpu=16, ram="16g"),
        pip_dependency_groups=["cpu"],
    )
"""

import fcntl
import gzip
import hashlib
import io
import json
import logging
import math
import os
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import fsspec
import requests

from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppInferenceConfig:
    """Configuration for Zephyr-based llama.cpp CPU inference."""

    input_path: str
    """Glob pattern for input files (e.g. ``gs://bucket/data/*.jsonl.gz``)."""

    output_path: str
    """Directory to write inference output."""

    gguf_model_path: str
    """Path to GGUF model file. Supports GCS (``gs://...``), HTTP(S) URLs
    (e.g. HuggingFace ``https://huggingface.co/.../resolve/main/model.gguf``),
    or any fsspec-compatible path."""

    llamacpp_binary_path: str
    """GCS path to directory containing the llama-server binary.
    Typically the output_path of a BuildLlamaCppConfig executor step."""

    # Format
    input_format: str = "jsonl.gz"
    """Input file format: ``"jsonl.gz"``, ``"jsonl.zst"``, ``"jsonl"``, or ``"parquet"``."""

    output_format: str = "jsonl.gz"
    """Output file format: ``"jsonl.gz"`` or ``"parquet"``."""

    # Prompting
    system_message: str | None = None
    """System message for chat completion."""

    template: str | None = None
    """Prompt template with ``{example}`` placeholder. If None, raw text is used."""

    prompt_column: str = "text"
    """Column name containing the input text."""

    generated_text_column: str = "generated_text"
    """Column name for the generated output text."""

    # Generation
    max_tokens: int = 4096
    """Maximum number of tokens to generate."""

    temperature: float = 0.0
    """Sampling temperature."""

    extra_generation_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs passed to the /v1/chat/completions request body."""

    # Server configuration — tune based on benchmark results
    threads_per_worker: int = 16
    """Threads for each llama-server instance."""

    context_length: int = 32768
    """Context window size for llama-server."""

    # Scaling
    num_workers: int = 32
    """Number of Zephyr workers (each gets a CPU allocation and its own server)."""

    cpu_per_worker: int = 16
    """CPU cores to request per Zephyr worker."""

    ram_per_worker: str = "16g"
    """RAM to request per Zephyr worker."""

    # Sharding
    records_per_shard: int = 100
    """Target number of records per shard. Smaller = better checkpointing."""

    # Timeouts
    server_startup_timeout: int = 120
    """Seconds to wait for llama-server to become healthy."""

    request_timeout: int = 3600
    """Seconds to wait for a single completion request. Must be large enough
    for the longest expected document: a 28k-token prompt with 4096 gen tokens
    takes ~762s at 16 threads on an uncontended Xeon 8481C, but with multiple
    workers sharing a node the effective time can be 2-3x longer."""


# ---------------------------------------------------------------------------
# Persistent server cache (lives for the lifetime of a Zephyr worker)
# ---------------------------------------------------------------------------

_SERVER_CACHE: dict[str, tuple[subprocess.Popen, str]] = {}


def _download_file(remote_path: str, local_path: str) -> str:
    """Download a file from GCS or HTTP(S), with atomic write and cross-process locking.

    Multiple Zephyr workers may share a physical node and race to download the
    same file. We use a lock file + atomic rename to ensure only one worker
    downloads at a time and others never see a partially-written file.
    """
    if os.path.isfile(local_path):
        logger.debug("Already cached: %s", local_path)
        return local_path

    parent_dir = os.path.dirname(local_path)
    os.makedirs(parent_dir, exist_ok=True)

    # Acquire an exclusive lock so only one process on this node downloads.
    lock_path = local_path + ".lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            # Re-check after acquiring lock — another process may have finished.
            if os.path.isfile(local_path):
                logger.debug("Already cached (after lock): %s", local_path)
                return local_path

            logger.info("Downloading %s -> %s", remote_path, local_path)

            # Write to a temp file in the same directory, then atomically rename.
            fd, tmp_path = tempfile.mkstemp(dir=parent_dir, suffix=".tmp")
            try:
                if remote_path.startswith(("http://", "https://")):
                    resp = requests.get(remote_path, stream=True, timeout=600)
                    resp.raise_for_status()
                    with os.fdopen(fd, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                            f.write(chunk)
                else:
                    with fsspec.open(remote_path, "rb") as remote_f:
                        with os.fdopen(fd, "wb") as local_f:
                            local_f.write(remote_f.read())
                # Make executable before rename so other processes never see
                # a non-executable file (also avoids "Text file busy" from
                # chmod-ing a running binary).
                os.chmod(tmp_path, 0o755)
                os.rename(tmp_path, local_path)
            except BaseException:
                os.unlink(tmp_path)
                raise
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)

    return local_path


def _get_or_start_server(config: LlamaCppInferenceConfig) -> str:
    """Start llama-server if not already running, return server URL.

    The server persists across shards via the module-level _SERVER_CACHE,
    avoiding model reload overhead on every shard.
    """
    cache_key = config.gguf_model_path
    if cache_key in _SERVER_CACHE:
        proc, url = _SERVER_CACHE[cache_key]
        if proc.poll() is None:
            return url
        logger.warning("Cached server process died (rc=%s), restarting...", proc.returncode)
        del _SERVER_CACHE[cache_key]

    # Download GGUF model. Include a hash of the remote path so that
    # switching models does not reuse stale local files.
    model_filename = os.path.basename(config.gguf_model_path)
    model_hash = hashlib.md5(config.gguf_model_path.encode()).hexdigest()[:8]
    local_model = _download_file(
        config.gguf_model_path,
        f"/tmp/llamacpp-models-{model_hash}/{model_filename}",
    )

    # Download llama-server binary. Include a hash of the GCS path in the
    # local directory so different build versions don't collide — prevents
    # stale binaries from being reused after a rebuild.
    binary_gcs = os.path.join(config.llamacpp_binary_path, "llama-server")
    bin_hash = hashlib.md5(binary_gcs.encode()).hexdigest()[:8]
    local_binary = _download_file(binary_gcs, f"/tmp/llamacpp-bin-{bin_hash}/llama-server")
    # Only chmod if not already executable — avoids "Text file busy" when
    # another worker on the same node is already running the binary.
    current_mode = os.stat(local_binary).st_mode
    if not (current_mode & stat.S_IEXEC):
        os.chmod(local_binary, current_mode | stat.S_IEXEC)

    # Find a free port. Multiple workers may share a physical node, so we
    # cannot use a fixed port. Bind-and-release is racy but sufficient here
    # because each worker starts exactly one server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    cmd = [
        local_binary,
        "--model",
        local_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--threads",
        str(config.threads_per_worker),
        "--ctx-size",
        str(config.context_length),
        "--n-predict",
        str(config.max_tokens),
    ]
    logger.info("Starting llama-server: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be healthy
    url = f"http://127.0.0.1:{port}"
    t0 = time.monotonic()
    for attempt in range(1, config.server_startup_timeout + 1):
        if proc.poll() is not None:
            output = proc.stdout.read().decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"llama-server exited during startup (rc={proc.returncode}): {output}")
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                # Double-check: send a tiny completion to verify model is loaded
                test_resp = requests.post(
                    f"{url}/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                    timeout=30,
                )
                if test_resp.status_code != 503:
                    elapsed = time.monotonic() - t0
                    logger.info("llama-server healthy in %.1fs on port %d", elapsed, port)
                    break
        except requests.ConnectionError:
            pass

        if attempt % 10 == 0:
            logger.info("Waiting for llama-server... (%ds)", attempt)
        time.sleep(1)
    else:
        proc.kill()
        raise TimeoutError(f"llama-server not healthy after {config.server_startup_timeout}s")

    _SERVER_CACHE[cache_key] = (proc, url)
    return url


# ---------------------------------------------------------------------------
# Record counting (shared with inference_v2)
# ---------------------------------------------------------------------------


def _count_records(input_path: str, input_format: str) -> int:
    """Count total records across all input files to compute shard count."""
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


def _process_shard(records: Iterator[dict]) -> Iterator[dict]:
    """Process an entire shard through llama-server.

    Called by Zephyr ``map_shard``. Each invocation:
    1. Gets (or starts) the persistent llama-server.
    2. For each record, formats a chat completion request and POSTs it.
    3. Yields records with the generated text column added.

    FUTURE OPTIMIZATION: Currently requests are sent sequentially (one at a time).
    llama-server supports continuous batching via the ``--parallel N`` flag, which
    allows it to process multiple requests concurrently. Using a ThreadPoolExecutor
    to send N requests in parallel while the server batches them could yield a 2-3x
    throughput improvement per worker, since the server can overlap prefill and
    generation across requests. To implement:
      1. Add ``--parallel N`` to the server launch command in _get_or_start_server
      2. Use concurrent.futures.ThreadPoolExecutor(max_workers=N) here to POST
         multiple requests simultaneously
      3. Collect results and yield in order
    """
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: LlamaCppInferenceConfig = ctx.get_shared("config")

    record_list = list(records)
    if not record_list:
        return

    server_url = _get_or_start_server(config)

    t0 = time.monotonic()
    total_gen_tokens = 0

    for record in record_list:
        text = record.get(config.prompt_column, "")
        if not text:
            record[config.generated_text_column] = ""
            yield record
            continue

        # Apply template
        if config.template is not None:
            text = config.template.format(example=text)

        # Build chat messages
        messages: list[dict[str, str]] = []
        if config.system_message:
            messages.append({"role": "system", "content": config.system_message})
        messages.append({"role": "user", "content": text})

        # POST to /v1/chat/completions
        request_body = {
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            **config.extra_generation_kwargs,
        }

        try:
            resp = requests.post(
                f"{server_url}/v1/chat/completions",
                json=request_body,
                timeout=config.request_timeout,
            )
            resp.raise_for_status()
            result = resp.json()

            generated_text = result["choices"][0]["message"]["content"]
            gen_tokens = result.get("usage", {}).get("completion_tokens", 0)
            total_gen_tokens += gen_tokens
        except Exception:
            logger.exception("Request failed for record (prompt_column=%s)", config.prompt_column)
            generated_text = ""

        record[config.generated_text_column] = generated_text
        yield record

    elapsed = time.monotonic() - t0
    if elapsed > 0 and total_gen_tokens > 0:
        logger.info(
            "Shard: %d records, %.1fs, %d gen tokens, %.1f tok/s",
            len(record_list),
            elapsed,
            total_gen_tokens,
            total_gen_tokens / elapsed,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_inference_llamacpp(config: LlamaCppInferenceConfig) -> None:
    """Zephyr-based inference with llama.cpp CPU servers.

    Resolves input files, pre-counts records for shard sizing, builds a Zephyr
    pipeline with resharding, and executes with CPU-provisioned workers.
    """
    from fray.v2 import ResourceConfig
    from zephyr import Dataset, ZephyrContext, load_file, load_jsonl

    if config.input_format == "parquet":
        loader = load_file
    else:
        loader = load_jsonl

    total_records = _count_records(config.input_path, config.input_format)
    num_shards = max(config.num_workers, math.ceil(total_records / config.records_per_shard))
    logger.info(
        "Inference llama.cpp: input=%s, %d records -> %d shards (%d records/shard), " "model=%s, workers=%d, threads=%d",
        config.input_path,
        total_records,
        num_shards,
        config.records_per_shard,
        config.gguf_model_path,
        config.num_workers,
        config.threads_per_worker,
    )

    output_pattern = f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}"
    ds = Dataset.from_files(config.input_path).flat_map(loader).reshard(num_shards).map_shard(_process_shard)

    if config.output_format == "parquet":
        ds = ds.write_parquet(f"{output_pattern}.parquet", skip_existing=True)
    else:
        ds = ds.write_jsonl(f"{output_pattern}.{config.output_format}", skip_existing=True)

    # Retry loop for preemption resilience (same pattern as inference_v2)
    max_retries = 10
    output_files: list[str] = []
    for attempt in range(1, max_retries + 1):
        try:
            with ZephyrContext(
                name="inference-llamacpp",
                num_workers=config.num_workers,
                resources=ResourceConfig.with_cpu(
                    cpu=config.cpu_per_worker,
                    ram=config.ram_per_worker,
                ),
                chunk_size=config.records_per_shard,
            ) as ctx:
                ctx.put("config", config)
                output_files = list(ctx.execute(ds))
            break
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    "Attempt %d/%d failed with %s: %s. Retrying...",
                    attempt,
                    max_retries,
                    type(e).__name__,
                    str(e)[:200],
                )
                time.sleep(30)
                continue
            logger.error("All %d attempts exhausted. Last error: %s", max_retries, e)
            raise

    # Write stats
    stats = {
        "model": config.gguf_model_path,
        "num_workers": config.num_workers,
        "threads_per_worker": config.threads_per_worker,
        "cpu_per_worker": config.cpu_per_worker,
        "total_records": total_records,
        "records_per_shard": config.records_per_shard,
        "num_shards": num_shards,
        "output_files": len(output_files),
    }
    stats_path = f"{config.output_path}/inference_llamacpp_stats.json"
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        "Inference llama.cpp complete: %d output files -> %s",
        len(output_files),
        config.output_path,
    )
