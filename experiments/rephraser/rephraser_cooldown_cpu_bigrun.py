# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser CPU inference on marin-big-run (~59k idle CPUs on v4 nodes).

CPU-ONLY pipeline — all steps use ResourceConfig.with_cpu(). No TPU resources
are requested anywhere. Workers use idle CPUs on the existing v4 TPU nodes.

Pipeline: download WARCs → filter → build llama.cpp → download GGUF →
inference (3000 workers) → postprocess → tokenize. No training.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster big-run --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_cooldown_cpu_bigrun.py
"""

import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat

from experiments.llama import llama3_tokenizer
from marin.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.generation.build_llamacpp import BuildLlamaCppConfig, build_llamacpp
from marin.generation.inference_llamacpp import LlamaCppInferenceConfig, run_inference_llamacpp
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

from experiments.rephraser.rephraser_cooldown import (
    SPECS,
    SYSTEM_MESSAGE,
    USER_TEMPLATE_FMT,
    load_warc_paths,
    spec_hash,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GGUF_SOURCE_URL = (
    "https://huggingface.co/MichaelR207/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M-GGUF/resolve/main/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"
)
GGUF_FILENAME = "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"

REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# big-run has ~31,080 CPUs available.
# 3000 workers × 16 CPUs = 48,000 CPUs — Ray will schedule what fits,
# remaining workers queue until CPUs free up.
THREADS_PER_WORKER = 16
CPU_PER_WORKER = 16
RAM_PER_WORKER = "16g"
NUM_INFERENCE_WORKERS = 2500
CONTEXT_LENGTH = 32768

# ---------------------------------------------------------------------------
# Step 0: Build llama.cpp (runs once, cached by executor)
# ---------------------------------------------------------------------------
build_llamacpp_step = ExecutorStep(
    name="tools/llamacpp-build-v3",
    description="Build llama.cpp from source and upload llama-server binary to GCS.",
    fn=build_llamacpp,
    config=BuildLlamaCppConfig(output_path=this_output_path()),
    resources=ResourceConfig.with_cpu(cpu=8, ram="16g"),
    pip_dependency_groups=["cpu"],
)


# ---------------------------------------------------------------------------
# Step 0b: Download GGUF model to GCS (runs once, cached)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DownloadGGUFConfig:
    source_url: str
    output_path: str
    filename: str = "model.gguf"


def download_gguf_to_gcs(config: DownloadGGUFConfig) -> None:
    """Download a GGUF model from HuggingFace and upload to GCS."""
    import fsspec
    import requests as req

    dest = os.path.join(config.output_path, config.filename)
    logger.info("Downloading GGUF: %s -> %s", config.source_url, dest)

    resp = req.get(config.source_url, stream=True, timeout=600)
    resp.raise_for_status()
    with fsspec.open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)

    logger.info("GGUF model uploaded to %s", dest)


_gguf_url_hash = spec_hash(GGUF_SOURCE_URL)[:8]

download_gguf_step = ExecutorStep(
    name=f"models/gguf-{_gguf_url_hash}",
    description="Download GGUF model from HuggingFace and cache in GCS.",
    fn=download_gguf_to_gcs,
    config=DownloadGGUFConfig(
        source_url=GGUF_SOURCE_URL,
        output_path=this_output_path(),
        filename=GGUF_FILENAME,
    ),
    resources=ResourceConfig.with_cpu(cpu=2, ram="4g"),
)

# ---------------------------------------------------------------------------
# Step 1: Download & Extract HTML from WARCs
# SAME name as other rephraser experiments — reuses cached output if present
# ---------------------------------------------------------------------------
warc_paths = load_warc_paths(WARC_MANIFEST)

download_warcs = ExecutorStep(
    name="raw/commoncrawl/rephraser_sweep_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="64g"),
    pip_dependency_groups=["cpu"],
)

# Step 1b: Pre-filter HTML by token length
filter_html = ExecutorStep(
    name="filtered/rephraser_sweep_batch0_v2",
    description=f"Filter HTML documents exceeding {CONTEXT_LENGTH - 4096} tokens.",
    fn=filter_by_token_length,
    config=FilterByTokenLengthConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=REPHRASER_TOKENIZER,
        text_column="html",
        max_tokens=CONTEXT_LENGTH - 4096,
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Steps 2-4: Per-spec pipeline (inference → postprocess → tokenize)
# No training — tokenized data can be transferred for cooldown training later.
# ---------------------------------------------------------------------------
all_tokenize_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: CPU Inference via llama.cpp (3000 workers, CPU-only)
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_cpu",
        description=f"Run rephraser inference via llama.cpp CPU for spec {sid}.",
        fn=run_inference_llamacpp,
        config=LlamaCppInferenceConfig(
            input_path=filter_html / "*.jsonl.gz",
            output_path=this_output_path(),
            gguf_model_path=download_gguf_step / GGUF_FILENAME,
            llamacpp_binary_path=output_path_of(build_llamacpp_step),
            input_format="jsonl.gz",
            output_format="jsonl.gz",
            system_message=SYSTEM_MESSAGE,
            template=user_template,
            prompt_column="html",
            max_tokens=4096,
            temperature=0.0,
            threads_per_worker=THREADS_PER_WORKER,
            num_workers=NUM_INFERENCE_WORKERS,
            cpu_per_worker=CPU_PER_WORKER,
            ram_per_worker=RAM_PER_WORKER,
            context_length=CONTEXT_LENGTH,
            records_per_shard=10,
        ),
        resources=ResourceConfig.with_cpu(cpu=CPU_PER_WORKER, ram=RAM_PER_WORKER),
        pip_dependency_groups=["cpu"],
    )

    # Step 3: Post-process
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_spec_{sid}_cpu",
        description=f"Post-process extraction output for spec {sid} (CPU inference).",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
        pip_dependency_groups=["cpu"],
    )

    # Step 4: Tokenize with Meta-Llama-3.1-8B (nemotron-compatible)
    tokenize_step = ExecutorStep(
        name=f"tokenized/rephraser_spec_{sid}_cooldown_cpu",
        description=f"Tokenize extracted text for spec {sid} (llama3 tokenizer, CPU inference).",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[postprocess_step / "*.jsonl.gz"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(llama3_tokenizer),
            format=TextLmDatasetFormat(),
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )

    all_tokenize_steps.append(tokenize_step)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_tokenize_steps,
        description="Rephraser CPU inference (big-run): llama.cpp on 3000 workers, CPU-only.",
    )
