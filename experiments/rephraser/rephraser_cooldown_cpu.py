# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser cooldown experiment using llama.cpp CPU inference.

Same pipeline as rephraser_cooldown.py but replaces the vLLM TPU inference step
with llama.cpp on CPU workers. This enables massive horizontal scaling on idle
CPUs across the cluster, removing the TPU bottleneck.

Pipeline:
  1. Download & extract WARCs (reuses cached output from other sweeps)
  1b. Filter HTML by token length (reuses cached output)
  0. Build llama.cpp (runs once, cached)
  2. CPU inference via llama-server (drop-in replacement)
  3. Post-process extraction
  4. Tokenize
  5. Extract NemotronCooldown tokens
  6. Cooldown training with rephraser data mixed in

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-east5-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_cooldown_cpu.py
"""

import logging
import os
from dataclasses import dataclass, replace

from fray.cluster import ResourceConfig
from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.remote import remote
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
from marin.processing.tokenize.data_configs import lm_mixture_data_config, step_to_lm_mixture_component
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

import experiments.rephraser.rephraser_cooldown as cooldown_module
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    CooldownTrainingConfig,
    ExtractCooldownConfig,
    NEMOTRON_MIX_WEIGHTS,
    NUM_TRAIN_STEPS,
    RESUME_STEP,
    SEQ_LEN,
    SPECS,
    SYSTEM_MESSAGE,
    USER_TEMPLATE_FMT,
    extract_cooldown_data,
    load_warc_paths,
    run_cooldown_training,
    spec_hash,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GGUF model for CPU inference. Downloaded once to GCS via an ExecutorStep,
# then workers fetch from GCS (fast, in-region) instead of HuggingFace.
GGUF_SOURCE_URL = (
    "https://huggingface.co/MichaelR207/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-Q4_K_M-GGUF/resolve/main/"
    "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"
)
GGUF_FILENAME = "qwen3-1.7b-rephraser-sft-mid-ckpt5000-q4_k_m.gguf"

logger = logging.getLogger(__name__)

# Tokenizer for pre-filtering HTML (must match the rephraser model's tokenizer)
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# Override CHECKPOINT_PATH to use the correct bucket for training.
# run_cooldown_training reads this as a module-level constant from rephraser_cooldown.
CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)
cooldown_module.CHECKPOINT_PATH = CHECKPOINT_PATH

# ---------------------------------------------------------------------------
# CPU inference tuning parameters.
# Validated by Stage 2+3 benchmarks: 16 threads is the sweet spot —
# more parallel servers beats wider single-server threading.
# 64 workers x 16 CPUs = 1024 CPUs (~5 TPU nodes worth of idle CPUs).
# Ray schedules workers as resources become available.
# ---------------------------------------------------------------------------
THREADS_PER_WORKER = 16
CPU_PER_WORKER = 16
RAM_PER_WORKER = "16g"
NUM_INFERENCE_WORKERS = 512
CONTEXT_LENGTH = 32768

# ---------------------------------------------------------------------------
# Step 0: Build llama.cpp (runs once, cached by executor)
# ---------------------------------------------------------------------------
build_llamacpp_step = ExecutorStep(
    name="tools/llamacpp-build-v3",
    description="Build llama.cpp from source and upload llama-server binary to GCS.",
    fn=build_llamacpp,
    config=BuildLlamaCppConfig(output_path=this_output_path()),
)

# ---------------------------------------------------------------------------
# Step 0b: Download GGUF model to GCS (runs once, cached by executor)
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


# Step name includes a hash of the source URL so changing models invalidates the cache.
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
)

# ---------------------------------------------------------------------------
# Step 1: Download & Extract HTML from WARCs
# SAME name as other rephraser experiments — reuses cached output
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
)

# Step 1b: Pre-filter HTML by token length
# SAME name as other sweeps — reuses cached output
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
)

# ---------------------------------------------------------------------------
# NemotronCooldown extraction (shared across all specs)
# ---------------------------------------------------------------------------
nemotron_steps = tokenize_nemotron()
starcoderdata_step = dclm_components_llama3["starcoderdata"]
proofpile_2_step = dclm_components_llama3["proofpile_2"]

nemotron_base_data = lm_mixture_data_config(
    components={**nemotron_steps, "starcoderdata": starcoderdata_step, "proofpile_2": proofpile_2_step},
    weights=NEMOTRON_MIX_WEIGHTS,
    shuffle=True,
)
nemotron_base_data = replace(nemotron_base_data, permutation_type="linear")

extract_cooldown_step = ExecutorStep(
    name="tokenized/nemotron_cooldown_1e20",
    description="Extract exact cooldown tokens (steps 35k-45k) from the original 1e20 nemotron run.",
    fn=extract_cooldown_data,
    config=ExtractCooldownConfig(
        nemotron_data_config=nemotron_base_data,
        start_step=RESUME_STEP,
        end_step=NUM_TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        output_path=this_output_path(),
    ),
)

# Validation sets (Paloma + Uncheatable Eval)
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_component_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}

# Pre-build the cooldown component (shared across specs)
cooldown_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=output_path_of(extract_cooldown_step),
        format=TextLmDatasetFormat(),
        tags=["nemotron_cooldown"],
    ),
    cache_dir=output_path_of(extract_cooldown_step),
    format=TextLmDatasetFormat(),
    tags=["nemotron_cooldown"],
)

# ---------------------------------------------------------------------------
# Steps 2-7: Per-spec pipeline (inference → postprocess → tokenize → train)
# ---------------------------------------------------------------------------
all_train_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: CPU Inference via llama.cpp (replaces TPU inference)
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_cpu",
        description=f"Run rephraser inference via llama.cpp CPU for spec {sid}.",
        fn=remote(run_inference_llamacpp, pip_dependency_groups=["cpu"]),
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
    )

    # Step 7: Cooldown training with rephraser data mixed in
    rephraser_component = step_to_lm_mixture_component(tokenize_step, include_raw_paths=False)

    train_step = ExecutorStep(
        name=f"cooldown-rephraser-{sid}-cpu",
        description=f"Cooldown training for spec {sid}: NemotronCooldown + rephraser mix (CPU inference).",
        fn=run_cooldown_training,
        config=CooldownTrainingConfig(
            rephraser_tokenized_path=tokenize_step,
            cooldown_tokenized_path=extract_cooldown_step,
            output_path=this_output_path(),
            cooldown_component=cooldown_component,
            rephraser_component=rephraser_component,
            rephraser_name=f"rephraser_{sid}",
            spec_id=sid,
            validation_configs=validation_component_configs,
        ),
    )

    all_train_steps.append(train_step)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_train_steps,
        description="Rephraser cooldown (CPU inference via llama.cpp): full pipeline including training.",
    )
