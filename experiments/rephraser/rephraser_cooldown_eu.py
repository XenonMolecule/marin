# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser data pipeline for cooldown experiment on eu-west4-a (v6e-8 TPUs).

Same data pipeline as rephraser_cooldown.py (WARC download → filter → inference
→ postprocess → tokenize) but targeting eu-west4-a with v6e-8 TPUs. Stops at
tokenization — no NemotronCooldown extraction or training.

The tokenized output can be copied to us-central1 for use in cooldown training,
or training can be added later.

Launch (eu-west4-a):
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster eu-west4-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_cooldown_eu.py
"""

import os

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat

from experiments.llama import llama3_tokenizer
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.remote import remote
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path, versioned
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

# Import shared prompt config and helpers from the rephraser cooldown experiment
from experiments.rephraser.rephraser_cooldown import (
    SPECS,
    SYSTEM_MESSAGE,
    USER_TEMPLATE_FMT,
    load_warc_paths,
    spec_hash,
)

# ---------------------------------------------------------------------------
# Configuration — eu-west4-a with v6e-8 TPUs
# ---------------------------------------------------------------------------
REPHRASER_MODEL = "gs://marin-eu-west4/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# ---------------------------------------------------------------------------
# Step 1: Download & Extract HTML from WARCs
# SAME name as rephraser_cooldown.py / rephraser_sweep — reuses cached output
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
    description=f"Filter HTML documents exceeding {32768 - 4096} tokens.",
    fn=filter_by_token_length,
    config=FilterByTokenLengthConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=REPHRASER_TOKENIZER,
        text_column="html",
        max_tokens=32768 - 4096,
    ),
)

# ---------------------------------------------------------------------------
# Steps 2-5: Per-spec pipeline (inference on v6e-8, tokenize with llama3)
# Final artifact = tokenized dataset (no training)
# ---------------------------------------------------------------------------
all_tokenize_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference on v6e-8 TPUs (eu-west4 model weights)
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_v2",
        description=f"Run rephraser inference_v2 for spec {sid} (v6e-8).",
        fn=remote(run_inference_v2, pip_dependency_groups=["vllm"]),
        config=InferenceV2Config(
            input_path=filter_html / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name=REPHRASER_MODEL,
            input_format="jsonl.gz",
            output_format="jsonl.gz",
            engine_kwargs={
                "max_model_len": 32768,
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            system_message=SYSTEM_MESSAGE,
            template=user_template,
            prompt_column="html",
            apply_chat_template=True,
            max_doc_tokens=32768 - 4096,
            tensor_parallel_size=4,
            tpu_type="v6e-8",
            num_workers=16,
            records_per_shard=500,
        ),
    )

    # Step 3: Post-process
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_spec_{sid}_v2",
        description=f"Post-process extraction output for spec {sid}.",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    # Step 4: Tokenize with Meta-Llama-3.1-8B (nemotron-compatible)
    # Same step name as rephraser_cooldown.py (_cooldown suffix) — reuses cached output
    tokenize_step = ExecutorStep(
        name=f"tokenized/rephraser_spec_{sid}_cooldown",
        description=f"Tokenize extracted text for spec {sid} (llama3 tokenizer).",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[postprocess_step / "*.jsonl.gz"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(llama3_tokenizer),
            format=TextLmDatasetFormat(),
        ),
    )

    all_tokenize_steps.append(tokenize_step)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_tokenize_steps,
        description="Rephraser cooldown data pipeline (eu-west4-a v6e-8): inference + tokenize only.",
    )
