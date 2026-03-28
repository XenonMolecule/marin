# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser inference-only sweep on eu-west4-a with v6e-8 TPUs.

Mirrors rephraser_sweep.py but runs only the inference pipeline (no training):
  download WARCs → filter HTML → inference → postprocess

Targets the eu-west4-a cluster with v6e-8 TPUs for comparing inference speed
against the us-central1 v5p-8 baseline.

Optimizations over the us-central1 config:
  - load_format="runai_streamer": stream model weights from GCS (no FUSE)
  - batch_size=64 (up from 32): v6e-8 has enough HBM for larger batches
  - num_instances=(1, 32): scale to more replicas when nodes are available

Launch (eu-west4-a):
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster eu-west4-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/rephraser_sweep_eu.py

Dry run (verify DAG, no execution):
    python experiments/rephraser/rephraser_sweep_eu.py --dry_run
"""

import hashlib
import os

from fray.cluster import ResourceConfig

from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.remote import remote
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Rephraser model: checkpoint already copied to eu-west4 bucket for low-latency loading.
REPHRASER_MODEL = "gs://marin-eu-west4/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1000"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

# WARC paths: shared with the us-central1 sweep
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# Same specs as the us-central1 sweep for apples-to-apples comparison
SPECS = [
    "Extract the main text content from this HTML page.",
]

# Prompt structure (identical to rephraser_sweep.py)
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

USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spec_hash(spec_text: str) -> str:
    """Stable 8-char ID from spec content. Order-independent."""
    return hashlib.sha256(spec_text.encode()).hexdigest()[:8]


def load_warc_paths(manifest_path: str) -> list[str]:
    with open(manifest_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------------------
# Step 1: Download & Extract HTML from WARCs
# ---------------------------------------------------------------------------
warc_paths = load_warc_paths(WARC_MANIFEST)

download_warcs = ExecutorStep(
    name="raw/commoncrawl/rephraser_sweep_eu_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 1b: Pre-filter HTML by token length
# ---------------------------------------------------------------------------
filter_html = ExecutorStep(
    name="filtered/rephraser_sweep_eu_batch0",
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
# Steps 2-3: Per-spec inference + postprocess (no training)
# ---------------------------------------------------------------------------
all_final_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference — run rephraser on v6e-8 TPUs
    inference_step = ExecutorStep(
        name=f"documents/rephraser_eu_spec_{sid}",
        description=f"Run rephraser inference for spec {sid} on v6e-8.",
        fn=remote(run_inference, pip_dependency_groups=["vllm"]),
        config=TextGenerationInferenceConfig(
            input_path=filter_html / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name=REPHRASER_MODEL,
            engine_kwargs={
                "tensor_parallel_size": 4,  # All 4 chips on v6e-8
                "max_model_len": 32768,
                "enable_prefix_caching": True,
                "load_format": "runai_streamer",  # Stream weights from GCS (no FUSE)
            },
            generation_kwargs={
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            system_message=SYSTEM_MESSAGE,
            template=user_template,
            prompt_column="html",
            apply_chat_template=True,
            save_templated_prompt=False,
            max_doc_tokens=32768 - 4096,
            num_instances=(1, 32),  # Scale up to 32 replicas across available nodes
            batch_size=256,  # Push batch size aggressively for TPU utilization
            tensor_parallel_size=4,
            resource_config=ResourceConfig.with_tpu("v6e-8"),
            generated_text_column_name="generated_text",
            checkpoint_id_column="id",
            filetype="jsonl.gz",
            output_filetype_override="parquet",
        ),
    )

    # Step 3: Post-process — strip <think> tokens, filter [NO_USEFUL_CONTENT]
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_eu_spec_{sid}",
        description=f"Post-process extraction output for spec {sid}.",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.parquet",
            output_path=this_output_path(),
        ),
    )

    all_final_steps.append(postprocess_step)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_final_steps,
        description="Rephraser inference-only sweep on eu-west4-a v6e-8 TPUs.",
    )
