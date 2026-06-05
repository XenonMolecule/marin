# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser spec sweep: compare M extraction specs via midtraining.

Downloads Common Crawl WARC files, runs inference with a trained rephraser model
for each spec independently, post-processes the output, tokenizes, and midtrains
Llama-3.2-1B on each spec's data. Evaluation (CORE_TASKS) is built into the
training step via Levanter's eval harness. Compare specs on W&B.

Key design:
  - Per-spec independence: each spec flows through its own inference → postprocess
    → tokenize → midtrain pipeline. Adding/removing a spec never invalidates others.
  - Content-hash spec IDs: spec order in SPECS list doesn't matter.
  - WARC batch extensibility: add more WARC batches later without re-downloading old ones.

Launch (us-central1):
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_sweep.py

Dry run (verify DAG, no execution):
    python experiments/rephraser/rephraser_sweep.py --dry_run
"""

import hashlib
import math
import os

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path, versioned
from marin.execution.remote import remote
from marin.generation.inference import TextGenerationInferenceConfig, run_inference
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_3_2_1b

# ---------------------------------------------------------------------------
# Configuration (user-editable section)
# ---------------------------------------------------------------------------

# Rephraser model: GCS path to trained checkpoint.
# Replace with the actual path to your trained rephraser model's HF export.
# The checkpoint lives on us-central1 (same region as the cluster for low-latency loading).
REPHRASER_MODEL = "gs://marin-us-central1/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1000"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"  # Same tokenizer as the fine-tuned rephraser

# WARC paths: one S3-style Common Crawl path per line
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# Specs: list of extraction prompt strings (order-independent, content-hashed).
# Each spec describes HOW to extract text from HTML. The rephraser model was trained
# with DSPy-style prompts where the spec goes into the [[ ## extraction_spec ## ]] field.
SPECS = [
    "Extract the main text content from this HTML page.",
    # Add more specs here — each gets its own independent pipeline.
    # Example:
    # "Grab the principal textual material from the HTML input and output it as clear, "
    # "formatted text. Convert headings to Markdown ATX syntax.",
]

# ---------------------------------------------------------------------------
# Full prompt structure (system + user messages)
#
# This matches the DSPy-style prompt format used during rephraser training.
# The system message is IDENTICAL for all specs; only the user message changes
# (different spec text per step, different HTML per document).
#
# Example of a complete 3-message conversation (system, user, assistant):
#
# [system]:
#   Your input fields are:
#   1. `html` (str):
#   2. `extraction_spec` (str):
#   Your output fields are:
#   1. `text` (str):
#   All interactions will be structured in the following way, with the
#   appropriate values filled in.
#
#   [[ ## html ## ]]
#   {html}
#
#   [[ ## extraction_spec ## ]]
#   {extraction_spec}
#
#   [[ ## text ## ]]
#   {text}
#
#   [[ ## completed ## ]]
#   In adhering to this structure, your objective is:
#           Extract the main content text from a given HTML document.
#
# [user]:
#   [[ ## html ## ]]
#   <div id="target" style="opacity: 0"></div>
#
#   [[ ## extraction_spec ## ]]
#   Grab the principal textual material from the HTML input and output it
#   as clear, formatted text. ...
#
#   Respond with the corresponding output fields, starting with the field
#   `[[ ## text ## ]]`, and then ending with the marker for
#   `[[ ## completed ## ]]`.
#
# [assistant]:
#   <think>...</think>[[ ## text ## ]]
#   ---
#   url: ""
#   title: ""
#   ---
#
#   [[ ## completed ## ]]
# ---------------------------------------------------------------------------
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

# User prompt template:
#   {example} is substituted by the generation pipeline with the HTML content
#   {spec} is baked in at step-construction time (one template per spec)
#
# Double braces {{example}} escape to {example} after the first .format(spec=...)
USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)

# Base model for midtraining (matches DCLM 1B experiment family)
BASE_MODEL_HF = "meta-llama/Llama-3.2-1B"
BASE_MODEL_CONFIG = llama_3_2_1b
MIDTRAIN_SEQ_LEN = 4096

# Training hyperparameters — adapted from DCLM 1B/1x (exp1077):
#   From-scratch DCLM: LR=3e-3, batch=256, wd=0.033, z_loss=1e-4, min_lr_ratio=0.1
#   We use ~10x lower LR for midtraining (continued pretraining from checkpoint).
MIDTRAIN_BATCH_SIZE = 64
MIDTRAIN_LR = 3e-4
MIDTRAIN_MIN_LR_RATIO = 0.1
MIDTRAIN_WARMUP_FRACTION = 0.05
MIDTRAIN_WEIGHT_DECAY = 0.033


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
    name="raw/commoncrawl/rephraser_sweep_batch0",
    description="Download WARC files from Common Crawl and extract HTML.",
    fn=download_and_extract_warcs,
    config=WarcDownloadConfig(
        warc_paths=versioned(tuple(warc_paths)),  # tuple for hashability
        output_path=this_output_path(),
    ),
)

# ---------------------------------------------------------------------------
# Step 1b: Pre-filter HTML by token length
#
# Discard documents whose HTML exceeds max_doc_tokens BEFORE inference.
# This avoids wasting TPU prefill compute on documents that would just be
# truncated inside vLLM anyway. Distributed via Zephyr (one shard per input
# file). With 10k+ WARCs there are plenty of shards for full parallelism.
# ---------------------------------------------------------------------------
filter_html = ExecutorStep(
    name="filtered/rephraser_sweep_batch0_v2",
    description=f"Filter HTML documents exceeding {32768 - 4096} tokens (Zephyr-distributed).",
    fn=filter_by_token_length,
    config=FilterByTokenLengthConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
        tokenizer=REPHRASER_TOKENIZER,
        text_column="html",
        max_tokens=32768 - 4096,  # 28672: matches max_doc_tokens in inference
    ),
)

# ---------------------------------------------------------------------------
# Steps 2-5: Per-spec pipeline (independent per spec)
#
# This follows the same parallel sweep pattern as
# experiments/exp_qwen3_1_7b_rephraser_sweep.py:
#   all_steps: list[ExecutorStep] = []
#   for <sweep_var> in <grid>:
#       ...build per-config steps...
#       all_steps.append(final_step)
#   executor_main(steps=all_steps)
#
# The executor auto-discovers all upstream dependencies from each final step.
# With --max_concurrent N, at most N specs run simultaneously.
#
# TODO(shared-vllm): When M grows to 10+, consider switching to a shared vLLM
# server for prefix caching. Currently each spec's inference step runs its own
# vLLM instance, so the system+HTML prefix (~20k tokens) is re-computed M times
# per document. A shared vLLM server with automatic prefix caching (APC) would
# compute the prefix once and reuse it across specs, yielding ~5x throughput
# improvement at M=100. The tradeoff is implementation complexity (Ray Serve
# lifecycle, HTTP client, health checks) vs the current simple per-step design.
# See the plan file for a detailed tradeoff analysis.
# ---------------------------------------------------------------------------
all_train_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference — run the rephraser model on each HTML document with this spec
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}",
        description=f"Run rephraser inference for spec {sid}.",
        fn=remote(run_inference, pip_dependency_groups=["vllm"]),
        config=TextGenerationInferenceConfig(
            input_path=filter_html / "*.jsonl.gz",
            output_path=this_output_path(),
            model_name=REPHRASER_MODEL,
            engine_kwargs={
                "tensor_parallel_size": 4,  # Shard across all 4 chips on v5p-8
                "max_model_len": 32768,  # Model's trained context window (generation eats into this budget)
                "enable_prefix_caching": True,
            },
            generation_kwargs={
                "temperature": 0.0,  # Deterministic extraction
                "max_tokens": 4096,  # Output budget (within the 32768 total)
            },
            system_message=SYSTEM_MESSAGE,
            template=user_template,
            prompt_column="html",
            apply_chat_template=True,
            save_templated_prompt=False,
            max_doc_tokens=32768 - 4096,  # 28672: max_model_len minus max generation tokens
            # num_instances controls Ray Data auto-scaling of vLLM replicas.
            # Each replica gets its own TPU (v5p-8 = 4 chips). Ray schedules
            # up to max replicas based on cluster capacity.
            # TODO: increase max to 32+ when cluster time-share pressure eases.
            num_instances=(1, 16),
            batch_size=256,
            tensor_parallel_size=4,
            resource_config=ResourceConfig.with_tpu("v5p-8"),
            generated_text_column_name="generated_text",
            checkpoint_id_column="id",
            filetype="jsonl.gz",  # Must match input format (filter_html outputs .jsonl.gz)
            output_filetype_override="parquet",  # Write output as parquet for efficient checkpointing
        ),
    )

    # Step 3: Post-process — strip <think> tokens, filter [NO_USEFUL_CONTENT], etc.
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_spec_{sid}",
        description=f"Post-process extraction output for spec {sid}.",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.parquet",
            output_path=this_output_path(),
        ),
    )

    # Step 4: Tokenize — prepare cleaned text for Levanter training
    tokenize_step = ExecutorStep(
        name=f"tokenized/rephraser_spec_{sid}",
        description=f"Tokenize extracted text for spec {sid}.",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[postprocess_step / "*.jsonl.gz"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(BASE_MODEL_HF),
            format=TextLmDatasetFormat(),  # Plain text, text_key="text"
        ),
    )

    # Step 5: Midtrain with CORE_TASKS eval
    data_config = lm_data_config(training_set=tokenize_step)

    # Token estimate calibrated from first run: 10,718,377 tokens from 2 WARCs
    # = ~5.36M tokens/WARC. This depends on spec verbosity and postprocessing
    # filter rate, but 5M/WARC is a reasonable default.
    est_tokens = len(warc_paths) * 5_000_000
    est_sequences = est_tokens // MIDTRAIN_SEQ_LEN
    num_train_steps = max(1, math.ceil(est_sequences / MIDTRAIN_BATCH_SIZE))

    train_step = default_train(
        name=f"midtrain-spec-{sid}-v2",
        tokenized=data_config,
        model_config=BASE_MODEL_CONFIG,
        train_config=SimpleTrainConfig(
            resources=ResourceConfig.with_tpu("v5p-8"),
            train_batch_size=MIDTRAIN_BATCH_SIZE,
            num_train_steps=num_train_steps,
            learning_rate=MIDTRAIN_LR,
            min_lr_ratio=MIDTRAIN_MIN_LR_RATIO,
            train_seq_len=MIDTRAIN_SEQ_LEN,
            initialize_from_hf=BASE_MODEL_HF,
            reset_data_loader_on_init=True,
            lr_schedule="cosine",
            warmup=MIDTRAIN_WARMUP_FRACTION,
            weight_decay=MIDTRAIN_WEIGHT_DECAY,
            z_loss_weight=1e-4,
            steps_per_task_eval=num_train_steps,  # Eval once at end
        ),
        tags=["rephraser-sweep", f"spec-{sid}"],
        eval_harness_tasks=CORE_TASKS,
    )

    all_train_steps.append(train_step)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=all_train_steps,  # Executor auto-discovers all upstream deps
        description="Rephraser spec sweep: midtrain + eval for each extraction spec.",
    )
