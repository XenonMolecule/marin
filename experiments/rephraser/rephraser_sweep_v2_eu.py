# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser spec sweep using inference_v2 on eu-west4-a (v6e-8 TPUs).

Same pipeline as rephraser_sweep_v2.py but targeting eu-west4-a:
  - Model weights from eu-west4 bucket
  - v6e-8 TPUs (8 chips, tensor_parallel_size=4)
  - Upstream steps (download, filter) run on eu-west4-a and cache separately

Launch (eu-west4-a):
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster eu-west4-a --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN $HF_TOKEN \
        -- python experiments/rephraser/rephraser_sweep_v2_eu.py

Dry run (verify DAG, no execution):
    python experiments/rephraser/rephraser_sweep_v2_eu.py --dry_run
"""

import hashlib
import math
import os

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_3_2_1b
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path, versioned
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

# ---------------------------------------------------------------------------
# Configuration — eu-west4-a with v6e-8 TPUs
# ---------------------------------------------------------------------------

REPHRASER_MODEL = "gs://marin-eu-west4/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"

WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

SPECS = [
    "Extract the main text content from this HTML page.",
]

# ---------------------------------------------------------------------------
# Prompt structure (identical to rephraser_sweep_v2.py)
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

USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)

BASE_MODEL_HF = "meta-llama/Llama-3.2-1B"
BASE_MODEL_CONFIG = llama_3_2_1b
MIDTRAIN_SEQ_LEN = 4096

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
        warc_paths=versioned(tuple(warc_paths)),
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="64g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 1b: Pre-filter HTML by token length
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
        max_tokens=32768 - 4096,
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Steps 2-5: Per-spec pipeline using inference_v2 on v6e-8
# ---------------------------------------------------------------------------
all_train_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference — inference_v2 on v6e-8 TPUs
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_v2",
        description=f"Run rephraser inference_v2 for spec {sid} (v6e-8).",
        fn=run_inference_v2,
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
        pip_dependency_groups=["vllm"],
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
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g"),
        pip_dependency_groups=["cpu"],
    )

    # Step 4: Tokenize
    tokenize_step = ExecutorStep(
        name=f"tokenized/rephraser_spec_{sid}_v2",
        description=f"Tokenize extracted text for spec {sid}.",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[postprocess_step / "*.jsonl.gz"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=ensure_versioned(BASE_MODEL_HF),
            format=TextLmDatasetFormat(),
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu"],
    )

    # Step 5: Midtrain with CORE_TASKS eval (v6e-8 for training too)
    data_config = lm_data_config(training_set=tokenize_step)

    est_tokens = len(warc_paths) * 5_000_000
    est_sequences = est_tokens // MIDTRAIN_SEQ_LEN
    num_train_steps = max(1, math.ceil(est_sequences / MIDTRAIN_BATCH_SIZE))

    train_step = default_train(
        name=f"midtrain-spec-{sid}-v3",
        tokenized=data_config,
        model_config=BASE_MODEL_CONFIG,
        train_config=SimpleTrainConfig(
            resources=ResourceConfig.with_tpu("v6e-8"),
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
            steps_per_task_eval=num_train_steps,
            steps_per_export=num_train_steps,
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
        steps=all_train_steps,
        description="Rephraser spec sweep (inference_v2, eu-west4-a v6e-8): midtrain + eval.",
    )
