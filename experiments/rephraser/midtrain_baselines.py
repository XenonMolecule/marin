# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining baselines for Llama 3.2 1B on two data sources.

Baseline 1 — DCLM: raw web text from mlfoundations/dclm-baseline-1.0.
Baseline 2 — Rephraser: cleaned assistant-extracted text from
    MichaelR207/rephraser_late_check_0225.

Both train Llama 3.2 1B for ~100M tokens (381 steps) with the same
hyperparameters used in rephraser_sweep.py, then evaluate on CORE_TASKS.
Compare these numbers against the rephraser sweep midtraining results
and the zero-shot baseline from llama_3_2_1b_baseline_eval.py.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/rephraser/midtrain_baselines.py
"""

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import SimpleTrainConfig, default_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama_3_2_1b
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, ensure_versioned, executor_main, this_output_path, versioned
from marin.processing.tokenize import TokenizeConfig, lm_data_config, tokenize
from marin.transform.clean_rephraser_messages import CleanRephraserMessagesConfig, clean_rephraser_messages

# ---------------------------------------------------------------------------
# Shared constants (match rephraser_sweep.py for apples-to-apples comparison)
# ---------------------------------------------------------------------------
BASE_MODEL_HF = "meta-llama/Llama-3.2-1B"
BASE_MODEL_CONFIG = llama_3_2_1b

TOKEN_BUDGET = 100_000_000
SEQ_LEN = 4096
BATCH_SIZE = 64  # Matches rephraser_sweep.py (MIDTRAIN_BATCH_SIZE)
NUM_TRAIN_STEPS = TOKEN_BUDGET // (SEQ_LEN * BATCH_SIZE)  # 381

LR = 3e-4
MIN_LR_RATIO = 0.1
WARMUP_FRACTION = 0.05
WEIGHT_DECAY = 0.033

TRAIN_CONFIG = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=LR,
    min_lr_ratio=MIN_LR_RATIO,
    train_seq_len=SEQ_LEN,
    initialize_from_hf=BASE_MODEL_HF,
    reset_data_loader_on_init=True,
    lr_schedule="cosine",
    warmup=WARMUP_FRACTION,
    weight_decay=WEIGHT_DECAY,
    z_loss_weight=1e-4,
    steps_per_task_eval=NUM_TRAIN_STEPS,  # Eval once at end
    steps_per_export=100,  # Checkpoint periodically to survive preemptions
)

# ===========================================================================
# Baseline 1: DCLM (raw web text)
#
# Downloads 3 shard files from DCLM (~300M+ tokens), tokenizes, and
# midtrains. Training stops after 381 steps (~100M tokens).
# ===========================================================================
dclm_download = ExecutorStep(
    name="raw/dclm-baseline-1.0-subset",
    description="Download a small subset of DCLM for baseline midtraining.",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision=versioned("a3b142c"),
        gcs_output_path=this_output_path(),
        hf_urls_glob=versioned(
            [
                "global-shard_01_of_10/local-shard_0_of_10/shard_0000000[0-2]_processed.jsonl.zst",
            ]
        ),
        wait_for_completion=True,
    ),
)

dclm_tokenize = ExecutorStep(
    name="tokenized/dclm_baseline_100m",
    description="Tokenize DCLM subset for midtraining.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[dclm_download / "**/*.jsonl.zst"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(BASE_MODEL_HF),
        format=TextLmDatasetFormat(),
    ),
)

dclm_data = lm_data_config(training_set=dclm_tokenize)

dclm_train = default_train(
    name="midtrain-dclm-baseline-100m",
    tokenized=dclm_data,
    model_config=BASE_MODEL_CONFIG,
    train_config=TRAIN_CONFIG,
    tags=["midtrain-baseline", "dclm"],
    eval_harness_tasks=CORE_TASKS,
)

# ===========================================================================
# Baseline 2: Rephraser (cleaned assistant extractions)
#
# Downloads the rephraser mid-check dataset, extracts and cleans assistant
# text (strips <think>, DSPy markers), tokenizes, and midtrains.
# ===========================================================================
rephraser_download = ExecutorStep(
    name="raw/rephraser_late_check_0225",
    description="Download rephraser late-check dataset from HuggingFace.",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="MichaelR207/rephraser_late_check_0225",
        revision=versioned("2194850"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)

rephraser_clean = ExecutorStep(
    name="processed/rephraser_late_cleaned",
    description="Extract and clean assistant text from rephraser messages.",
    fn=clean_rephraser_messages,
    config=CleanRephraserMessagesConfig(
        input_path=rephraser_download / "data/*.parquet",
        output_path=this_output_path(),
    ),
)

rephraser_tokenize = ExecutorStep(
    name="tokenized/rephraser_late_cleaned_100m",
    description="Tokenize cleaned rephraser text for midtraining.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[rephraser_clean / "*.jsonl.gz"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(BASE_MODEL_HF),
        format=TextLmDatasetFormat(),
    ),
)

rephraser_data = lm_data_config(training_set=rephraser_tokenize)

rephraser_train = default_train(
    name="midtrain-rephraser-baseline-100m",
    tokenized=rephraser_data,
    model_config=BASE_MODEL_CONFIG,
    train_config=TRAIN_CONFIG,
    tags=["midtrain-baseline", "rephraser"],
    eval_harness_tasks=CORE_TASKS,
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[dclm_train, rephraser_train],
        description="Midtraining baselines: DCLM web text vs rephraser-extracted text (100M tokens each).",
    )
