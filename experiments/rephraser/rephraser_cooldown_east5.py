# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rephraser cooldown experiment on us-east5-a (v5p-8 TPUs) — full pipeline including training.

Same pipeline as rephraser_cooldown.py (WARC download → filter → inference →
postprocess → tokenize → NemotronCooldown extraction → cooldown training) but
targeting us-east5-a. All GCS paths use gs://marin-us-east5/.

Launch (us-east5-a):
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-east5-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/rephraser_cooldown_east5.py
"""

import os
from dataclasses import replace

from fray.cluster import ResourceConfig
from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets import tokenize_nemotron
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from marin.datakit.download.commoncrawl.download_warc import WarcDownloadConfig, download_and_extract_warcs
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_mixture_data_config, step_to_lm_mixture_component
from marin.transform.filter_by_token_length import FilterByTokenLengthConfig, filter_by_token_length
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

# Import shared config, training functions, and helpers from the main cooldown experiment
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
# Configuration — us-east5-a with v5p-8 TPUs
# ---------------------------------------------------------------------------
REPHRASER_MODEL = "gs://marin-us-east5/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
REPHRASER_TOKENIZER = "Qwen/Qwen3-8B"
WARC_MANIFEST = os.path.join(os.path.dirname(__file__), "warc_paths.txt")

# Override CHECKPOINT_PATH to point at us-east5 bucket.
# run_cooldown_training reads this as a module-level constant from rephraser_cooldown,
# so we patch it there before any training function runs.
CHECKPOINT_PATH = (
    "gs://marin-us-east5/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)
cooldown_module.CHECKPOINT_PATH = CHECKPOINT_PATH

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

    # Step 2: Inference on v5p-8 TPUs (us-east5 model weights)
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_v2",
        description=f"Run rephraser inference_v2 for spec {sid} (v5p-8).",
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
            tpu_type="v5p-8",
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

    # Step 7: Cooldown training with rephraser data mixed in
    rephraser_component = step_to_lm_mixture_component(tokenize_step, include_raw_paths=False)

    train_step = ExecutorStep(
        name=f"cooldown-rephraser-{sid}-v2",
        description=f"Cooldown training for spec {sid}: NemotronCooldown + rephraser mix.",
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
        description="Rephraser cooldown (us-east5-a v5p-8): full pipeline including training.",
    )
