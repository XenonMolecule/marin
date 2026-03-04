# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM-filtered cooldown: filter raw WARCs with the full DCLM pipeline, then cooldown.

Instead of using pre-filtered DCLM data from HuggingFace (dclm_cooldown.py), this
experiment starts from the same raw Common Crawl WARCs used for rephraser experiments,
applies the complete DCLM-Baseline filtering pipeline (RefinedWeb heuristics + FastText
quality classifier), and mixes the surviving text into the cooldown phase. This gives
a fair comparison between rephraser extraction and classical DCLM filtering on identical
source HTML.

Pipeline:
  WARCs → extract plain text (resiliparse) → DCLM filters → tokenize → mix w/ nemotron → train

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/dclm_filtered_cooldown.py

Dry run:
    python experiments/rephraser/dclm_filtered_cooldown.py --dry_run
"""

import logging
from dataclasses import dataclass, replace
from datetime import timedelta

import jmp

from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from marin.download.download_url import (
    DownloadUrlToGcsConfig,
    download_url_to_gcs,
)
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    this_output_path,
    versioned,
)
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from marin.transform.dclm_filter import DclmFilterConfig, dclm_filter
from marin.transform.extract_text_from_html import ExtractTextConfig, extract_text_from_html

# Import shared constants and steps from the rephraser cooldown experiment
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    CHECKPOINT_PATH as _CHECKPOINT_PATH_CENTRAL1,
    COOLDOWN_STEPS,
    LEARNING_RATE,
    SEQ_LEN,
    _read_token_count,
    cooldown_component,
    cooldown_optimizer,
    download_warcs,
    extract_cooldown_step,
    scaling_1e20_qwen3,
    validation_component_configs,
)

# The original checkpoint lives on us-central1. We copy it to whichever cluster
# we're running on and override the path here.
CHECKPOINT_PATH = _CHECKPOINT_PATH_CENTRAL1.replace(
    "gs://marin-us-central1/", "gs://marin-us-east1/"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1: Download DCLM resources (models + ban lists)
# ---------------------------------------------------------------------------

download_lid_model = ExecutorStep(
    name="resources/dclm/lid_176",
    description="Download FastText language ID model (lid.176.bin).",
    fn=download_url_to_gcs,
    config=DownloadUrlToGcsConfig(
        url=versioned("https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"),
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=2, ram="4g"),
    pip_dependency_groups=["cpu"],
)

download_quality_model = ExecutorStep(
    name="resources/dclm/fasttext_oh_eli5",
    description="Download FastText quality classifier.",
    fn=download_url_to_gcs,
    config=DownloadUrlToGcsConfig(
        url=versioned(
            "https://huggingface.co/mlfoundations/fasttext-oh-eli5/resolve/main/"
            "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        ),
        output_path=this_output_path(),
        filename="fasttext_oh_eli5.bin",
    ),
    resources=ResourceConfig.with_cpu(cpu=2, ram="8g"),
    pip_dependency_groups=["cpu"],
)

# Ban lists are pre-uploaded to GCS (the curated domain list is 118MB, too large for GitHub).
BANLISTS_GCS_PATH = "gs://marin-us-central2/resources/dclm/banlists"

# ---------------------------------------------------------------------------
# Step 2: Extract plain text from HTML using resiliparse
# ---------------------------------------------------------------------------

extract_text = ExecutorStep(
    name="processed/dclm_text_from_warcs",
    description="Extract plain text from WARC HTML records using resiliparse.",
    fn=extract_text_from_html,
    config=ExtractTextConfig(
        input_path=download_warcs / "*.jsonl.gz",
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 3: Apply DCLM filters
# ---------------------------------------------------------------------------

filter_step = ExecutorStep(
    name="filtered/dclm_full_pipeline",
    description="Apply full DCLM-Baseline filtering pipeline.",
    fn=dclm_filter,
    config=DclmFilterConfig(
        input_path=extract_text / "*.jsonl.gz",
        output_path=this_output_path(),
        lid_model_path=download_lid_model,
        quality_model_path=download_quality_model,
        banlists_path=versioned(BANLISTS_GCS_PATH),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu", "dclm"],
)

# ---------------------------------------------------------------------------
# Step 4: Tokenize
# ---------------------------------------------------------------------------

tokenize_step = ExecutorStep(
    name="tokenized/dclm_filtered_warcs_llama3",
    description="Tokenize DCLM-filtered text with Meta-Llama-3.1-8B tokenizer.",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[filter_step / "*.jsonl.gz"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
    pip_dependency_groups=["cpu"],
)

# ---------------------------------------------------------------------------
# Step 5: Cooldown training (DCLM-filtered + NemotronCooldown mix)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DclmFilteredCooldownConfig:
    """Config for cooldown training with DCLM-filtered data mixed into NemotronCooldown."""

    dclm_tokenized_path: str
    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    dclm_component: DatasetComponent
    validation_configs: dict[str, DatasetComponent] | None = None


def run_dclm_filtered_cooldown_training(config: DclmFilteredCooldownConfig):
    """Mix DCLM-filtered web text into NemotronCooldown and train.

    Same weight computation as rephraser_cooldown: proportional to token counts
    so each source gets ~1 epoch over the cooldown phase.
    """
    dclm_tokens = _read_token_count(config.dclm_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    cooldown_weight = float(cooldown_tokens)
    dclm_weight = float(dclm_tokens)
    total_tokens = cooldown_tokens + dclm_tokens
    dclm_frac = dclm_tokens / total_tokens

    logger.info("=== DCLM-Filtered Cooldown Training ===")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info("Model params: ~1,384,579,840 (Qwen3 1.385B)")
    logger.info(f"DCLM-filtered tokenized path: {config.dclm_tokenized_path}")
    logger.info(f"NemotronCooldown tokenized path: {config.cooldown_tokenized_path}")
    logger.info(f"DCLM-filtered token count: {dclm_tokens:,}")
    logger.info(f"DCLM-filtered sequence count: {dclm_tokens // SEQ_LEN:,}")
    logger.info(f"NemotronCooldown token count: {cooldown_tokens:,}")
    logger.info(f"NemotronCooldown sequence count: {cooldown_tokens // SEQ_LEN:,}")
    logger.info(f"Total cooldown sequences: {COOLDOWN_STEPS * BATCH_SIZE:,}")
    logger.info(f"DCLM-filtered fraction of total: {dclm_frac:.4f}")
    logger.info(f"Computed DCLM-filtered weight: {dclm_weight:.6f}")
    logger.info(f"Computed cooldown weight: {cooldown_weight:.6f}")
    logger.info(f"Cooldown steps: {COOLDOWN_STEPS}")
    logger.info(f"LR: {LEARNING_RATE:.6f} -> 0 (linear decay over {COOLDOWN_STEPS} steps)")
    logger.info("TPU: v4-8")
    logger.info("=== Comparison Target (original 1e20 run) ===")
    logger.info("Original eval/loss=2.828, bpb=0.966, paloma=1.1105, uncheatable=0.8727")

    components = {
        "nemotron_cooldown": config.cooldown_component,
        "dclm_filtered": config.dclm_component,
    }
    weights = {
        "nemotron_cooldown": cooldown_weight,
        "dclm_filtered": dclm_weight,
    }

    data = LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )

    if config.validation_configs:
        new_components = {
            **data.components,
            **{k: v for k, v in config.validation_configs.items() if k not in data.components},
        }
        new_weights = {
            **data.train_weights,
            **{name: 0.0 for name in config.validation_configs if name not in data.train_weights},
        }
        data = replace(data, components=new_components, train_weights=new_weights)

    inner_config = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "dclm-filtered-cooldown",
                    f"dclm-filtered-tokens={dclm_tokens}",
                    f"dclm-filtered-frac={dclm_frac:.4f}",
                    f"cooldown-steps={COOLDOWN_STEPS}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=2,  # v4-8 has 30.75G HBM/chip; microbatch to avoid OOM
            num_train_steps=COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=COOLDOWN_STEPS)],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (
                        ResourceAxis.REPLICA_DCN,
                        ResourceAxis.REPLICA,
                        ResourceAxis.DATA,
                    ),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=cooldown_optimizer,
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        # eval_harness disabled: crashes on multi-host v6e-32 with device_put error.
        # Eval already completed successfully (eval loss=2.827). HF checkpoint at step-9758 saved.
        # eval_harness=train_lm.LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(CORE_TASKS)),
        # eval_harness_steps=COOLDOWN_STEPS,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v6e-32"),
        output_path=config.output_path,
    )

    logger.info(f"Launching DCLM-filtered cooldown training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------

dclm_filtered_component = step_to_lm_mixture_component(tokenize_step, include_raw_paths=False)

dclm_filtered_train_step = ExecutorStep(
    name="cooldown-dclm-filtered-v1",
    description="Cooldown training: NemotronCooldown + DCLM-filtered web text mix.",
    fn=run_dclm_filtered_cooldown_training,
    config=DclmFilteredCooldownConfig(
        dclm_tokenized_path=tokenize_step,
        cooldown_tokenized_path=extract_cooldown_step,
        output_path=this_output_path(),
        cooldown_component=cooldown_component,
        dclm_component=dclm_filtered_component,
        validation_configs=validation_component_configs,
    ),
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[dclm_filtered_train_step],
        description="DCLM-filtered cooldown: DCLM pipeline on WARCs → tokenize → cooldown training.",
    )
