# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""DCLM cooldown baseline: mix raw DCLM web text into the cooldown phase.

Same setup as rephraser_cooldown.py but substituting DCLM (raw web text) for
rephraser-extracted text. This gives a controlled comparison:
  - Baseline (nemotron-only cooldown): defined in rephraser_cooldown.py
  - Rephraser: NemotronCooldown + rephraser-extracted text
  - DCLM: NemotronCooldown + raw DCLM web text  <-- this file

Downloads ~100M tokens of DCLM, tokenizes with the nemotron-compatible tokenizer
(Meta-Llama-3.1-8B), and mixes into the cooldown phase. The DCLM download step
reuses the same name as midtrain_baselines.py, so if that data is already cached
on GCS it won't re-download.

NOTE: The midtrain_baselines DCLM pipeline has never been run end-to-end, so
watch for bugs in the download/tokenize steps.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/dclm_cooldown.py
"""

import logging
from dataclasses import dataclass, replace
from datetime import timedelta

import jmp

from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from haliax.partitioning import ResourceAxis

from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
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

# Import shared constants and steps from the rephraser cooldown experiment
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    COOLDOWN_STEPS,
    LEARNING_RATE,
    SEQ_LEN,
    _read_token_count,
    cooldown_component,
    cooldown_optimizer,
    extract_cooldown_step,
    scaling_1e20_qwen3,
    validation_component_configs,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DCLM download + tokenize pipeline
# ---------------------------------------------------------------------------

# Download 3 shard files from DCLM (~300M+ tokens raw).
# SAME step name as midtrain_baselines.py — reuses cached output if available.
dclm_download = ExecutorStep(
    name="raw/dclm-baseline-1.0-subset",
    description="Download a small subset of DCLM for baseline cooldown.",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="mlfoundations/dclm-baseline-1.0",
        revision=versioned("a3b142c"),
        gcs_output_path=this_output_path(),
        hf_urls_glob=versioned(
            [
                "global-shard_01_of_10/local-shard_0_of_10/" "shard_0000000[0-2]_processed.jsonl.zst",
            ]
        ),
        wait_for_completion=True,
    ),
)

# Tokenize with Meta-Llama-3.1-8B (nemotron-compatible tokenizer).
# midtrain_baselines.py uses Llama-3.2-1B — different tokenizer, so we need
# a separate tokenize step with a distinct name.
dclm_tokenize = ExecutorStep(
    name="tokenized/dclm_baseline_100m_llama3",
    description="Tokenize DCLM subset with nemotron-compatible tokenizer (Meta-Llama-3.1-8B).",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[dclm_download / "**/*.jsonl.zst"],
        validation_paths=[],
        cache_path=this_output_path(),
        tokenizer=ensure_versioned(llama3_tokenizer),
        format=TextLmDatasetFormat(),
    ),
)


# ---------------------------------------------------------------------------
# DCLM Cooldown Training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DclmCooldownConfig:
    """Config for cooldown training with DCLM data mixed into NemotronCooldown."""

    dclm_tokenized_path: str
    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    dclm_component: DatasetComponent
    validation_configs: dict[str, DatasetComponent] | None = None


def run_dclm_cooldown_training(config: DclmCooldownConfig):
    """Mix DCLM raw web text into NemotronCooldown and train.

    Same weight computation as rephraser_cooldown: proportional to token counts
    so each source gets ~1 epoch over the cooldown phase.
    """
    dclm_tokens = _read_token_count(config.dclm_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    cooldown_weight = float(cooldown_tokens)
    dclm_weight = float(dclm_tokens)
    total_tokens = cooldown_tokens + dclm_tokens
    dclm_frac = dclm_tokens / total_tokens

    logger.info("=== DCLM Cooldown Training ===")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info("Model params: ~1,384,579,840 (Qwen3 1.385B)")
    logger.info(f"DCLM tokenized path: {config.dclm_tokenized_path}")
    logger.info(f"NemotronCooldown tokenized path: {config.cooldown_tokenized_path}")
    logger.info(f"DCLM token count: {dclm_tokens:,}")
    logger.info(f"DCLM sequence count: {dclm_tokens // SEQ_LEN:,}")
    logger.info(f"NemotronCooldown token count: {cooldown_tokens:,}")
    logger.info(f"NemotronCooldown sequence count: {cooldown_tokens // SEQ_LEN:,}")
    logger.info(f"Total cooldown sequences: {COOLDOWN_STEPS * BATCH_SIZE:,}")
    logger.info(f"DCLM fraction of total: {dclm_frac:.4f}")
    logger.info(f"Computed DCLM weight: {dclm_weight:.6f}")
    logger.info(f"Computed cooldown weight: {cooldown_weight:.6f}")
    logger.info(f"Cooldown steps: {COOLDOWN_STEPS}")
    logger.info(f"LR: {LEARNING_RATE:.6f} -> 0 (linear decay over {COOLDOWN_STEPS} steps)")
    logger.info("TPU: v5p-8")
    logger.info("=== Comparison Target (original 1e20 run) ===")
    logger.info("Original eval/loss=2.828, bpb=0.966, paloma=1.1105, uncheatable=0.8727")

    components = {
        "nemotron_cooldown": config.cooldown_component,
        "dclm": config.dclm_component,
    }
    weights = {
        "nemotron_cooldown": cooldown_weight,
        "dclm": dclm_weight,
    }

    data = LmDataConfig(
        components=components,
        train_weights=weights,
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )

    # Add validation configs (weight 0) for eval during training
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
                    "dclm-cooldown",
                    f"dclm-tokens={dclm_tokens}",
                    f"dclm-frac={dclm_frac:.4f}",
                    f"cooldown-steps={COOLDOWN_STEPS}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[],
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
        eval_harness=train_lm.LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(CORE_TASKS)),
        eval_harness_steps=COOLDOWN_STEPS,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    logger.info(f"Launching DCLM cooldown training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------
dclm_component = step_to_lm_mixture_component(dclm_tokenize, include_raw_paths=False)

dclm_train_step = ExecutorStep(
    name="cooldown-dclm-100m-v2",
    description="Cooldown training: NemotronCooldown + DCLM raw web text mix.",
    fn=run_dclm_cooldown_training,
    config=DclmCooldownConfig(
        dclm_tokenized_path=dclm_tokenize,
        cooldown_tokenized_path=extract_cooldown_step,
        output_path=this_output_path(),
        cooldown_component=cooldown_component,
        dclm_component=dclm_component,
        validation_configs=validation_component_configs,
    ),
    # No resources/pip_dependency_groups — run_dclm_cooldown_training calls
    # run_levanter_train_lm internally, which handles TPU allocation.
)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[dclm_train_step],
        description="DCLM cooldown baseline: mix raw DCLM web text into nemotron 1e20 cooldown phase.",
    )
