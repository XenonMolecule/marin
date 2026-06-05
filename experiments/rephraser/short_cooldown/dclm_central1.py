# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Eval-resume: short DCLM cooldown on us-central1 (v5p-8).

Training completed on us-east1-d (v6e-32) but the eval harness crashed.
This script resumes from the orbax checkpoint on us-central1 using v5p-8
(single-host) to finish the remaining training steps and run lm-eval.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN <your-hf-token> \
        -- python experiments/rephraser/short_cooldown/dclm_central1.py
"""

import logging
from dataclasses import replace
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat, UrlDatasetSourceConfig
from levanter.main import train_lm
from levanter.main.train_lm import TrainLmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.defaults import default_validation_sets
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    SEQ_LEN,
    _read_token_count,
    cooldown_optimizer,
    scaling_1e20_qwen3,
)
from experiments.rephraser.short_cooldown._common import (
    LEARNING_RATE,
    SHORT_COOLDOWN_STEPS,
)
from experiments.rephraser.short_cooldown.dclm import ShortDclmCooldownConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths on us-central1 (copied from us-east1)
# ---------------------------------------------------------------------------

COOLDOWN_1B_PATH = "gs://marin-us-central1/tokenized/nemotron_cooldown_1e20_short_1b-413400"
DCLM_TOKENIZED_PATH = "gs://marin-us-central1/tokenized/dclm_baseline_100m_llama3-f42a23"

CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)

import experiments.rephraser.rephraser_cooldown as cooldown_module

cooldown_module.CHECKPOINT_PATH = CHECKPOINT_PATH

# ---------------------------------------------------------------------------
# Data components
# ---------------------------------------------------------------------------

cooldown_1b_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=COOLDOWN_1B_PATH,
        format=TextLmDatasetFormat(),
        tags=["nemotron_cooldown"],
    ),
    cache_dir=COOLDOWN_1B_PATH,
    format=TextLmDatasetFormat(),
    tags=["nemotron_cooldown"],
)

dclm_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=DCLM_TOKENIZED_PATH,
        format=TextLmDatasetFormat(),
        tags=["dclm"],
    ),
    cache_dir=DCLM_TOKENIZED_PATH,
    format=TextLmDatasetFormat(),
    tags=["dclm"],
)

# Validation sets
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_component_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def run_short_dclm_v5p8(config: ShortDclmCooldownConfig):
    """Short DCLM cooldown on v5p-8 (eval resume)."""
    dclm_tokens = _read_token_count(config.dclm_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    cooldown_weight = float(cooldown_tokens)
    dclm_weight = float(dclm_tokens)
    total_tokens = cooldown_tokens + dclm_tokens
    dclm_frac = dclm_tokens / total_tokens

    logger.info("=== Short DCLM Cooldown (v5p-8, us-central1) ===")
    logger.info(f"DCLM tokens: {dclm_tokens:,} ({dclm_frac:.1%} of mix)")
    logger.info(f"Nemotron tokens: {cooldown_tokens:,}")
    logger.info(f"Steps: {SHORT_COOLDOWN_STEPS}, LR: {LEARNING_RATE:.6f} -> 0")

    data = LmDataConfig(
        components={
            "nemotron_cooldown": config.cooldown_component,
            "dclm": config.dclm_component,
        },
        train_weights={
            "nemotron_cooldown": cooldown_weight,
            "dclm": dclm_weight,
        },
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

    inner = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=[
                    "short-cooldown",
                    f"cooldown-steps={SHORT_COOLDOWN_STEPS}",
                    "dclm-cooldown",
                    f"dclm-tokens={dclm_tokens}",
                    f"dclm-frac={dclm_frac:.4f}",
                    "v5p-8",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=SHORT_COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[],
            ),
            mesh=MeshConfig(
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                }
            ),
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN,
        model=scaling_1e20_qwen3,
        optimizer=cooldown_optimizer,
        initialize_from_checkpoint_path=CHECKPOINT_PATH,
        eval_harness=train_lm.LmEvalHarnessConfig(task_spec=convert_to_levanter_task_config(CORE_TASKS)),
        eval_harness_steps=SHORT_COOLDOWN_STEPS - 1,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

train_step = ExecutorStep(
    name="short-cooldown-dclm",
    description="Short cooldown: 1B nemotron + ~300M DCLM eval resume on v5p-8.",
    fn=run_short_dclm_v5p8,
    config=ShortDclmCooldownConfig(
        dclm_tokenized_path=DCLM_TOKENIZED_PATH,
        cooldown_tokenized_path=COOLDOWN_1B_PATH,
        output_path=this_output_path(),
        cooldown_component=cooldown_1b_component,
        dclm_component=dclm_component,
        validation_configs=validation_component_configs,
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[train_step],
        description="Short DCLM cooldown eval resume on us-central1 (v5p-8).",
    )
