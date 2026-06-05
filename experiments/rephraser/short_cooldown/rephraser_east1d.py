# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only: short rephraser cooldown on us-east1-d (v6e-32).

Data was processed on eu-west4-a / us-east5-a. Rephraser tokenized data and
the 1B nemotron extraction have been copied to gs://marin-us-east1/.
This script runs ONLY the training step on v6e-32 TPUs.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster us-east1 --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/short_cooldown/rephraser_east1d.py
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
    SPECS,
    _read_token_count,
    _validate_mixin_fraction,
    cooldown_optimizer,
    scaling_1e20_qwen3,
    spec_hash,
)
from experiments.rephraser.short_cooldown._common import (
    LEARNING_RATE,
    SHORT_COOLDOWN_STEPS,
)
from experiments.rephraser.short_cooldown.rephraser import ShortRephraserCooldownConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths on us-east1
# ---------------------------------------------------------------------------

COOLDOWN_1B_PATH = "gs://marin-us-east1/tokenized/nemotron_cooldown_1e20_short_1b-413400"

sid = spec_hash(SPECS[0])  # d7d976d3
REPHRASER_TOKENIZED_PATH = f"gs://marin-us-east1/tokenized/rephraser_spec_{sid}_cooldown-02c17e"

CHECKPOINT_PATH = (
    "gs://marin-us-east1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
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

rephraser_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=REPHRASER_TOKENIZED_PATH,
        format=TextLmDatasetFormat(),
        tags=["rephraser"],
    ),
    cache_dir=REPHRASER_TOKENIZED_PATH,
    format=TextLmDatasetFormat(),
    tags=["rephraser"],
)

# Validation sets
validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_component_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
}


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
def run_short_rephraser_v6e32(config: ShortRephraserCooldownConfig):
    """Short rephraser cooldown on v6e-32."""
    rephraser_tokens = _read_token_count(config.rephraser_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    _validate_mixin_fraction(
        mixin_name="rephraser",
        mixin_tokens=rephraser_tokens,
        cooldown_tokens=cooldown_tokens,
        num_train_steps=SHORT_COOLDOWN_STEPS,
        max_fraction=config.max_mixin_fraction,
    )

    cooldown_weight = float(cooldown_tokens)
    rephraser_weight = float(rephraser_tokens)
    total_tokens = cooldown_tokens + rephraser_tokens
    rephraser_frac = rephraser_tokens / total_tokens

    logger.info("=== Short Rephraser Cooldown (v6e-32, us-east1-d) ===")
    logger.info(f"Rephraser tokens: {rephraser_tokens:,} ({rephraser_frac:.1%} of mix)")
    logger.info(f"Nemotron tokens: {cooldown_tokens:,}")
    logger.info(f"Steps: {SHORT_COOLDOWN_STEPS}, LR: {LEARNING_RATE:.6f} -> 0")

    data = LmDataConfig(
        components={
            "nemotron_cooldown": config.cooldown_component,
            config.rephraser_name: config.rephraser_component,
        },
        train_weights={
            "nemotron_cooldown": cooldown_weight,
            config.rephraser_name: rephraser_weight,
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
                    "rephraser-cooldown",
                    f"spec-{config.spec_id}",
                    f"rephraser-tokens={rephraser_tokens}",
                    f"rephraser-frac={rephraser_frac:.4f}",
                    "v6e-32",
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
        eval_harness_steps=SHORT_COOLDOWN_STEPS,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu("v6e-32"),
        output_path=config.output_path,
    )
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

train_step = ExecutorStep(
    name=f"short-cooldown-rephraser-{sid}-v2",
    description=f"Short cooldown v2 (token-count fix): 1B nemotron + ~300M rephraser (spec {sid}) on v6e-32.",
    fn=run_short_rephraser_v6e32,
    config=ShortRephraserCooldownConfig(
        rephraser_tokenized_path=REPHRASER_TOKENIZED_PATH,
        cooldown_tokenized_path=COOLDOWN_1B_PATH,
        output_path=this_output_path(),
        cooldown_component=cooldown_1b_component,
        rephraser_component=rephraser_component,
        rephraser_name=f"rephraser_{sid}",
        spec_id=sid,
        validation_configs=validation_component_configs,
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[train_step],
        description="Short rephraser cooldown on us-east1-d (v6e-32).",
    )
