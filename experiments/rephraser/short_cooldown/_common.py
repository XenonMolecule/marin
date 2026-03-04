# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared infrastructure for short (~1.3B token) cooldown experiments.

Short cooldown: ~1.31B total training tokens (vs ~2.56B in the full cooldown).
The mixed-in data is ~25% of the total (vs ~10% in full), while training takes
roughly half the steps (5,000 vs 9,759).

All 3 conditions share:
  - Same checkpoint (step 35,000 from exp2166)
  - Same model, optimizer, LR schedule (linear decay to 0 over 5,000 steps)
  - Same training budget (~1.31B tokens = 5,000 steps * 64 batch * 4096 seq)
  - Same 1B nemotron cooldown tokens (first component)

Only the mixin data varies:
  - rephraser.py: ~300M rephraser-extracted tokens
  - dclm.py: ~300M DCLM raw web text tokens
  - nemotron_only.py: ~300M additional nemotron tokens (control)
"""

import logging
from dataclasses import dataclass, replace
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

from experiments.defaults import default_validation_sets
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    LEARNING_RATE,
    RESUME_STEP,
    SEQ_LEN,
    ExtractCooldownConfig,
    _read_token_count,
    cooldown_optimizer,
    extract_cooldown_data,
    nemotron_base_data,
    scaling_1e20_qwen3,
)

# Checkpoint copied from us-central1 to eu-west4 for training on v6e-8.
# Original: gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/checkpoints/step-35000
CHECKPOINT_PATH = (
    "gs://marin-eu-west4/checkpoints"
    "/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/step-35000"
)
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Short cooldown constants
# ---------------------------------------------------------------------------
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 262,144

# Total training budget: 5,000 steps * 262,144 = 1,310,720,000 tokens (~1.31B)
SHORT_COOLDOWN_STEPS = 5_000

# 1B nemotron tokens for the mixed conditions (shared across all 3).
# ceil(1B / 262,144) = 3,815 steps of data from the original 1e20 run.
SHORT_NEMOTRON_END_STEP = RESUME_STEP + 3_815  # step 38,815
# Actual nemotron tokens: 3,815 * 262,144 = 1,000,079,360 (~1.000B)

# Extra 300M nemotron tokens for the nemotron-only baseline.
# ceil(300M / 262,144) = 1,145 steps of additional data.
EXTRA_NEMOTRON_END_STEP = SHORT_NEMOTRON_END_STEP + 1_145  # step 39,960
# Actual extra tokens: 1,145 * 262,144 = 300,154,880 (~300M)


# ---------------------------------------------------------------------------
# Extraction steps
# ---------------------------------------------------------------------------

# 1B nemotron tokens (steps 35,000 -> 38,815), shared across all 3 conditions
extract_short_cooldown_step = ExecutorStep(
    name="tokenized/nemotron_cooldown_1e20_short_1b",
    description="Extract ~1B nemotron cooldown tokens (steps 35k-38.8k).",
    fn=extract_cooldown_data,
    config=ExtractCooldownConfig(
        nemotron_data_config=nemotron_base_data,
        start_step=RESUME_STEP,
        end_step=SHORT_NEMOTRON_END_STEP,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="128g"),
    pip_dependency_groups=["cpu"],
)

# Extra 300M nemotron tokens (steps 38,815 -> 39,960), only for nemotron-only baseline
extract_extra_nemotron_step = ExecutorStep(
    name="tokenized/nemotron_cooldown_1e20_short_extra_300m",
    description="Extract ~300M extra nemotron tokens (steps 38.8k-40k) for baseline.",
    fn=extract_cooldown_data,
    config=ExtractCooldownConfig(
        nemotron_data_config=nemotron_base_data,
        start_step=SHORT_NEMOTRON_END_STEP,
        end_step=EXTRA_NEMOTRON_END_STEP,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        output_path=this_output_path(),
    ),
    resources=ResourceConfig.with_cpu(cpu=8, ram="128g"),
    pip_dependency_groups=["cpu"],
)


# ---------------------------------------------------------------------------
# Shared DatasetComponents
# ---------------------------------------------------------------------------

short_cooldown_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=output_path_of(extract_short_cooldown_step),
        format=TextLmDatasetFormat(),
        tags=["nemotron_cooldown"],
    ),
    cache_dir=output_path_of(extract_short_cooldown_step),
    format=TextLmDatasetFormat(),
    tags=["nemotron_cooldown"],
)

extra_nemotron_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=output_path_of(extract_extra_nemotron_step),
        format=TextLmDatasetFormat(),
        tags=["nemotron_extra"],
    ),
    cache_dir=output_path_of(extract_extra_nemotron_step),
    format=TextLmDatasetFormat(),
    tags=["nemotron_extra"],
)


# ---------------------------------------------------------------------------
# Validation sets (Paloma + Uncheatable Eval)
# ---------------------------------------------------------------------------

validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
validation_component_configs = {
    name: step_to_lm_mixture_component(step, include_raw_paths=False)
    for name, step in validation_steps.items()
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_validation_configs(data: LmDataConfig) -> LmDataConfig:
    """Add validation sets (weight=0) to a data config for eval during training."""
    new_components = {
        **data.components,
        **{k: v for k, v in validation_component_configs.items() if k not in data.components},
    }
    new_weights = {
        **data.train_weights,
        **{name: 0.0 for name in validation_component_configs if name not in data.train_weights},
    }
    return replace(data, components=new_components, train_weights=new_weights)


def build_short_cooldown_pod_config(
    data: LmDataConfig,
    output_path: str,
    tags: list[str],
) -> TrainLmOnPodConfig:
    """Build a TrainLmOnPodConfig for short cooldown training.

    LR schedule: linear decay from LEARNING_RATE -> 0 over SHORT_COOLDOWN_STEPS.
    The optimizer's decay=1.0 + warmup=0 + min_lr_ratio=0.0 means the full
    cooldown completes in exactly SHORT_COOLDOWN_STEPS steps.
    """
    inner = TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project="marin",
                tags=["short-cooldown", f"cooldown-steps={SHORT_COOLDOWN_STEPS}"] + tags,
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=SHORT_COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=SHORT_COOLDOWN_STEPS)],
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
    return TrainLmOnPodConfig(
        train_config=inner,
        resources=ResourceConfig.with_tpu("v6e-8"),
        output_path=output_path,
    )
