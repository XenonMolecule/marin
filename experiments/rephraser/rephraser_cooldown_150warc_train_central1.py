# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only experiment for the 150-WARC rephraser cooldown on us-central1 (v5p-8).

All tokenized data and the checkpoint already exist on gs://marin-us-central1/.
This script runs the full training (9,759 steps) on v5p-8 TPUs.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN <your-hf-token> \
        -- python experiments/rephraser/rephraser_cooldown_150warc_train_central1.py
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

from experiments.defaults import default_validation_sets
from experiments.evals.task_configs import CORE_TASKS, convert_to_levanter_task_config
from experiments.llama import llama3_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
)
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

import experiments.rephraser.rephraser_cooldown as cooldown_module
from experiments.rephraser.rephraser_cooldown import (
    BATCH_SIZE,
    COOLDOWN_STEPS,
    CooldownTrainingConfig,
    LEARNING_RATE,
    SEQ_LEN,
    _read_token_count,
    _validate_mixin_fraction,
    cooldown_optimizer,
    scaling_1e20_qwen3,
    spec_hash,
    SPECS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths on us-central1 (copied from us-east1)
# ---------------------------------------------------------------------------

REPHRASER_TOKENIZED_PATH = "gs://marin-us-central1/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e"

COOLDOWN_TOKENIZED_PATH = "gs://marin-us-central1/tokenized/nemotron_cooldown_1e20-666089"

CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0" "/checkpoints/step-35000"
)
cooldown_module.CHECKPOINT_PATH = CHECKPOINT_PATH

# ---------------------------------------------------------------------------
# Build data components pointing at pre-existing tokenized data
# ---------------------------------------------------------------------------

cooldown_component = DatasetComponent(
    source=UrlDatasetSourceConfig(
        train_urls=[],
        validation_urls=[],
        cache_dir=COOLDOWN_TOKENIZED_PATH,
        format=TextLmDatasetFormat(),
        tags=["nemotron_cooldown"],
    ),
    cache_dir=COOLDOWN_TOKENIZED_PATH,
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
# Training function using v5p-8 TPUs
# ---------------------------------------------------------------------------
def run_cooldown_training_v5p8(config: CooldownTrainingConfig):
    """Cooldown training on v5p-8 TPUs for us-central1 (eval resume)."""
    rephraser_tokens = _read_token_count(config.rephraser_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    _validate_mixin_fraction(
        mixin_name="rephraser",
        mixin_tokens=rephraser_tokens,
        cooldown_tokens=cooldown_tokens,
        num_train_steps=COOLDOWN_STEPS,
        max_fraction=config.max_mixin_fraction,
    )

    cooldown_weight = float(cooldown_tokens)
    rephraser_weight = float(rephraser_tokens)
    total_tokens = cooldown_tokens + rephraser_tokens
    rephraser_frac = rephraser_tokens / total_tokens

    logger.info("=== Rephraser Cooldown Training (v5p-8, us-central1) ===")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Rephraser token count: {rephraser_tokens:,}")
    logger.info(f"NemotronCooldown token count: {cooldown_tokens:,}")
    logger.info(f"Rephraser fraction of total: {rephraser_frac:.4f}")
    logger.info(f"Cooldown steps: {COOLDOWN_STEPS}")
    logger.info(f"LR: {LEARNING_RATE:.6f} -> 0 (linear decay over {COOLDOWN_STEPS} steps)")
    logger.info("TPU: v5p-8")

    components = {
        "nemotron_cooldown": config.cooldown_component,
        config.rephraser_name: config.rephraser_component,
    }
    weights = {
        "nemotron_cooldown": cooldown_weight,
        config.rephraser_name: rephraser_weight,
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
                    "rephraser-cooldown",
                    f"spec-{config.spec_id}",
                    f"rephraser-tokens={rephraser_tokens}",
                    f"rephraser-frac={rephraser_frac:.4f}",
                    f"cooldown-steps={COOLDOWN_STEPS}",
                    "v5p-8",
                    "150warc",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=BATCH_SIZE,
            num_train_steps=COOLDOWN_STEPS,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=3),
                keep=[dict(every=COOLDOWN_STEPS)],
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
        eval_harness_steps=COOLDOWN_STEPS - 1,
    )

    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=ResourceConfig.with_tpu("v5p-8"),
        output_path=config.output_path,
    )

    logger.info(f"Launching cooldown training with resources: {pod_config.resources}")
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

sid = spec_hash(SPECS[0])  # d7d976d3

train_step = ExecutorStep(
    name=f"cooldown-rephraser-{sid}-150warc-v2",
    description=f"150-WARC cooldown v2 (token-count fix) for spec {sid} on v5p-8.",
    fn=run_cooldown_training_v5p8,
    config=CooldownTrainingConfig(
        rephraser_tokenized_path=REPHRASER_TOKENIZED_PATH,
        cooldown_tokenized_path=COOLDOWN_TOKENIZED_PATH,
        output_path=this_output_path(),
        cooldown_component=cooldown_component,
        rephraser_component=rephraser_component,
        rephraser_name=f"rephraser_{sid}",
        spec_id=sid,
        validation_configs=validation_component_configs,
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[train_step],
        description="150-WARC rephraser cooldown training on us-central1 (v5p-8).",
    )
