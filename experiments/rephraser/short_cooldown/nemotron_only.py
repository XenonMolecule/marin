# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short cooldown with nemotron-only data: ~1B nemotron + ~300M extra nemotron ≈ 1.3B tokens.

Control condition: same total training budget as rephraser.py and dclm.py, but the
mixin data is additional nemotron tokens instead of rephraser/DCLM text.

The 1B nemotron tokens (steps 35k-38.8k) are identical to the other conditions.
The extra 300M (steps 38.8k-40k) continues from where the 1B extraction left off,
ensuring no overlap.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster eu-west4-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/short_cooldown/nemotron_only.py

Dry run:
    python experiments/rephraser/short_cooldown/nemotron_only.py --dry_run
"""

import logging
from dataclasses import dataclass

from levanter.data.text import DatasetComponent, LmDataConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.training.training import run_levanter_train_lm

from experiments.llama import llama3_tokenizer
from experiments.rephraser.rephraser_cooldown import _read_token_count
from experiments.rephraser.short_cooldown._common import (
    LEARNING_RATE,
    SHORT_COOLDOWN_STEPS,
    add_validation_configs,
    build_short_cooldown_pod_config,
    extra_nemotron_component,
    extract_extra_nemotron_step,
    extract_short_cooldown_step,
    short_cooldown_component,
    validation_component_configs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config + function
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShortNemotronOnlyCooldownConfig:
    """Config for short cooldown training with nemotron-only data (control)."""

    cooldown_tokenized_path: str
    extra_nemotron_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    extra_nemotron_component: DatasetComponent
    validation_configs: dict[str, DatasetComponent] | None = None


def run_short_nemotron_only_cooldown(config: ShortNemotronOnlyCooldownConfig):
    """Short cooldown: 1B nemotron + 300M extra nemotron, 5,000 steps, LR -> 0."""
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")
    extra_tokens = _read_token_count(config.extra_nemotron_tokenized_path, split="train")

    cooldown_weight = float(cooldown_tokens)
    extra_weight = float(extra_tokens)
    total_tokens = cooldown_tokens + extra_tokens

    logger.info("=== Short Nemotron-Only Cooldown Training (control) ===")
    logger.info(f"Nemotron cooldown tokens: {cooldown_tokens:,}")
    logger.info(f"Nemotron extra tokens: {extra_tokens:,}")
    logger.info(f"Total nemotron tokens: {total_tokens:,}")
    logger.info(f"Steps: {SHORT_COOLDOWN_STEPS}, LR: {LEARNING_RATE:.6f} -> 0")

    data = LmDataConfig(
        components={
            "nemotron_cooldown": config.cooldown_component,
            "nemotron_extra": config.extra_nemotron_component,
        },
        train_weights={
            "nemotron_cooldown": cooldown_weight,
            "nemotron_extra": extra_weight,
        },
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )
    data = add_validation_configs(data)

    pod_config = build_short_cooldown_pod_config(
        data=data,
        output_path=config.output_path,
        tags=[
            "nemotron-only-cooldown",
            "baseline",
            f"total-nemotron-tokens={total_tokens}",
        ],
    )
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

nemotron_only_train_step = ExecutorStep(
    name="short-cooldown-nemotron-only",
    description="Short cooldown: 1B + 300M nemotron only (control).",
    fn=run_short_nemotron_only_cooldown,
    config=ShortNemotronOnlyCooldownConfig(
        cooldown_tokenized_path=extract_short_cooldown_step,
        extra_nemotron_tokenized_path=extract_extra_nemotron_step,
        output_path=this_output_path(),
        cooldown_component=short_cooldown_component,
        extra_nemotron_component=extra_nemotron_component,
        validation_configs=validation_component_configs,
    ),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[nemotron_only_train_step],
        description="Short cooldown: 1B + 300M nemotron only (control), 5k steps, LR -> 0.",
    )
