# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short cooldown with DCLM data: ~1B nemotron + ~300M DCLM ≈ 1.3B tokens.

Same setup as rephraser.py but substituting DCLM (raw web text) for rephraser-
extracted text. This gives a controlled comparison:
  - rephraser.py: NemotronCooldown + rephraser-extracted text
  - dclm.py: NemotronCooldown + raw DCLM web text  <-- this file
  - nemotron_only.py: NemotronCooldown + extra nemotron (control)

The DCLM download + tokenize steps reuse cached outputs from dclm_cooldown.py.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster eu-west4-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/short_cooldown/dclm.py

Dry run:
    python experiments/rephraser/short_cooldown/dclm.py --dry_run
"""

import logging
from dataclasses import dataclass

from levanter.data.text import DatasetComponent, LmDataConfig

from experiments.llama import llama3_tokenizer
from experiments.rephraser.dclm_cooldown import dclm_component, dclm_tokenize
from experiments.rephraser.rephraser_cooldown import _read_token_count
from experiments.rephraser.short_cooldown._common import (
    LEARNING_RATE,
    SHORT_COOLDOWN_STEPS,
    add_validation_configs,
    build_short_cooldown_pod_config,
    extract_short_cooldown_step,
    short_cooldown_component,
    validation_component_configs,
)
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.training.training import run_levanter_train_lm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config + function
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShortDclmCooldownConfig:
    """Config for short cooldown training with DCLM data mixed in."""

    dclm_tokenized_path: str
    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    dclm_component: DatasetComponent
    validation_configs: dict[str, DatasetComponent] | None = None


def run_short_dclm_cooldown(config: ShortDclmCooldownConfig):
    """Short cooldown: 1B nemotron + ~300M DCLM, 5,000 steps, LR -> 0."""
    dclm_tokens = _read_token_count(config.dclm_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    cooldown_weight = float(cooldown_tokens)
    dclm_weight = float(dclm_tokens)
    total_tokens = cooldown_tokens + dclm_tokens
    dclm_frac = dclm_tokens / total_tokens

    logger.info("=== Short DCLM Cooldown Training ===")
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
    data = add_validation_configs(data)

    pod_config = build_short_cooldown_pod_config(
        data=data,
        output_path=config.output_path,
        tags=[
            "dclm-cooldown",
            f"dclm-tokens={dclm_tokens}",
            f"dclm-frac={dclm_frac:.4f}",
        ],
    )
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

dclm_train_step = ExecutorStep(
    name="short-cooldown-dclm",
    description="Short cooldown: 1B nemotron + ~300M DCLM raw web text.",
    fn=run_short_dclm_cooldown,
    config=ShortDclmCooldownConfig(
        dclm_tokenized_path=dclm_tokenize,
        cooldown_tokenized_path=extract_short_cooldown_step,
        output_path=this_output_path(),
        cooldown_component=short_cooldown_component,
        dclm_component=dclm_component,
        validation_configs=validation_component_configs,
    ),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    executor_main(
        steps=[dclm_train_step],
        description="Short cooldown: 1B nemotron + ~300M DCLM, 5k steps, LR -> 0.",
    )
