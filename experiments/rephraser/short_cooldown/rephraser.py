# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short cooldown with rephraser data: ~1B nemotron + ~300M rephraser ≈ 1.3B tokens.

Same model/checkpoint/optimizer as the full rephraser_cooldown.py, but trains for
5,000 steps instead of 9,759. This makes the rephraser data ~25% of the total mix
(vs ~10% in the full version) and halves training time.

The rephraser data pipeline (download -> filter -> inference -> postprocess -> tokenize)
reuses cached outputs from the full cooldown experiment via identical step names.

Launch (eu-west4-a, v6e-8):
    uv run lib/marin/src/marin/run/ray_run.py \\
        --cluster eu-west4-a --no_wait \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN <your-hf-token> \\
        -- python experiments/rephraser/short_cooldown/rephraser.py

Dry run:
    python experiments/rephraser/short_cooldown/rephraser.py --dry_run
"""

import logging
from dataclasses import dataclass

from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat

from experiments.llama import llama3_tokenizer
from experiments.rephraser.rephraser_cooldown import (
    MAX_MIXIN_FRACTION,
    REPHRASER_MODEL,
    SPECS,
    SYSTEM_MESSAGE,
    USER_TEMPLATE_FMT,
    _read_token_count,
    _validate_mixin_fraction,
    filter_html,
    spec_hash,
)
from experiments.rephraser.short_cooldown._common import (
    LEARNING_RATE,
    SHORT_COOLDOWN_STEPS,
    add_validation_configs,
    build_short_cooldown_pod_config,
    extract_short_cooldown_step,
    short_cooldown_component,
    validation_component_configs,
)
from marin.execution.remote import remote
from marin.execution.executor import (
    ExecutorStep,
    ensure_versioned,
    executor_main,
    this_output_path,
)
from marin.generation.inference_v2 import InferenceV2Config, run_inference_v2
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component
from marin.training.training import run_levanter_train_lm
from marin.transform.postprocess_extraction import PostProcessExtractionConfig, postprocess_extraction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training config + function
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShortRephraserCooldownConfig:
    """Config for short cooldown training with rephraser data mixed in."""

    rephraser_tokenized_path: str
    cooldown_tokenized_path: str
    output_path: str
    cooldown_component: DatasetComponent
    rephraser_component: DatasetComponent
    rephraser_name: str
    spec_id: str
    validation_configs: dict[str, DatasetComponent] | None = None
    max_mixin_fraction: float = MAX_MIXIN_FRACTION


def run_short_rephraser_cooldown(config: ShortRephraserCooldownConfig):
    """Short cooldown: 1B nemotron + ~300M rephraser, 5,000 steps, LR -> 0."""
    rephraser_tokens = _read_token_count(config.rephraser_tokenized_path, split="train")
    cooldown_tokens = _read_token_count(config.cooldown_tokenized_path, split="train")

    _validate_mixin_fraction(
        mixin_name="rephraser",
        mixin_tokens=rephraser_tokens,
        cooldown_tokens=cooldown_tokens,
        num_train_steps=SHORT_COOLDOWN_STEPS,
        max_fraction=config.max_mixin_fraction,
    )

    # Weights proportional to token counts: ~1 epoch of each source.
    cooldown_weight = float(cooldown_tokens)
    rephraser_weight = float(rephraser_tokens)
    total_tokens = cooldown_tokens + rephraser_tokens
    rephraser_frac = rephraser_tokens / total_tokens

    logger.info("=== Short Rephraser Cooldown Training ===")
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
    data = add_validation_configs(data)

    pod_config = build_short_cooldown_pod_config(
        data=data,
        output_path=config.output_path,
        tags=[
            "rephraser-cooldown",
            f"spec-{config.spec_id}",
            f"rephraser-tokens={rephraser_tokens}",
            f"rephraser-frac={rephraser_frac:.4f}",
        ],
    )
    run_levanter_train_lm(pod_config)


# ---------------------------------------------------------------------------
# Pipeline: reuse cached rephraser data, new short nemotron extraction
# ---------------------------------------------------------------------------

# Per-spec pipeline steps (same names as rephraser_cooldown.py -> cached outputs reused)
all_train_steps: list[ExecutorStep] = []

for spec_text in SPECS:
    sid = spec_hash(spec_text)
    user_template = USER_TEMPLATE_FMT.format(spec=spec_text)

    # Step 2: Inference (reuses cached output)
    inference_step = ExecutorStep(
        name=f"documents/rephraser_spec_{sid}_v2",
        description=f"Run rephraser inference_v2 for spec {sid}.",
        fn=remote(run_inference_v2, pip_dependency_groups=["vllm"]),
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

    # Step 3: Post-process (reuses cached output)
    postprocess_step = ExecutorStep(
        name=f"processed/rephraser_spec_{sid}_v2",
        description=f"Post-process extraction output for spec {sid}.",
        fn=postprocess_extraction,
        config=PostProcessExtractionConfig(
            input_path=inference_step / "*.jsonl.gz",
            output_path=this_output_path(),
        ),
    )

    # Step 4: Tokenize (reuses cached output)
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

    # Step 7: Short cooldown training
    rephraser_component = step_to_lm_mixture_component(tokenize_step, include_raw_paths=False)

    train_step = ExecutorStep(
        name=f"short-cooldown-rephraser-{sid}",
        description=f"Short cooldown: 1B nemotron + ~300M rephraser (spec {sid}).",
        fn=run_short_rephraser_cooldown,
        config=ShortRephraserCooldownConfig(
            rephraser_tokenized_path=tokenize_step,
            cooldown_tokenized_path=extract_short_cooldown_step,
            output_path=this_output_path(),
            cooldown_component=short_cooldown_component,
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
        description="Short cooldown: 1B nemotron + ~300M rephraser, 5k steps, LR -> 0.",
    )
