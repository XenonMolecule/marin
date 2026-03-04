# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only experiment for the 150-WARC rephraser cooldown.

The inference + tokenization was completed on us-east5-a. The tokenized data
has been copied to gs://marin-us-central1/. This script runs ONLY the training
step on us-central1, where TPU resources are more stable.

Launch:
    uv run lib/marin/src/marin/run/ray_run.py \
        --cluster us-central1 --no_wait \
        -e WANDB_API_KEY $WANDB_API_KEY \
        -e HF_TOKEN <your-hf-token> \
        -- python experiments/rephraser/rephraser_cooldown_150warc_train.py
"""

from levanter.data.text import DatasetComponent, TextLmDatasetFormat, UrlDatasetSourceConfig

from experiments.defaults import default_validation_sets
from experiments.llama import llama3_tokenizer
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
)
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

import experiments.rephraser.rephraser_cooldown as cooldown_module
from experiments.rephraser.rephraser_cooldown import (
    CooldownTrainingConfig,
    run_cooldown_training,
    spec_hash,
    SPECS,
)

# ---------------------------------------------------------------------------
# Paths to already-processed data on us-central1
# ---------------------------------------------------------------------------

# Tokenized rephraser data (150 WARCs, copied from east5)
REPHRASER_TOKENIZED_PATH = (
    "gs://marin-us-central1/tokenized/rephraser_spec_d7d976d3_cooldown-02c17e"
)

# NemotronCooldown tokens (already cached on central1)
COOLDOWN_TOKENIZED_PATH = (
    "gs://marin-us-central1/tokenized/nemotron_cooldown_1e20-666089"
)

# Training checkpoint (central1)
CHECKPOINT_PATH = (
    "gs://marin-us-central1/exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0"
    "/checkpoints/step-35000"
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
    name: step_to_lm_mixture_component(step, include_raw_paths=False)
    for name, step in validation_steps.items()
}

# ---------------------------------------------------------------------------
# Training step — uses the same run_cooldown_training as the other experiments
# ---------------------------------------------------------------------------

sid = spec_hash(SPECS[0])  # d7d976d3

train_step = ExecutorStep(
    name=f"cooldown-rephraser-{sid}-150warc",
    description=f"150-WARC cooldown training for spec {sid}: NemotronCooldown + rephraser mix.",
    fn=run_cooldown_training,
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
        description="150-WARC rephraser cooldown training on us-central1.",
    )
