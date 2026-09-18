# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run lm-eval tasks on one HF checkpoint through Levanter's TPU-native harness.

The scaling-sweep eval children (`mmlu/run_mmlu_eval.py`, `cultural/run_cultural_eval.py`)
used to hand an `EvaluationConfig` to `marin.evaluation.run.evaluate`, which dispatched to
`LevanterLmEvalEvaluator`. The 2026-08-13 upstream sync replaced marin's lm-eval evaluators
with Evalchemy over a served model (#7281) and removed that entry point; the evaluator module
left behind no longer imports. Levanter's harness itself is unchanged, so this helper builds
`EvalHarnessMainConfig` exactly as the evaluator did and calls `run_eval_harness_main`
in-process on the TPU the child already holds. Results land at `<output_dir>/results.json`
with the same shape the sweeps' consolidators read.

Call only after the offline HF environment (HF_HOME, HF_DATASETS_OFFLINE, ...) is in place:
importing this module pulls in levanter and transformers.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os

import jmp
import levanter.eval_harness as eval_harness
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.evaluation.evaluation_config import EvalTaskConfig, convert_to_levanter_task_config
from rigging.filesystem import filesystem as marin_filesystem

import levanter

logger = logging.getLogger(__name__)

# The evaluator's hardcoded context budget; sl_verb 5-shot prompts fit (see mmlu_tasks_set).
MAX_LENGTH = 2048
SAMPLES_LOGGED_PER_BENCHMARK = 20
WANDB_PROJECT = "marin"
RESULTS_FILENAME = "results.json"


def _json_default(value):
    """Best-effort JSON serialization for objects the harness returns."""
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return list(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return repr(value)


def run_levanter_lm_eval(
    checkpoint: str,
    evals: list[EvalTaskConfig],
    output_dir: str,
    run_name: str | None,
    max_eval_instances: int | None = None,
) -> str:
    """Score `evals` on the HF checkpoint at `checkpoint`; returns the results.json path.

    `checkpoint` is an HF-format export dir (gs://.../hf/step-N/) holding config.json,
    safetensors and the tokenizer. The wandb run is named `<run_name>_lmeval_<tasks>`.
    """
    model_name = run_name or os.path.basename(checkpoint.rstrip("/"))
    wandb_name = model_name + "_lmeval_" + "-".join(e.name for e in evals)
    trainer_config = TrainerConfig(
        tracker=WandbConfig(project=WANDB_PROJECT, name=wandb_name),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=1,
    )
    try:
        model_config = HFCheckpointConverter.from_hf(checkpoint).LevConfigClass()
    except Exception:
        # from_hf probes every registered config's default HF reference and raises on
        # architectures it cannot map that way (the OLMIX swarm proxies export as qwen3).
        # The checkpoint's own config IS local — resolve LevConfig directly from its
        # model_type via the registry, the same fallback olmo_bpb/run_olmo_bpb_eval uses.
        import json as _json

        import fsspec as _fsspec
        from levanter.models.lm_model import LmConfig as _LmConfig

        with _fsspec.open(f"{checkpoint.rstrip('/')}/config.json", "r") as _f:
            _mt = _json.load(_f)["model_type"]
        # reference_checkpoint must point at the LOCAL export: hf_checkpoint_converter()
        # eagerly infers a tokenizer from it at construction, and the class default is a
        # Hub repo id — a hard failure under the eval children's HF_HUB_OFFLINE=1.
        model_config = _LmConfig.get_known_choices()[_mt](reference_checkpoint=checkpoint)
    tasks = convert_to_levanter_task_config(evals)
    logger.info("Levanter lm-eval: checkpoint=%s tasks=%s", checkpoint, tasks)

    config = eval_harness.EvalHarnessMainConfig(
        eval_harness=eval_harness.LmEvalHarnessConfig(
            task_spec=tasks,
            max_examples=max_eval_instances,
            log_samples=True,
            max_length=MAX_LENGTH,
            apply_chat_template=False,
            confirm_run_unsafe_code=True,
            sample_logging=eval_harness.SampleLoggingConfig(max_samples_per_benchmark=SAMPLES_LOGGED_PER_BENCHMARK),
        ),
        tokenizer=checkpoint,
        checkpoint_path=checkpoint,
        checkpoint_is_hf=True,
        trainer=trainer_config,
        model=model_config,
    )
    results = eval_harness.run_eval_harness_main(config)

    results_path = f"{output_dir.rstrip('/')}/{RESULTS_FILENAME}"
    with marin_filesystem("gcs").open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    levanter.tracker.current_tracker().finish()
    logger.info("Wrote %s", results_path)
    return results_path
