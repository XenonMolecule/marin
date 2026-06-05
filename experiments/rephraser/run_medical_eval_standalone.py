# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native standalone eval child for the medical 0.6B-Base re-runs.

Runs **inside one TPU iris job** (submitted by `launch_medical_evals.py`).
Receives one HF-checkpoint path via CLI, detects the local region, asserts
the checkpoint is region-local, builds a `ModelConfig` + the medical eval
task list, and calls `LMEvaluationHarnessEvaluator.evaluate(...)` IN-PROCESS.

WHY THIS FILE EXISTS
--------------------
Companion to `run_medical_sft_standalone.py`. The training standalone
produces an HF checkpoint; this script evaluates that checkpoint on the
same MMLU-medical + MediQA suite the original instruct runs used.

PATH HYGIENE & 1:1 PARITY
-------------------------
- Eval task list matches `medical_extraction_sft_v2.py:130-155` exactly
  (7 MMLU-* generative 5-shot subtasks + mediqa_qa2019_lite 0-shot).
- `apply_chat_template=False` matches the original recipe.
- engine_kwargs match the original: max_model_len=8192, max_gen_toks=512.
- We call `LMEvaluationHarnessEvaluator.evaluate(...)` directly — the
  in-process method that uses `VllmEnvironment` + `lm_eval.simple_evaluate`.
  We deliberately bypass `launch_evaluate(...)` (the Ray-submit wrapper)
  because the Marin executor is offline.
- No `gs://` paths are hardcoded for the model; the coordinator passes
  `--model-rel-path` which is resolved against `MARIN_PREFIX` →
  `REGION_TO_BUCKET[region]` at boot, exactly the same pattern as the
  training standalone uses for tokenized caches.

USAGE
-----
Normally invoked by `launch_medical_evals.py`, but can be run by hand for
debug:

    python experiments/rephraser/run_medical_eval_standalone.py \\
        --model-rel-path checkpoints/medical-sft-base/medical-extraction-lr5e-6_bs32-qwen3-0.6b-base-rerun/hf \\
        --model-name medical-ext-lr5e-6_bs32-qwen3-0.6b-base
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging

import fsspec
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.evaluators.evaluator import ModelConfig
from marin.evaluation.evaluators.lm_evaluation_harness_evaluator import LMEvaluationHarnessEvaluator

from experiments.scaling_law_sweeps import region_tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Default location for the per-run eval summary index. Tiny metadata files,
# central home is fine.
DEFAULT_EVAL_RESULTS_PREFIX = "gs://marin-us-central1/metadata/medical_sft_base_eval_results/"

# Eval task list — 1:1 with `medical_extraction_sft_v2.py:130-155`. The
# `_generative` task aliases use lm-eval-harness's generative MMLU variants
# (no logprob-based multiple choice, which is broken on TPU per
# https://github.com/vllm-project/vllm/issues/8499 — see the comment in
# `LMEvaluationHarnessEvaluator` line 25).
MEDICAL_EVALS: list[EvalTaskConfig] = [
    EvalTaskConfig(name="mmlu_anatomy_generative", num_fewshot=5, task_alias="mmlu_anatomy_gen_5shot"),
    EvalTaskConfig(
        name="mmlu_clinical_knowledge_generative",
        num_fewshot=5,
        task_alias="mmlu_clinical_knowledge_gen_5shot",
    ),
    EvalTaskConfig(
        name="mmlu_college_medicine_generative",
        num_fewshot=5,
        task_alias="mmlu_college_medicine_gen_5shot",
    ),
    EvalTaskConfig(
        name="mmlu_medical_genetics_generative",
        num_fewshot=5,
        task_alias="mmlu_medical_genetics_gen_5shot",
    ),
    EvalTaskConfig(
        name="mmlu_professional_medicine_generative",
        num_fewshot=5,
        task_alias="mmlu_professional_medicine_gen_5shot",
    ),
    EvalTaskConfig(
        name="mmlu_college_biology_generative",
        num_fewshot=5,
        task_alias="mmlu_college_biology_gen_5shot",
    ),
    EvalTaskConfig(
        name="mmlu_high_school_biology_generative",
        num_fewshot=5,
        task_alias="mmlu_high_school_biology_gen_5shot",
    ),
    EvalTaskConfig(name="mediqa_qa2019_lite", num_fewshot=0, task_alias="mediqa_qa2019_lite_0shot"),
]

# Engine kwargs — 1:1 with the original recipe (medical_extraction_sft_v2.py
# uses these via `EVAL_ENGINE_KWARGS`).
DEFAULT_ENGINE_KWARGS: dict = {"max_model_len": 8192, "max_gen_toks": 512}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-rel-path",
        required=True,
        help=(
            "HF checkpoint path RELATIVE to the local region bucket, e.g. "
            "'checkpoints/medical-sft-base/medical-extraction-lr5e-6_bs32-"
            "qwen3-0.6b-base-rerun/hf'. Resolved at boot via MARIN_PREFIX → "
            "REGION_TO_BUCKET[region]."
        ),
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help=("Model name for W&B / output_path naming. e.g. " "'medical-extraction-lr5e-6_bs32-qwen3-0.6b-base'."),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Where to write the eval results. Defaults to "
            "<model-path>/eval/lm_eval_harness/ (same layout as the original "
            "Ray-based evals, just under our region-local bucket)."
        ),
    )
    parser.add_argument(
        "--eval-results-prefix",
        default=DEFAULT_EVAL_RESULTS_PREFIX,
        help="Where to write the per-run summary JSON for downstream aggregation.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help=(
            "If set, apply the tokenizer's chat template at eval time. The "
            "medical re-runs MUST leave this OFF to match the original instruct "
            "runs (which also had it OFF) — the SFT data is plain text, no chat."
        ),
    )
    parser.add_argument("--max-eval-instances", type=int, default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Local-path safety
# ---------------------------------------------------------------------------
def _assert_model_local(model_path: str, region: str) -> None:
    """HARD invariant — same as `run_medical_sft_standalone._assert_cache_local`.

    vLLM downloads the HF checkpoint into the worker's local FS at startup.
    If `model_path` were in a different region's bucket, that download would
    silently pay cross-region egress (often gigabytes per checkpoint). Fail
    loud instead.
    """
    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    if not model_path.startswith(expected_prefix):
        raise ValueError(
            f"model_path={model_path!r} is not in the local region's bucket "
            f"(expected prefix {expected_prefix!r}). Cross-region reads "
            f"forbidden — pre-copy the checkpoint into {expected_prefix} first."
        )


def _resolve_local_model_path(model_rel_path: str, region: str) -> str:
    """Compose the LOCAL gs:// model path. Mirrors `_resolve_local_cache_dir` in the training standalone."""
    bucket = region_tracker.REGION_TO_BUCKET[region]
    return f"{bucket}/{model_rel_path.strip('/')}"


def _is_hf_reference(name: str) -> bool:
    """Heuristic: a `vendor/model-name` HF reference vs a GCS rel-path.

    HF refs match the pattern `<owner>/<repo>` with no further slashes (e.g.
    `Qwen/Qwen3-0.6B-Base`). GCS rel-paths typically have multiple path
    components like `checkpoints/medical-sft-base/<run-name>/hf`. The
    `--baseline-hf-model` flow uses HF refs and skips the local-region assert.
    """
    return name.count("/") == 1 and not name.startswith("gs://")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    # 1. Region detection — same MARIN_PREFIX path as the training side.
    region = region_tracker.detect_current_region()
    logger.info("Standalone medical EVAL child boot — region=%s, model=%s", region, args.model_name)

    # 2. Resolve model path. For HF references (e.g. "Qwen/Qwen3-0.6B-Base"
    #    used to evaluate an unmodified Base model as a baseline), pass the
    #    name straight to vLLM and skip the local-region assert — vLLM will
    #    download from HF Hub. For GCS rel-paths, resolve locally and assert.
    if _is_hf_reference(args.model_rel_path):
        model_path = args.model_rel_path
        logger.info("Detected HF reference: %s (skipping local-region assert)", model_path)
    else:
        model_path = _resolve_local_model_path(args.model_rel_path, region)
        _assert_model_local(model_path, region)
        logger.info("Local model path: %s", model_path)

    # 3. Build the ModelConfig + output path.
    #    For GCS model paths (post-SFT checkpoints), default to writing eval
    #    results alongside the model. For HF references (baseline evals), the
    #    "model_path" is just an HF id like "Qwen/Qwen3-0.6B-Base" — using it
    #    as a path would make output_path a RELATIVE local fs path that
    #    ephemeral TPU workers throw away. Route those to a region-local
    #    GCS bucket instead so results survive.
    if args.output_path:
        output_path = args.output_path
    elif model_path.startswith("gs://"):
        output_path = f"{model_path.rstrip('/')}/eval/lm_eval_harness"
    else:
        # HF reference — write to a region-local baselines prefix.
        bucket = region_tracker.REGION_TO_BUCKET[region]
        output_path = f"{bucket}/eval_baselines/{args.model_name}/lm_eval_harness"
    logger.info("Eval output path: %s", output_path)

    model = ModelConfig(
        name=args.model_name,
        path=model_path,
        engine_kwargs=dict(DEFAULT_ENGINE_KWARGS),
        generation_params=None,
        apply_chat_template=args.apply_chat_template,  # default False — matches original
        base_eval_run_name=args.model_name,
    )

    # 4. Run eval IN-PROCESS — no Ray submit. `LMEvaluationHarnessEvaluator.evaluate`
    #    drives `VllmEnvironment` + `lm_eval.simple_evaluate` directly. The
    #    Ray-submit wrapper (`launch_evaluate`) is deliberately NOT used because
    #    the Marin executor is offline.
    evaluator = LMEvaluationHarnessEvaluator()
    evaluator.evaluate(
        model=model,
        evals=MEDICAL_EVALS,
        output_path=output_path,
        max_eval_instances=args.max_eval_instances,
        wandb_tags=[
            "medical",
            "eval",
            "qwen3-0.6b-base",
            "rerun=base-fix",
            f"model={args.model_name}",
        ],
    )
    logger.info("Eval finished cleanly.")

    # 5. Write a per-run summary so the report-rewrite step has a flat index.
    summary = {
        "model_name": args.model_name,
        "model_path": model_path,
        "model_rel_path": args.model_rel_path,
        "region": region,
        "output_path": output_path,
        "evals": [e.task_alias or e.name for e in MEDICAL_EVALS],
        "apply_chat_template": args.apply_chat_template,
        "engine_kwargs": DEFAULT_ENGINE_KWARGS,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    summary_path = f"{args.eval_results_prefix.rstrip('/')}/{args.model_name}.json"
    try:
        with fsspec.open(summary_path, "w") as f:
            f.write(json.dumps(summary, indent=2))
        logger.info("Wrote eval summary: %s", summary_path)
    except Exception as e:
        logger.warning("Failed to write eval summary at %s: %s", summary_path, e)

    # 6. DONE marker so the coordinator can skip already-evaluated checkpoints.
    done_marker = f"{output_path.rstrip('/')}/.medical_eval_DONE"
    try:
        with fsspec.open(done_marker, "w") as f:
            f.write(
                json.dumps(
                    {
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "model_name": args.model_name,
                        "region": region,
                    }
                )
            )
        logger.info("Wrote DONE marker: %s", done_marker)
    except Exception as e:
        logger.warning("Failed to write DONE marker at %s: %s", done_marker, e)


if __name__ == "__main__":
    main()
