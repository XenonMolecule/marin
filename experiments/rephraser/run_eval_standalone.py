# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native standalone eval child for any (domain, model) combo.

Generalizes `run_medical_eval_standalone.py` to also handle code and math.
Domain-specific eval task lists are baked in below; everything else
(local-region assert, in-process LM-eval-harness driver, summary writes)
is identical.

VAL / TEST SPLITS
-----------------
This script runs ALL subtasks for the chosen domain. Val/test aggregation
is done at report time — splitting in the standalone itself would make
re-aggregation under a different split (e.g. moving a subtask from val to
test) require a re-run, which is wasteful.

Conventions baked into the report aggregator:
  - medical: VAL = {anatomy, college_biology, high_school_biology, medical_genetics};
             TEST = {clinical_knowledge, college_medicine, professional_medicine};
             mediqa is supplementary.
  - math:    VAL = {prealgebra, algebra, num_theory, counting_and_probability};
             TEST = {geometry, intermediate_algebra, precalc};
             gsm8k is supplementary.
  - code:    VAL = {humaneval}; TEST = {mbpp_0shot, mbpp_3shot}.

USAGE
-----
    python experiments/rephraser/run_eval_standalone.py \\
        --domain code \\
        --model-rel-path Qwen/Qwen3-8B-Base \\
        --model-name baseline-qwen3-8b-base-code
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
# Per-domain task catalog
# ---------------------------------------------------------------------------
# Medical: 7 MMLU-* generative 5-shot subtasks + mediqa 0-shot.
# 1:1 with `medical_extraction_sft_v2.py:130-155` and
# `run_medical_eval_standalone.py:71-104`.
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

# Math: 7 minerva_math_* topics (4-shot) + gsm8k_platinum_cot (8-shot).
# 1:1 with the published 14B recipe at `math_14b_top3_sft.py:103-118` and
# `math_extraction_minerva_eval.py:32-46`. Using `hendrycks_math_*` 0-shot
# instead would produce numbers that are not directly comparable to the
# published 14B table — Minerva-style prompting + 4-shot is the load-bearing
# difference there. `gsm8k_platinum_cot` is the cleaned platinum test set;
# again 1:1 with the published recipe.
MATH_EVALS: list[EvalTaskConfig] = [
    EvalTaskConfig(name="minerva_math_algebra", num_fewshot=4, task_alias="minerva_math_algebra_4shot"),
    EvalTaskConfig(
        name="minerva_math_counting_and_prob",
        num_fewshot=4,
        task_alias="minerva_math_counting_and_prob_4shot",
    ),
    EvalTaskConfig(
        name="minerva_math_geometry",
        num_fewshot=4,
        task_alias="minerva_math_geometry_4shot",
    ),
    EvalTaskConfig(
        name="minerva_math_intermediate_algebra",
        num_fewshot=4,
        task_alias="minerva_math_intermediate_algebra_4shot",
    ),
    EvalTaskConfig(
        name="minerva_math_num_theory",
        num_fewshot=4,
        task_alias="minerva_math_num_theory_4shot",
    ),
    EvalTaskConfig(
        name="minerva_math_prealgebra",
        num_fewshot=4,
        task_alias="minerva_math_prealgebra_4shot",
    ),
    EvalTaskConfig(name="minerva_math_precalc", num_fewshot=4, task_alias="minerva_math_precalc_4shot"),
    EvalTaskConfig(name="gsm8k_platinum_cot", num_fewshot=8, task_alias="gsm8k_platinum_cot_8shot"),
]

# Code: HumanEval 0-shot + MBPP 0/3-shot. 1:1 with
# `baseline_14b_eval.py:22-26` and `code_extraction_sft_v3_14b_sweep.py:72-74`.
CODE_EVALS: list[EvalTaskConfig] = [
    EvalTaskConfig(name="humaneval", num_fewshot=0, task_alias="humaneval_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=0, task_alias="mbpp_0shot"),
    EvalTaskConfig(name="mbpp", num_fewshot=3, task_alias="mbpp_3shot"),
]

# Medical, *logprob* variant: standard `mmlu_*` tasks (single-token loglikelihood
# of A/B/C/D). Avoids the format-following collapse we observed with the
# generative variant on Base models — empty-string responses for ~50-90% of
# questions due to a `::` template artifact at the boundary between in-context
# examples and the test prompt. Only viable if vLLM-TPU's loglikelihood path
# works; smoke-test on Qwen3-0.6B-Base before committing.
MEDICAL_LOGPROB_EVALS: list[EvalTaskConfig] = [
    EvalTaskConfig(name="mmlu_anatomy", num_fewshot=5, task_alias="mmlu_anatomy_logprob_5shot"),
    EvalTaskConfig(name="mmlu_clinical_knowledge", num_fewshot=5, task_alias="mmlu_clinical_knowledge_logprob_5shot"),
    EvalTaskConfig(name="mmlu_college_medicine", num_fewshot=5, task_alias="mmlu_college_medicine_logprob_5shot"),
    EvalTaskConfig(name="mmlu_medical_genetics", num_fewshot=5, task_alias="mmlu_medical_genetics_logprob_5shot"),
    EvalTaskConfig(
        name="mmlu_professional_medicine", num_fewshot=5, task_alias="mmlu_professional_medicine_logprob_5shot"
    ),
    EvalTaskConfig(name="mmlu_college_biology", num_fewshot=5, task_alias="mmlu_college_biology_logprob_5shot"),
    EvalTaskConfig(name="mmlu_high_school_biology", num_fewshot=5, task_alias="mmlu_high_school_biology_logprob_5shot"),
]

DOMAIN_EVALS: dict[str, list[EvalTaskConfig]] = {
    "medical": MEDICAL_EVALS,
    "medical-logprob": MEDICAL_LOGPROB_EVALS,
    "math": MATH_EVALS,
    "code": CODE_EVALS,
}

# Default summary-prefix per domain. Matches existing layout for medical so
# the aggregator script doesn't need a special case.
DOMAIN_SUMMARY_PREFIX: dict[str, str] = {
    "medical": "gs://marin-us-central1/metadata/medical_sft_base_eval_results/",
    "medical-logprob": "gs://marin-us-central1/metadata/medical_logprob_sft_base_eval_results/",
    "math": "gs://marin-us-central1/metadata/math_sft_base_eval_results/",
    "code": "gs://marin-us-central1/metadata/code_sft_base_eval_results/",
}

# Engine kwargs — 1:1 with the original recipe (`baseline_14b_eval.py:27`).
DEFAULT_ENGINE_KWARGS: dict = {"max_model_len": 8192, "max_gen_toks": 512}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", choices=["code", "math", "medical", "medical-logprob"], required=True)
    parser.add_argument(
        "--model-rel-path",
        required=True,
        help="HF checkpoint path RELATIVE to local region bucket, OR an HF reference like 'Qwen/Qwen3-8B-Base'.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name for W&B / output_path naming.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write eval results. Defaults to <model-path>/eval/lm_eval_harness/.",
    )
    parser.add_argument(
        "--eval-results-prefix",
        default=None,
        help="Where to write the per-run summary JSON. Defaults to per-domain prefix in DOMAIN_SUMMARY_PREFIX.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="If set, apply tokenizer's chat template at eval time. OFF by default for Base models.",
    )
    parser.add_argument("--max-eval-instances", type=int, default=None)
    return parser.parse_args(argv)


def _assert_model_local(model_path: str, region: str) -> None:
    expected_prefix = region_tracker.REGION_TO_BUCKET[region] + "/"
    if not model_path.startswith(expected_prefix):
        raise ValueError(
            f"model_path={model_path!r} is not in the local region's bucket "
            f"(expected prefix {expected_prefix!r}). Cross-region reads forbidden."
        )


def _resolve_local_model_path(model_rel_path: str, region: str) -> str:
    bucket = region_tracker.REGION_TO_BUCKET[region]
    return f"{bucket}/{model_rel_path.strip('/')}"


def _is_hf_reference(name: str) -> bool:
    return name.count("/") == 1 and not name.startswith("gs://")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    region = region_tracker.detect_current_region()
    logger.info(
        "Standalone EVAL child boot — domain=%s, region=%s, model=%s",
        args.domain,
        region,
        args.model_name,
    )

    if _is_hf_reference(args.model_rel_path):
        model_path = args.model_rel_path
        logger.info("Detected HF reference: %s (skipping local-region assert)", model_path)
    else:
        model_path = _resolve_local_model_path(args.model_rel_path, region)
        _assert_model_local(model_path, region)
        logger.info("Local model path: %s", model_path)

    if args.output_path:
        output_path = args.output_path
    elif model_path.startswith("gs://"):
        output_path = f"{model_path.rstrip('/')}/eval/lm_eval_harness"
    else:
        bucket = region_tracker.REGION_TO_BUCKET[region]
        output_path = f"{bucket}/eval_baselines/{args.model_name}/lm_eval_harness"
    logger.info("Eval output path: %s", output_path)

    evals = DOMAIN_EVALS[args.domain]
    eval_results_prefix = args.eval_results_prefix or DOMAIN_SUMMARY_PREFIX[args.domain]

    model = ModelConfig(
        name=args.model_name,
        path=model_path,
        engine_kwargs=dict(DEFAULT_ENGINE_KWARGS),
        generation_params=None,
        apply_chat_template=args.apply_chat_template,
        base_eval_run_name=args.model_name,
    )

    evaluator = LMEvaluationHarnessEvaluator()
    evaluator.evaluate(
        model=model,
        evals=evals,
        output_path=output_path,
        max_eval_instances=args.max_eval_instances,
        wandb_tags=[
            args.domain,
            "eval",
            "qwen3-base",
            f"model={args.model_name}",
        ],
    )
    logger.info("Eval finished cleanly.")

    summary = {
        "domain": args.domain,
        "model_name": args.model_name,
        "model_path": model_path,
        "model_rel_path": args.model_rel_path,
        "region": region,
        "output_path": output_path,
        "evals": [e.task_alias or e.name for e in evals],
        "apply_chat_template": args.apply_chat_template,
        "engine_kwargs": DEFAULT_ENGINE_KWARGS,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    summary_path = f"{eval_results_prefix.rstrip('/')}/{args.model_name}.json"
    try:
        with fsspec.open(summary_path, "w") as f:
            f.write(json.dumps(summary, indent=2))
        logger.info("Wrote eval summary: %s", summary_path)
    except Exception as e:
        logger.warning("Failed to write eval summary at %s: %s", summary_path, e)

    done_marker = f"{output_path.rstrip('/')}/.{args.domain}_eval_DONE"
    try:
        with fsspec.open(done_marker, "w") as f:
            f.write(
                json.dumps(
                    {
                        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
                        "domain": args.domain,
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
