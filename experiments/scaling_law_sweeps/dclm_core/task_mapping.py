# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Map DCLM CORE task names (MosaicML eval-gauntlet convention) onto
EleutherAI lm-evaluation-harness task names.

DCLM's actual eval runner uses llm-foundry's `build_icl_evaluators`, NOT
lm-eval-harness. Task semantics overlap but prompt formatting, few-shot
sampling, and metric definitions can differ. This mapping is best-effort
and must be calibrated against a DCLM-published reference number before
trusted.

Each entry below specifies:
  - `dclm`: the task name as it appears in DCLM's eval_meta_data.csv
  - `lm_eval`: the corresponding lm-evaluation-harness task name
  - `num_fewshot`: shots per DCLM's meta-data (authoritative — overrides
    lm-eval's default to match DCLM)
  - `metric`: which key to pull from lm-eval's per-task results dict
    ('acc_norm' is typically what MosaicML calls "accuracy" for MC tasks;
    'acc' for language-modeling / generative tasks; for `lambada` it's
    'acc')

Tasks not present in stock lm-eval-harness are flagged. We may need to
add custom YAML task files for the bigbench_* members if the stock
versions are missing or score differently.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskMapEntry:
    dclm: str  # DCLM/MosaicML task name (key in eval_meta_data.csv)
    lm_eval: str  # lm-eval-harness task name
    num_fewshot: int  # shots — matches DCLM's eval_meta_data.csv
    metric: str  # metric key in lm-eval's per-task results dict
    notes: str = ""


# 22 CORE tasks. Few-shot counts copied verbatim from DCLM's eval_meta_data.csv.
# lm-eval names verified against EleutherAI upstream registry (commit
# 0ef7548 era — what marin's Stanford-CRFM fork is based on). Tasks marked
# (?) need verification during Phase-2 calibration.
CORE_TASK_MAP: tuple[TaskMapEntry, ...] = (
    TaskMapEntry("hellaswag_zeroshot", "hellaswag", 0, "acc_norm"),
    TaskMapEntry(
        "jeopardy",
        "jeopardy",
        10,
        "acc",
        notes="Custom loglikelihood task in custom_tasks/jeopardy/ (mosaicml/llm-foundry@v0.9.0). "
        "acc = is_greedy = DCLM's InContextLearningLMAccuracy (language_modeling), NOT generate_until.",
    ),
    TaskMapEntry(
        "bigbench_qa_wikidata",
        "bigbench_qa_wikidata_dclm",
        10,
        "acc",
        notes="Custom loglikelihood task in custom_tasks/bigbench_qa_wikidata/ (llm-foundry gauntlet). "
        "acc = is_greedy = DCLM's language_modeling scoring, NOT the slower/looser generate_until.",
    ),
    TaskMapEntry("arc_easy", "arc_easy", 10, "acc_norm"),
    TaskMapEntry("arc_challenge", "arc_challenge", 10, "acc_norm"),
    TaskMapEntry("copa", "copa", 0, "acc"),
    TaskMapEntry(
        "commonsense_qa",
        "commonsense_qa_dclm4",
        10,
        "acc",
        notes="Custom 4-choice task in custom_tasks/commonsense_qa/ (llm-foundry gauntlet). "
        "Stock lm-eval commonsense_qa is 5-choice (baseline 20%); DCLM is 4-choice (baseline 25%).",
    ),
    TaskMapEntry("piqa", "piqa", 10, "acc_norm"),
    TaskMapEntry("openbook_qa", "openbookqa", 0, "acc_norm"),
    TaskMapEntry("lambada_openai", "lambada_openai", 0, "acc"),
    TaskMapEntry("hellaswag", "hellaswag", 10, "acc_norm"),
    TaskMapEntry(
        "winograd",
        "winograd",
        0,
        "acc",
        notes="Custom schema task in custom_tasks/winograd/ (llm-foundry gauntlet WSC273 = 273 rows). "
        "Scores P(continuation | option-substituted context) via lm-eval's winogrande-style role "
        "inversion (!function preprocess_winograd); matches DCLM's icl_task_type=schema.",
    ),
    TaskMapEntry("winogrande", "winogrande", 0, "acc"),
    TaskMapEntry(
        "bigbench_dyck_languages",
        "bigbench_dyck_languages_dclm",
        10,
        "acc",
        notes="Custom loglikelihood task in custom_tasks/bigbench_dyck_languages/ (llm-foundry gauntlet). "
        "acc = is_greedy = DCLM's language_modeling scoring, NOT generate_until.",
    ),
    TaskMapEntry(
        "agi_eval_lsat_ar",
        "agi_eval_lsat_ar_dclm4",
        3,
        "acc_norm",
        notes="Custom 4-choice task in custom_tasks/agi_eval_lsat_ar/ (llm-foundry gauntlet). "
        "Stock lm-eval agieval_lsat_ar is 5-choice (baseline 20%); DCLM is 4-choice (baseline 25%).",
    ),
    TaskMapEntry("bigbench_cs_algorithms", "bigbench_cs_algorithms_dclm", 10, "acc"),
    TaskMapEntry("bigbench_operators", "bigbench_operators_dclm", 10, "acc"),
    TaskMapEntry("bigbench_repeat_copy_logic", "bigbench_repeat_copy_logic_dclm", 10, "acc"),
    TaskMapEntry(
        "squad",
        "squad_dclm",
        10,
        "acc",
        notes="Custom task in custom_tasks/squad/ (llm-foundry gauntlet, full SQuAD v1.1 dev = 10570). "
        "output_type=loglikelihood → acc = is_greedy = DCLM's InContextLearningLMAccuracy "
        "(teacher-forced greedy-token-match), NOT the slower/looser generate_until.",
    ),
    TaskMapEntry(
        "coqa",
        "coqa_dclm",
        0,
        "acc",
        notes="Custom task in custom_tasks/coqa/ (llm-foundry gauntlet, per-question = 7983). "
        "output_type=loglikelihood → acc = is_greedy = DCLM's InContextLearningLMAccuracy "
        "(teacher-forced greedy-token-match), NOT the slower/looser generate_until.",
    ),
    TaskMapEntry("boolq", "boolq", 10, "acc"),
    TaskMapEntry(
        "bigbench_language_identification",
        "bigbench_language_identification",
        10,
        "acc",
        notes="Custom 4-choice task in custom_tasks/bigbench_language_identification/ (MosaicML "
        "llm-foundry gauntlet jsonl). Stock lm-eval's *_multiple_choice pulls hails/bigbench with "
        "ELEVEN choices (baseline ~9%, ~10x slower); DCLM's CoreV2 is FOUR-choice (baseline 25%).",
    ),
)

# A given DCLM task name may collide with another's lm-eval target (e.g.
# hellaswag at 0-shot vs hellaswag at 10-shot both map to lm-eval `hellaswag`).
# We disambiguate via task_alias when building lm-eval TaskConfigs.

assert len(CORE_TASK_MAP) == 22, f"Expected 22 CORE tasks, got {len(CORE_TASK_MAP)}"
