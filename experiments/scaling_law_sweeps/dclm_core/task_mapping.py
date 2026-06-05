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
        "exact_match",
        notes="Custom lm-eval task in custom_tasks/jeopardy/ (vendored from mosaicml/llm-foundry@v0.9.0).",
    ),
    TaskMapEntry(
        "bigbench_qa_wikidata",
        "bigbench_qa_wikidata_generate_until",
        10,
        "exact_match",
        notes="Only the generate_until variant exists in marin's lm-eval pin (verified via registry probe).",
    ),
    TaskMapEntry("arc_easy", "arc_easy", 10, "acc_norm"),
    TaskMapEntry("arc_challenge", "arc_challenge", 10, "acc_norm"),
    TaskMapEntry("copa", "copa", 0, "acc"),
    TaskMapEntry("commonsense_qa", "commonsense_qa", 10, "acc"),
    TaskMapEntry("piqa", "piqa", 10, "acc_norm"),
    TaskMapEntry("openbook_qa", "openbookqa", 0, "acc_norm"),
    TaskMapEntry("lambada_openai", "lambada_openai", 0, "acc"),
    TaskMapEntry("hellaswag", "hellaswag", 10, "acc_norm"),
    TaskMapEntry(
        "winograd",
        "winograd",
        0,
        "acc",
        notes="Custom task in custom_tasks/winograd/ — stock lm-eval wsc273 fetches a "
        "URL that 302-redirects and the dataset loader doesn't follow it. We "
        "vendor WSCollection.xml + convert to jsonl.",
    ),
    TaskMapEntry("winogrande", "winogrande", 0, "acc"),
    TaskMapEntry(
        "bigbench_dyck_languages",
        "bigbench_dyck_languages_generate_until",
        10,
        "exact_match",
        notes="DCLM's dyck is a completion task. lm-eval's generate_until variant fits.",
    ),
    TaskMapEntry("agi_eval_lsat_ar", "agieval_lsat_ar", 3, "acc_norm"),
    TaskMapEntry("bigbench_cs_algorithms", "bigbench_cs_algorithms_generate_until", 10, "exact_match"),
    TaskMapEntry("bigbench_operators", "bigbench_operators_generate_until", 10, "exact_match"),
    TaskMapEntry("bigbench_repeat_copy_logic", "bigbench_repeat_copy_logic_generate_until", 10, "exact_match"),
    TaskMapEntry(
        "squad",
        "squad_completion",
        10,
        "contains",
        notes="lm-eval's `squad_completion` only reports `contains` (not exact_match). "
        "DCLM uses exact_match; for calibration we may need to swap to a different "
        "squad variant or post-process generations against gold to compute em.",
    ),
    TaskMapEntry(
        "coqa",
        "coqa",
        0,
        "f1",
        notes="DCLM's coqa is generative QA. lm-eval coqa reports f1 + em — DCLM uses what? Verify in calibration.",
    ),
    TaskMapEntry("boolq", "boolq", 10, "acc"),
    TaskMapEntry(
        "bigbench_language_identification",
        "bigbench_language_identification_multiple_choice",
        10,
        "acc",
        notes="lm-eval's multiple_choice variant matches DCLM's task_type.",
    ),
)

# A given DCLM task name may collide with another's lm-eval target (e.g.
# hellaswag at 0-shot vs hellaswag at 10-shot both map to lm-eval `hellaswag`).
# We disambiguate via task_alias when building lm-eval TaskConfigs.

assert len(CORE_TASK_MAP) == 22, f"Expected 22 CORE tasks, got {len(CORE_TASK_MAP)}"
