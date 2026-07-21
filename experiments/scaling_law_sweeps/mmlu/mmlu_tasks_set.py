# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU for the curation scaling sweeps, as the CRFM fork's `mmlu_sl_verb` variant.

We run `mmlu_sl_verb` ("sl" = scaling law), NOT stock `mmlu`. Both are
`output_type: multiple_choice` (logprob rank-classification), so both run natively in
the Levanter harness — the note in `olmes_base/olmes_tasks_set.py` claiming MMLU
cannot is wrong; `exp600_tootsie.py` has run stock MMLU in-loop through this very
harness. Two reasons to prefer sl_verb here:

  * METRICS. sl_verb emits `acc, acc_norm, bpb, logprob, choice_logprob,
    choice_prob_norm, choice_logprob_norm`. This sweep spans 1e17..2e21 FLOPs at
    d512..d2432, where raw MMLU accuracy is pinned at the 25% random baseline — a
    flat line of noise. The soft metrics (choice_logprob, choice_prob_norm) stay
    informative down there, which is the entire point of the variant.
    `exp1337_eval_suite.py` uses it for exactly this reason (the Delphi
    "mmlu-emergence" figure fits a soft-metric power law, then maps soft->hard).
  * DATASET. sl_verb loads `hails/mmlu_no_train`, which is pure parquet with NO
    loading script. Stock `mmlu` loads `cais/mmlu`, which still carries a legacy
    `hendrycks_test.py` — the exact shape that killed wsc273 and social_iqa under
    marin's `datasets>=3.1` pin. sl_verb sidesteps the question entirely.

SHOT COUNTS. sl_verb's YAML sets no `num_fewshot`, but it does ship
`fewshot_split: dev` + `fewshot_config.sampler: first_n` — it is built to be handed a
shot count by the caller, and MMLU's `dev` split holds exactly 5 examples per subject.
That is the canonical 5-shot MMLU protocol, and it is what the sweep runs.

0-shot is defined here but NOT run by default. It buys no extra soft-metric signal:
sl_verb emits all seven metrics at EVERY shot count, so `choice_logprob` /
`choice_prob_norm` come out of 5-shot just as well. Context length is not a reason to
prefer it either — measured against the evaluator's hardcoded 2048-token `max_length`
(`levanter_lm_eval_evaluator.py`) with the Qwen3 tokenizer, the worst subject
(professional_law: 1534 docs, the longest prompts in MMLU) has a median 5-shot context
of 1699 tokens and only 7/1534 docs (0.5%) over 2048. Truncation is therefore ~0.05% of
MMLU and not worth a second sweep. Keep `--shots 0` for one-off comparisons.

marin's `EvalTaskConfig.num_fewshot` is a required field and
`convert_to_levanter_task_config` always forwards it, so a shot count is always passed
explicitly; there is no "inherit the YAML default" path.

The two shot counts must be evaluated as SEPARATE runs, never together in one
`evals=[...]`: they share the lm-eval task name `mmlu_sl_verb` and would collide in the
harness's task dict. exp1337 submits one step per shot count for this reason, and
`launch_mmlu_manifest` submits one child per (checkpoint, shot count) into its own
results directory.

`mmlu_sl_verb` is a GROUP: 56 subject tasks -> 4 subgroups (stem / other /
social_sciences / humanities) -> one group row carrying the aggregate metrics
(size-weighted for acc/acc_norm, unweighted mean for the soft metrics).

CONSOLIDATION NOTES, verified against a real results.json (2026-07-15):
  * `results` holds 58 rows and the group row is keyed by the ALIAS —
    `mmlu_sl_verb_0shot` / `mmlu_sl_verb_5shot`, NOT `mmlu_sl_verb`. Subject rows are
    aliased the same way (`mmlu_sl_verb_anatomy_5shot`). Read the aliased group row;
    do not flat-average every key.
  * Metrics are suffixed `,none` (e.g. `acc,none`, `choice_prob_norm,none`); the
    `*_stderr,none` companions come back as the string "N/A", so they cannot be parsed
    as floats.
  * `outputs` (logged samples) is a GLOBAL list duplicated into EVERY row — anatomy's
    row carries abstract_algebra's prompts. It is not per-subject; do not analyze it
    per task. It is also why results.json runs ~8MB.

Both the cache builder and the runner import this module so the cached datasets and
the evaluated task list cannot drift apart.
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

MMLU_SL_VERB_TASK_NAME = "mmlu_sl_verb"

MMLU_SL_VERB_BY_SHOTS: dict[int, EvalTaskConfig] = {
    0: EvalTaskConfig(MMLU_SL_VERB_TASK_NAME, 0, task_alias="mmlu_sl_verb_0shot"),
    5: EvalTaskConfig(MMLU_SL_VERB_TASK_NAME, 5, task_alias="mmlu_sl_verb_5shot"),
}

SUPPORTED_SHOTS: tuple[int, ...] = tuple(MMLU_SL_VERB_BY_SHOTS)


def task_for_shots(num_fewshot: int) -> EvalTaskConfig:
    """The single `mmlu_sl_verb` task config at `num_fewshot`, for one eval run.

    Raises ValueError on any shot count the sweep does not define, rather than
    silently constructing an unblessed variant whose results would not line up with
    the rest of the sweep.
    """
    if num_fewshot not in MMLU_SL_VERB_BY_SHOTS:
        raise ValueError(f"Unsupported shot count {num_fewshot}; expected one of {SUPPORTED_SHOTS}.")
    return MMLU_SL_VERB_BY_SHOTS[num_fewshot]


def unique_dataset_task_names() -> list[str]:
    """Distinct lm-eval task names to instantiate when building the dataset cache.

    Shot count does not change the dataset (0/5-shot both read `hails/mmlu_no_train`;
    5-shot additionally reads its `dev` split, which the cache already holds), so this
    is the one group name. Resolving the group fans out to all 56 subject configs.
    """
    return [MMLU_SL_VERB_TASK_NAME]
