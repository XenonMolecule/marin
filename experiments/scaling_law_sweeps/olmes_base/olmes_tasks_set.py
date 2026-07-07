# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The OLMo base-model "easy" downstream suite, as runnable stock lm-eval tasks.

This is the OLMES ``core_9mcqa`` suite (allenai/olmes) — the multiple-choice
downstream tasks OLMo/OLMES use for rapid base-model assessment — expressed with
STOCK lm-eval task names so the Levanter LM-eval harness scores them exactly as
`default_eval` would. We run them 0-shot (rank-classification), matching the cheap
in-loop "easy" setting rather than the heavier 5-shot OLMES-standard formulation.

Membership vs the marin CORE_TASKS suite:
  * OLMES core-9: arc_easy, arc_challenge, boolq, commonsense_qa (csqa),
    hellaswag, openbookqa, piqa, social_iqa, winogrande.
  * We add `sciq` (an OLMo-1 in-loop core task, cheap, not in CORE_TASKS).
  * MMLU / generative (coqa, squad, drop, naturalqs, jeopardy) and the code/math
    bpb tasks in the OLMo "base" suites are NOT logprob-MC and cannot run in the
    Levanter harness — same reason CORE_TASKS omits squadv2.

`social_iqa`'s STOCK loader (`allenai/social_i_qa`) is a dataset SCRIPT, dead under
marin's `datasets>=3.1` pin (identical failure to wsc273). We keep the task by
SHADOWING it with a custom lm-eval YAML (see olmes_custom_tasks) that points at the
`lighteval/siqa` parquet mirror — byte-identical rows, same prompt/choices/target
as stock siqa.yaml. So it is faithful data, loaded via parquet instead of a script.
"""

from marin.evaluation.evaluation_config import EvalTaskConfig

OLMES_BASE_EASY = (
    EvalTaskConfig("arc_easy", 0),
    EvalTaskConfig("arc_challenge", 0),
    EvalTaskConfig("boolq", 0),
    EvalTaskConfig("commonsense_qa", 0),  # OLMES "csqa"
    EvalTaskConfig("hellaswag", 0),
    EvalTaskConfig("openbookqa", 0),
    EvalTaskConfig("piqa", 0),
    EvalTaskConfig("sciq", 0),  # OLMo-1 core; new vs CORE_TASKS
    EvalTaskConfig("social_iqa", 0),  # shadowed -> lighteval/siqa parquet (see olmes_custom_tasks)
    EvalTaskConfig("winogrande", 0),
)

# All ten run today: social_iqa via the custom parquet task, the rest stock.
OLMES_BASE_EASY_RUNNABLE = OLMES_BASE_EASY


def unique_dataset_task_names() -> list[str]:
    """Distinct lm-eval task names behind OLMES_BASE_EASY_RUNNABLE (each name maps to
    one dataset; shot count does not change the dataset)."""
    seen: dict[str, None] = {}
    for entry in OLMES_BASE_EASY_RUNNABLE:
        seen.setdefault(entry.name, None)
    return list(seen)
