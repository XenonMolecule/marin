# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The runnable subset of the standard pretraining CORE_TASKS.

`wsc273` is excluded: its stock lm-eval task loads `winograd_wsc` via a dataset
loading SCRIPT, which marin's pinned `datasets>=3.1.0` has removed support for
("Dataset scripts are no longer supported"). This breaks the stock task in
`default_eval`'s own environment too (in-loop CORE would fail it identically),
so running the remaining 12 entries is exactly what default_eval can run today.
DCLM only got wsc273 by reprocessing it into a bespoke local JSONL — which is the
non-stock variant we are deliberately NOT reusing.

Both the offline dataset-cache builder and the eval runner import this set so the
cache and the eval task list can never drift apart.
"""

from experiments.evals.task_configs import CORE_TASKS

EXCLUDED_TASK_NAMES: tuple[str, ...] = ("wsc273",)

CORE_TASKS_RUNNABLE = tuple(e for e in CORE_TASKS if e.name not in EXCLUDED_TASK_NAMES)


def unique_dataset_task_names() -> list[str]:
    """Distinct lm-eval task names behind CORE_TASKS_RUNNABLE (shot count does not
    change the dataset, so hellaswag's 0/10-shot dedupe to one)."""
    seen: dict[str, None] = {}
    for entry in CORE_TASKS_RUNNABLE:
        seen.setdefault(entry.name, None)
    return list(seen)
