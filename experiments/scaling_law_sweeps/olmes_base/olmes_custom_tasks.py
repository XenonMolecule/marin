# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install this package's custom lm-eval task YAMLs onto lm_eval's TaskManager.

Levanter's LM-eval harness constructs `lm_eval.tasks.TaskManager()` with NO
`include_path`. To make our `custom_tasks/` YAMLs (currently just a parquet-backed
`social_iqa` that shadows the datasets>=3-dead stock loader) discoverable, the shared
`lm_eval_include_path` patch defaults `include_path` to every registered directory.

Both the offline dataset-cache builder and the eval runner call
`install_custom_task_path()` so the cache and the eval resolve the identical
`social_iqa` definition (and thus the identical dataset).
"""

from __future__ import annotations

from pathlib import Path

from experiments.scaling_law_sweeps.lm_eval_include_path import install_include_paths

_CUSTOM_TASKS_DIR = Path(__file__).parent / "custom_tasks"


def install_custom_task_path() -> None:
    """Register `custom_tasks/` so no-arg `TaskManager()` constructions pick it up. Idempotent."""
    install_include_paths(_CUSTOM_TASKS_DIR)
