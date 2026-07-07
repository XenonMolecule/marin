# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install this package's custom lm-eval task YAMLs onto lm_eval's TaskManager.

Levanter's LM-eval harness constructs `lm_eval.tasks.TaskManager()` with NO
`include_path`. To make our `custom_tasks/` YAMLs (currently just a parquet-backed
`social_iqa` that shadows the datasets>=3-dead stock loader) discoverable, we patch
`TaskManager.__init__` to default `include_path` to our dir — the same mechanism the
DCLM-CORE runner uses for its vendored YAMLs.

Both the offline dataset-cache builder and the eval runner call
`install_custom_task_path()` so the cache and the eval resolve the identical
`social_iqa` definition (and thus the identical dataset).
"""

from __future__ import annotations

from pathlib import Path

_CUSTOM_TASKS_DIR = Path(__file__).parent / "custom_tasks"


def install_custom_task_path() -> None:
    """Patch `lm_eval.tasks.TaskManager.__init__` so no-arg constructions pick up
    our custom_tasks/ dir. Idempotent."""
    import lm_eval.tasks as tasks

    if getattr(tasks.TaskManager.__init__, "_olmes_patched", False):
        return

    resolved_path = str(_CUSTOM_TASKS_DIR)
    original_init = tasks.TaskManager.__init__

    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            include_path = resolved_path
        original_init(self, *args, include_path=include_path, **kwargs)

    patched_init._olmes_patched = True  # type: ignore[attr-defined]
    tasks.TaskManager.__init__ = patched_init  # type: ignore[method-assign]
