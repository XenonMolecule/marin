# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Register extra lm-eval task directories with the harness's no-arg ``TaskManager()``.

Levanter's harness constructs ``lm_eval.tasks.TaskManager()`` without an ``include_path``,
so custom task YAMLs (OLMES's parquet-backed ``social_iqa``, the cultural suite's
generated tasks) are invisible unless the constructor default is patched. Every suite
that needs custom YAMLs registers its directory here; the patch is installed once and
hands the accumulated list to every later no-arg construction.
"""

from __future__ import annotations

from pathlib import Path

_REGISTERED_DIRS: list[str] = []


def install_include_paths(*dirs: str | Path) -> list[str]:
    """Add ``dirs`` to the include paths of every future no-arg ``TaskManager()``.

    Returns the full registered list. Idempotent per directory; the constructor patch is
    applied at most once per process.
    """
    import lm_eval.tasks as tasks  # noqa: PLC0415  # optional dep: lm_eval

    for d in dirs:
        resolved = str(Path(d).resolve())
        if resolved not in _REGISTERED_DIRS:
            _REGISTERED_DIRS.append(resolved)

    if getattr(tasks.TaskManager.__init__, "_marin_include_paths", False):
        return list(_REGISTERED_DIRS)

    original_init = tasks.TaskManager.__init__

    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            include_path = list(_REGISTERED_DIRS)
        original_init(self, *args, include_path=include_path, **kwargs)

    patched_init._marin_include_paths = True  # type: ignore[attr-defined]
    tasks.TaskManager.__init__ = patched_init  # type: ignore[method-assign]
    return list(_REGISTERED_DIRS)
