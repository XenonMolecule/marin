# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Try to load each of the 22 DCLM CORE tasks via lm-eval-harness — no model,
no inference, just the dataset/config materialization path. Report which load
cleanly and which fail with what error.

Why: when smoke-testing the full pipeline, every dataset issue (dead URLs,
missing tasks, trust_remote_code prompts, etc.) requires re-running the
whole stack (JAX init, model load, all earlier tasks) just to find the next
broken task. This probe cuts that loop down to ~30 seconds.

Usage:
    uv run --with "lm-eval[math,api]@git+...stanford-crfm...@d5e3391f..." \
           --with torch --with transformers --with sentencepiece \
        python -m experiments.scaling_law_sweeps.dclm_core.probe_task_loads
"""

from __future__ import annotations

import logging
import sys
import traceback

from experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval import (
    _install_custom_task_path,
)
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _install_custom_task_path()

    import lm_eval.tasks as lm_eval_tasks

    mgr = lm_eval_tasks.TaskManager()
    all_tasks = set(mgr.all_tasks)

    print(f"\nlm-eval-harness fork: {len(all_tasks)} tasks registered\n")
    print(f"Probing {len(CORE_TASK_MAP)} DCLM CORE tasks...\n")

    ok: list[str] = []
    fail: dict[str, str] = {}

    for entry in CORE_TASK_MAP:
        lm_eval_name = entry.lm_eval
        label = f"{entry.dclm:>42s}  →  {lm_eval_name:<55s}"

        if lm_eval_name not in all_tasks:
            print(f"  ✗  {label} REGISTRY MISS")
            fail[entry.dclm] = "task name not in registry"
            continue

        # Try to actually instantiate the task dict — this triggers dataset
        # download / config materialization, which is where most failures live.
        try:
            task_dict = lm_eval_tasks.get_task_dict([lm_eval_name], mgr)
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            short = (str(e).splitlines() or [""])[0][:200]
            print(f"  ✗  {label} LOAD FAIL: {type(e).__name__}: {short}")
            fail[entry.dclm] = f"{type(e).__name__}: {short}"
            continue

        if not task_dict:
            print(f"  ✗  {label} EMPTY task_dict returned")
            fail[entry.dclm] = "empty task_dict"
            continue

        # Get the first instantiated task — peek at its docs from whichever
        # split exists (test / validation / training, in that order).
        try:
            first_task = next(iter(task_dict.values()))
            got_doc = False
            last_err: Exception | None = None
            for method_name in ("test_docs", "validation_docs", "training_docs"):
                try:
                    docs = getattr(first_task, method_name, lambda: None)()
                    if docs is None:
                        continue
                    sample = next(iter(docs))
                    if sample is not None:
                        got_doc = True
                        break
                except StopIteration:
                    continue
                except Exception as e:
                    last_err = e
                    continue
            if not got_doc:
                raise last_err or RuntimeError("no docs from any split")
        except Exception as e:
            short = (str(e).splitlines() or [""])[0][:200]
            print(f"  ✗  {label} DOC FETCH FAIL: {type(e).__name__}: {short}")
            fail[entry.dclm] = f"docs: {type(e).__name__}: {short}"
            continue

        print(f"  ✓  {label} OK")
        ok.append(entry.dclm)

    print()
    print("=" * 78)
    print(f"OK:   {len(ok)} / {len(CORE_TASK_MAP)}")
    print(f"FAIL: {len(fail)} / {len(CORE_TASK_MAP)}")
    if fail:
        print("\nFailures:")
        for dclm_task, reason in fail.items():
            print(f"  - {dclm_task}: {reason}")
    print("=" * 78)

    sys.exit(0 if not fail else 1)


if __name__ == "__main__":
    main()
