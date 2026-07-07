# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the DCLM CORE v2 eval tasks to a flat decontamination source.

For each unique task in `CORE_TASK_MAP` we materialize its scored eval docs
via the same lm-eval-harness fork the eval runner uses (custom jeopardy /
winograd tasks included), then write one JSONL record per doc with a single
`text` field = the rendered example (question/context + gold answer). That
output directory becomes `decontaminate_source` for
`marin.processing.classification.decon` in DECONTAMINATE mode.

We render **question + gold answer** (the "full example", GPT-3 / Dolma
convention = Option 2). Under n-gram matching (n>=13) short answer strings
cannot form an n-gram, so including the answer is a no-op on multiple-choice
tasks and only adds genuine long-answer spans on the generative tasks
(squad / coqa / jeopardy / bigbench `generate_until`).

Few-shot count does NOT change the doc set — it only prepends exemplars at
request time — so tasks that differ only in `num_fewshot` (hellaswag 0-shot
vs 10-shot) share one rendered file, keyed by the lm-eval task name.

Usage (local smoke: 5 docs/task, prints samples + counts, no GCS):

    uv run --with "lm-eval[math,api]@git+https://github.com/stanford-crfm/lm-evaluation-harness" \\
           --with torch --with transformers --with sentencepiece --with datasets \\
        python -m experiments.scaling_law_sweeps.dclm_core.dump_core_v2_decon_source \\
            --output-dir scratch/decon_core_v2 --limit 5 --print-samples

    # Full dump to GCS (decontaminate_source for the decon run):
    ... --output-dir gs://marin-us-central1/decontamination/dclm_core_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP

logger = logging.getLogger(__name__)

# Splits to try, in priority order. CORE is scored on test where available;
# a few tasks expose only validation/train.
_SPLIT_METHODS = ("test_docs", "validation_docs", "training_docs")

_CUSTOM_TASKS_DIR = Path(__file__).parent / "custom_tasks"


# NOTE: these two helpers are inlined (rather than imported from
# run_dclm_core_eval) so this dumper depends only on the `eval` uv extra
# (lm-eval + transformers) and NOT the `tpu` stack (jax/levanter), which
# run_dclm_core_eval pulls in at module import. Keep them in sync if the
# custom-task layout changes.
def _materialize_resolved_custom_tasks() -> str:
    """Stage custom_tasks/ to a temp dir, rewriting relative `data_files` in
    any YAML to absolute paths so lm-eval resolves them regardless of CWD."""
    tmp = Path(tempfile.mkdtemp(prefix="dclm_core_decon_tasks_"))
    shutil.copytree(_CUSTOM_TASKS_DIR, tmp, dirs_exist_ok=True)

    for yaml_file in tmp.rglob("*.yaml"):
        text = yaml_file.read_text()
        orig = text
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("data_files:"):
                rel = stripped.split(":", 1)[1].strip()
                if rel and not rel.startswith("/") and not rel.startswith("hf://"):
                    abs_path = yaml_file.parent / rel
                    text = text.replace(line, line.replace(stripped, f"data_files: {abs_path}"))
        if text != orig:
            yaml_file.write_text(text)

    return str(tmp)


def _install_custom_task_path() -> None:
    """Patch lm_eval's TaskManager so the vendored jeopardy/winograd YAMLs are
    findable when it is constructed with no include_path."""
    resolved_path = _materialize_resolved_custom_tasks()

    import lm_eval.tasks as tasks

    original_init = tasks.TaskManager.__init__

    def patched_init(self, *args, include_path=None, **kwargs):
        if include_path is None:
            include_path = resolved_path
        original_init(self, *args, include_path=include_path, **kwargs)

    tasks.TaskManager.__init__ = patched_init


def _gold_answer(task, doc) -> str:
    """Render the gold answer for one doc as a string.

    Normalizes the three lm-eval target shapes: an int index into the
    multiple-choice list (only here do we consult `doc_to_choice`, which
    generative tasks don't define), a list of acceptable answers, or a bare
    string.
    """
    target = task.doc_to_target(doc)

    if isinstance(target, int):
        choices = task.doc_to_choice(doc)
        return str(choices[target]) if 0 <= target < len(choices) else ""
    if isinstance(target, (list, tuple)):
        return " ".join(str(t) for t in target)
    return str(target)


def _render_text(task, doc) -> str:
    """question/context + gold answer, newline-joined (Option 2 full example)."""
    ctx = task.doc_to_text(doc)
    if not isinstance(ctx, str):
        ctx = str(ctx)
    answer = _gold_answer(task, doc)
    return f"{ctx}\n{answer}" if answer else ctx


def _iter_docs(task, limit: int | None) -> Iterator[dict]:
    """Yield docs from the first available split (test > validation > train)."""
    for method in _SPLIT_METHODS:
        fn = getattr(task, method, None)
        if fn is None:
            continue
        docs = fn()
        if docs is None:
            continue
        count = 0
        for doc in docs:
            yield doc
            count += 1
            if limit and count >= limit:
                return
        return


def _open_for_write(path: str):
    if path.startswith("gs://"):
        return marin_filesystem("gcs").open(path, "w")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w")


def _unique_lm_eval_tasks() -> list[str]:
    """De-duplicated lm-eval task names across the 22-entry CORE map."""
    seen: dict[str, None] = {}
    for entry in CORE_TASK_MAP:
        seen.setdefault(entry.lm_eval, None)
    return list(seen)


def dump(output_dir: str, limit: int | None, print_samples: bool) -> dict[str, int]:
    """Render every unique CORE task to `<output_dir>/<lm_eval_task>.jsonl`.

    Returns a {task_name: doc_count} map.
    """
    _install_custom_task_path()

    import lm_eval.tasks as lm_eval_tasks

    mgr = lm_eval_tasks.TaskManager()
    registry = set(mgr.all_tasks)

    counts: dict[str, int] = {}
    for task_name in _unique_lm_eval_tasks():
        if task_name not in registry:
            raise ValueError(f"CORE task {task_name!r} not in lm-eval registry — fix task_mapping first")

        task_dict = lm_eval_tasks.get_task_dict([task_name], mgr)
        task = next(iter(task_dict.values()))

        out_path = f"{output_dir.rstrip('/')}/{task_name}.jsonl"
        written = 0
        with _open_for_write(out_path) as f:
            for doc in _iter_docs(task, limit):
                text = _render_text(task, doc)
                # id/task carry provenance so the inspection pass can attribute a
                # flagged corpus span back to the specific eval item it matched.
                record = {"id": f"{task_name}:{written}", "task": task_name, "text": text}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                if print_samples and written == 0:
                    logger.info("[%s] sample text:\n%s\n---", task_name, text[:600])
                written += 1
        counts[task_name] = written
        logger.info("  %-50s %6d docs -> %s", task_name, written, out_path)

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Local or gs:// dir for per-task JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Cap docs per task (smoke test). Omit for full dump.")
    parser.add_argument("--print-samples", action="store_true", help="Log the first rendered text per task.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    counts = dump(args.output_dir, args.limit, args.print_samples)

    total = sum(counts.values())
    logger.info("=" * 70)
    logger.info("Rendered %d tasks, %d total docs -> %s", len(counts), total, args.output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
