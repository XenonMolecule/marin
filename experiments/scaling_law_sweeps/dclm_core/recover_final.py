# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover a final Core JSON by re-aggregating from per-task partials on disk.

Use this when an upstream bug (e.g. the 2026-05-23 task-filter aggregation
bug) caused a final JSON to be written with only a subset of the 22 CORE
tasks. This script reads ALL partial JSONs in
`gs://.../data_curation_core_results/partial/{run_name}/`, merges them via
the same logic the runner uses, and writes a corrected final JSON to
`gs://.../data_curation_core_results/{run_name}.json` (overwriting the bad
one).

Designed to run inside an Iris job in us-central1 so GCS reads are free.

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.recover_final \\
        --run-name curation-high_quality_3000-expFM_natural-9e+19-d512-L6-B512 \\
        [--run-name <another>] ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from dataclasses import dataclass

from experiments.scaling_law_sweeps.dclm_core.centering import compute_core
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP, TaskMapEntry

logger = logging.getLogger(__name__)


# --- Inlined helpers so this module doesn't pull haliax/jax via run_dclm_core_eval. ---


@dataclass(frozen=True)
class _TaskAlias:
    task_alias: str
    num_fewshot: int


def build_task_configs(_limit=None) -> tuple[list[_TaskAlias], dict[str, TaskMapEntry]]:
    """Mirror of run_dclm_core_eval.build_task_configs, sans the levanter TaskConfig dep."""
    tasks: list[_TaskAlias] = []
    alias_to_entry: dict[str, TaskMapEntry] = {}
    for entry in CORE_TASK_MAP:
        alias = f"{entry.dclm}_{entry.num_fewshot}shot"
        tasks.append(_TaskAlias(task_alias=alias, num_fewshot=entry.num_fewshot))
        alias_to_entry[alias] = entry
    return tasks, alias_to_entry


def extract_dclm_results(
    lm_eval_results: dict,
    alias_to_entry: dict[str, TaskMapEntry],
) -> tuple[dict[str, float], dict[str, str]]:
    """Mirror of run_dclm_core_eval.extract_dclm_results."""
    results_section = lm_eval_results.get("results", {})
    raw: dict[str, float] = {}
    log: dict[str, str] = {}
    for alias, entry in alias_to_entry.items():
        task_results = results_section.get(alias)
        if task_results is None:
            log[entry.dclm] = f"MISSING_TASK_RESULTS (looked for alias {alias!r})"
            continue
        for key, val in task_results.items():
            if key.startswith(entry.metric + ",") and not key.endswith("_stderr,none"):
                if isinstance(val, (int, float)):
                    raw[entry.dclm] = float(val)
                    log[entry.dclm] = f"OK from {key} = {val}"
                    break
        else:
            log[entry.dclm] = f"NO_METRIC_MATCH (have keys: {sorted(task_results.keys())})"
    return raw, log


def _merge_lm_eval_outputs(partials: list[dict]) -> dict:
    merged: dict = {}
    aliased_keys = {"results", "configs", "versions", "samples", "n-shot", "higher_is_better"}
    for partial in partials:
        for key, value in partial.items():
            if key in aliased_keys and isinstance(value, dict):
                merged.setdefault(key, {}).update(value)
            else:
                merged.setdefault(key, value)
    return merged


def _resolve_partial_dir(output_json: str, run_name: str) -> str:
    parent = output_json.rsplit("/", 1)[0]
    return f"{parent}/partial/{run_name}"


def _partial_path(partial_dir: str, task_alias: str) -> str:
    return f"{partial_dir.rstrip('/')}/{task_alias}.json"


def _gcs_read(path: str) -> bytes | None:
    """Read a GCS object via rigging.filesystem (or local file). Returns None if missing."""
    if not path.startswith("gs://"):
        try:
            with open(path, "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None
    try:
        # Lazy import: rigging is available on v5p-8/v4-8 eval workers but not always on smaller pools.
        from rigging.filesystem import filesystem as marin_filesystem

        with marin_filesystem("gcs").open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        return None


def _gcs_write(path: str, content: bytes) -> None:
    if not path.startswith("gs://"):
        with open(path, "wb") as f:
            f.write(content)
        return
    from rigging.filesystem import filesystem as marin_filesystem

    with marin_filesystem("gcs").open(path, "wb") as f:
        f.write(content)


def _read_partial(path: str) -> dict | None:
    raw = _gcs_read(path)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as e:
        logger.warning("Could not parse partial %s: %s", path, e)
        return None


def _write_json(path: str, data: dict) -> None:
    body = json.dumps(data, indent=2, default=_json_default).encode()
    _gcs_write(path, body)


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def recover_one(run_name: str, output_prefix: str) -> dict:
    """Recover one config; return a small report dict."""
    output_json = f"{output_prefix.rstrip('/')}/{run_name}.json"
    partial_dir = _resolve_partial_dir(output_json, run_name)
    logger.info("=== %s ===", run_name)
    logger.info("partial dir: %s", partial_dir)
    logger.info("output:      %s", output_json)

    full_tasks, alias_to_entry = build_task_configs(None)
    t0 = time.time()
    partials: list[dict] = []
    for task in full_tasks:
        p = _partial_path(partial_dir, task.task_alias)
        loaded = _read_partial(p)
        if loaded is None:
            logger.warning("  missing partial %s", task.task_alias)
            continue
        partials.append(loaded)
    logger.info("loaded %d / %d partials in %.0fs", len(partials), len(full_tasks), time.time() - t0)

    if len(partials) < len(CORE_TASK_MAP):
        logger.error("Aborting — only %d / %d CORE partials present", len(partials), len(CORE_TASK_MAP))
        return {
            "run_name": run_name,
            "status": "skipped_incomplete",
            "partials": len(partials),
            "expected": len(CORE_TASK_MAP),
        }

    combined = _merge_lm_eval_outputs(partials)
    raw, extr_log = extract_dclm_results(combined, alias_to_entry)
    dclm = compute_core(raw)
    core = dclm.get("Core")
    logger.info("Core = %s", core)

    output = {
        "checkpoint": None,
        "run_name": run_name,
        "tokenizer": None,
        "args": {"limit": None, "max_length": 2048},
        "partial_dir": partial_dir,
        "extraction_log": extr_log,
        "dclm": dclm,
        "lm_eval_raw": combined,
        "_recovered_by": f"recover_final.py at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
    }
    _write_json(output_json, output)
    logger.info("✓ wrote final → %s", output_json)
    _write_summary(output_json, run_name, partial_dir, dclm, extr_log)
    return {"run_name": run_name, "status": "recovered", "core": core, "partials": len(partials)}


def _write_summary(output_json: str, run_name: str, partial_dir: str, dclm: dict, extr_log: dict) -> None:
    """Write a small `<output>_summary.json` next to the big final so callers
    can pull the Core score without downloading the multi-GB lm_eval_raw."""
    summary = {
        "run_name": run_name,
        "checkpoint": None,
        "tokenizer": None,
        "partial_dir": partial_dir,
        "dclm": dclm,
        "extraction_log": extr_log,
    }
    summary_path = output_json.rsplit(".json", 1)[0] + "_summary.json"
    _write_json(summary_path, summary)
    logger.info("  ✓ wrote summary → %s", summary_path)


def summarize_only(run_name: str, output_prefix: str) -> dict:
    """Read an existing valid final and emit just the small summary sibling."""
    output_json = f"{output_prefix.rstrip('/')}/{run_name}.json"
    logger.info("=== summarize: %s ===", run_name)
    raw = _gcs_read(output_json)
    if raw is None:
        raise RuntimeError(f"Could not read final at {output_json}")
    d = json.loads(raw)
    dclm = d.get("dclm", {}) or {}
    extr = d.get("extraction_log", {}) or {}
    partial_dir = d.get("partial_dir") or _resolve_partial_dir(output_json, run_name)
    _write_summary(output_json, run_name, partial_dir, dclm, extr)
    return {"run_name": run_name, "status": "summarized", "core": dclm.get("Core")}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", action="append", required=True, help="Repeat once per config.")
    p.add_argument(
        "--output-prefix",
        default="gs://marin-us-central1/metadata/data_curation_core_results",
        help="Parent dir holding finals + partial/.",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Don't re-aggregate; just read the existing final and emit "
        "a `<run>_summary.json` sibling. Use this for finals that "
        "are already valid and just need the small summary written.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    reports = []
    for run in args.run_name:
        try:
            if args.summarize_only:
                reports.append(summarize_only(run, args.output_prefix))
            else:
                reports.append(recover_one(run, args.output_prefix))
        except Exception as e:
            logger.error("Recovery failed for %s: %s", run, e, exc_info=True)
            reports.append({"run_name": run, "status": "error", "error": str(e)})

    logger.info("=" * 60)
    logger.info("Recovery summary:")
    for r in reports:
        logger.info("  %s", r)
    ok_statuses = {"recovered", "summarized"}
    failed = [r for r in reports if r.get("status") not in ok_statuses]
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
