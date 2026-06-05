# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate per-task partial JSONs into a final CORE result.

Polls a list of run_names; for each:
  - lists `gs://.../partial/{run_name}/*.json`
  - if all 22 expected aliases present AND no final JSON exists, merge them
    and write `gs://.../{run_name}.json` with the Core score.

This runs CPU-only (no JAX/vLLM); cheap to loop. Designed to be a long-running
sidecar while Levanter and vLLM children race to fill partials.

Usage (single shot):
    python -m experiments.scaling_law_sweeps.dclm_core.aggregate_partials --run-name <name>

Usage (loop over a manifest of (run_name, output_json_path) pairs):
    python -m experiments.scaling_law_sweeps.dclm_core.aggregate_partials \\
        --watch --interval 120 --manifest /tmp/aggregator_manifest.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experiments.scaling_law_sweeps.dclm_core.centering import compute_core
from experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval import (
    _json_default,
    _merge_lm_eval_outputs,
    _resolve_partial_dir,
    build_task_configs,
    extract_dclm_results,
)
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP

# --- Local-friendly GCS helpers (use gcloud CLI; works on laptops where gcsfs SSL is finicky) ---


def _path_exists(path: str) -> bool:
    if path.startswith("gs://"):
        return subprocess.run(["gcloud", "storage", "ls", path], capture_output=True).returncode == 0
    return Path(path).exists()


def _read_partial(path: str) -> dict | None:
    if not _path_exists(path):
        return None
    if path.startswith("gs://"):
        result = subprocess.run(["gcloud", "storage", "cat", path], capture_output=True)
        if result.returncode != 0:
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.warning("Bad JSON in %s: %s", path, e)
            return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def _write_final(path: str, data: dict) -> None:
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(data, tf, indent=2, default=_json_default)
            tmp_path = tf.name
        try:
            subprocess.run(["gcloud", "storage", "cp", tmp_path, path], check=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)


logger = logging.getLogger(__name__)


def _expected_aliases() -> list[str]:
    return [f"{e.dclm}_{e.num_fewshot}shot" for e in CORE_TASK_MAP]


def try_aggregate(run_name: str, output_json: str, *, force: bool = False) -> bool:
    """Aggregate if all 22 partials are present. Returns True if aggregation occurred."""
    partial_dir = _resolve_partial_dir(output_json, run_name)
    if _path_exists(output_json) and not force:
        return False  # already aggregated

    expected = _expected_aliases()
    partials: dict[str, dict] = {}
    missing: list[str] = []
    for alias in expected:
        ppath = f"{partial_dir.rstrip('/')}/{alias}.json"
        p = _read_partial(ppath)
        if p is None:
            missing.append(alias)
        else:
            partials[alias] = p

    if missing:
        logger.info(
            "[%s] %d/%d partials present; missing: %s",
            run_name,
            len(partials),
            len(expected),
            ", ".join(missing[:3]) + ("..." if len(missing) > 3 else ""),
        )
        return False

    logger.info("[%s] all 22 partials present — aggregating", run_name)
    _tasks, alias_to_entry = build_task_configs(None)
    combined_lm_eval_results = _merge_lm_eval_outputs(list(partials.values()))
    raw, extraction_log = extract_dclm_results(combined_lm_eval_results, alias_to_entry)
    dclm_aggregation = compute_core(raw)
    logger.info(
        "[%s] Core=%s Core_v2=%s missing_for_core=%s",
        run_name,
        dclm_aggregation.get("Core"),
        dclm_aggregation.get("Core_v2"),
        dclm_aggregation.get("missing_tasks_for_core"),
    )

    output = {
        "checkpoint": None,  # not known at aggregation time
        "run_name": run_name,
        "partial_dir": partial_dir,
        "extraction_log": extraction_log,
        "dclm": dclm_aggregation,
        "lm_eval_raw": combined_lm_eval_results,
        "aggregated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _open_for_write(output_json) as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("[%s] wrote final JSON to %s", run_name, output_json)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest", default=None, help="Path to a manifest file: lines of '<run_name>\\t<output_json_path>'."
    )
    p.add_argument(
        "--run-name", default=None, help="Single-shot mode: aggregate ONLY this run_name. Requires --output-json."
    )
    p.add_argument("--output-json", default=None, help="For single-shot mode.")
    p.add_argument("--watch", action="store_true", help="Loop forever, polling every --interval seconds.")
    p.add_argument("--interval", type=int, default=120, help="Poll interval in seconds (default 120).")
    p.add_argument("--force", action="store_true", help="Re-aggregate even if final JSON exists.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    if args.run_name:
        if not args.output_json:
            p.error("--run-name requires --output-json")
        manifest = [(args.run_name, args.output_json)]
    elif args.manifest:
        manifest = []
        with open(args.manifest) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    logger.warning("Skipping malformed manifest line: %s", line)
                    continue
                manifest.append((parts[0], parts[1]))
    else:
        p.error("Provide either --run-name + --output-json or --manifest")

    logger.info("Manifest has %d entries", len(manifest))

    while True:
        any_aggregated = False
        for run_name, output_json in manifest:
            try:
                if try_aggregate(run_name, output_json, force=args.force):
                    any_aggregated = True
            except Exception as e:
                logger.exception("[%s] aggregation failed: %s", run_name, e)
        if not args.watch:
            sys.exit(0)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
