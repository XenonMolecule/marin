# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap CORE scores from per-example sample logs.

Single-run script: reads one run's 22 per-task partial JSONs (or a final
aggregated JSON), resamples examples within each task B times with
replacement, applies the same centering+aggregation as `centering.compute_core`
on each resampled raw_results dict, and reports mean/stdev of the resulting
Core_v2 distribution.

The per-task metric is discovered from the partial's own `results[alias]`
dict (the keys lm-eval-harness writes, e.g. "acc_norm,none"). This stays
in lockstep with whatever metric `extract_dclm_results` would pick during
the deterministic aggregation: if you swap acc -> acc_norm in
task_mapping.CORE_TASK_MAP, the partial's results dict reflects the swap
automatically and the bootstrap follows.

CLI override available for ad-hoc tweaks without editing task_mapping:
    --metric-override piqa=acc_norm --metric-override boolq=acc

Designed to run in-region on a GCS-colocated worker (us-central1 for the
data_curation_core_results bucket) to avoid sample-level egress. The squad
partial alone is ~21 MB; pulling 22 partials across all runs would be
nontrivial otherwise.

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.bootstrap_core \\
        --run-input gs://marin-us-central1/metadata/data_curation_core_results/partial/<run_name>/ \\
        --output-json gs://marin-us-central1/metadata/data_curation_core_bootstrap/<run_name>_bootstrap.json \\
        --n-bootstrap 1000 \\
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import gcsfs
import numpy as np

from experiments.scaling_law_sweeps.dclm_core.centering import compute_core
from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP, TaskMapEntry

# --- Light-weight GCS helpers (no levanter/haliax dependency) ---

_GCS_FS: gcsfs.GCSFileSystem | None = None


def _fs() -> gcsfs.GCSFileSystem:
    global _GCS_FS
    if _GCS_FS is None:
        _GCS_FS = gcsfs.GCSFileSystem()
    return _GCS_FS


def _path_exists(path: str) -> bool:
    if path.startswith("gs://"):
        return _fs().exists(path)
    return Path(path).exists()


def _read_partial(path: str) -> dict | None:
    """Try to load a JSON partial. Returns None if missing or unreadable."""
    try:
        if path.startswith("gs://"):
            with _fs().open(path, "r") as f:
                return json.load(f)
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Could not read %s: %s", path, e)
        return None


def _open_for_write(path: str):
    if path.startswith("gs://"):
        return _fs().open(path, "w")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return open(path, "w")


def _json_default(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskSamples:
    """Per-example scores for one CORE task."""

    dclm: str
    alias: str
    metric: str  # the metric key used (e.g. "acc_norm", "exact_match")
    scores: np.ndarray  # shape (N,), float — one score per example


def _expected_aliases() -> list[tuple[str, TaskMapEntry]]:
    return [(f"{e.dclm}_{e.num_fewshot}shot", e) for e in CORE_TASK_MAP]


def _discover_metric_from_results(
    results_for_alias: dict,
    fallback: str,
) -> str:
    """Pick the metric key the deterministic path would have used.

    lm-eval writes keys like "acc,none", "acc_norm,none", "acc_stderr,none".
    We strip the ",none" suffix and drop the *_stderr siblings. If the fallback
    metric is among the candidates we prefer it; otherwise return the first.
    """
    candidates = {
        k.split(",")[0]
        for k in results_for_alias.keys()
        if not k.split(",")[0].endswith("_stderr") and not k.startswith("alias")
    }
    if fallback in candidates:
        return fallback
    if not candidates:
        raise KeyError(f"No metric candidates in results dict (keys: {list(results_for_alias)})")
    # Stable, deterministic pick: alphabetical.
    return sorted(candidates)[0]


def _extract_sample_score(sample: dict, metric: str) -> float:
    """Get one example's score, looking at top-level then `metrics` sub-dict."""
    if metric in sample:
        return float(sample[metric])
    metrics = sample.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        return float(metrics[metric])
    raise KeyError(f"Metric {metric!r} not in sample (top-level keys: {list(sample)[:10]}...)")


def _load_run_samples(
    run_input: str,
    metric_overrides: dict[str, str],
) -> dict[str, TaskSamples]:
    """Load per-task per-example scores from a partial dir or a final JSON.

    Returns {dclm_task_name: TaskSamples}, with one entry per task present
    in the input. Missing tasks are simply absent from the returned dict —
    downstream bootstrap will pass an incomplete raw_results to compute_core
    which already handles the "missing tasks" case.
    """
    by_task: dict[str, TaskSamples] = {}

    # Treat as partial dir if it ends with / or doesn't look like a .json.
    is_partial_dir = run_input.endswith("/") or not run_input.endswith(".json")

    if is_partial_dir:
        partial_dir = run_input.rstrip("/")
        for alias, entry in _expected_aliases():
            path = f"{partial_dir}/{alias}.json"
            partial = _read_partial(path)
            if partial is None:
                logger.info("[%s] missing partial at %s", entry.dclm, path)
                continue
            ts = _samples_from_lm_eval_dict(partial, alias, entry, metric_overrides)
            if ts is not None:
                by_task[entry.dclm] = ts
    else:
        # Final aggregated JSON: lm_eval_raw embeds the merged samples/results dicts.
        final = _read_partial(run_input)
        if final is None:
            raise FileNotFoundError(run_input)
        lm_eval_raw = final.get("lm_eval_raw")
        if lm_eval_raw is None:
            raise ValueError(f"{run_input} has no `lm_eval_raw` block")
        for alias, entry in _expected_aliases():
            ts = _samples_from_lm_eval_dict(lm_eval_raw, alias, entry, metric_overrides)
            if ts is not None:
                by_task[entry.dclm] = ts

    return by_task


def _samples_from_lm_eval_dict(
    lm_eval_dict: dict,
    alias: str,
    entry: TaskMapEntry,
    metric_overrides: dict[str, str],
) -> TaskSamples | None:
    """Pull per-example scores for one task out of an lm-eval-style dict.

    The dict can be a single-task partial (results+samples keyed by alias)
    or a merged lm_eval_raw (same shape, but with all 22 aliases).
    """
    results = lm_eval_dict.get("results", {}).get(alias)
    samples = lm_eval_dict.get("samples", {}).get(alias)
    if results is None or samples is None:
        return None

    metric = metric_overrides.get(entry.dclm)
    if metric is None:
        metric = _discover_metric_from_results(results, fallback=entry.metric)

    scores = np.fromiter(
        (_extract_sample_score(s, metric) for s in samples),
        dtype=np.float64,
        count=len(samples),
    )
    return TaskSamples(dclm=entry.dclm, alias=alias, metric=metric, scores=scores)


def bootstrap(
    task_samples: dict[str, TaskSamples],
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run B bootstrap iters.

    Returns:
        core_dist: shape (B,) — Core_v2 per iter, NaN when missing tasks
        extended_dist: shape (B,) — Extended_v2 per iter (uses all 22 meta
            tasks; NaN if any are missing)
        per_task_means: {dclm_name: shape-(B,) bootstrap mean of task acc}
    """
    rng = np.random.default_rng(seed)
    per_task_means: dict[str, np.ndarray] = {name: np.empty(n_bootstrap, dtype=np.float64) for name in task_samples}
    core_dist = np.empty(n_bootstrap, dtype=np.float64)
    extended_dist = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        raw: dict[str, float] = {}
        for name, ts in task_samples.items():
            n = ts.scores.shape[0]
            idx = rng.integers(0, n, size=n)
            mean = float(ts.scores[idx].mean())
            raw[name] = mean
            per_task_means[name][b] = mean
        agg = compute_core(raw)
        core_dist[b] = _as_float(agg.get("Core_v2"))
        extended_dist[b] = _as_float(agg.get("Extended_v2"))

    return core_dist, extended_dist, per_task_means


def _as_float(value) -> float:
    return value if isinstance(value, (int, float)) else float("nan")


def _parse_overrides(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--metric-override expects TASK=METRIC, got {item!r}")
        task, metric = item.split("=", 1)
        out[task.strip()] = metric.strip()
    return out


def _summarize(dist: np.ndarray) -> dict:
    finite = dist[np.isfinite(dist)]
    if finite.size == 0:
        return {"mean": None, "stdev": None, "n_valid": 0}
    return {
        "mean": float(finite.mean()),
        "stdev": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_valid": int(finite.size),
    }


def run_one(
    run_input: str,
    output_json: str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    metric_overrides: dict[str, str] | None = None,
    run_name: str | None = None,
    force: bool = False,
) -> dict | None:
    """Bootstrap one run; write the result JSON. Returns the summary dict or None if skipped."""
    if metric_overrides is None:
        metric_overrides = {}

    if _path_exists(output_json) and not force:
        logger.info("Output %s already exists; pass force=True to overwrite. Skipping.", output_json)
        return None

    rn = run_name or _derive_run_name(run_input)

    logger.info("[%s] loading samples from %s", rn, run_input)
    task_samples = _load_run_samples(run_input, metric_overrides)
    if not task_samples:
        raise RuntimeError(f"No tasks loaded from {run_input}")
    logger.info("[%s] loaded %d/22 tasks", rn, len(task_samples))

    missing = [e.dclm for _, e in _expected_aliases() if e.dclm not in task_samples]
    if missing:
        logger.warning("[%s] missing tasks (Core_v2 will be NaN): %s", rn, missing)

    t0 = time.time()
    core_dist, extended_dist, per_task_means = bootstrap(task_samples, n_bootstrap=n_bootstrap, seed=seed)
    logger.info("[%s] bootstrap of %d iters took %.1fs", rn, n_bootstrap, time.time() - t0)

    core_summary = _summarize(core_dist)
    extended_summary = _summarize(extended_dist)
    logger.info("[%s] Core_v2: %s ± %s (n=%d)", rn, core_summary["mean"], core_summary["stdev"], core_summary["n_valid"])

    output = {
        "run_name": rn,
        "run_input": run_input,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "metric_overrides": metric_overrides,
        "n_tasks_present": len(task_samples),
        "missing_tasks": missing,
        "per_task_metric": {ts.dclm: ts.metric for ts in task_samples.values()},
        "per_task_n_samples": {ts.dclm: int(ts.scores.shape[0]) for ts in task_samples.values()},
        "per_task_observed_mean": {ts.dclm: float(ts.scores.mean()) for ts in task_samples.values()},
        "per_task_bootstrap_stdev": {name: float(arr.std(ddof=1)) for name, arr in per_task_means.items()},
        "core_mean": core_summary["mean"],
        "core_stdev": core_summary["stdev"],
        "core_n_valid": core_summary["n_valid"],
        "extended_mean": extended_summary["mean"],
        "extended_stdev": extended_summary["stdev"],
        "extended_n_valid": extended_summary["n_valid"],
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _open_for_write(output_json) as f:
        json.dump(output, f, indent=2, default=_json_default)
    logger.info("[%s] wrote bootstrap result to %s", rn, output_json)
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run-input",
        required=True,
        help="Partial dir (.../<run_name>/) or final JSON (.../<run_name>.json), " "either gs:// or local.",
    )
    p.add_argument("--output-json", required=True, help="Where to write the bootstrap result JSON (gs:// or local).")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--metric-override",
        action="append",
        default=[],
        help="Per-task metric override, e.g. piqa=acc_norm. Repeatable.",
    )
    p.add_argument("--run-name", default=None, help="Defaults to the basename of --run-input.")
    p.add_argument("--force", action="store_true", help="Overwrite output JSON if it already exists.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_one(
        run_input=args.run_input,
        output_json=args.output_json,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        metric_overrides=_parse_overrides(args.metric_override),
        run_name=args.run_name,
        force=args.force,
    )


def _derive_run_name(run_input: str) -> str:
    s = run_input.rstrip("/")
    if s.endswith(".json"):
        return s.rsplit("/", 1)[-1][: -len(".json")]
    return s.rsplit("/", 1)[-1]


if __name__ == "__main__":
    main()
