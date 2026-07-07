# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit isoFLOP result JSONs against their wandb run finals.

The scaling-law plots are built from the per-cell **result JSONs** under a GCS
results prefix (one `<run_name>.json` per (method, budget, width) cell). Each JSON
embeds an `eval` snapshot taken by a post-training eval pass. wandb separately
records the run's own in-training final metrics. When the two disagree the plot
shows a value that the run never actually produced -- e.g. a stale eval snapshot
taken from a non-final checkpoint, which manifests as a spurious bump in the
isoFLOP curve (a fake "overfit") even though wandb is clean and monotonic.

This tool cross-checks every result JSON against the matching wandb run and flags:

- ``METRIC_MISMATCH``  -- a tracked metric differs by more than ``--tol`` between
  the JSON and the wandb final. This is the one that corrupts the plot.
- ``STEP_MISMATCH``    -- the JSON's ``eval.step`` differs from the wandb final
  ``_step`` (the eval ran on a different checkpoint than the run finished on).
- ``NO_EVAL_STEP``     -- the JSON carries no ``eval.step`` (can't prove which
  checkpoint was evaluated; correlated with stale snapshots).
- ``UNDERTRAINED``     -- the wandb ``_step`` is well short of the planned
  ``train_steps`` (the run never reached its budget).
- ``MULTI_GENERATION`` -- more than one wandb run shares the display name; the
  JSON might have been written by a different generation than the surviving one.
- ``NO_WANDB``         -- no wandb run matches the JSON's ``run_name``.

It is **read-only**: it lists + reads GCS JSONs and queries the wandb API. It never
submits jobs, mutates the registry, or rewrites results. Safe to run any time,
including concurrently with a live sweep.

    python experiments/scaling_law_sweeps/audit_results_vs_wandb.py
    python experiments/scaling_law_sweeps/audit_results_vs_wandb.py --all          # show OK cells too
    python experiments/scaling_law_sweeps/audit_results_vs_wandb.py --out audit.json
    python experiments/scaling_law_sweeps/audit_results_vs_wandb.py \
        --results-prefix gs://.../some_other_results/ --wandb-group some-group

Exit code is non-zero when any cell trips an error-level flag (METRIC_MISMATCH,
STEP_MISMATCH, NO_WANDB), so it can gate a notebook refresh or CI check. Pass
``--strict`` to also fail on warning-level flags.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import pandas as pd
import wandb

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX,
    DEFAULT_WANDB_GROUP,
)

DEFAULT_WANDB_ENTITY = "marin-community"
DEFAULT_WANDB_PROJECT = "marin"

# The metrics that actually drive the plots. macro_loss over the frozen
# Uncheatable-Eval snapshot is the isoFLOP y-axis; eval/loss is the overall
# validation loss. Both must agree with wandb or the plot is wrong.
DEFAULT_METRICS = (
    "eval/uncheatable_eval/macro_loss",
    "eval/uncheatable_eval/micro_loss",
    "eval/loss",
)

# A metric gap below this is just float/eval-noise; above it the plot diverges
# from what the run produced. 0.005 == the sweep's own TREND-outlier threshold.
DEFAULT_TOL = 0.005

# Fraction of planned train_steps the wandb run must reach to not be UNDERTRAINED.
UNDERTRAINED_FRACTION = 0.98


class Flag(StrEnum):
    METRIC_MISMATCH = "METRIC_MISMATCH"
    STEP_MISMATCH = "STEP_MISMATCH"
    NO_EVAL_STEP = "NO_EVAL_STEP"
    UNDERTRAINED = "UNDERTRAINED"
    MULTI_GENERATION = "MULTI_GENERATION"
    NO_WANDB = "NO_WANDB"


class Diagnosis(StrEnum):
    """Which side of a METRIC_MISMATCH is the stale one -- drives the fix."""

    JSON_STALE = "JSON_STALE"  # result JSON evaluated a non-final ckpt; regenerate it
    WANDB_STALE = "WANDB_STALE"  # wandb generation froze early (zombie); JSON is truth
    AMBIGUOUS = "AMBIGUOUS"  # disagreement, but direction unclear -- inspect by hand
    OK = "OK"


# Flags that mean the plot/data is actually wrong (vs. merely suspicious).
ERROR_FLAGS = frozenset({Flag.METRIC_MISMATCH, Flag.STEP_MISMATCH, Flag.NO_WANDB})


@dataclass
class CellAudit:
    """Audit outcome for one result JSON vs its wandb run."""

    run_name: str
    json_path: str
    flags: list[str] = field(default_factory=list)
    # per-metric (json_value, wandb_value, abs_delta); only metrics present somewhere
    metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    json_eval_step: int | None = None
    wandb_step: int | None = None
    wandb_eval_step: int | None = None
    planned_train_steps: int | None = None
    wandb_state: str | None = None
    wandb_url: str | None = None
    n_generations: int = 0
    completed_at: str | None = None
    diagnosis: str = Diagnosis.OK
    notes: list[str] = field(default_factory=list)

    @property
    def is_error(self) -> bool:
        return any(f in ERROR_FLAGS for f in self.flags)

    @property
    def worst_metric_delta(self) -> float:
        deltas = [m["delta"] for m in self.metrics.values() if m.get("delta") is not None]
        return max(deltas) if deltas else 0.0


def download_result_jsons(results_prefix: str, dest_dir: str) -> dict[str, str]:
    """Bulk-copy every ``*.json`` under the prefix to dest_dir.

    Uses ``gcloud storage`` (not gcsfs) because the gcloud CLI carries its own
    auth/TLS trust, which gcsfs's aiohttp client does not in every environment.
    Returns {gs_path: local_path}.
    """
    glob = results_prefix.rstrip("/") + "/*.json"
    listing = subprocess.run(["gcloud", "storage", "ls", glob], capture_output=True, text=True)
    if listing.returncode != 0:
        raise RuntimeError(f"gcloud storage ls failed: {listing.stderr.strip()}")
    gs_paths = [ln.strip() for ln in listing.stdout.splitlines() if ln.strip().endswith(".json")]
    if not gs_paths:
        return {}
    # One batched copy of all matching objects into the temp dir.
    cp = subprocess.run(["gcloud", "storage", "cp", glob, dest_dir], capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"gcloud storage cp failed: {cp.stderr.strip()}")
    return {gs: os.path.join(dest_dir, os.path.basename(gs)) for gs in gs_paths}


def load_json(local_path: str) -> dict:
    with open(local_path) as f:
        return json.load(f)


def index_wandb_runs(entity: str, project: str, group: str, api_key: str | None) -> dict[str, list]:
    """Return {display_name: [runs...]} for every run in the group (one API sweep)."""
    api = wandb.Api(api_key=api_key) if api_key else wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300)
    by_name: dict[str, list] = {}
    for r in runs:
        by_name.setdefault(r.name, []).append(r)
    return by_name


def pick_final_generation(runs: list):
    """Of N generations sharing a name, the one trained furthest (max history step)."""
    return max(runs, key=lambda r: (getattr(r, "lastHistoryStep", None) or r.summary.get("_step") or -1))


def wandb_final_from_history(run, metrics: tuple[str, ...]) -> tuple[dict[str, tuple[float, int]], int | None]:
    """Final logged value (and its step) for each metric, read from run *history*.

    wandb's ``run.summary`` is the last value *flushed*, which a crashed or
    preempted run freezes at the pre-crash step -- so it lies about the final
    metric. The history time-series, in contrast, runs through to the true last
    step (resumed generations append to it). We therefore read the value at the
    maximum step where each metric was logged. Eval metrics are co-logged, so one
    combined ``history`` call usually covers them; any metric missing from that
    call (logged on a different cadence) is back-filled with its own query.

    Returns ({metric: (value, step)}, last_history_step).
    """
    last_step = getattr(run, "lastHistoryStep", None)
    values: dict[str, tuple[float, int]] = {}

    def _ingest(df: pd.DataFrame, wanted: tuple[str, ...]) -> None:
        if df is None or len(df) == 0 or "_step" not in df.columns:
            return
        for m in wanted:
            if m not in df.columns:
                continue
            sub = df.dropna(subset=[m])
            if len(sub):
                row = sub.loc[sub["_step"].idxmax()]
                values[m] = (float(row[m]), int(row["_step"]))

    try:
        _ingest(run.history(keys=list(metrics), samples=10000), metrics)
    except (ValueError, KeyError):
        pass
    for m in metrics:
        if m not in values:
            try:
                _ingest(run.history(keys=[m], samples=10000), (m,))
            except (ValueError, KeyError):
                continue
    return values, last_step


def diagnose(audit: CellAudit, ev: dict, wandb_finals: dict[str, float], primary_metric: str) -> str:
    """Decide which side of a metric mismatch is stale, to direct the fix.

    JSON_STALE: the result JSON is worse than wandb *and* wandb reached its
    planned step -- the post-hoc eval ran on a non-final checkpoint, so the JSON
    should be regenerated. WANDB_STALE: wandb's last *history* step is short of
    the planned step (the run truly never finished) while the JSON's eval ran
    at/after that step -- the JSON is the trustworthy value. Otherwise AMBIGUOUS.

    Note: ``wandb_finals`` are read from history (true final), not summary, so a
    merely-crashed run whose history reached the planned step is NOT WANDB_STALE.
    """
    jv = ev.get(primary_metric)
    wv = wandb_finals.get(primary_metric)
    if jv is None or wv is None:
        return Diagnosis.AMBIGUOUS
    planned = audit.planned_train_steps
    wandb_reached = planned is None or (audit.wandb_step or 0) >= UNDERTRAINED_FRACTION * planned
    json_at_final = (
        audit.json_eval_step is not None
        and planned is not None
        and (audit.json_eval_step >= UNDERTRAINED_FRACTION * planned)
    )
    if jv > wv and wandb_reached:
        return Diagnosis.JSON_STALE
    if wv > jv and not wandb_reached and (json_at_final or audit.json_eval_step is None):
        return Diagnosis.WANDB_STALE
    return Diagnosis.AMBIGUOUS


def audit_cell(
    data: dict,
    json_path: str,
    wandb_by_name: dict[str, list],
    metrics: tuple[str, ...],
    tol: float,
) -> CellAudit:
    """Cross-check one result JSON against its wandb run."""
    plan = data.get("plan", {})
    ev = data.get("eval", {})
    run_name = plan.get("run_name") or os.path.basename(json_path)[: -len(".json")]

    audit = CellAudit(
        run_name=run_name,
        json_path=json_path,
        json_eval_step=ev.get("step"),
        planned_train_steps=plan.get("train_steps"),
        completed_at=data.get("run", {}).get("completed_at"),
    )

    if audit.json_eval_step is None:
        audit.flags.append(Flag.NO_EVAL_STEP)

    generations = wandb_by_name.get(run_name, [])
    audit.n_generations = len(generations)
    if not generations:
        audit.flags.append(Flag.NO_WANDB)
        for m in metrics:
            jv = ev.get(m)
            if jv is not None:
                audit.metrics[m] = {"json": jv, "wandb": None, "delta": None}
        return audit

    if len(generations) > 1:
        audit.flags.append(Flag.MULTI_GENERATION)
        audit.notes.append(f"{len(generations)} wandb generations share this name")

    run = pick_final_generation(generations)
    audit.wandb_state = run.state
    # Build the URL by hand with a literal '+' (wandb's run.url percent-encodes it,
    # which breaks these run ids that embed budgets like 3e+20).
    audit.wandb_url = f"https://wandb.ai/{run.entity}/{run.project}/runs/{run.id}"

    # Read finals from history, NOT summary -- summary is frozen at the pre-crash
    # step for crashed/preempted runs and misreports the final metric.
    finals, last_step = wandb_final_from_history(run, metrics)
    audit.wandb_step = last_step
    final_values = {m: v for m, (v, _) in finals.items()}
    # The step the primary metric was actually evaluated at (its last eval point).
    audit.wandb_eval_step = finals.get(metrics[0], (None, None))[1]

    mismatched = False
    for m in metrics:
        jv = ev.get(m)
        wv = final_values.get(m)
        delta = abs(jv - wv) if (jv is not None and wv is not None) else None
        audit.metrics[m] = {"json": jv, "wandb": wv, "delta": delta}
        if delta is not None and delta > tol:
            mismatched = True
    if mismatched:
        audit.flags.append(Flag.METRIC_MISMATCH)
        audit.diagnosis = diagnose(audit, ev, final_values, metrics[0])

    if audit.json_eval_step is not None and audit.wandb_eval_step is not None:
        if audit.json_eval_step != audit.wandb_eval_step:
            audit.flags.append(Flag.STEP_MISMATCH)
            audit.notes.append(f"json eval.step={audit.json_eval_step} vs wandb eval step={audit.wandb_eval_step}")

    if audit.planned_train_steps and audit.wandb_step is not None:
        if audit.wandb_step < UNDERTRAINED_FRACTION * audit.planned_train_steps:
            audit.flags.append(Flag.UNDERTRAINED)
            audit.notes.append(f"wandb final history step={audit.wandb_step} < planned {audit.planned_train_steps}")

    return audit


def format_row(a: CellAudit, metrics: tuple[str, ...]) -> str:
    short = a.run_name.replace("curation-", "").replace("-expFM_natural", "")
    flagstr = ",".join(a.flags) if a.flags else "OK"
    # show the plot-critical metric delta inline
    primary = metrics[0]
    pm = a.metrics.get(primary, {})
    jv, wv, dl = pm.get("json"), pm.get("wandb"), pm.get("delta")
    metricstr = ""
    if jv is not None or wv is not None:
        j = f"{jv:.4f}" if isinstance(jv, (int, float)) else "n/a"
        w = f"{wv:.4f}" if isinstance(wv, (int, float)) else "n/a"
        d = f"Δ{dl:.4f}" if isinstance(dl, (int, float)) else ""
        metricstr = f"  json={j} wandb={w} {d}"
    return f"  [{flagstr}] {short}{metricstr}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX, help="GCS prefix of result JSONs.")
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metric keys (under the JSON 'eval' dict / wandb summary) to compare.",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Max allowed |json-wandb| metric gap.")
    parser.add_argument("--all", action="store_true", help="Print OK cells too, not just flagged ones.")
    parser.add_argument("--out", help="Write the full machine-readable audit (JSON) here.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on warning-level flags too.")
    parser.add_argument(
        "--wandb-api-key",
        default=os.environ.get("WANDB_API_KEY"),
        help="Defaults to $WANDB_API_KEY.",
    )
    args = parser.parse_args()

    metrics = tuple(args.metrics)

    print(f"Listing + downloading result JSONs under {args.results_prefix} ...")
    with tempfile.TemporaryDirectory(prefix="audit_results_") as tmp:
        local_by_gs = download_result_jsons(args.results_prefix, tmp)
        print(f"  found {len(local_by_gs)} result JSONs")

        print(f"Indexing wandb runs in {args.wandb_entity}/{args.wandb_project} group={args.wandb_group} ...")
        wandb_by_name = index_wandb_runs(args.wandb_entity, args.wandb_project, args.wandb_group, args.wandb_api_key)
        print(f"  indexed {sum(len(v) for v in wandb_by_name.values())} runs over {len(wandb_by_name)} names")

        audits: list[CellAudit] = []
        for gs_path, local_path in local_by_gs.items():
            try:
                data = load_json(local_path)
            except (json.JSONDecodeError, OSError) as e:
                a = CellAudit(run_name=os.path.basename(gs_path)[: -len(".json")], json_path=gs_path)
                a.notes.append(f"failed to read/parse: {e}")
                a.flags.append(Flag.NO_WANDB)  # treat unreadable as an error-level flag
                audits.append(a)
                continue
            audits.append(audit_cell(data, gs_path, wandb_by_name, metrics, args.tol))

    errors = [a for a in audits if a.is_error]
    warns = [a for a in audits if a.flags and not a.is_error]
    clean = [a for a in audits if not a.flags]

    # ---- report ----
    errors.sort(key=lambda a: a.worst_metric_delta, reverse=True)
    warns.sort(key=lambda a: a.run_name)

    print()
    print("=" * 78)
    print(f"AUDIT: {len(audits)} cells | {len(errors)} ERROR | {len(warns)} WARN | {len(clean)} OK")
    print(f"metrics={list(metrics)} tol={args.tol}")
    print("=" * 78)

    if errors:
        print(f"\n### ERRORS ({len(errors)}) -- plot/data is wrong here, sorted by worst metric gap")
        for a in errors:
            diag = f"  <{a.diagnosis}>" if a.diagnosis != Diagnosis.OK else ""
            print(format_row(a, metrics) + diag)
            if a.wandb_url:
                print(f"        {a.wandb_url}")
            for n in a.notes:
                print(f"        - {n}")

    if warns:
        print(f"\n### WARNINGS ({len(warns)})")
        for a in warns:
            print(format_row(a, metrics))
            if a.wandb_url:
                print(f"        {a.wandb_url}")
            for n in a.notes:
                print(f"        - {n}")

    if args.all and clean:
        print(f"\n### OK ({len(clean)})")
        for a in sorted(clean, key=lambda a: a.run_name):
            print(format_row(a, metrics))

    if args.out:
        with open(args.out, "w") as f:
            json.dump([asdict(a) for a in audits], f, indent=2)
        print(f"\nWrote full audit to {args.out}")

    print()
    if errors:
        print(f"RESULT: FAIL -- {len(errors)} cell(s) need regeneration (result JSON out of sync with wandb).")
    elif warns and args.strict:
        print(f"RESULT: FAIL (strict) -- {len(warns)} warning(s).")
    else:
        print("RESULT: PASS -- every result JSON agrees with its wandb final within tolerance.")

    sys.exit(1 if (errors or (warns and args.strict)) else 0)


if __name__ == "__main__":
    main()
