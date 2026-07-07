# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repair stale isoFLOP result JSONs from their wandb run history.

Companion to ``audit_results_vs_wandb.py``. The audit finds cells whose result
JSON disagrees with the run's true wandb final; this tool *fixes* the
``JSON_STALE`` ones -- where a post-hoc eval pass scored a non-final checkpoint,
so the plotted point is worse than what the run achieved.

The repair reads the run's **last logged eval step from history** (not the frozen
``summary``) and rewrites the JSON's *entire* ``eval`` block from that one
authoritative snapshot -- every score (``eval/loss``, all ``eval/paloma/*``,
``eval/lima/*``, every ``eval/uncheatable_eval/*`` plus their macro/micro/bpb),
not just the isoFLOP metric -- so all downstream numbers move together and stay
mutually consistent. It stamps ``eval.step`` so the cell stops tripping
``NO_EVAL_STEP`` and future audits can verify the checkpoint.

Safety rails:

- **Only ``JSON_STALE`` cells are touched.** Each target is re-diagnosed with the
  audit logic at run time; cells where wandb is the stale side (``WANDB_STALE``),
  ambiguous, or already clean are skipped. This makes it impossible to overwrite
  a good JSON with a zombie/early-checkpoint value. ``--force`` overrides, for
  explicitly named runs only.
- **Dry-run by default.** Nothing is written without ``--apply``; the default run
  prints a per-key before/after diff.
- **Backups.** Before each overwrite the original object is copied to
  ``<path>.bak-<UTC-stamp>`` in the same bucket (disable with ``--no-backup``).

    # see what would change across every stale cell:
    python experiments/scaling_law_sweeps/repair_results_from_wandb.py

    # actually repair them:
    python experiments/scaling_law_sweeps/repair_results_from_wandb.py --apply

    # repair specific runs only:
    python experiments/scaling_law_sweeps/repair_results_from_wandb.py \
        --runs curation-high_quality_10k-expFM_natural-3e+20-d1536-L16-B256 --apply
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone

from experiments.scaling_law_sweeps.audit_results_vs_wandb import (
    DEFAULT_METRICS,
    DEFAULT_TOL,
    DEFAULT_WANDB_ENTITY,
    DEFAULT_WANDB_PROJECT,
    Diagnosis,
    audit_cell,
    download_result_jsons,
    index_wandb_runs,
    load_json,
    pick_final_generation,
    wandb_final_from_history,
)
from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX,
    DEFAULT_WANDB_GROUP,
)

# JSON eval keys that are timing/bookkeeping, not science -- left untouched.
NON_SCIENCE_EVAL_KEYS = frozenset({"step", "eval/loading_time", "eval/total_time"})


@dataclass
class RepairPlan:
    """What a single cell's repair would change."""

    run_name: str
    json_path: str
    diagnosis: str
    final_step: int | None = None
    changes: dict[str, tuple[float, float]] = field(default_factory=dict)  # key -> (old, new)
    missing_in_history: list[str] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def n_changed(self) -> int:
        return sum(1 for old, new in self.changes.values() if old != new)


def science_keys(eval_dict: dict) -> list[str]:
    """Eval keys worth restoring: numeric loss/bpb metrics, not timing fields."""
    return [
        k
        for k, v in eval_dict.items()
        if isinstance(v, (int, float)) and k not in NON_SCIENCE_EVAL_KEYS and (k.endswith("loss") or k.endswith("bpb"))
    ]


def build_plan(data: dict, json_path: str, run, audit_diagnosis: str, tol: float) -> RepairPlan:
    """Compute the eval-block rewrite for one cell from its wandb history final."""
    plan = RepairPlan(run_name=data.get("plan", {}).get("run_name", ""), json_path=json_path, diagnosis=audit_diagnosis)
    eval_dict = data.get("eval", {})
    keys = science_keys(eval_dict)
    finals, _ = wandb_final_from_history(run, tuple(keys))

    steps = {step for _, step in finals.values()}
    plan.final_step = max(steps) if steps else None
    for k in keys:
        if k in finals:
            new = finals[k][0]
            plan.changes[k] = (eval_dict[k], new)
        else:
            plan.missing_in_history.append(k)
    return plan


def apply_plan(data: dict, plan: RepairPlan, gs_path: str, local_dir: str, backup: bool) -> None:
    """Write the repaired JSON back to GCS, backing up the original first."""
    if backup:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        bak = f"{gs_path}.bak-{stamp}"
        _gcloud("cp", gs_path, bak)
        print(f"        backed up -> {bak}")

    new = dict(data)
    new_eval = dict(data.get("eval", {}))
    for k, (_old, new_val) in plan.changes.items():
        new_eval[k] = new_val
    if plan.final_step is not None:
        new_eval["step"] = plan.final_step
    new["eval"] = new_eval
    run_meta = dict(new.get("run", {}))
    run_meta["repaired_at"] = datetime.now(timezone.utc).isoformat()
    run_meta["repaired_from_wandb_step"] = plan.final_step
    new["run"] = run_meta

    local = os.path.join(local_dir, os.path.basename(gs_path))
    with open(local, "w") as f:
        json.dump(new, f, indent=2)
    _gcloud("cp", local, gs_path)
    print(f"        wrote {plan.n_changed} updated scores -> {gs_path}")


def _gcloud(*args: str) -> None:
    res = subprocess.run(["gcloud", "storage", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gcloud storage {' '.join(args)} failed: {res.stderr.strip()}")


def format_plan(plan: RepairPlan, tol: float) -> str:
    short = plan.run_name.replace("curation-", "").replace("-expFM_natural", "")
    big = sorted(
        ((k, o, n) for k, (o, n) in plan.changes.items() if abs(o - n) > tol),
        key=lambda x: abs(x[1] - x[2]),
        reverse=True,
    )
    head = f"  {short}  <{plan.diagnosis}>  final_step={plan.final_step}  {plan.n_changed} scores change"
    lines = [head]
    for k, o, n in big[:8]:
        lines.append(f"        {k:48s} {o:.4f} -> {n:.4f}  (Δ{abs(o - n):.4f})")
    if len(big) > 8:
        lines.append(f"        ... and {len(big) - 8} more changing > {tol}")
    if plan.missing_in_history:
        lines.append(f"        ! {len(plan.missing_in_history)} key(s) absent from history, kept as-is")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    parser.add_argument("--wandb-entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics the audit uses to diagnose JSON_STALE (first is primary).",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Repair only these run_names (else: every JSON_STALE cell found by the audit).",
    )
    parser.add_argument("--apply", action="store_true", help="Actually write (default: dry-run diff only).")
    parser.add_argument("--no-backup", action="store_true", help="Skip the GCS backup of each original.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Repair even cells not diagnosed JSON_STALE. Only honored with explicit --runs.",
    )
    parser.add_argument("--wandb-api-key", default=os.environ.get("WANDB_API_KEY"))
    args = parser.parse_args()

    metrics = tuple(args.metrics)

    print(f"Listing + downloading result JSONs under {args.results_prefix} ...")
    with tempfile.TemporaryDirectory(prefix="repair_results_") as tmp:
        local_by_gs = download_result_jsons(args.results_prefix, tmp)
        print(f"  found {len(local_by_gs)} result JSONs")

        print(f"Indexing wandb runs in {args.wandb_entity}/{args.wandb_project} group={args.wandb_group} ...")
        wandb_by_name = index_wandb_runs(args.wandb_entity, args.wandb_project, args.wandb_group, args.wandb_api_key)

        # Map run_name -> (gs_path, data); restrict to --runs if given.
        by_run: dict[str, tuple[str, dict]] = {}
        for gs_path, local_path in local_by_gs.items():
            try:
                data = load_json(local_path)
            except (json.JSONDecodeError, OSError):
                continue
            run_name = data.get("plan", {}).get("run_name") or os.path.basename(gs_path)[: -len(".json")]
            by_run[run_name] = (gs_path, data)

        if args.runs:
            targets = [r for r in args.runs if r in by_run]
            for missing in set(args.runs) - set(targets):
                print(f"  WARNING: requested run not found in results: {missing}")
        else:
            targets = sorted(by_run)

        plans: list[RepairPlan] = []
        for run_name in targets:
            gs_path, data = by_run[run_name]
            generations = wandb_by_name.get(run_name, [])
            if not generations:
                plans.append(
                    RepairPlan(run_name=run_name, json_path=gs_path, diagnosis="NO_WANDB", skip_reason="no wandb run")
                )
                continue
            audit = audit_cell(data, gs_path, wandb_by_name, metrics, args.tol)
            forced = args.force and bool(args.runs)
            if audit.diagnosis != Diagnosis.JSON_STALE and not forced:
                # Not a JSON-side problem; never overwrite a good/zombie-conflicted JSON.
                if args.runs:  # only chatter about explicitly requested ones
                    plans.append(
                        RepairPlan(
                            run_name=run_name,
                            json_path=gs_path,
                            diagnosis=audit.diagnosis,
                            skip_reason=f"diagnosis={audit.diagnosis} (use --force to override)",
                        )
                    )
                continue
            run = pick_final_generation(generations)
            plan = build_plan(data, gs_path, run, audit.diagnosis, args.tol)
            plans.append(plan)

        repairable = [p for p in plans if p.skip_reason is None and p.n_changed]
        skipped = [p for p in plans if p.skip_reason is not None]

        print()
        print("=" * 78)
        mode = "APPLY" if args.apply else "DRY-RUN"
        print(f"REPAIR [{mode}]: {len(repairable)} cell(s) to repair | {len(skipped)} skipped")
        print("=" * 78)

        for p in sorted(repairable, key=lambda p: -p.n_changed):
            print(format_plan(p, args.tol))
            if args.apply:
                gs_path, data = by_run[p.run_name]
                apply_plan(data, p, gs_path, tmp, backup=not args.no_backup)

        if skipped:
            print(f"\n### SKIPPED ({len(skipped)})")
            for p in skipped:
                short = p.run_name.replace("curation-", "").replace("-expFM_natural", "")
                print(f"  {short}: {p.skip_reason}")

        print()
        if not args.apply:
            print(f"DRY-RUN: nothing written. Re-run with --apply to repair {len(repairable)} cell(s).")
        else:
            print(f"DONE: repaired {len(repairable)} cell(s).")

    sys.exit(0)


if __name__ == "__main__":
    main()
