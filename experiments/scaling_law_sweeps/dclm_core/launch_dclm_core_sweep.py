# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Enumerate the curation sweep, plan DCLM CORE eval runs, launch them.

Reads run summaries from the Fixed-Model sweep
(`data_curation_fixed_model_results/`, experiment_tag = `expFM_natural`),
filters to the user's priority methods, sorts by budget descending, and for
each checkpoint either prints the plan (`--dry-run`), prints the bash command
(`--emit-commands`), or submits an Iris child job (`--launch`).

## Handoff one-liner (run from any laptop with iris auth)

    iris --cluster marin job run \\
        -e WANDB_API_KEY -e HF_TOKEN \\
        -- python -m experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep \\
               --pilot --launch

That single command:
  1. Bundles the local workspace and ships it to the iris controller.
  2. Submits a small parent job that runs THIS script.
  3. The parent enumerates pilot plans, then submits ONE child per checkpoint.
  4. Each child reserves a 4-chip TPU slice (v5p-8 / v4-8 / v6e-4, whichever
     has capacity in the checkpoint's region), runs `run_dclm_core_eval.py`,
     and writes `gs://.../data_curation_core_results/{run_name}.json`.
  5. Skip-if-done: any run whose output already exists is skipped.

## Pilot scope (per the approved plan)

Methods in priority order (top-budget anchor first when --pilot):
  1. high_quality_3000
  2. dclm
  3. nemotron_full_bos_fixed
  4. resiliparse_dedup

`--priority-methods` expands to all budgets for those four methods.
`--all-fm` covers the entire Fixed-Model sweep.

## Per-checkpoint plan

  - run_name (from summary's plan.run_name)
  - region (from summary's run.region — child is HARD-pinned here to avoid
    cross-region reads, per CLAUDE.md)
  - hf_checkpoint_gcs (run.output_path + /hf/step-<final>/)
  - output_json_gcs (parallel sibling prefix data_curation_core_results/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass

from iris.client.client import IrisClient
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


PRIORITY_BAND_MAP = {
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}

# Single-host 4-chip TPU variants. Fits any model up to ~7B in bf16. We do
# NOT include v6e-8 (8 chips) because mixing chip counts inside a
# device_variant_constraint trips iris's per-VM committed_tpu accounting and
# causes /dev/vfio collisions. See launch_curation_sweep.py for the full note.
DEFAULT_TPU_VARIANTS: tuple[str, ...] = ("v5p-8", "v4-8", "v6e-4")

# --- Configuration ---------------------------------------------------------

PRIORITY_METHODS: tuple[str, ...] = (
    "high_quality_3000",
    "dclm",
    "nemotron_full_bos_fixed",
    "resiliparse_dedup",
)

FM_SUMMARIES_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
CORE_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_core_results/"

# WARC sweep is Phase 4b territory; configured here for completeness but not
# enabled by default.
WARC_SUMMARIES_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"


# --- Data classes ----------------------------------------------------------


@dataclass(frozen=True)
class CheckpointPlan:
    run_name: str
    method: str
    experiment_tag: str
    budget_flops: float
    region: str
    output_path: str  # parent dir on GCS (no /hf/step-N appended)
    summary_path: str  # the JSON we read it from
    output_json_path: str  # where the CORE result will land

    def hf_dir(self) -> str:
        """The /hf/ directory; the final step subdir is resolved at run time."""
        return f"{self.output_path.rstrip('/')}/hf/"


# --- GCS helpers (gsutil/gcloud-based for portability) ---------------------


def _gcs_ls(prefix: str) -> list[str]:
    """List immediate children of a GCS prefix. Falls back to empty on error."""
    cmd = ["gcloud", "storage", "ls", prefix]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _gcs_cat(path: str) -> bytes:
    cmd = ["gcloud", "storage", "cat", path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"gcs cat failed: {path}: {result.stderr.decode(errors='replace')}")
    return result.stdout


def _gcs_exists(path: str) -> bool:
    cmd = ["gcloud", "storage", "ls", path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _resolve_final_step(hf_dir: str) -> str | None:
    """Find the largest step-N under hf_dir. Returns the full GCS dir path."""
    entries = _gcs_ls(hf_dir)
    steps: list[tuple[int, str]] = []
    for entry in entries:
        e = entry.rstrip("/")
        leaf = e.split("/")[-1]
        if leaf.startswith("step-"):
            try:
                n = int(leaf[len("step-") :])
            except ValueError:
                continue
            steps.append((n, e + "/"))
    if not steps:
        return None
    steps.sort()
    return steps[-1][1]


# --- Enumeration ----------------------------------------------------------


def enumerate_plans(
    summaries_prefix: str,
    method_filter: set[str] | None,
    *,
    budget_desc: bool = True,
) -> list[CheckpointPlan]:
    """List CheckpointPlans from a summaries prefix, optionally filtered by method."""
    plans: list[CheckpointPlan] = []
    summary_paths = [p for p in _gcs_ls(summaries_prefix) if p.endswith(".json")]
    logger.info("Found %d summary JSONs under %s", len(summary_paths), summaries_prefix)

    # Filename-level pre-filter: summaries are named curation-{method}-{tag}-{budget}-...
    # Skip cating JSONs whose filename method isn't in the filter.
    if method_filter is not None:

        def _filename_method(path: str) -> str | None:
            leaf = path.rsplit("/", 1)[-1]
            if not leaf.startswith("curation-"):
                return None
            # Strip the leading "curation-" and find the longest method-name match
            stem = leaf[len("curation-") :]
            for m in sorted(method_filter, key=len, reverse=True):
                if stem.startswith(m + "-"):
                    return m
            return None

        kept = [p for p in summary_paths if _filename_method(p) is not None]
        logger.info(
            "Filename-level filter kept %d of %d (methods: %s)", len(kept), len(summary_paths), sorted(method_filter)
        )
        summary_paths = kept

    for sp in summary_paths:
        try:
            doc = json.loads(_gcs_cat(sp))
        except Exception as e:
            logger.warning("Skipping unreadable summary %s: %s", sp, e)
            continue
        plan_section = doc.get("plan", {})
        method = plan_section.get("method_name") or doc.get("method", {}).get("name")
        if method is None:
            continue
        if method_filter is not None and method not in method_filter:
            continue
        run_section = doc.get("run", {})
        run_name = plan_section.get("run_name") or plan_section.get("run_name_core")
        if run_name is None:
            continue
        plans.append(
            CheckpointPlan(
                run_name=run_name,
                method=method,
                experiment_tag=plan_section.get("experiment_tag", ""),
                budget_flops=float(plan_section.get("budget_flops", 0.0)),
                region=run_section.get("region", ""),
                output_path=run_section.get("output_path", ""),
                summary_path=sp,
                output_json_path=f"{CORE_RESULTS_PREFIX}{run_name}.json",
            )
        )
    if budget_desc:
        # Within each method, top budget first. Method order = priority list.
        method_rank = {m: i for i, m in enumerate(PRIORITY_METHODS)}
        plans.sort(
            key=lambda p: (
                method_rank.get(p.method, 99),
                -p.budget_flops,
                p.run_name,
            )
        )
    return plans


def select_pilot(plans: list[CheckpointPlan], top_n: int = 1) -> list[CheckpointPlan]:
    """Top-N highest-budget runs per priority method.

    top_n=1 (default) = one anchor per method (smallest meaningful pilot).
    top_n=3 = top 3 budgets per method = 12 runs total for the 4 priority methods.
    `plans` must already be sorted by (method_priority, -budget) for this to be correct.
    """
    counts: dict[str, int] = {}
    pilot: list[CheckpointPlan] = []
    for plan in plans:
        if plan.method not in PRIORITY_METHODS:
            continue
        if counts.get(plan.method, 0) >= top_n:
            continue
        pilot.append(plan)
        counts[plan.method] = counts.get(plan.method, 0) + 1
    return pilot


# --- Command emission -----------------------------------------------------


def emit_run_command(plan: CheckpointPlan, hf_step_dir: str) -> str:
    return (
        "python -m experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval "
        f"--hf-checkpoint {hf_step_dir} "
        f"--output-json {plan.output_json_path} "
        f"--run-name {plan.run_name}"
    )


# --- Iris submission -------------------------------------------------------


def submit_one(
    client: IrisClient,
    plan: CheckpointPlan,
    hf_step_dir: str,
    *,
    priority_band: int,
    wandb_api_key: str,
    hf_token: str,
    tpu_variants: tuple[str, ...] = DEFAULT_TPU_VARIANTS,
    preemptible: bool = True,
    max_length: int = 2048,
    limit: int | None = None,
    allow_eu_fallback: bool = False,
    name_suffix: str = "",
    task_filter: str | None = None,
    log_samples: bool = True,
) -> str:
    """Submit one CheckpointPlan as an Iris TPU job. Returns the iris job id.

    Region is HARD-pinned to the checkpoint's region (plan.region) so the eval
    reads model weights from the local bucket only (no cross-region egress).

    The child runs `run_dclm_core_eval.py` with the resolved HF step subdir and
    writes the result JSON directly to plan.output_json_path (gs://).

    Pass `task_filter` (comma-separated DCLM task names) to spawn a tail-only
    child that runs alongside the full-22 main job; both share the partial dir
    via `run-name`, so whichever lands a task's partial first wins.
    """
    cmd_args = [
        "python",
        "-m",
        "experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval",
        "--hf-checkpoint",
        hf_step_dir,
        "--output-json",
        plan.output_json_path,
        "--run-name",
        plan.run_name,
        "--max-length",
        str(max_length),
    ]
    if log_samples:
        cmd_args += ["--log-samples"]
    if task_filter is not None:
        cmd_args += ["--task-filter", task_filter]
    if limit is not None:
        cmd_args += ["--limit", str(limit)]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "HF_TOKEN": hf_token,
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "PYTHONUNBUFFERED": "1",
        # WandB init occasionally stalls past the 90s default on TPU egress.
        "WANDB_INIT_TIMEOUT": "300",
        # Default rigging cap is 10GB and the largest curation checkpoints
        # (d=2432, ~15GB) trip it when iris lands a child cross-region.
        # 25GB headroom covers any single checkpoint we eval; at $0.02/GB
        # same-continent US egress that's ≤$0.50 per cross-region read.
        "MARIN_MIRROR_BUDGET_GB": "25",
    }

    # Region strategy: SOFT preference (mode=PREFERRED) for cheap regions, never
    # a hard pin. This is how `launch_curation_sweep.submit_one` schedules its
    # training children, and they land smoothly while my earlier HARD-pin
    # version would sit pending for hours when home was contested. iris's
    # scheduler treats PREFERRED as a tiebreaker — it'll go cross-region if
    # necessary, but won't time out waiting.
    #
    # Cost note: if iris lands a job in europe-west4 to read a us-east5
    # checkpoint, that's ~$0.40 cross-region egress for a 4-8GB checkpoint.
    # At 20 evals it's bounded under $10 worst case. Tradeoff accepted vs
    # losing the whole overnight to pending.
    ALL_REGIONS = ["us-east5", "us-central1", "us-central2", "us-east1", "us-west4", "europe-west4"]
    preferred = [plan.region] + [r for r in ALL_REGIONS if r != plan.region]
    if not allow_eu_fallback:
        # Even SOFT mode can let iris land in EU; drop it from the list when
        # the caller doesn't want trans-Atlantic egress at scale.
        preferred = [r for r in preferred if r != "europe-west4"]
    region_constraint = Constraint.create(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=preferred,
        mode=1,  # CONSTRAINT_MODE_PREFERRED (soft)
    )

    constraints = [
        preemptible_constraint(preemptible),
        region_constraint,
    ]
    if len(set(tpu_variants)) > 1:
        constraints.append(device_variant_constraint(list(tpu_variants)))

    primary_tpu = tpu_variants[0]

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"dclm-core-{plan.run_name}{name_suffix}"[:200],
        resources=ResourceSpec(
            # Match the curation training spec exactly — those land on TPU
            # smoothly while my earlier 64GB-disk / 8-cpu / 64GB-mem requests
            # were getting stuck pending. The training pool's autoscaler is
            # already provisioning VMs at this profile so we slot right in.
            cpu=32,
            memory="256GB",
            disk="50GB",
            device=tpu_device(primary_tpu),
        ),
        environment=EnvironmentSpec(
            extras=["tpu", "eval"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=20,
        max_retries_failure=5,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _iris_client() -> IrisClient:
    """Build IrisClient for child submission.

    Two modes:
      - **Direct (laptop)**: caller provides IRIS_CONTROLLER_ADDRESS (e.g.,
        from an open SSH tunnel) AND IRIS_WORKSPACE pointing at the marin
        repo root. We bundle the workspace ourselves and submit children.
      - **Parent-job**: launcher runs as an iris parent job; the harness sets
        IRIS_CONTROLLER_ADDRESS and IRIS_BUNDLE_ID; children inherit the
        bundle. (Tried first since it auto-sets BUNDLE_ID; if that's set,
        we skip workspace bundling.)

    Direct laptop one-liner (the handoff entry point):
        # In one terminal — keep the tunnel open:
        gcloud compute ssh iris-controller-marin --zone us-central1-a \\
            --project hai-gcp-models -- -L 10000:localhost:10000 -N

        # In another:
        IRIS_CONTROLLER_ADDRESS=http://127.0.0.1:10000 \\
        IRIS_WORKSPACE=$(pwd) \\
        WANDB_API_KEY=... HF_TOKEN=... \\
        uv run python -m experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep \\
            --pilot --top-n 3 --launch
    """
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set. See module docstring for the laptop one-liner.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    if bundle_id:
        return IrisClient.remote(controller_address, bundle_id=bundle_id)
    workspace = os.environ.get("IRIS_WORKSPACE")
    if not workspace:
        raise RuntimeError(
            "IRIS_WORKSPACE not set (and no inherited IRIS_BUNDLE_ID). "
            "Point it at the marin repo root so the workspace can be bundled."
        )
    from pathlib import Path

    return IrisClient.remote(controller_address, workspace=Path(workspace))


# --- CLI ------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    scope = ap.add_mutually_exclusive_group(required=True)
    scope.add_argument(
        "--pilot",
        action="store_true",
        help="Top-N runs per priority method = the smallest meaningful batch. "
        "Default N=1 (one anchor); set with --top-n.",
    )
    scope.add_argument(
        "--priority-methods", action="store_true", help="All budgets for the 4 priority methods in the FM sweep."
    )
    scope.add_argument("--all-fm", action="store_true", help="All FM sweep runs (no method filter).")
    ap.add_argument(
        "--top-n", type=int, default=1, help="With --pilot: top N highest-budget runs per priority method (default 1)."
    )
    ap.add_argument("--summaries-prefix", default=FM_SUMMARIES_PREFIX, help="GCS prefix containing summary JSONs.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print plans only, don't launch.")
    mode.add_argument(
        "--emit-commands", action="store_true", help="Print bash commands per plan (resolves final HF step subdir)."
    )
    mode.add_argument(
        "--launch", action="store_true", help="Actually submit Iris children for each plan (the handoff entry point)."
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip plans whose output_json already exists on GCS (default ON).",
    )
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument(
        "--child-priority",
        default="batch",
        choices=sorted(PRIORITY_BAND_MAP),
        help="Priority band for child Iris jobs. Default: batch (don't preempt other work).",
    )
    ap.add_argument(
        "--non-preemptible",
        action="store_true",
        help="Use non-preemptible TPU pool. Default uses preemptible (cheaper, often more capacity).",
    )
    ap.add_argument(
        "--allow-eu-fallback",
        action="store_true",
        help="Allow children to land on europe-west4 v6e-4 if their primary region is saturated. "
        "Each child then pays ~$0.32-0.64 cross-region read for the checkpoint. "
        "OK at pilot scale, costly at 100x+ — opt in deliberately.",
    )
    ap.add_argument(
        "--name-suffix",
        default="",
        help="Suffix appended to child job names (e.g. '-v2'). "
        "Use to dodge JobAlreadyExists after a kill+resubmit cycle, "
        "since iris kills are async and the old name may linger.",
    )
    ap.add_argument("--max-length", type=int, default=2048, help="Max sequence length for eval. DCLM uses 2048.")
    ap.add_argument(
        "--limit", type=int, default=None, help="Cap each task to N examples (smoke testing). None = full eval."
    )
    args = ap.parse_args()

    method_filter: set[str] | None
    if args.pilot or args.priority_methods:
        method_filter = set(PRIORITY_METHODS)
    else:
        method_filter = None

    plans = enumerate_plans(args.summaries_prefix, method_filter)
    if args.pilot:
        plans = select_pilot(plans, top_n=args.top_n)
    logger.info("Selected %d plans", len(plans))

    # In --launch mode, set up the Iris client once and capture credentials.
    client = None
    wandb_api_key = None
    hf_token = None
    if args.launch:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        if not wandb_api_key or not hf_token:
            logger.error("--launch requires WANDB_API_KEY and HF_TOKEN in env " "(needed for the child runs).")
            sys.exit(2)
        client = _iris_client()

    submitted: list[tuple[str, str]] = []  # (run_name, job_id)
    skipped: list[str] = []
    incomplete: list[str] = []

    for i, plan in enumerate(plans):
        if args.skip_existing and _gcs_exists(plan.output_json_path):
            logger.info("[%2d] SKIP (exists): %s", i, plan.run_name)
            skipped.append(plan.run_name)
            continue

        if args.dry_run:
            logger.info(
                "[%2d] %s | method=%s | budget=%.0e | region=%s | output=%s",
                i,
                plan.run_name,
                plan.method,
                plan.budget_flops,
                plan.region,
                plan.output_path,
            )
            continue

        hf_step_dir = _resolve_final_step(plan.hf_dir())
        if hf_step_dir is None:
            logger.warning("[%2d] NO HF SUBDIR found in %s -- run may not have completed", i, plan.hf_dir())
            incomplete.append(plan.run_name)
            continue

        if args.emit_commands:
            print(emit_run_command(plan, hf_step_dir))
            continue

        # args.launch
        try:
            job_id = submit_one(
                client,
                plan,
                hf_step_dir,
                priority_band=PRIORITY_BAND_MAP[args.child_priority],
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                preemptible=not args.non_preemptible,
                max_length=args.max_length,
                limit=args.limit,
                allow_eu_fallback=args.allow_eu_fallback,
                name_suffix=args.name_suffix,
            )
        except Exception as e:
            # JobAlreadyExists or transient submit failures: log and continue so
            # one collision doesn't abort the rest of the batch. Re-runnable with
            # --name-suffix to dodge stale names.
            logger.error("[%2d] SUBMIT FAILED for %s: %s", i, plan.run_name, e)
            incomplete.append(plan.run_name)
            continue
        submitted.append((plan.run_name, job_id))
        logger.info("[%2d] LAUNCHED %s -> iris job %s (region=%s)", i, plan.run_name, job_id, plan.region)

    if args.launch:
        logger.info("Summary: submitted=%d skipped=%d incomplete=%d", len(submitted), len(skipped), len(incomplete))
        if submitted:
            logger.info("Job IDs:")
            for run_name, job_id in submitted:
                logger.info("  %s -> %s", job_id, run_name)


if __name__ == "__main__":
    main()
