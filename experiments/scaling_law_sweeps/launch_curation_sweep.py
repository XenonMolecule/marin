# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coordinator for the data-curation IsoFLOP sweep.

A lightweight CPU-only iris job that enumerates all planned training runs
(via `curation_plan.enumerate_plans`) and submits each as an INDEPENDENT iris
TPU child job running `run_curation_train_standalone.py`. Each child:

  - Floats across regions (SOFT preference for ALL_REGIONS prevents parent
    region inheritance — see `experiments/baseline_collection/launch_adaptive.py`
    lines 109-121 for the canonical pattern this mirrors).
  - Gets BOTH a v4 and a v5p TPU type via `--tpu v4-X,v5p-Y` comma list, so
    iris schedules on whichever has capacity.
  - Region-locks itself on first-write via the tracker.

Usage (from inside an iris parent job, normally launched as):

    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 4 --job-name curation-coordinator \\
        -- python experiments/scaling_law_sweeps/launch_curation_sweep.py \\
        --methods all --experiments all --child-priority batch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import fsspec

from experiments.scaling_law_sweeps import region_tracker
from iris.client.client import IrisClient
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    get_tpu_topology,
    tpu_device,
)
from iris.rpc import job_pb2
from rigging.filesystem import REGION_TO_DATA_BUCKET

from experiments.scaling_law_sweeps import curation_plan

logger = logging.getLogger(__name__)


def _vm_count(tpu_variant: str) -> int:
    """Return iris's vm_count for a TPU shape (e.g. v4-16 -> 2, v5p-16 -> 2).

    Single source of truth for vm_count is iris's TPU_TOPOLOGIES table
    (lib/iris/src/iris/cluster/types.py). Used to filter TPU alternatives down
    to those matching the primary's vm_count — required for multi-host
    coscheduling correctness.
    """
    return get_tpu_topology(tpu_variant).vm_count

# The TPU-side child script that each iris job will run.
SCRIPT = "experiments/scaling_law_sweeps/run_curation_train_standalone.py"

# All regions the SOFT REGION constraint considers acceptable. By preferring
# ALL of them with mode=PREFERRED, we suppress parent-region inheritance and
# let iris pick whichever region has capacity for the requested TPU types.
ALL_REGIONS = sorted(REGION_TO_DATA_BUCKET.keys())

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def submit_one(
    client: IrisClient,
    plan: curation_plan.PlannedRun,
    *,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    wandb_project: str,
    wandb_entity: str,
    wandb_group: str,
    tracker_prefix: str,
    allowed_regions: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    run_suffix: str = "",
    wandb_mode: str = "auto",
    force_primary_tpu: str | None = None,
) -> str:
    """Submit one PlannedRun as an iris TPU job. Returns the iris job id.

    Per iris CLI pattern (`iris job run --tpu v4-X,v5p-Y`): set the primary
    TPU via `tpu_device(primary)` and add `device_variant_constraint([v4, v5p])`
    as a constraint so the scheduler can pick EITHER variant when capacity is
    contested. Both variants must share the same vm_count (v4-8 and v5p-8 both
    have vm_count=1, so this works for our small-end plans).
    """
    # v6e included as fallback when the plan fits a vm_count=1 v6e slice.
    # iris requires all variants in a single device_variant_constraint to share
    # vm_count, so v6e-16+ (vm_count=4) can't mix with v4-16/v5p-16 (vm_count=2).
    #
    # IMPORTANT: all variants in the list must have the SAME chip count as the
    # primary. Mixing chip counts (e.g., primary=v4-8 with 4 chips, variant=v6e-8
    # with 8 chips) causes an iris co-scheduling bug: iris still charges our
    # task `committed_tpu=4` when placing on v6e-8, so it thinks 4 chips are
    # "free" and can land another 4-chip task on the same VM. That second task
    # then collides with ours on `/dev/vfio/0` (TPU devices aren't partitionable
    # within a VM — libtpu grabs the whole slice). Until iris's scheduler
    # stops treating chips as fungible within a VM, we keep variants at matching
    # chip counts: v4-8 + v5p-8 + v6e-4 are all 4-chip slices. v6e-8 (8 chips)
    # is intentionally excluded.
    if force_primary_tpu:
        # Smoke/debug override: pin to a specific TPU shape, no alternatives.
        # Use when you want iris to only consider one shape (e.g. testing
        # multi-host on v5p-16 without also considering v4-32 in a different
        # region). vm_count is derived from that shape alone.
        tpu_variants = [force_primary_tpu]
        primary_tpu = force_primary_tpu
    else:
        tpu_variants = [plan.v4_tpu, plan.v5p_tpu]
        if plan.v6e_tpu:
            tpu_variants.append(plan.v6e_tpu)
        primary_tpu = plan.v4_tpu  # arbitrary — constraint allows any of them

    # vm_count compatibility filter for multi-host. iris's adjust_tpu_replicas
    # auto-scales replicas using the PRIMARY device's vm_count. If we then land
    # on a variant with a different vm_count via device_variant_constraint, the
    # replicas value is wrong (e.g., primary v4-32 vm_count=4, but landing on
    # v5p-16 vm_count=2 — iris asks for 4 same-tpu-name workers and never finds
    # them). Filter alternatives to those matching the primary's vm_count;
    # single-host plans (vm_count=1) keep all alternatives.
    primary_vm_count = _vm_count(primary_tpu)
    tpu_variants = [v for v in tpu_variants if _vm_count(v) == primary_vm_count]

    cmd_args = [
        "python",
        SCRIPT,
        *plan.to_cli_args(),
        "--tracker-prefix",
        tracker_prefix,
        "--wandb-mode",
        wandb_mode,
        *(("--run-suffix", run_suffix) if run_suffix else ()),
        "--wandb-project",
        wandb_project,
        "--wandb-entity",
        wandb_entity,
        "--wandb-group",
        wandb_group,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        # WandB's default init_timeout is 90s, which we've seen time out on
        # TPU workers with slow egress to wandb.ai. Bump to 5 min.
        "WANDB_INIT_TIMEOUT": "300",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if extra_env:
        env_vars.update(extra_env)

    # Region constraint:
    #   - Default (`allowed_regions=None`): SOFT preference for ALL_REGIONS.
    #     Prevents parent-region inheritance (iris client.py:645) without
    #     restricting the autoscaler. Same pattern as launch_adaptive.py:109-121.
    #   - When `allowed_regions` is provided: HARD restriction to that subset.
    #     Use this to pin children to regions whose buckets already hold the
    #     pre-copied tokenized caches (zero cross-region reads).
    if allowed_regions:
        region_constraint = Constraint(
            key=WellKnownAttribute.REGION,
            op=ConstraintOp.IN,
            values=tuple(allowed_regions),
        )  # default mode = CONSTRAINT_MODE_REQUIRED (hard)
    else:
        region_constraint = Constraint(
            key=WellKnownAttribute.REGION,
            op=ConstraintOp.IN,
            values=tuple(ALL_REGIONS),
            mode=1,  # CONSTRAINT_MODE_PREFERRED (soft)
        )

    constraints = [
        preemptible_constraint(True),
        region_constraint,
    ]
    # If the v4 and v5p TPU types differ (the common case), add a constraint
    # that allows EITHER variant — the iris scheduler will pick whichever has
    # capacity. Same pattern as `iris job run --tpu v4-X,v5p-Y` CLI flag.
    if len(set(tpu_variants)) > 1:
        constraints.append(device_variant_constraint(tpu_variants))

    # Multi-host TPU support: pass replicas=1 + tpu-name coscheduling ONLY for
    # multi-host shapes (vm_count > 1). iris's `adjust_tpu_replicas`
    # (lib/iris/src/iris/cluster/types.py:711) auto-scales replicas=1 ->
    # vm_count for multi-host topologies (v4-16 -> 2 VMs, v4-32 -> 4 VMs).
    # `coscheduling=group_by="tpu-name"` gang-schedules all replicas onto the
    # same TPU slice so libtpu can wire up multi-host JAX.
    #
    # For single-host (vm_count=1), we OMIT both kwargs. Yesterday's working
    # single-host submissions didn't set them, and adding
    # coscheduling=tpu-name with replicas=1 may interact poorly with iris's
    # scheduler (observed: consistent /dev/vfio busy libtpu collisions on
    # v5p-8 when coscheduling was set, while yesterday's bare submissions to
    # the same pool worked). Matches fray's iris_backend.py:67 pattern:
    # `resolve_coscheduling` only returns a config when replicas > 1.
    submit_kwargs: dict = {}
    if primary_vm_count > 1:
        submit_kwargs["replicas"] = 1
        submit_kwargs["coscheduling"] = CoschedulingConfig(group_by="tpu-name")
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"curation-{plan.run_name_core}"[:200],
        resources=ResourceSpec(
            cpu=plan.cpu,
            memory=f"{plan.memory_gb}GB",
            disk=f"{plan.disk_gb}GB",
            device=tpu_device(primary_tpu),
        ),
        environment=EnvironmentSpec(
            extras=["tpu"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        **submit_kwargs,
        max_retries_preemption=100,
        # max_retries_failure raised from 3 -> 10. Observed: transient TPU VMs
        # with stale libtpu state (/dev/vfio/N busy) kept failing retries 3x
        # and iris gave up before routing to a clean VM. 10 gives iris enough
        # attempts to evict bad workers from the pool and land us on healthy
        # capacity. Bounded enough that a genuinely broken plan still fails
        # quickly.
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    return str(job.job_id)


def is_run_already_complete(
    plan: curation_plan.PlannedRun,
    tracker_prefix: str,
    run_suffix: str = "",
) -> bool:
    """Check if a run has already completed (done marker written by prior child).

    Flow:
      1. Read the region tracker to find which region this run claimed.
         If no claim exists, the run never started — not done.
      2. Compute the checkpoint output_path in that region's bucket.
      3. Check for `.data_curation_DONE` marker. Return True only if present.

    Both tracker key and output_path use `run_name_core + "-" + suffix` when
    a suffix is set (matches what the child writes in main()), so
    skip-if-done catches runs completed under the same suffix — not runs
    from a DIFFERENT suffix.

    Safe (no false positives): we only skip when we can PROVE the run
    completed — any filesystem/network hiccup returns False → re-submit,
    and the child's own region-lock check prevents duplicate work.
    """
    # Reconstruct the effective run_name and tracker key using the suffix —
    # the child's main() does the same: `run_name = plan.run_name_core + "-" + suffix`.
    suffix = run_suffix.strip()
    run_name = plan.run_name_core + (f"-{suffix}" if suffix else "")
    run_key = f"{plan.method_name}__{plan.experiment_tag}__{run_name}.region"
    tracker_path = f"{tracker_prefix.rstrip('/')}/{run_key}"
    try:
        fs, urlpath = fsspec.core.url_to_fs(tracker_path)
        if not fs.exists(urlpath):
            return False  # never claimed → never ran → not done
        with fs.open(urlpath, "rb") as f:
            obj = json.loads(f.read().decode())
        region = obj.get("region")
        if region not in region_tracker.REGION_TO_BUCKET:
            return False
    except Exception as e:
        logger.debug("Tracker read failed for %s: %s", run_name, e)
        return False

    bucket = region_tracker.REGION_TO_BUCKET[region]
    done_marker = f"{bucket}/checkpoints/isoflop-curation/{run_name}/.data_curation_DONE"
    try:
        fs, urlpath = fsspec.core.url_to_fs(done_marker)
        return fs.exists(urlpath)
    except Exception as e:
        logger.debug("Done-marker check failed for %s: %s", plan.run_name_core, e)
        return False


def submit_all(
    client: IrisClient,
    plans: list[curation_plan.PlannedRun],
    *,
    tracker_prefix: str,
    skip_if_done: bool = True,
    **submit_kwargs,
) -> tuple[list[str], list[curation_plan.PlannedRun]]:
    """Eager-submit all plans. Iris queues whatever can't immediately schedule.

    When `skip_if_done=True` (default), we skip any plan whose output path
    contains a `.data_curation_DONE` marker from a prior successful run.

    Returns (submitted_job_ids, skipped_plans).
    """
    submitted: list[str] = []
    skipped: list[curation_plan.PlannedRun] = []
    run_suffix = submit_kwargs.get("run_suffix", "")
    for i, plan in enumerate(plans):
        if skip_if_done and is_run_already_complete(plan, tracker_prefix, run_suffix=run_suffix):
            skipped.append(plan)
            logger.info("[%d/%d] SKIP (already done): %s", i + 1, len(plans), plan.run_name_core)
            continue
        try:
            job_id = submit_one(client, plan, tracker_prefix=tracker_prefix, **submit_kwargs)
            submitted.append(job_id)
            logger.info("[%d/%d] submitted %s → %s", i + 1, len(plans), plan.run_name_core, job_id)
        except Exception as e:
            logger.exception("[%d/%d] failed to submit %s: %s", i + 1, len(plans), plan.run_name_core, e)
    return submitted, skipped


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        help="Curation methods. Use 'all' for every registered method, or names from curation_plan.METHODS.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["A", "B", "all"],
        help="A=natural epoching; B=pinned T_target.",
    )
    parser.add_argument(
        "--t-targets",
        nargs="+",
        type=float,
        default=list(curation_plan.DEFAULT_T_TARGETS),
        help="T_target values for Experiment B (extension lever).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=float,
        default=list(curation_plan.BUDGETS),
        help="Compute (FLOPs) budgets to sweep.",
    )
    parser.add_argument(
        "--min-slice-tokens",
        type=float,
        default=curation_plan.MIN_SLICE_TOKENS_DEFAULT,
        help="Safety floor on Experiment B slice size (unique tokens).",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help="If set, limit the total number of children submitted (smoke testing).",
    )
    parser.add_argument(
        "--filter-name-contains",
        type=str,
        default=None,
        help="If set, only submit plans whose run_name_core contains this substring. "
        "Useful for targeted smoke launches (e.g. '--filter-name-contains d2560-L26-B8' "
        "to launch a specific multi-host candidate).",
    )
    parser.add_argument(
        "--force-primary-tpu",
        type=str,
        default=None,
        help="If set, override every plan's primary TPU shape with this value (e.g. "
        "'v5p-16'). vm_count filtering then drops alternatives whose vm_count differs. "
        "Use for smoke tests to pin to a specific shape/region regardless of the plan's "
        "default. Example: '--force-primary-tpu v5p-16' to test multi-host without waiting "
        "on fresh v4-32 provisioning.",
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
        help="Priority band for child TPU jobs. Default 'batch' so children yield to higher-priority work.",
    )
    parser.add_argument(
        "--wandb-project",
        default="marin",
    )
    parser.add_argument(
        "--wandb-entity",
        default="marin-community",
    )
    parser.add_argument(
        "--wandb-group",
        default="data-curation-isoflop",
        help="WandB group so all runs land in one comparison view.",
    )
    parser.add_argument(
        "--tracker-prefix",
        default="gs://marin-us-central1/metadata/region_locks/data_curation_isoflop/",
        help="GCS prefix for region-lock tracker files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan, do not submit.",
    )
    parser.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Skip the post-submit `while True: sleep(3600)`. Useful for testing the coordinator outside iris.",
    )
    parser.add_argument(
        "--no-skip-if-done",
        action="store_true",
        help="DISABLE skip-if-already-done. Every plan will be submitted even if a prior "
        "successful run's done marker exists. Normally off — we want to skip finished runs.",
    )
    parser.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="If set, HARD-restrict child TPU jobs to this subset of regions. "
        "Use this to pin runs to regions whose buckets already contain the pre-copied "
        "tokenized caches. Unset (default) = SOFT preference for all regions (floats).",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="Optional suffix appended to every child's run_name. Useful to force "
        "fresh WandB runs + checkpoint dirs when iterating on a config.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "offline_no_sync"],
        default="auto",
        help="Passed through to every child's --wandb-mode flag. See " "run_curation_train_standalone.py for semantics.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    methods = curation_plan.resolve_methods(args.methods)
    experiments = curation_plan.resolve_experiments(args.experiments, args.t_targets)
    plans = curation_plan.enumerate_plans(
        methods,
        experiments,
        budgets=tuple(args.budgets),
        min_slice_tokens=args.min_slice_tokens,
    )
    if args.filter_name_contains is not None:
        plans = [p for p in plans if args.filter_name_contains in p.run_name_core]
    if args.max_count is not None:
        plans = plans[: args.max_count]

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        return

    # Acquire iris client from the parent job's env.
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError(
            "IRIS_CONTROLLER_ADDRESS not set — this coordinator must run inside an iris job. "
            "Use `iris --cluster marin job run -- python this_script.py ...`."
        )
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    # Required env vars for child training jobs.
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required (set via -e on iris job run).")
    hf_token = os.environ.get("HF_TOKEN")  # optional for non-gated models

    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    logger.info("Submitting up to %d children (priority=%s)...", len(plans), args.child_priority)
    submitted, skipped = submit_all(
        client,
        plans,
        tracker_prefix=args.tracker_prefix,
        skip_if_done=not args.no_skip_if_done,
        child_priority_band=child_priority_band,
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        allowed_regions=args.allowed_regions,
        run_suffix=args.run_suffix,
        wandb_mode=args.wandb_mode,
        force_primary_tpu=args.force_primary_tpu,
    )
    logger.info(
        "Result: %d submitted, %d skipped (already done), %d failed",
        len(submitted),
        len(skipped),
        len(plans) - len(submitted) - len(skipped),
    )

    if args.no_keep_alive:
        return

    # Stay alive so iris's parent-keeps-children-alive contract holds. Same as
    # launch_adaptive.py:316 — if the parent dies, iris will treat the entire
    # job hierarchy as orphaned and may kill the children.
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
