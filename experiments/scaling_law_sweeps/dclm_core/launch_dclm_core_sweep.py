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
  - region (from summary's run.region — child is HARD-pinned here by default to
    avoid cross-region reads, per CLAUDE.md; pass --region-float to relax to a
    SOFT preference for end-of-run stragglers stuck pending in a contested home)
  - hf_checkpoint_gcs (run.output_path + /hf/step-<final>/)
  - output_json_gcs (parallel sibling prefix data_curation_core_results/)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
from rigging.filesystem import filesystem as marin_filesystem

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
    hidden_dim: int  # model width; (method, hidden_dim, budget_flops) is the cell key
    region: str
    output_path: str  # parent dir on GCS (no /hf/step-N appended)
    summary_path: str  # the training-run JSON we read this plan from
    output_json_path: str  # where the full (sample-heavy) CORE result lands — a ttl= prefix
    scores_json_path: str  # where the small scores-only summary lands — a PERMANENT prefix

    def cell_key(self) -> tuple[str, int, float]:
        """Identity of the isoflop cell this checkpoint fills."""
        return (self.method, self.hidden_dim, self.budget_flops)

    def hf_dir(self) -> str:
        """The /hf/ directory; the final step subdir is resolved at run time."""
        return f"{self.output_path.rstrip('/')}/hf/"


# --- GCS helpers ----------------------------------------------------------
# fsspec/gcsfs, NOT the gcloud CLI: iris worker containers don't ship gcloud, so
# the launcher must use the same filesystem the eval child uses.


def _gcs_ls(prefix: str) -> list[str]:
    """List immediate children of a GCS prefix as gs:// paths. Empty on missing."""
    try:
        entries = marin_filesystem("gcs").ls(prefix, detail=False)
    except FileNotFoundError:
        return []
    # gcsfs strips the gs:// scheme; re-add it for downstream consumers (the
    # resolved hf/step dir is handed to the child as --hf-checkpoint).
    return [e if e.startswith("gs://") else f"gs://{e}" for e in entries]


def _gcs_cat(path: str) -> bytes:
    with marin_filesystem("gcs").open(path, "rb") as f:
        return f.read()


def _gcs_exists(path: str) -> bool:
    return marin_filesystem("gcs").exists(path)


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
    results_prefix: str = CORE_RESULTS_PREFIX,
    samples_prefix: str | None = None,
    budget_desc: bool = True,
) -> list[CheckpointPlan]:
    """List CheckpointPlans from a summaries prefix, optionally filtered by method.

    ``results_prefix`` holds the small permanent scores summaries; ``samples_prefix``
    (defaults to ``results_prefix``) holds the sample-heavy finals/partials and is
    where a ttl= prefix belongs so the raw outputs auto-expire.
    """
    samples_prefix = samples_prefix or results_prefix
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
                hidden_dim=int(plan_section.get("hidden_dim", 0)),
                region=run_section.get("region", ""),
                output_path=run_section.get("output_path", ""),
                summary_path=sp,
                output_json_path=f"{samples_prefix}{run_name}.json",
                scores_json_path=f"{results_prefix}{run_name}_summary.json",
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


# Suffixes that mark a re-run or variance variant of an otherwise-canonical run
# (a seed sweep, a persistent-cache re-export, a manual retry). These are dropped
# when a plain run exists at the same cell. NOT a batch-size token (`-B128`):
# distinct batch sizes are legitimately different training configs.
_RERUN_SUFFIXES: tuple[str, ...] = ("-seed", "-pcache", "-retry", "-v2", "-v3")


def _is_rerun_variant(run_name: str) -> bool:
    leaf = run_name.rsplit("/", 1)[-1]
    return any(tag in leaf for tag in _RERUN_SUFFIXES)


def _canonical_rank(run_name: str) -> tuple[int, int, str]:
    """Deterministic sort key for the rare cell that has ONLY re-run variants and
    no plain run: prefer the shorter, then lexicographically-first name."""
    leaf = run_name.rsplit("/", 1)[-1]
    return (1 if _is_rerun_variant(run_name) else 0, len(leaf), leaf)


def drop_rerun_variants(plans: list[CheckpointPlan]) -> list[CheckpointPlan]:
    """Drop re-run / variance variants, keeping every distinct training config.

    The 10k sweep has a few cells with extra summaries: high_quality's
    `pcache-v1` re-export and fineweb_edu's `seed7/13/21` sweep are re-runs of a
    plain run and are dropped. Distinct batch sizes (B64 vs B128 at d512/2e19) are
    NOT re-runs — they are different training configs the canonical figure may
    plot separately — so both are kept. Every drop is logged for --dry-run review.
    """
    by_cell: dict[tuple[str, int, float], list[CheckpointPlan]] = {}
    for p in plans:
        by_cell.setdefault(p.cell_key(), []).append(p)
    kept: list[CheckpointPlan] = []
    dropped = 0
    for cell, group in by_cell.items():
        plain = [p for p in group if not _is_rerun_variant(p.run_name)]
        variants = [p for p in group if _is_rerun_variant(p.run_name)]
        if not plain:
            # No plain run at this cell — every summary here is a re-run variant
            # (e.g. a file whose name says `-B64` but whose plan.run_name is
            # `-B64-pcache-v1`). Keep one deterministically and log the pick.
            ordered = sorted(group, key=lambda p: _canonical_rank(p.run_name))
            kept.append(ordered[0])
            dropped += len(ordered) - 1
            if len(ordered) > 1:
                method, hidden_dim, budget = cell
                logger.info(
                    "DROP rerun variants (no plain run) method=%s d=%d budget=%.0e: keep %s | drop [%s]",
                    method,
                    hidden_dim,
                    budget,
                    ordered[0].run_name,
                    ", ".join(p.run_name for p in ordered[1:]),
                )
            continue
        kept.extend(plain)
        if variants:
            dropped += len(variants)
            method, hidden_dim, budget = cell
            logger.info(
                "DROP rerun variants method=%s d=%d budget=%.0e: keep [%s] | drop [%s]",
                method,
                hidden_dim,
                budget,
                ", ".join(p.run_name for p in plain),
                ", ".join(p.run_name for p in variants),
            )
    logger.info("Dropped %d re-run variants: %d plans → %d", dropped, len(plans), len(kept))
    return kept


def filter_to_cells(plans: list[CheckpointPlan], cell_specs: list[str]) -> list[CheckpointPlan]:
    """Keep only plans at the given (budget, width) cells.

    Each cell spec is ``"<budget>:<width>"`` (e.g. ``"9e+17:1024"``). Matching is on the
    run-name substring ``-<budget>-d<width>-`` so it's exact and float-safe. Used to run a
    focused wave at a specific set of isoflop cells (e.g. the fastpipe-aligned scales)
    instead of the whole sweep.
    """
    if not cell_specs:
        return plans
    subs = []
    for c in cell_specs:
        budget, width = c.split(":")
        subs.append(f"-{budget}-d{width}-")
    kept = [p for p in plans if any(s in p.run_name for s in subs)]
    logger.info("Cell filter (%d cells): %d plans → %d", len(cell_specs), len(plans), len(kept))
    return kept


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
    region_float: bool = False,
    allow_eu_fallback: bool = False,
    name_suffix: str = "",
    task_filter: str | None = None,
    log_samples: bool = True,
) -> str:
    """Submit one CheckpointPlan as an Iris TPU job. Returns the iris job id.

    By default the child is HARD-pinned to the region its checkpoint was trained
    in (plan.region) so the eval reads model weights from the local bucket only:
    zero cross-region egress, and no exposure to rigging's cumulative
    mirror-budget failures on large (d>=2432, ~15GB) checkpoints. Set
    `region_float=True` to relax this to a SOFT preference that lets iris migrate
    a job to any region with capacity (primary region still preferred first) —
    use that only for end-of-run stragglers stuck pending in a contested home.

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
        "--summary-json",
        plan.scores_json_path,
        "--run-name",
        plan.run_name,
        "--max-length",
        str(max_length),
    ]
    # Point the child at the byte-identical dataset cache in ITS OWN region's
    # bucket (the same bucket the checkpoint lives in) so task loading reads
    # locally, never the HF Hub. hf_step_dir = gs://<bucket>/checkpoints/... .
    ckpt_bucket = hf_step_dir.split("/")[2]
    cmd_args += ["--dataset-cache-gcs", f"gs://{ckpt_bucket}/eval_datasets/dclm_core_hf_cache/"]
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
        # NOTE: do NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE here. The eval
        # datasets run offline via the in-region GCS cache (HF_DATASETS_OFFLINE,
        # set in run_dclm_core_eval), but the model load still needs the *base*
        # model's config/tokenizer via _name_or_path, which isn't in the local
        # hub cache — forcing hub-offline makes transformers hard-error
        # ("run the library in offline mode") on every job. The Hub cold-start
        # rate limit ("We had to rate limit you", 1000 req/5min) is instead
        # ridden out by the in-process retry in run_dclm_core_eval
        # (_load_hf_with_retry) plus iris job retries — it is transient, not fatal.
        # WandB init occasionally stalls past the 90s default on TPU egress.
        "WANDB_INIT_TIMEOUT": "300",
        # Default rigging cap is 10GB and the largest curation checkpoints
        # (d=2432, ~15GB) trip it when iris lands a child cross-region.
        # 25GB headroom covers any single checkpoint we eval; at $0.02/GB
        # same-continent US egress that's ≤$0.50 per cross-region read.
        "MARIN_MIRROR_BUDGET_GB": "25",
    }

    # Region strategy: HARD-pin to the checkpoint's training region by default so
    # the eval reads weights from the local bucket only — no cross-region egress,
    # and no cumulative mirror-budget failures on the largest (d>=2432, ~15GB)
    # checkpoints. `region_float=True` relaxes this to a SOFT preference for
    # end-of-run stragglers: iris keeps the training region first but may migrate
    # a job to any region with capacity rather than sit pending.
    #
    # Cost note (float mode only): if iris lands a job in europe-west4 to read a
    # us-east5 checkpoint, that's ~$0.40 cross-region egress for a 4-8GB
    # checkpoint. EU is dropped from the candidate list unless allow_eu_fallback.
    if region_float:
        ALL_REGIONS = ["us-east5", "us-central1", "us-central2", "us-east1", "us-west4", "europe-west4"]
        preferred = [plan.region] + [r for r in ALL_REGIONS if r != plan.region]
        if not allow_eu_fallback:
            preferred = [r for r in preferred if r != "europe-west4"]
        region_constraint = Constraint.create(
            key=WellKnownAttribute.REGION,
            op=ConstraintOp.IN,
            values=preferred,
            mode=job_pb2.CONSTRAINT_MODE_PREFERRED,
        )
    else:
        region_constraint = Constraint.create(
            key=WellKnownAttribute.REGION,
            op=ConstraintOp.IN,
            values=[plan.region],
            mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
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
    scope.add_argument(
        "--methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help="Explicit method names to eval (exact plan.method_name match), e.g. the 3000-scale "
        "biased+random comparison set: dclm nemotron_full_bos_fixed high_quality_3000 "
        "resiliparse_dedup fineweb_edu med_quality_3000 dclm_random_3000 "
        "nemotron_full_random_3000 resiliparse_random_dedup_3000.",
    )
    ap.add_argument(
        "--top-n", type=int, default=1, help="With --pilot: top N highest-budget runs per priority method (default 1)."
    )
    ap.add_argument("--summaries-prefix", default=FM_SUMMARIES_PREFIX, help="GCS prefix containing summary JSONs.")
    ap.add_argument(
        "--results-prefix",
        default=CORE_RESULTS_PREFIX,
        help="PERMANENT GCS prefix where the small scores-only summaries land (default: the "
        "3000-WARC FM prefix). Point at a dedicated prefix (e.g. data_curation_10k_core_results/) "
        "to keep a sweep's scores in their own namespace.",
    )
    ap.add_argument(
        "--samples-prefix",
        default=None,
        help="GCS prefix where the sample-heavy full finals + per-task partials land (defaults to "
        "--results-prefix). Point at a ttl= prefix (e.g. gs://marin-us-central1/tmp/ttl=30d/dclm_10k_core/) "
        "so the ~360GB of --log-samples output auto-expires while the scores summaries persist forever.",
    )
    ap.add_argument(
        "--dedup-cells",
        action="store_true",
        help="Drop re-run / variance variants (seed sweeps, pcache re-exports, retries) when a "
        "plain run exists at the same cell, while keeping distinct batch-size configs (B64 and "
        "B128 are both kept). Logs every drop. For the 10k sweep this takes 250 summaries → 246.",
    )
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
        "--region-float",
        action="store_true",
        help="Relax child region pinning from HARD (train region only) to a SOFT "
        "preference that lets iris migrate a job to any region with capacity. "
        "Default OFF: each eval runs in the region its checkpoint was trained in "
        "(no cross-region egress). Turn on for end-of-run stragglers stuck pending.",
    )
    ap.add_argument(
        "--allow-eu-fallback",
        action="store_true",
        help="With --region-float, also allow children to land on europe-west4 v6e-4 "
        "if their primary region is saturated. Each child then pays ~$0.32-0.64 "
        "cross-region read for the checkpoint. No effect without --region-float.",
    )
    ap.add_argument(
        "--name-suffix",
        default="",
        help="Suffix appended to child job names (e.g. '-v2'). "
        "Use to dodge JobAlreadyExists after a kill+resubmit cycle, "
        "since iris kills are async and the old name may linger.",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=0,
        help="Throttle submission: after this many launches, pause --wave-delay before continuing. "
        "0 = submit all at once. Use ~40 to keep cold-start HF requests under the Hub's 1000-req/5min "
        "API limit when launching a large fleet.",
    )
    ap.add_argument(
        "--wave-delay",
        type=float,
        default=330.0,
        help="Seconds to pause between waves (should exceed HF's 5-min rate window). Default 330.",
    )
    ap.add_argument("--max-length", type=int, default=2048, help="Max sequence length for eval. DCLM uses 2048.")
    ap.add_argument(
        "--limit", type=int, default=None, help="Cap each task to N examples (smoke testing). None = full eval."
    )
    ap.add_argument(
        "--no-log-samples",
        dest="log_samples",
        action="store_false",
        default=True,
        help="Don't persist per-example prompts/generations in the partials/finals. Scores only. "
        "Use when you only need the Core_v2 numbers (smaller output, no large cross-region sample writes).",
    )
    ap.add_argument(
        "--no-keepalive",
        dest="keepalive",
        action="store_false",
        default=True,
        help="Submit-and-exit instead of blocking until children finish. Children submitted from "
        "inside a parent iris job are lifecycle-bound descendants and get reaped if the parent "
        "exits early, so keep-alive is ON by default. Only safe to disable in direct-laptop mode.",
    )
    ap.add_argument(
        "--keepalive-poll", type=float, default=180.0, help="Keep-alive: seconds between scores-count polls."
    )
    ap.add_argument(
        "--keepalive-max",
        type=float,
        default=6 * 3600.0,
        help="Keep-alive: max seconds to block before exiting even if children remain (default 6h).",
    )
    args = ap.parse_args()

    method_filter: set[str] | None
    if args.methods:
        method_filter = set(args.methods)
    elif args.pilot or args.priority_methods:
        method_filter = set(PRIORITY_METHODS)
    else:
        method_filter = None

    plans = enumerate_plans(
        args.summaries_prefix, method_filter, results_prefix=args.results_prefix, samples_prefix=args.samples_prefix
    )
    if args.dedup_cells:
        plans = drop_rerun_variants(plans)
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
        if args.skip_existing and _gcs_exists(plan.scores_json_path):
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
                region_float=args.region_float,
                allow_eu_fallback=args.allow_eu_fallback,
                name_suffix=args.name_suffix,
                log_samples=args.log_samples,
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

        # Throttle: HF's Hub API caps at 1000 requests / 5 min per token, and each
        # eval makes several HF calls at cold-start (AutoConfig resolution + the
        # model-type probe's tokenizer fetch). Submitting the whole fleet at once
        # blows the cap and jobs die with "We had to rate limit you". Pausing
        # between waves keeps concurrent startups — and thus the HF request rate —
        # under the limit.
        if args.wave_size and len(submitted) % args.wave_size == 0:
            logger.info(
                "Wave of %d launched; pausing %.0fs to stay under HF rate limit...", args.wave_size, args.wave_delay
            )
            time.sleep(args.wave_delay)

    if args.launch:
        logger.info("Summary: submitted=%d skipped=%d incomplete=%d", len(submitted), len(skipped), len(incomplete))
        if submitted:
            logger.info("Job IDs:")
            for run_name, job_id in submitted:
                logger.info("  %s -> %s", job_id, run_name)

        # Keep-alive: children submitted via IrisClient from inside a parent job are
        # named as descendants of the parent (`/user/parent/child`) and are bound to
        # the parent's lifecycle — if this launcher returns while children are still
        # running, the controller reaps them as orphans. So block here until every
        # selected plan has produced its scores summary (or --keepalive-max elapses),
        # polling the results prefix. Set --no-keepalive to submit-and-exit (only safe
        # when children are top-level, e.g. direct-laptop mode).
        if args.keepalive:
            expected = {p.scores_json_path for p in plans}
            deadline = time.time() + args.keepalive_max
            logger.info(
                "Keep-alive: waiting for %d scores under %s (poll=%.0fs, max=%.0fs)",
                len(expected),
                args.results_prefix,
                args.keepalive_poll,
                args.keepalive_max,
            )
            while time.time() < deadline:
                present = {p for p in _gcs_ls(args.results_prefix) if p in expected}
                done = len(present)
                logger.info("Keep-alive: %d / %d scores present", done, len(expected))
                if done >= len(expected):
                    logger.info("Keep-alive: all scores present — exiting cleanly.")
                    break
                time.sleep(args.keepalive_poll)
            else:
                logger.warning("Keep-alive: deadline reached with %d / %d done — exiting.", done, len(expected))


if __name__ == "__main__":
    main()
