# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris launcher: run LIMA eval on every completed expFM fixed-model run.

For each training-summary JSON under
`gs://marin-us-central1/metadata/data_curation_fixed_model_results/`:

1. Scan all 5 marin-region buckets for `checkpoints/isoflop-curation/<run_name>/`.
   The first bucket that contains a `step-*` directory is considered the
   checkpoint's home region.
2. Submit a one-shot Iris child in that region (HARD `--region` constraint) to
   run `run_lima_eval_standalone.py`. The child is single-host — LIMA is tiny
   and one forward pass fits on any v*-8 TPU.
3. Default `--child-priority batch` so evals yield to training. Flexible TPU
   type via `--tpu-types v4-8,v5e-8,v5p-8,v5litepod-8` — Iris picks whichever
   is free.
4. Results land at
   `gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/<run>.json`
   (side-car file so we don't race with training-time summary writes; plotter
   merges them).

Usage (submit a parent that fans out children):

    iris --cluster marin job run --priority production --no-wait \\
        --memory 2GB --cpu 2 --job-name lima-eval-coordinator \\
        --extra marin:cpu -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/scaling_law_sweeps/launch_lima_eval.py

    # Or to target a subset:
    python experiments/scaling_law_sweeps/launch_lima_eval.py \\
        --filter-name-contains d1536 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# All regions we've ever trained in (see region_tracker.REGION_TO_BUCKET).
# Regions we're willing to schedule single-host LIMA eval children in.
# Full empirical mapping below (see REGION_TO_TPU for each region's v*-8 variant).
CANDIDATE_REGIONS: tuple[str, ...] = (
    "us-central1", "us-east5", "us-central2", "us-east1", "europe-west4",
)

# Summary dir — where the training summaries land. We read this to enumerate
# completed runs (each summary implies one final checkpoint).
DEFAULT_SUMMARIES_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"

# Output dir for eval results. Separate dir so we don't RMW the training
# summary JSONs (avoids any write-race risk).
DEFAULT_EVAL_OUTPUT_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/"

# TPU shapes the eval child can run on. Any single-host v*-8 works; LIMA is
# 1330 docs so inference is trivial even at 998M params.
DEFAULT_TPU_TYPES: tuple[str, ...] = ("v5p-8", "v4-8", "v5e-8", "v5litepod-8")

# Per-region TPU variant. Each GCP region only has a subset of TPU generations
# physically provisioned; we pick one known-available variant per region so
# the scheduler can place children without "no groups in region" errors.
# Empirically verified from running training + LIMA eval on each region.
REGION_TO_TPU: dict[str, str] = {
    "us-central1": "v5p-8",
    "us-east5": "v5p-8",
    "us-central2": "v4-8",
    "us-east1": "v6e-8",
    "europe-west4": "v6e-8",
}

SCRIPT = "experiments/scaling_law_sweeps/run_lima_eval_standalone.py"

PRIORITY_BAND_MAP = {
    "production": "PRIORITY_BAND_PRODUCTION",
    "interactive": "PRIORITY_BAND_INTERACTIVE",
    "batch": "PRIORITY_BAND_BATCH",
}


@dataclass(frozen=True)
class RunToEval:
    run_name: str
    region: str  # where the checkpoint lives


def _list_summaries(prefix: str) -> list[str]:
    import fsspec

    fs, _ = fsspec.core.url_to_fs(prefix)
    entries = fs.ls(prefix.rstrip("/"))
    run_names = []
    for e in entries:
        if not e.endswith(".json"):
            continue
        # Strip the bucket path and .json, e.g.
        # "gs://.../curation-dclm-...-B256.json" -> "curation-dclm-...-B256"
        name = e.rsplit("/", 1)[-1][:-5]
        run_names.append(name)
    return sorted(run_names)


def _find_checkpoint_region(run_name: str) -> str | None:
    """Return the region whose checkpoint dir contains the HIGHEST step.

    Training runs sometimes flip between regions (preempted in one, resumed
    in another), so the run_name exists in multiple regions but only ONE of
    them has the true final checkpoint. Picking the first-match region is
    wrong — we'd eval on an early partial checkpoint and get garbage loss.

    Strategy: probe every candidate region, collect (max_step, region) pairs,
    return the region whose max_step is largest. Tie-break doesn't matter
    (identical checkpoints).
    """
    import fsspec
    import re

    from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

    best: tuple[int, str] | None = None
    for region in CANDIDATE_REGIONS:
        bucket = REGION_TO_BUCKET[region]
        path = f"{bucket}/checkpoints/isoflop-curation/{run_name}/checkpoints/"
        fs, _ = fsspec.core.url_to_fs(path)
        try:
            if not fs.exists(path):
                continue
            entries = fs.ls(path)
            steps = []
            for e in entries:
                mm = re.search(r"step-(\d+)/?$", e)
                if mm:
                    steps.append(int(mm.group(1)))
            if not steps:
                continue
            max_step = max(steps)
            if best is None or max_step > best[0]:
                best = (max_step, region)
        except Exception as e:
            logger.debug("Probe %s failed: %s", path, e)
    return best[1] if best else None


def _regions_with_lima_cache(lima_hash: str) -> set[str]:
    """Return the set of regions whose bucket has a valid LIMA cache.

    Safeguard: the launcher must not submit eval children to regions that
    don't have the LIMA cache yet -- those children would hit preflight and
    fail fast, wasting TPU allocation time.
    """
    import fsspec

    from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

    ready: set[str] = set()
    for region in CANDIDATE_REGIONS:
        bucket = REGION_TO_BUCKET[region]
        stats = f"{bucket}/tokenized/lima_text-{lima_hash}/validation/.stats.json"
        fs, _ = fsspec.core.url_to_fs(stats)
        try:
            if fs.exists(stats):
                ready.add(region)
        except Exception as e:
            logger.debug("LIMA probe %s failed: %s", stats, e)
    return ready


def _already_evaluated(run_name: str, output_prefix: str) -> bool:
    import fsspec

    path = f"{output_prefix.rstrip('/')}/{run_name}.json"
    try:
        fs, _ = fsspec.core.url_to_fs(path)
        return fs.exists(path)
    except Exception:
        return False


def _submit_one(
    client,
    run: RunToEval,
    *,
    priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    tpu_types: list[str],
    memory: str,
    output_prefix: str,
    lima_hash: str,
) -> str:
    """Submit one LIMA-eval child pinned to the checkpoint's region.

    Iris `CoschedulingConfig` not needed — single-host eval.
    """
    from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device

    # Child derives its read-region from MARIN_PREFIX (auto-set by iris
    # based on where the worker landed). We enforce region via the HARD
    # Iris `Constraint(REGION IN ...)` below, so MARIN_PREFIX will match.
    cmd = [
        "python",
        SCRIPT,
        "--run-name",
        run.run_name,
        "--output-prefix",
        output_prefix,
        "--lima-hash",
        lima_hash,
    ]
    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    constraints = [
        Constraint(
            key=WellKnownAttribute.REGION,
            op=ConstraintOp.IN,
            values=(run.region,),
        ),
    ]
    # Pick the TPU variant for this region (each region only has a subset
    # of TPU generations physically provisioned).
    primary_tpu = REGION_TO_TPU.get(run.region, tpu_types[0])
    resources = ResourceSpec(cpu=4.0, memory=memory, disk="50GB", device=tpu_device(primary_tpu))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"lima-eval-{run.run_name[:60]}",
        resources=resources,
        environment=EnvironmentSpec(extras=["tpu"], env_vars=env_vars),
        constraints=constraints,
        max_retries_preemption=20,
        max_retries_failure=2,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summaries-prefix", default=DEFAULT_SUMMARIES_PREFIX)
    p.add_argument("--output-prefix", default=DEFAULT_EVAL_OUTPUT_PREFIX)
    p.add_argument("--lima-hash", default="41ca0d")
    p.add_argument(
        "--filter-name-contains",
        nargs="*",
        default=[],
        help=(
            "Only evaluate runs whose run_name contains ALL of these substrings (AND). "
            "E.g., '--filter-name-contains resiliparse d1536' targets Resiliparse 998M only."
        ),
    )
    p.add_argument(
        "--skip-already-done",
        action="store_true",
        default=True,
        help="Skip runs whose eval-result file already exists (default: on).",
    )
    p.add_argument(
        "--no-skip-already-done",
        dest="skip_already_done",
        action="store_false",
        help="Disable skip — re-run all evals.",
    )
    p.add_argument(
        "--child-priority",
        choices=list(PRIORITY_BAND_MAP.keys()),
        default="batch",
        help="Priority band for eval children. Default batch so evals yield to training.",
    )
    p.add_argument("--memory", default="64GB")
    p.add_argument("--tpu-types", nargs="+", default=list(DEFAULT_TPU_TYPES))
    p.add_argument("--max-count", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Skip the post-submit keep-alive sleep (default: coordinator sleeps forever).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    from iris.client.client import IrisClient
    from iris.rpc import job_pb2

    run_names = _list_summaries(args.summaries_prefix)
    logger.info("Discovered %d completed runs under %s", len(run_names), args.summaries_prefix)

    if args.filter_name_contains:
        before = len(run_names)
        for needle in args.filter_name_contains:
            run_names = [r for r in run_names if needle in r]
        logger.info("After AND-filter %s: %d runs (was %d)", args.filter_name_contains, len(run_names), before)

    # For each run, find which region has its checkpoint + whether already evaluated.
    to_eval: list[RunToEval] = []
    skipped_already = 0
    skipped_no_ckpt = 0
    for r in run_names:
        if args.skip_already_done and _already_evaluated(r, args.output_prefix):
            skipped_already += 1
            continue
        region = _find_checkpoint_region(r)
        if region is None:
            skipped_no_ckpt += 1
            logger.warning("No checkpoint found across regions for %s — skipping", r)
            continue
        to_eval.append(RunToEval(run_name=r, region=region))

    logger.info(
        "Plan: %d runs to eval (%d skipped as already-done, %d skipped due to missing checkpoint)",
        len(to_eval),
        skipped_already,
        skipped_no_ckpt,
    )
    if args.max_count is not None:
        to_eval = to_eval[: args.max_count]
        logger.info("Capped to --max-count=%d", args.max_count)

    # Summarize region breakdown.
    from collections import Counter

    region_counts = Counter(r.region for r in to_eval)
    logger.info("Region breakdown: %s", dict(region_counts))

    if args.dry_run:
        for r in to_eval[:10]:
            logger.info("  DRY %s → region=%s", r.run_name, r.region)
        if len(to_eval) > 10:
            logger.info("  ... and %d more", len(to_eval) - 10)
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set. Must run this inside an Iris parent.")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY required.")
    hf_token = os.environ.get("HF_TOKEN")

    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }[args.child_priority]

    submitted: list[tuple[str, str, str]] = []
    for r in to_eval:
        try:
            jid = _submit_one(
                client,
                r,
                priority_band=priority_band,
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                tpu_types=args.tpu_types,
                memory=args.memory,
                output_prefix=args.output_prefix,
                lima_hash=args.lima_hash,
            )
            submitted.append((r.run_name, r.region, jid))
            logger.info("submitted %s (region=%s) → %s", r.run_name, r.region, jid)
        except Exception as e:
            logger.exception("failed to submit %s: %s", r.run_name, e)

    logger.info("Total submitted: %d / %d", len(submitted), len(to_eval))

    if not args.no_keep_alive:
        logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
