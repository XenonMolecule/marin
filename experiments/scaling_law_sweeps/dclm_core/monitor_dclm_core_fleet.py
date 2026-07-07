# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent coordinator: launch + babysit all DCLM CORE 10k-natural evals.

Runs as a LONG-LIVED Iris job. It must stay alive for the whole sweep because the
eval children it submits are nested under it — if the coordinator exits, the
controller orphan-kills them (the hard-won coordinator-keep-alive lesson).

Each cycle:
  1. Enumerate the canonical runs (enumerate_plans + drop_rerun_variants).
  2. done = runs whose final JSON already exists on GCS.
  3. live = runs with a running/pending/scheduled Iris job this coordinator owns
     (job name contains ``{run_name}-m``).
  4. needy = planned - done - live  (never-launched OR terminal-failed).
  5. Resubmit each needy run with a fresh monotonic suffix ``-m{cycle}`` so it
     never collides with a dead job of the same run.
  6. Log a status line; sleep; repeat until every run is done, then report a few
     more cycles and exit.

Datasets are read offline from the pre-mirrored in-region cache (submit_one wires
``--dataset-cache-gcs``), so the only residual HF calls are model-config loads;
the first cycle throttles submission in waves to stay under HF's rate limit.

Launch (direct laptop mode OR as an interactive Iris job that stays up):

    iris --cluster marin job run --region us-central1 --priority interactive \\
        --cpu 2 --memory 8GB --disk 20GB --enable-extra-resources \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python -m experiments.scaling_law_sweeps.dclm_core.monitor_dclm_core_fleet \\
               --summaries-prefix gs://marin-us-central1/metadata/data_curation_10k_natural_results/ \\
               --results-prefix  gs://marin-us-central1/metadata/data_curation_10k_core_results/
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep import (
    CORE_RESULTS_PREFIX,
    FM_SUMMARIES_PREFIX,
    PRIORITY_BAND_MAP,
    CheckpointPlan,
    _gcs_exists,
    _iris_client,
    _resolve_final_step,
    drop_rerun_variants,
    enumerate_plans,
    filter_to_cells,
    submit_one,
)

logger = logging.getLogger(__name__)

_LIVE_STATES = {
    job_pb2.JOB_STATE_RUNNING,
    job_pb2.JOB_STATE_PENDING,
    job_pb2.JOB_STATE_BUILDING,
}
_JOB_PREFIX = "/michaelryan/dclm-core"


def _live_run_names(client, prefix: JobName) -> set[str]:
    """Run names that currently have a coordinator-owned live job.

    Children are submitted in parent-job mode, so their job_ids are NESTED under
    this coordinator's own name (``/<coord>/dclm-core-{run_name}-m{cycle}``) — we
    must list under the coordinator's own prefix, not a top-level one. The
    ``{run_name}-m`` boundary won't false-match a sibling cell (…-B64 vs …-B128)."""
    live: set[str] = set()
    jobs = client.list_jobs(prefix=prefix)
    for status in jobs:
        if status.state not in _LIVE_STATES:
            continue
        job_id = status.job_id
        marker = job_id.split("dclm-core-", 1)[-1]
        # marker == "{run_name}-m{cycle}"; strip the -m{cycle} tail.
        if "-m" in marker:
            live.add(marker.rsplit("-m", 1)[0])
    return live


def _resubmit(client, plan: CheckpointPlan, cycle: int, creds: dict) -> str | None:
    hf_step_dir = _resolve_final_step(plan.hf_dir())
    if hf_step_dir is None:
        logger.warning("no hf/step dir for %s — skipping", plan.run_name)
        return None
    try:
        return submit_one(
            client,
            plan,
            hf_step_dir,
            priority_band=PRIORITY_BAND_MAP["batch"],
            wandb_api_key=creds["wandb"],
            hf_token=creds["hf"],
            name_suffix=f"-m{cycle}",
        )
    except Exception as e:
        logger.error("resubmit failed for %s: %s", plan.run_name, e)
        return None


def run(
    summaries_prefix: str,
    results_prefix: str,
    *,
    samples_prefix: str | None,
    cells: list[str] | None,
    poll_interval: float,
    wave_size: int,
    wave_delay: float,
    max_in_flight: int,
    max_cycles: int,
) -> None:
    creds = {"wandb": os.environ["WANDB_API_KEY"], "hf": os.environ["HF_TOKEN"]}
    client = _iris_client()

    # Children nest under this coordinator's own job name in parent-job mode, so
    # live-detection must query that prefix. Fall back to the top-level prefix
    # when running outside an iris job (direct/laptop mode).
    info = get_job_info()
    own_prefix = info.task_id.require_task()[0] if info is not None else JobName.from_string(_JOB_PREFIX)
    logger.info("Live-detection prefix: %s", own_prefix.to_wire())

    plans = drop_rerun_variants(
        enumerate_plans(summaries_prefix, None, results_prefix=results_prefix, samples_prefix=samples_prefix)
    )
    plans = filter_to_cells(plans, cells or [])
    logger.info("Babysitting %d canonical runs", len(plans))

    cycle = 0
    done_streak = 0
    while cycle < max_cycles:
        # Done-marker is the PERMANENT scores summary (not the ttl= final), so a run
        # stays "done" even after its sample-heavy final auto-expires.
        done = {p.run_name for p in plans if _gcs_exists(p.scores_json_path)}
        live = _live_run_names(client, own_prefix)
        needy = [p for p in plans if p.run_name not in done and p.run_name not in live]

        # CONCURRENCY CAP: keep at most max_in_flight jobs alive at once. Even with
        # datasets offline, each cold-start makes ~4 model-config HF calls; running
        # all 245 at once keeps HF's 1000-req/5min limit saturated so nothing gets
        # through. Capping to ~50 keeps the call rate well under the limit — jobs
        # start clean, complete, and free slots for the next batch. All 245 still
        # get driven to done, just pipelined through the cap.
        budget = max(0, max_in_flight - len(live))
        to_submit = needy[:budget]

        logger.info(
            "cycle %d: done=%d live=%d needy=%d cap=%d submitting=%d / %d",
            cycle,
            len(done),
            len(live),
            len(needy),
            max_in_flight,
            len(to_submit),
            len(plans),
        )

        if not needy and len(done) == len(plans):
            logger.info("ALL %d runs complete.", len(plans))
            done_streak += 1
            if done_streak >= 2:
                return
        else:
            done_streak = 0

        for i, plan in enumerate(to_submit):
            job_id = _resubmit(client, plan, cycle, creds)
            if job_id:
                logger.info("  resubmitted %s -> %s", plan.run_name, job_id)
            # Stagger even within the cap so the batch's model-config calls don't
            # all land in the same instant.
            if wave_size and (i + 1) % wave_size == 0 and i + 1 < len(to_submit):
                logger.info("  wave of %d submitted; pausing %.0fs", wave_size, wave_delay)
                time.sleep(wave_delay)

        cycle += 1
        time.sleep(poll_interval)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-prefix", default=FM_SUMMARIES_PREFIX)
    ap.add_argument("--results-prefix", default=CORE_RESULTS_PREFIX, help="PERMANENT prefix for scores summaries.")
    ap.add_argument(
        "--samples-prefix",
        default=None,
        help="ttl= prefix for the sample-heavy finals/partials (defaults to --results-prefix).",
    )
    ap.add_argument("--poll-interval", type=float, default=600.0, help="Seconds between babysit cycles.")
    ap.add_argument("--wave-size", type=int, default=25, help="Within a cycle, submit N, pause --wave-delay, repeat.")
    ap.add_argument("--wave-delay", type=float, default=300.0, help="Seconds to pause between waves.")
    ap.add_argument(
        "--max-in-flight",
        type=int,
        default=50,
        help="Max jobs alive at once. Caps concurrent cold-start model-config HF calls under the "
        "1000-req/5min limit; all 245 still get driven to done, pipelined through the cap.",
    )
    ap.add_argument("--max-cycles", type=int, default=500, help="Safety cap on babysit cycles.")
    ap.add_argument(
        "--cells",
        default=None,
        help="Comma-separated <budget>:<width> cells to restrict the sweep to (e.g. "
        "'9e+17:1024,3e+18:1536'). Runs only those isoflop cells instead of the full sweep.",
    )
    args = ap.parse_args()

    run(
        args.summaries_prefix,
        args.results_prefix,
        samples_prefix=args.samples_prefix,
        cells=[c.strip() for c in args.cells.split(",")] if args.cells else None,
        poll_interval=args.poll_interval,
        wave_size=args.wave_size,
        wave_delay=args.wave_delay,
        max_in_flight=args.max_in_flight,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
