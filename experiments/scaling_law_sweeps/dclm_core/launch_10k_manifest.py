# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch DCLM CORE (22-task, Core_v2) evals over an explicit checkpoint manifest.

Sibling of ``launch_dclm_core_sweep.py``. That launcher re-enumerates checkpoints
from the Fixed-Model summary JSONs and applies its own cell-dedup; this one instead
reads an explicit, already-curated CSV manifest (run_name, region, output_path,
hf_dir) so the exact set the user picked is what runs — no re-derivation.

Everything downstream is reused verbatim from ``launch_dclm_core_sweep``:
  * ``_resolve_final_step`` picks the largest step-N under each ``hf_dir``.
  * ``submit_one`` HARD-pins each child to its checkpoint's region (zero
    cross-region reads), points it at the byte-identical offline HF dataset cache
    in that region's bucket (``gs://<bucket>/eval_datasets/dclm_core_hf_cache/``)
    so task loading never hits the HF Hub, and runs ``run_dclm_core_eval.py``.
  * Wave throttling keeps concurrent cold-starts under the Hub's rate window
    (a belt-and-suspenders guard; the offline cache already avoids Hub calls).

Results are written IN-REGION (each checkpoint's own bucket) to avoid cross-region
write egress: a permanent per-run scores summary plus a ttl= full-result sibling.
The 245 tiny scores summaries can be gathered centrally afterward (KBs each).

## Handoff one-liner (run from any laptop with iris auth)

    iris --cluster marin job run \\
        -e WANDB_API_KEY -e HF_TOKEN \\
        -- python -m experiments.scaling_law_sweeps.dclm_core.launch_10k_manifest \\
               --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_dedup245.csv \\
               --launch --wave-size 40
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time

from experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep import (
    DEFAULT_TPU_VARIANTS,
    PRIORITY_BAND_MAP,
    CheckpointPlan,
    _gcs_exists,
    _iris_client,
    _resolve_final_step,
    submit_one,
)

logger = logging.getLogger(__name__)

# Permanent per-run scores summaries and ttl full-results both land in the
# checkpoint's OWN bucket so every read and write stays in-region.
SCORES_SUBPATH = "metadata/data_curation_10k_core_results/{run_name}_summary.json"
SAMPLES_SUBPATH = "tmp/ttl=30d/dclm_10k_core/{run_name}.json"


def plans_from_manifest(
    manifest_path: str,
    scores_subpath: str = SCORES_SUBPATH,
    samples_subpath: str = SAMPLES_SUBPATH,
) -> list[CheckpointPlan]:
    """Build CheckpointPlans from a run_name,region,output_path,hf_dir CSV.

    method / experiment_tag / budget_flops / hidden_dim are not used by
    ``submit_one`` (they only drive the sweep launcher's dedup/sort) and are left
    as inert defaults. output_path and region are the fields that matter here.
    """
    plans: list[CheckpointPlan] = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            run_name = row["run_name"].strip()
            output_path = row["output_path"].strip().rstrip("/")
            bucket = output_path.split("/")[2]  # gs://<bucket>/...
            plans.append(
                CheckpointPlan(
                    run_name=run_name,
                    method="",
                    experiment_tag="",
                    budget_flops=0.0,
                    hidden_dim=0,
                    region=row["region"].strip(),
                    output_path=output_path,
                    summary_path=manifest_path,
                    output_json_path=f"gs://{bucket}/{samples_subpath.format(run_name=run_name)}",
                    scores_json_path=f"gs://{bucket}/{scores_subpath.format(run_name=run_name)}",
                )
            )
    return plans


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with columns run_name,region,output_path,hf_dir.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print resolved plans + HF step dirs, don't launch.")
    mode.add_argument("--launch", action="store_true", help="Submit an Iris child per checkpoint.")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="Skip runs whose scores JSON exists.")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--child-priority", default="batch", choices=sorted(PRIORITY_BAND_MAP))
    ap.add_argument("--non-preemptible", action="store_true")
    ap.add_argument("--log-samples", action="store_true", help="Persist per-example samples (big). Default off.")
    ap.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke testing).")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument(
        "--memory-gb",
        type=int,
        default=192,
        help="Child RAM request. 192 (default) EXCEEDS what a v5e VM can offer (192GB total), "
        "so v5e-only regions such as us-west4 can never schedule it. Use ~64 to run there.",
    )
    ap.add_argument(
        "--per-device-eval-parallelism",
        type=int,
        default=1,
        help="Sequences scored per device per step (effective eval batch = this x 4 on a 4-chip "
        "slice). Scores are unaffected; only throughput and memory. Raise for small checkpoints.",
    )
    ap.add_argument(
        "--hub-offline-after-load",
        action="store_true",
        help="Children set HF_HUB_OFFLINE=1 after model load so the task loop never reaches the "
        "Hub. Mitigates the 'Failed to load task' deaths caused by sustained 429s at fleet scale.",
    )
    ap.add_argument("--name-suffix", default="", help="Suffix on child job names to dodge JobAlreadyExists.")
    ap.add_argument(
        "--scores-subpath",
        default=SCORES_SUBPATH,
        help="Templated '{run_name}' scores-summary path under the checkpoint's bucket. Point a "
        "separate fleet elsewhere so it does not collide with the canonical curation results.",
    )
    ap.add_argument(
        "--samples-subpath",
        default=SAMPLES_SUBPATH,
        help="Templated '{run_name}' full-result path under the checkpoint's bucket.",
    )
    ap.add_argument(
        "--tpu-variants",
        default=",".join(DEFAULT_TPU_VARIANTS),
        help="Comma-separated TPU variants for the children. The default exists in us-east5 / "
        "us-central1 / europe-west4 but NOT in us-west4, which only has v5litepod-*. A child "
        "asking for a variant its pinned region lacks is rejected at SUBMIT time, so a "
        "us-west4 manifest must pass e.g. 'v5litepod-4'. Applies to the whole wave, so keep "
        "one wave per region when the hardware differs.",
    )
    ap.add_argument(
        "--wave-size",
        type=int,
        default=40,
        help="After this many launches, pause --wave-delay. 0 = all at once. Default 40.",
    )
    ap.add_argument("--wave-delay", type=float, default=330.0, help="Seconds between waves. Default 330.")
    ap.add_argument(
        "--keepalive-timeout",
        type=float,
        default=43200.0,
        help="Max seconds to hold the parent open waiting for children to finish (default 12h). "
        "Prevents iris from orphan-killing nested children when the parent exits.",
    )
    ap.add_argument("--keepalive-poll", type=float, default=300.0, help="Seconds between keep-alive GCS polls.")
    args = ap.parse_args()

    tpu_variants = tuple(v.strip() for v in args.tpu_variants.split(",") if v.strip())
    if not tpu_variants:
        logger.error("--tpu-variants resolved to an empty list.")
        sys.exit(2)

    plans = plans_from_manifest(args.manifest, args.scores_subpath, args.samples_subpath)
    logger.info("Loaded %d checkpoint plans from %s", len(plans), args.manifest)

    # us-west4 has only v5litepod-*; a child asking for an absent variant is rejected at
    # submit time and lands in `incomplete`, which normally means the benign "HF export
    # not written yet". Say so loudly instead.
    n_usw4 = sum(1 for p in plans if p.region == "us-west4")
    if n_usw4 and not any(v.startswith("v5litepod") for v in tpu_variants):
        logger.warning(
            "Manifest has %d us-west4 checkpoints but --tpu-variants=%s has no v5litepod-*; "
            "those children will be REJECTED at submit time.",
            n_usw4,
            args.tpu_variants,
        )

    client = wandb_api_key = hf_token = None
    if args.launch:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        if not wandb_api_key or not hf_token:
            logger.error("--launch requires WANDB_API_KEY and HF_TOKEN in env.")
            sys.exit(2)
        client = _iris_client()

    submitted: list[tuple[str, str]] = []
    submitted_scores: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []

    for i, plan in enumerate(plans):
        if args.skip_existing and _gcs_exists(plan.scores_json_path):
            logger.info("[%3d] SKIP (exists): %s", i, plan.run_name)
            skipped.append(plan.run_name)
            continue

        hf_step_dir = _resolve_final_step(plan.hf_dir())
        if hf_step_dir is None:
            logger.warning("[%3d] NO HF STEP found under %s", i, plan.hf_dir())
            incomplete.append(plan.run_name)
            continue

        if args.dry_run:
            logger.info("[%3d] %s | region=%s | %s", i, plan.run_name, plan.region, hf_step_dir)
            continue

        try:
            job_id = submit_one(
                client,
                plan,
                hf_step_dir,
                priority_band=PRIORITY_BAND_MAP[args.child_priority],
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                tpu_variants=tpu_variants,
                preemptible=not args.non_preemptible,
                max_length=args.max_length,
                per_device_eval_parallelism=args.per_device_eval_parallelism,
                hub_offline_after_load=args.hub_offline_after_load,
                memory_gb=args.memory_gb,
                limit=args.limit,
                name_suffix=args.name_suffix,
                log_samples=args.log_samples,
            )
        except Exception as e:
            logger.error("[%3d] SUBMIT FAILED for %s: %s", i, plan.run_name, e)
            incomplete.append(plan.run_name)
            continue
        submitted.append((plan.run_name, job_id))
        submitted_scores.append(plan.scores_json_path)
        logger.info("[%3d] LAUNCHED %s -> %s (region=%s)", i, plan.run_name, job_id, plan.region)

        if args.wave_size and len(submitted) % args.wave_size == 0:
            logger.info("Wave of %d launched; pausing %.0fs...", args.wave_size, args.wave_delay)
            time.sleep(args.wave_delay)

    logger.info("Summary: submitted=%d skipped=%d incomplete=%d", len(submitted), len(skipped), len(incomplete))
    if incomplete:
        logger.warning("Incomplete (no HF step / submit failed): %s", ", ".join(incomplete))

    # Keep the parent alive until every child's scores JSON lands. When this
    # launcher runs as an iris parent job, children are submitted NESTED under it;
    # if the parent exits, iris finalizes and orphan-KILLS any child that hasn't
    # detached yet. Hold the parent open (polling GCS for the small scores
    # summaries) so children survive, then exit cleanly once all are done or the
    # timeout elapses. Children are region-pinned + skip-if-done, so a stopped and
    # resubmitted parent resumes without redoing finished work.
    if args.launch and submitted_scores:
        start = time.time()
        pending = set(submitted_scores)
        logger.info(
            "Keep-alive: holding parent open until %d children finish (timeout %.1fh)...",
            len(pending),
            args.keepalive_timeout / 3600.0,
        )
        while pending and (time.time() - start) < args.keepalive_timeout:
            time.sleep(args.keepalive_poll)
            pending = {s for s in pending if not _gcs_exists(s)}
            logger.info(
                "Keep-alive: %d/%d scores present (%.0f min elapsed)",
                len(submitted_scores) - len(pending),
                len(submitted_scores),
                (time.time() - start) / 60.0,
            )
        if pending:
            logger.warning(
                "Keep-alive timed out with %d children still missing scores: %s",
                len(pending),
                ", ".join(sorted(pending)),
            )
        else:
            logger.info("Keep-alive: all %d children produced scores. Exiting.", len(submitted_scores))


if __name__ == "__main__":
    main()
