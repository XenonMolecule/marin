# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the OLMo Base-Easy bpb suite over an explicit checkpoint manifest.

The bpb analogue of `olmes_base.launch_olmes_manifest`: reads the same
(run_name, region, output_path, ...) CSV/TXT manifest and submits one iris child per
checkpoint running `run_olmo_bpb_eval.py`. Each child is HARD-pinned to its checkpoint's
region and points at that region's `olmo_in_loop_evals/` (oe-eval bpb requests) and the
shared `core_tasks_hub_cache/` (base HF configs) so weights + eval data read locally —
no cross-region egress, no HF Hub calls.

Results land IN-REGION at
`gs://<bucket>/metadata/olmo_bpb_results/<run_name>/results.json`.

Handoff (run via iris so the parent can submit children and keep them alive):

    iris --cluster marin job run -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.scaling_law_sweeps.olmo_bpb.launch_olmo_bpb_manifest \\
               --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_with_fastpipe_341.txt \\
               --launch

Smoke test (one checkpoint, tiny task cap):

    ... --manifest experiments/core_eval_manifests/olmo_bpb_SMOKE1.txt \\
        --launch --tasks lambada/bpb_0shot,gsm8k/gold_bpb_5shot --limit 8 --name-suffix -smoke
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, tpu_device
from iris.rpc import job_pb2

from experiments.scaling_law_sweeps.dclm_core.launch_dclm_core_sweep import (
    DEFAULT_TPU_VARIANTS,
    PRIORITY_BAND_MAP,
    _gcs_exists,
    _iris_client,
    _resolve_final_step,
)

logger = logging.getLogger(__name__)

RESULTS_SUBPATH = "metadata/olmo_bpb_results/{run_name}/"
CACHE_SUBPATH = "eval_datasets/olmo_in_loop_evals/"
HUB_CACHE_SUBPATH = "eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep


@dataclass(frozen=True)
class CheckpointRow:
    run_name: str
    region: str
    output_path: str  # gs://<bucket>/checkpoints/... (no /hf)
    # Templated `{run_name}` results dir, relative to the checkpoint's bucket. Overridable so a
    # side sweep (a different task subset, a reproducibility re-run) writes somewhere else
    # instead of overwriting the canonical `olmo_bpb_results/<run>/results.json`.
    results_subpath: str = RESULTS_SUBPATH

    @property
    def bucket(self) -> str:
        return self.output_path.split("/")[2]

    def hf_dir(self) -> str:
        return f"{self.output_path.rstrip('/')}/hf/"

    def output_dir(self) -> str:
        return f"gs://{self.bucket}/{self.results_subpath.format(run_name=self.run_name)}"

    def results_json(self) -> str:
        return f"{self.output_dir()}results.json"

    def cache_gcs(self) -> str:
        return f"gs://{self.bucket}/{CACHE_SUBPATH}"

    def hub_cache_gcs(self) -> str:
        return f"gs://{self.bucket}/{HUB_CACHE_SUBPATH}"


def rows_from_manifest(manifest_path: str, results_subpath: str = RESULTS_SUBPATH) -> list[CheckpointRow]:
    rows: list[CheckpointRow] = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                CheckpointRow(
                    run_name=row["run_name"].strip(),
                    region=row["region"].strip(),
                    output_path=row["output_path"].strip().rstrip("/"),
                    results_subpath=results_subpath,
                )
            )
    return rows


def submit_one(
    client,
    row: CheckpointRow,
    hf_step_dir: str,
    *,
    priority_band: int,
    wandb_api_key: str,
    hf_token: str,
    tasks: str,
    limit: int | None,
    name_suffix: str,
    memory_gb: int,
    merge: bool,
    done_marker: str | None,
    tpu_variants: tuple[str, ...],
    hub_offline: bool,
) -> str:
    cmd_args = [
        "python",
        "-m",
        "experiments.scaling_law_sweeps.olmo_bpb.run_olmo_bpb_eval",
        "--hf-checkpoint",
        hf_step_dir,
        "--output-dir",
        row.output_dir(),
        "--run-name",
        row.run_name,
        "--dataset-cache-gcs",
        row.cache_gcs(),
        "--hub-cache-gcs",
        row.hub_cache_gcs(),
        "--tasks",
        tasks,
    ]
    if limit is not None:
        cmd_args += ["--limit", str(limit)]
    if merge:
        cmd_args += ["--merge"]
    if done_marker:
        cmd_args += ["--done-marker", done_marker]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "HF_TOKEN": hf_token,
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        # Model/config/tokenizer load ONLINE (like CORE v2) — the eval container's transformers
        # can't reliably resolve a few repos (gpt2, mistral-regex fix) OFFLINE. HF_HOME points at
        # the staged hub cache (accelerator; cache hits avoid the Hub, misses fetch online).
        "HF_HOME": "/tmp/olmo_bpb_hf_home",
        "PYTHONUNBUFFERED": "1",
        "MARIN_MIRROR_BUDGET_GB": "25",
    }
    if hub_offline:
        # Zero Hub traffic: the eval requests are staged raw text and the checkpoint carries its
        # own config + tokenizer, so nothing here NEEDS the network. The historical blocker was
        # `HFCheckpointConverter.from_hf`, whose registry scan probes every registered config's
        # default repo and raises offline on a cache miss -- `run_olmo_bpb_eval` now catches that
        # and resolves LevConfig from the checkpoint's local config.json instead. Verify on one
        # checkpoint before a wave: an offline miss fails the child rather than falling back.
        env_vars["HF_HUB_OFFLINE"] = "1"
        env_vars["TRANSFORMERS_OFFLINE"] = "1"

    region_constraint = Constraint.create(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=[row.region],
        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
    )
    constraints = [preemptible_constraint(True), region_constraint]
    if len(set(tpu_variants)) > 1:
        constraints.append(device_variant_constraint(list(tpu_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"olmobpb-{row.run_name}{name_suffix}"[:200],
        resources=ResourceSpec(cpu=8, memory=f"{memory_gb}GB", disk="50GB", device=tpu_device(tpu_variants[0])),
        environment=EnvironmentSpec(extras=["tpu", "eval"], env_vars=env_vars),
        constraints=constraints,
        max_retries_preemption=20,
        max_retries_failure=5,
        priority_band=priority_band,
    )
    return str(job.job_id)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV/TXT with columns run_name,region,output_path,hf_dir.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--launch", action="store_true")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="Skip runs whose results.json exists.")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--child-priority", default="batch", choices=sorted(PRIORITY_BAND_MAP))
    ap.add_argument("--tasks", default="all", help="'all', 'qa_rc', 'all+qa_rc', or comma-separated dirs.")
    ap.add_argument("--limit", type=int, default=None, help="Cap each task to N docs (smoke test).")
    ap.add_argument("--merge", action="store_true", help="Child merges new tasks into existing results.json.")
    ap.add_argument(
        "--done-marker",
        default=None,
        help="Completion marker filename for keep-alive/skip in --merge mode (results.json pre-exists). "
        "e.g. _qa_rc.done",
    )
    ap.add_argument("--name-suffix", default="", help="Suffix on child job names to dodge JobAlreadyExists.")
    ap.add_argument(
        "--results-subpath",
        default=RESULTS_SUBPATH,
        help="Templated '{run_name}' results dir under the checkpoint's bucket. Point a side sweep "
        "elsewhere so it does not overwrite the canonical olmo_bpb_results/<run>/results.json.",
    )
    ap.add_argument("--max-count", type=int, default=None, help="Only submit the first N not-yet-done checkpoints.")
    ap.add_argument("--memory-gb", type=int, default=64, help="Child worker memory.")
    ap.add_argument(
        "--wave-size",
        type=int,
        default=8,
        help="Pause --wave-delay after this many launches. 0 = all at once. The HF Hub caps at "
        "1000 API requests / 5 min PER TOKEN and the model load is inherently online (levanter "
        "probes gpt2), so a big simultaneous launch rate-limits itself. Worse, it rate-limits "
        "TRAINING children too -- an unthrottled 138-child launch killed 2 runs out of the live "
        "sweep. Default 8 deliberately: this tool previously had no throttle at all.",
    )
    ap.add_argument("--wave-delay", type=float, default=420.0, help="Seconds between waves. Default 420.")
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
        "--hub-offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE on the children so they make NO Hub calls. "
        "The 1000-req/5-min cap is per TOKEN across every job -- eval waves have rate-limited "
        "live TRAINING children to death -- so going offline removes the whole failure mode "
        "rather than pacing around it. Requires the staged hub cache to cover every repo the "
        "load path touches; verify on one checkpoint first, because an offline cache miss is a "
        "hard failure instead of a silent online fetch.",
    )
    ap.add_argument("--keepalive-timeout", type=float, default=43200.0, help="Max seconds to hold the parent open.")
    ap.add_argument("--keepalive-poll", type=float, default=300.0, help="Seconds between keep-alive GCS polls.")
    args = ap.parse_args()

    rows = rows_from_manifest(args.manifest, results_subpath=args.results_subpath)
    logger.info("Loaded %d checkpoints from %s", len(rows), args.manifest)

    def completion_path(row: CheckpointRow) -> str:
        # In --merge mode results.json pre-exists, so completion is signalled by the done-marker.
        return f"{row.output_dir()}{args.done_marker}" if args.done_marker else row.results_json()

    client = wandb_api_key = hf_token = None
    if args.launch:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        hf_token = os.environ.get("HF_TOKEN")
        if not wandb_api_key or not hf_token:
            logger.error("--launch requires WANDB_API_KEY and HF_TOKEN in env.")
            sys.exit(2)
        client = _iris_client()

    submitted: list[tuple[str, str]] = []
    submitted_results: list[str] = []
    skipped: list[str] = []
    incomplete: list[str] = []
    # Kept separate from `incomplete` on purpose. "No HF checkpoint yet" is expected and
    # self-healing; "iris refused the submission" is a config error that will never heal and
    # silently drops the row from the wave. Conflating them hid 132 us-west4 evals -- every
    # one rejected for requesting a TPU variant that region does not have -- behind a count
    # that read as ordinary training lag.
    rejected: list[str] = []

    for i, row in enumerate(rows):
        if args.max_count is not None and len(submitted) >= args.max_count:
            logger.info("Reached --max-count=%d; stopping submission.", args.max_count)
            break
        if args.skip_existing and _gcs_exists(completion_path(row)):
            logger.info("[%3d] SKIP (exists): %s", i, row.run_name)
            skipped.append(row.run_name)
            continue

        hf_step_dir = _resolve_final_step(row.hf_dir())
        if hf_step_dir is None:
            logger.warning("[%3d] NO HF STEP under %s", i, row.hf_dir())
            incomplete.append(row.run_name)
            continue

        if args.dry_run:
            logger.info("[%3d] %s | region=%s | %s -> %s", i, row.run_name, row.region, hf_step_dir, row.results_json())
            continue

        try:
            job_id = submit_one(
                client,
                row,
                hf_step_dir,
                priority_band=PRIORITY_BAND_MAP[args.child_priority],
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                tasks=args.tasks,
                limit=args.limit,
                name_suffix=args.name_suffix,
                memory_gb=args.memory_gb,
                merge=args.merge,
                done_marker=args.done_marker,
                tpu_variants=tuple(v.strip() for v in args.tpu_variants.split(",") if v.strip()),
                hub_offline=args.hub_offline,
            )
        except Exception as e:
            logger.error("[%3d] SUBMIT REJECTED for %s (region=%s): %s", i, row.run_name, row.region, e)
            rejected.append(row.run_name)
            continue
        submitted.append((row.run_name, job_id))
        submitted_results.append(completion_path(row))
        logger.info("[%3d] LAUNCHED %s -> %s (region=%s)", i, row.run_name, job_id, row.region)

        if args.wave_size and len(submitted) % args.wave_size == 0:
            logger.info(
                "Wave of %d launched; pausing %.0fs to stay under the Hub rate limit...", args.wave_size, args.wave_delay
            )
            time.sleep(args.wave_delay)

    logger.info(
        "Summary: submitted=%d skipped=%d incomplete=%d rejected=%d",
        len(submitted),
        len(skipped),
        len(incomplete),
        len(rejected),
    )
    if incomplete:
        logger.warning("Incomplete (no HF checkpoint yet, will heal): %s", ", ".join(incomplete))
    if rejected:
        # Loud and per-region: a rejection is a config error, and the region is almost always
        # the discriminating fact (a variant the region lacks, a bucket it cannot read).
        by_region = Counter(r.region for r in rows if r.run_name in set(rejected))
        logger.error(
            "%d submissions were REJECTED and are NOT running. By region: %s. "
            "These rows will never complete until the cause is fixed and the wave re-run.",
            len(rejected),
            dict(by_region),
        )

    # Keep the parent alive until every child's results.json lands, so iris does not
    # orphan-kill the nested children when the parent exits.
    if args.launch and submitted_results:
        start = time.time()
        pending = set(submitted_results)
        logger.info(
            "Keep-alive: holding parent open for %d children (timeout %.1fh)...",
            len(pending),
            args.keepalive_timeout / 3600.0,
        )
        while pending and (time.time() - start) < args.keepalive_timeout:
            time.sleep(args.keepalive_poll)
            pending = {r for r in pending if not _gcs_exists(r)}
            logger.info(
                "Keep-alive: %d/%d results present (%.0f min)",
                len(submitted_results) - len(pending),
                len(submitted_results),
                (time.time() - start) / 60.0,
            )
        if pending:
            logger.warning("Keep-alive timed out; %d children missing results.", len(pending))
        else:
            logger.info("Keep-alive: all %d children produced results. Exiting.", len(submitted_results))


if __name__ == "__main__":
    main()
