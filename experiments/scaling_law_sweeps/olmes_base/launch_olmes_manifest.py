# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the OLMES base-easy suite over an explicit checkpoint manifest.

The OLMES analogue of `core_tasks.launch_core_tasks_manifest`: reads the same
(run_name, region, output_path, ...) CSV/TXT manifest and submits one iris child per
checkpoint running `run_olmes_eval.py`. Each child is HARD-pinned to its checkpoint's
region and points at that region's `olmes_base_hf_cache/` (datasets) and the shared
`core_tasks_hub_cache/` (model configs) so weights + eval data read locally — no
cross-region egress, no HF Hub calls.

Results land IN-REGION at
`gs://<bucket>/metadata/olmes_base_results/<run_name>/results.json`.

Handoff (run via iris so the parent can submit children and keep them alive):

    iris --cluster marin job run -e WANDB_API_KEY -e HF_TOKEN \\
        -- python -m experiments.scaling_law_sweeps.olmes_base.launch_olmes_manifest \\
               --manifest experiments/core_eval_manifests/checkpoint_manifest_10k_with_fastpipe_275.txt \\
               --launch
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
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

RESULTS_SUBPATH = "metadata/olmes_base_results/{run_name}/"
CACHE_SUBPATH = "eval_datasets/olmes_base_hf_cache/"
HUB_CACHE_SUBPATH = "eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep


@dataclass(frozen=True)
class CheckpointRow:
    run_name: str
    region: str
    output_path: str  # gs://<bucket>/checkpoints/... (no /hf)

    @property
    def bucket(self) -> str:
        return self.output_path.split("/")[2]

    def hf_dir(self) -> str:
        return f"{self.output_path.rstrip('/')}/hf/"

    def output_dir(self) -> str:
        return f"gs://{self.bucket}/{RESULTS_SUBPATH.format(run_name=self.run_name)}"

    def results_json(self) -> str:
        return f"{self.output_dir()}results.json"

    def cache_gcs(self) -> str:
        return f"gs://{self.bucket}/{CACHE_SUBPATH}"

    def hub_cache_gcs(self) -> str:
        return f"gs://{self.bucket}/{HUB_CACHE_SUBPATH}"


def rows_from_manifest(manifest_path: str) -> list[CheckpointRow]:
    rows: list[CheckpointRow] = []
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                CheckpointRow(
                    run_name=row["run_name"].strip(),
                    region=row["region"].strip(),
                    output_path=row["output_path"].strip().rstrip("/"),
                )
            )
    return rows


def submit_one(client, row: CheckpointRow, hf_step_dir: str, *, priority_band: int, wandb_api_key: str,
               hf_token: str, limit: int | None, name_suffix: str, memory_gb: int) -> str:
    cmd_args = [
        "python", "-m", "experiments.scaling_law_sweeps.olmes_base.run_olmes_eval",
        "--hf-checkpoint", hf_step_dir,
        "--output-dir", row.output_dir(),
        "--run-name", row.run_name,
        "--dataset-cache-gcs", row.cache_gcs(),
        "--hub-cache-gcs", row.hub_cache_gcs(),
    ]
    if limit is not None:
        cmd_args += ["--limit", str(limit)]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "HF_TOKEN": hf_token,
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        # Model/config/tokenizer load ONLINE (like CORE v2) — offline resolution of a few repos
        # (gpt2, mistral-regex fix) is unreliable in the eval container. Datasets stay offline
        # (runner sets HF_DATASETS_OFFLINE=1). HF_HOME points at the staged hub cache (accelerator).
        "HF_HOME": "/tmp/olmes_hf_home",
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        "MARIN_MIRROR_BUDGET_GB": "25",
    }

    region_constraint = Constraint.create(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=[row.region],
        mode=job_pb2.CONSTRAINT_MODE_REQUIRED,
    )
    constraints = [preemptible_constraint(True), region_constraint]
    if len(set(DEFAULT_TPU_VARIANTS)) > 1:
        constraints.append(device_variant_constraint(list(DEFAULT_TPU_VARIANTS)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"olmes-{row.run_name}{name_suffix}"[:200],
        resources=ResourceSpec(cpu=8, memory=f"{memory_gb}GB", disk="50GB", device=tpu_device(DEFAULT_TPU_VARIANTS[0])),
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
    ap.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke test).")
    ap.add_argument("--name-suffix", default="", help="Suffix on child job names to dodge JobAlreadyExists.")
    ap.add_argument("--max-count", type=int, default=None, help="Only submit the first N not-yet-done checkpoints.")
    ap.add_argument("--memory-gb", type=int, default=64, help="Child worker memory.")
    ap.add_argument("--keepalive-timeout", type=float, default=43200.0, help="Max seconds to hold the parent open.")
    ap.add_argument("--keepalive-poll", type=float, default=300.0, help="Seconds between keep-alive GCS polls.")
    args = ap.parse_args()

    rows = rows_from_manifest(args.manifest)
    logger.info("Loaded %d checkpoints from %s", len(rows), args.manifest)

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

    for i, row in enumerate(rows):
        if args.max_count is not None and len(submitted) >= args.max_count:
            logger.info("Reached --max-count=%d; stopping submission.", args.max_count)
            break
        if args.skip_existing and _gcs_exists(row.results_json()):
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
                client, row, hf_step_dir,
                priority_band=PRIORITY_BAND_MAP[args.child_priority],
                wandb_api_key=wandb_api_key, hf_token=hf_token,
                limit=args.limit, name_suffix=args.name_suffix, memory_gb=args.memory_gb,
            )
        except Exception as e:
            logger.error("[%3d] SUBMIT FAILED for %s: %s", i, row.run_name, e)
            incomplete.append(row.run_name)
            continue
        submitted.append((row.run_name, job_id))
        submitted_results.append(row.results_json())
        logger.info("[%3d] LAUNCHED %s -> %s (region=%s)", i, row.run_name, job_id, row.region)

    logger.info("Summary: submitted=%d skipped=%d incomplete=%d", len(submitted), len(skipped), len(incomplete))
    if incomplete:
        logger.warning("Incomplete: %s", ", ".join(incomplete))

    # Keep the parent alive until every child's results.json lands, so iris does not
    # orphan-kill the nested children when the parent exits.
    if args.launch and submitted_results:
        start = time.time()
        pending = set(submitted_results)
        logger.info("Keep-alive: holding parent open for %d children (timeout %.1fh)...",
                    len(pending), args.keepalive_timeout / 3600.0)
        while pending and (time.time() - start) < args.keepalive_timeout:
            time.sleep(args.keepalive_poll)
            pending = {r for r in pending if not _gcs_exists(r)}
            logger.info("Keep-alive: %d/%d results present (%.0f min)",
                        len(submitted_results) - len(pending), len(submitted_results), (time.time() - start) / 60.0)
        if pending:
            logger.warning("Keep-alive timed out; %d children missing results.", len(pending))
        else:
            logger.info("Keep-alive: all %d children produced results. Exiting.", len(submitted_results))


if __name__ == "__main__":
    main()
