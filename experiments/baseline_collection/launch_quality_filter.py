# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris launcher for the region-local Nemotron quality-filter + tokenize pipeline.

Fires one CPU Iris job per (region, preset) combination. Each child job runs
``run_quality_filter_region.py`` locally in its region, reading from the
pre-mirrored ``filtered/baseline_nemotron_full-347dfe/`` and writing filtered
JSONL + tokenized cache to the same region's bucket. Zero cross-region I/O.

Typical invocation from an Iris parent or from a laptop with Iris CLI:

    iris --cluster marin job run --priority production --no-wait \\
        --memory 2GB --cpu 2 --job-name quality-filter-coordinator \\
        -e WANDB_API_KEY ... -e HF_TOKEN ... \\
        -- python experiments/baseline_collection/launch_quality_filter.py \\
        --presets high medplus --regions us-central1 us-east1 us-east5 eu-west4 \\
        --output-hash v1

``--output-hash`` is applied uniformly across regions so all output dirs share
the same name (``filtered/baseline_nemotron_qhigh-v1/`` etc.) -- this is what
enables the determinism-check (sha256-diffing shard files across regions).
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    preemptible_constraint,
)
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

SCRIPT = "experiments/baseline_collection/run_quality_filter_region.py"

# Iris region labels (what the scheduler expects). Note europe-west4 uses
# the full label even though its bucket is gs://marin-eu-west4.
KNOWN_REGIONS = ("us-central1", "us-central2", "us-east1", "us-east5", "europe-west4")
KNOWN_PRESETS = ("high", "medplus")

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
}


def submit_one(
    client: IrisClient,
    region: str,
    preset: str,
    output_hash: str,
    *,
    priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    cpu: float = 8.0,
    memory: str = "32GB",
    disk: str = "100GB",
    skip_filter: bool = False,
    skip_tokenize: bool = False,
) -> str:
    """Submit a single (region, preset) CPU job. Returns the Iris job id."""
    cmd = [
        "python",
        SCRIPT,
        "--preset",
        preset,
        "--region",
        region,
        "--output-hash",
        output_hash,
    ]
    if skip_filter:
        cmd.append("--skip-filter")
    if skip_tokenize:
        cmd.append("--skip-tokenize")
    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    constraints = [
        preemptible_constraint(True),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=region),
    ]
    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd),
        name=f"quality-filter-{preset}-{region}",
        resources=ResourceSpec(cpu=cpu, memory=memory, disk=disk),
        environment=EnvironmentSpec(extras=["cpu"], env_vars=env_vars),
        constraints=constraints,
        max_retries_preemption=20,
        max_retries_failure=3,
        priority_band=priority_band,
    )
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--presets", nargs="+", default=list(KNOWN_PRESETS), choices=list(KNOWN_PRESETS))
    p.add_argument(
        "--regions",
        nargs="+",
        default=["us-central1", "us-east1", "us-east5", "europe-west4"],
        choices=list(KNOWN_REGIONS),
    )
    p.add_argument(
        "--output-hash",
        required=True,
        help="Output dataset version tag. Same string across regions.",
    )
    p.add_argument(
        "--priority",
        choices=list(PRIORITY_BAND_MAP.keys()),
        default="batch",
        help="Child priority band. Default 'batch' so these yield to training.",
    )
    p.add_argument("--cpu", type=float, default=8.0)
    p.add_argument("--memory", default="32GB")
    p.add_argument("--disk", default="100GB")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-filter",
        action="store_true",
        help="Pass --skip-filter to children; assumes filtered JSONL already exists in every region.",
    )
    p.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Pass --skip-tokenize to children; run the filter step only.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    plan = [(region, preset) for region in args.regions for preset in args.presets]
    logger.info("Planning %d jobs: %s", len(plan), plan)
    if args.dry_run:
        for region, preset in plan:
            logger.info("  DRY: preset=%s region=%s hash=%s", preset, region, args.output_hash)
        return

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set. Run this inside an Iris parent or set it manually.")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY required.")
    hf_token = os.environ.get("HF_TOKEN")

    band = PRIORITY_BAND_MAP[args.priority]
    submitted: list[tuple[str, str, str]] = []
    for region, preset in plan:
        try:
            jid = submit_one(
                client,
                region=region,
                preset=preset,
                output_hash=args.output_hash,
                priority_band=band,
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                cpu=args.cpu,
                memory=args.memory,
                disk=args.disk,
                skip_filter=args.skip_filter,
                skip_tokenize=args.skip_tokenize,
            )
            submitted.append((region, preset, jid))
            logger.info("submitted %s/%s -> %s", region, preset, jid)
        except Exception as e:
            logger.exception("failed to submit %s/%s: %s", region, preset, e)

    logger.info("Total submitted: %d / %d", len(submitted), len(plan))
    # Keep the coordinator alive so Iris doesn't garbage-collect the hierarchy.
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
