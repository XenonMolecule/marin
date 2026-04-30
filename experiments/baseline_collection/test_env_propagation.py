# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does parent MARIN_PREFIX propagate to a child in a different region?

Parent runs in region A (pinned via --region at submit time). Parent spawns a
child pinned HARD to region B. Both log their MARIN_PREFIX, detected region,
and IRIS_JOB_ENV. Comparing the two tells us whether env inheritance pins the
child to the parent's bucket regardless of physical region.

Submit:
    uv run iris --cluster marin job run --cpu 0.5 --memory 2GB \
        --region us-central1 --job-name test-env-propagation \
        -- python experiments/baseline_collection/test_env_propagation.py parent us-east5
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _log_env_snapshot(role: str) -> None:
    from rigging.filesystem import marin_prefix, marin_region, region_from_metadata

    logger.info("========== %s env snapshot ==========", role)
    logger.info("  MARIN_PREFIX (os.environ): %r", os.environ.get("MARIN_PREFIX"))
    logger.info("  IRIS_JOB_ENV: %r", os.environ.get("IRIS_JOB_ENV"))
    logger.info("  IRIS_TASK_ID: %r", os.environ.get("IRIS_TASK_ID"))
    logger.info("  IRIS_WORKER_ID: %r", os.environ.get("IRIS_WORKER_ID"))
    logger.info("  IRIS_WORKER_REGION: %r", os.environ.get("IRIS_WORKER_REGION"))
    logger.info("  region_from_metadata(): %r", region_from_metadata())
    logger.info("  marin_region(): %r", marin_region())
    logger.info("  marin_prefix() -> %r", marin_prefix())
    logger.info("========== end %s snapshot ==========", role)


def run_parent(child_region: str) -> None:
    _log_env_snapshot("PARENT")

    from iris.client.client import IrisClient
    from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    logger.info("Parent about to submit child pinned HARD to region=%s", child_region)

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — must run inside an Iris job")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)
    job = client.submit(
        entrypoint=Entrypoint.from_command(
            "python",
            "experiments/baseline_collection/test_env_propagation.py",
            "child",
            child_region,
        ),
        name="child",
        resources=ResourceSpec(cpu=0.5, memory="2GB", disk="2GB"),
        environment=EnvironmentSpec(extras=[], env_vars={}),
        constraints=[
            Constraint(
                key=WellKnownAttribute.REGION,
                op=ConstraintOp.IN,
                values=(child_region,),
            ),
        ],
        max_retries_failure=0,
        max_retries_preemption=3,
    )
    logger.info("Submitted child job: %s — waiting for completion", job.job_id)
    job.wait(timeout=1800, poll_interval=10, raise_on_failure=False, stream_logs=True)
    logger.info("Child finished.")


def run_child(expected_region: str) -> None:
    _log_env_snapshot("CHILD")
    from rigging.filesystem import region_from_metadata

    actual = region_from_metadata()
    logger.info("VERDICT: expected_region=%s  actual_metadata_region=%s", expected_region, actual)
    if actual != expected_region:
        logger.warning("Child was scheduled in %s but expected %s", actual, expected_region)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["parent", "child"])
    parser.add_argument("region", help="For parent: region to pin child to. For child: expected region.")
    args = parser.parse_args()
    if args.role == "parent":
        run_parent(args.region)
    else:
        run_child(args.region)


if __name__ == "__main__":
    main()
