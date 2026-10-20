# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a read-only host probe (memory, /dev/shm, top processes) on specific TPU slices.

Submits one tiny CPU child per ``--tpu-name``, pinned to that slice with a hard
``tpu-name`` constraint, and prints each child's output. Must run inside an iris job
(coordinator pattern) so it can reach the controller::

    uv run iris --cluster marin job run --priority interactive --cpu 1 --memory 1GB --extra cpu \
        --job-name probe-shm -- python experiments/one_off/probe_worker_host.py \
        --region us-central2 --tpu-names t1v-n-7ef43b99 t1v-n-e063905e
"""

import argparse
import logging
import os

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

PROBE_CMD = (
    "echo HOST=$(hostname); grep -E 'MemTotal|MemFree|MemAvailable|^Cached|Shmem:|Dirty' /proc/meminfo; echo; "
    "grep -E 'tmpfs|shm|iris|cache' /proc/mounts | cut -c1-140; echo; "
    "for m in $(grep -E 'tmpfs|iris|cache' /proc/mounts | awk '{print $2}'); do du -sh $m 2>/dev/null; done; echo; "
    "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null; echo; "
    "ls /proc | grep -cE '^[0-9]+$'"
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tpu-names", nargs="+", required=True)
    ap.add_argument("--region", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()

    client = IrisClient.remote(os.environ["IRIS_CONTROLLER_ADDRESS"], bundle_id=os.environ.get("IRIS_BUNDLE_ID"))
    jobs = {}
    for name in args.tpu_names:
        job = client.submit(
            entrypoint=Entrypoint.from_command("bash", "-c", PROBE_CMD),
            name=f"probe-{name}",
            resources=ResourceSpec(cpu=1, memory="512MB"),
            environment=EnvironmentSpec(extras=["cpu"]),
            constraints=[
                preemptible_constraint(False),
                Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=[args.region]),
                Constraint.create(key=WellKnownAttribute.TPU_NAME, op=ConstraintOp.EQ, value=name),
            ],
            max_retries_failure=0,
            priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
        )
        jobs[name] = job
        logger.info("submitted %s -> %s", name, job.job_id)
    for name, job in jobs.items():
        job.wait(timeout=args.timeout, poll_interval=10.0, raise_on_failure=False)
        print(f"===== {name} ({job.job_id}) =====")
        for entry in job.logs():
            print(entry.data)


if __name__ == "__main__":
    main()
