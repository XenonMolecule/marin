# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris coordinator: submit one at-rest query-test job per built BM25 index.

Each child runs :mod:`bm25_query_test` for one (dataset, collection), pinned to
the index's region so the mmap download stays in-region (no cross-region egress).
Launch as an Iris job so ``IRIS_CONTROLLER_ADDRESS`` is set. Keeps alive so the
children are not orphan-killed; stop it once results are gathered via bug-report.

    iris job run --cluster marin --region us-central1 --extra cpu -- \
        python -m experiments.infinigram.launch_bm25_query_test --datasets dclm --collections small
"""

import argparse
import logging
import os
import time

from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute, preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

from experiments.infinigram.bm25_sources import BM25_SPECS, get_bm25_target
from experiments.infinigram.targets import Collection

logger = logging.getLogger(__name__)

# Test child: disk to hold the index (downloaded once), RAM for that one-time
# download + bounded batched querying. resiliparse-300 is ~31 GiB.
TEST_RESOURCES = {"cpu": 4.0, "memory": "48g", "disk": "80g"}


def _submit(client: IrisClient, dataset: str, collection: Collection) -> str:
    target = get_bm25_target(dataset, collection)
    job = client.submit(
        entrypoint=Entrypoint.from_command(
            "python",
            "-m",
            "experiments.infinigram.bm25_query_test",
            "--dataset",
            dataset,
            "--collection",
            collection.value,
        ),
        name=f"bm25-qtest-{target.name}",
        resources=ResourceSpec(**TEST_RESOURCES),
        environment=EnvironmentSpec(extras=["cpu"], pip_packages=["bm25s"], env_vars={"PYTHONUNBUFFERED": "1"}),
        constraints=[
            preemptible_constraint(True),
            Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value=target.region),
        ],
        max_retries_preemption=5,
        max_retries_failure=0,
    )
    return str(job.job_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--collections", nargs="+", choices=[c.value for c in Collection], default=["small"])
    args = p.parse_args()

    controller = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — launch via `iris ... job run -- python <this>`")
    client = IrisClient.remote(controller, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    for ds in args.datasets:
        for col in args.collections:
            if ds not in BM25_SPECS or BM25_SPECS[ds].source(Collection(col)) is None:
                continue
            try:
                jid = _submit(client, ds, Collection(col))
                logger.info("submitted qtest %s-%s -> %s", ds, col, jid)
            except Exception:
                logger.exception("failed to submit qtest %s-%s", ds, col)

    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
