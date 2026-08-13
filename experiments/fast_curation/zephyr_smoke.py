# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Minimal Zephyr smoke test: does actor-group creation work on this cluster AT ALL?

Diagnostic for the lpv11 dedup hang, where three separate reshape attempts sat with a live
coordinator, no ``-workers-`` actor group ever registering as an iris job, and zero output --
across max_workers=200 and max_workers=8, on both a busy and a completely empty cluster.

This isolates the variable: 4 integers, 2 workers, no GCS reads, no marin step machinery. If this
job SUCCEEDS, Zephyr provisioning is healthy and the dedup hang is specific to that pipeline. If it
hangs the same way, the problem is Zephyr/iris on this cluster and no dedup tuning will help.

    python -m experiments.fast_curation.zephyr_smoke
"""
from __future__ import annotations

import logging
import time

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.monotonic()
    logger.info("SMOKE: building trivial pipeline")

    pipeline = Dataset.from_iterable([1, 2, 3, 4]).map(lambda x: x * 2)
    ctx = ZephyrContext(
        name="zephyr-smoke",
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="2g", disk="5g"),
    )

    logger.info("SMOKE: calling ctx.execute -- this is where the dedup reshape hangs")
    results = ctx.execute(pipeline).results
    logger.info("SMOKE: SUCCESS results=%s in %.1fs", sorted(results), time.monotonic() - t0)


if __name__ == "__main__":
    main()
