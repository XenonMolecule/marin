# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr pipeline: bootstrap the Option B scaling-law fit across a worker pool.

Runs as an Iris coordinator job. Computes the CENTRAL fit (Option B on the
original gold) once and writes it to `{output}/central_fit.json`, then fans out
B bootstrap fits via a Zephyr `Dataset.from_list(range(B)).map(fit_one_seed)`
worker pool, writing base64 winner grids to `{output}/seeds-*.jsonl.gz`.

Launch (Iris coordinator, interactive so it isn't preempted):

    uv run iris --cluster us-central1 job run \\
      --region us-central1 --cpu 2 --memory 8GB --disk 10GB \\
      --priority interactive --no-wait --job-name boot-optionB-coord \\
      -- python experiments/scaling_law_sweeps/run_bootstrap_optionB_zephyr.py \\
         --n-bootstraps 1000 --max-workers 100 \\
         --output gs://marin-us-central1/scratch/bootstrap_optionB/run_<ts>

The aggregator (`aggregate_bootstrap_optionB.py`) later pulls the jsonl shards +
central_fit.json and builds the combined trust-region npz for figure rendering.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import subprocess
import tempfile

from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.scaling_law_sweeps.bootstrap_optionB_lib import (
    N_RESTARTS_DEFAULT,
    fit_central,
    fit_one_seed,
)

logger = logging.getLogger(__name__)


def _write_json_gcs(obj: dict, gs_path: str) -> None:
    """Write a small JSON object to a gs:// path (fsspec, gcloud fallback)."""
    payload = json.dumps(obj, indent=2)
    try:
        import fsspec

        with fsspec.open(gs_path, "w") as f:
            f.write(payload)
        return
    except Exception:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        subprocess.run(["gcloud", "storage", "cp", tmp_path, gs_path], check=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--max-workers", type=int, default=100)
    p.add_argument("--n-restarts", type=int, default=N_RESTARTS_DEFAULT)
    p.add_argument("--worker-cpu", type=float, default=2.0)
    p.add_argument("--worker-ram", default="4g")
    p.add_argument(
        "--output", required=True, help="gs:// prefix for this run's outputs " "(central_fit.json + seeds-*.jsonl.gz)."
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    output = args.output.rstrip("/")

    # 1. Central fit (one Option B fit on the original gold). Cheap; do it on
    #    the coordinator so the figure never has to refit.
    logger.info("Computing central Option B fit (n_restarts=%d)…", args.n_restarts)
    central = fit_central(n_restarts=args.n_restarts)
    _write_json_gcs(central, f"{output}/central_fit.json")
    logger.info("Wrote %s/central_fit.json", output)
    for m, p in central["params"].items():
        logger.info(
            "  central %-14s E=%.3f C=%.3g β=%.3f B=%.3g δ=%.3f " "α=%.3f R_D=%.3f R_decay=%.2f",
            m,
            p[0],
            p[1],
            p[2],
            p[3],
            p[4],
            p[5],
            p[6],
            p[7],
        )

    # 2. Bootstrap fan-out via Zephyr worker pool.
    logger.info(
        "Fanning out %d bootstrap fits over <=%d workers " "(cpu=%.1f ram=%s, n_restarts=%d)…",
        args.n_bootstraps,
        args.max_workers,
        args.worker_cpu,
        args.worker_ram,
        args.n_restarts,
    )
    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram),
        coordinator_resources=ResourceConfig(cpu=2, ram="8g", preemptible=False),
        name="boot-optionB",
    )
    fit_fn = functools.partial(fit_one_seed, n_restarts=args.n_restarts)
    pipeline = (
        Dataset.from_list(list(range(args.n_bootstraps)))
        .map(fit_fn)
        .write_jsonl(f"{output}/seeds-{{shard:05d}}.jsonl.gz")
    )
    result = ctx.execute(pipeline, verbose=True)
    logger.info("Bootstrap complete. Output shards: %s", getattr(result, "results", "<n/a>"))
    logger.info("All outputs under %s", output)


if __name__ == "__main__":
    main()
