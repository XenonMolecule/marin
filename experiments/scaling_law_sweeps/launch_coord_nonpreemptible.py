# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a curation sweep coordinator on a non-preemptible worker.

iris CLI's `job run` doesn't expose a preemptible flag, so the coord lands
on preemptible TPU workers' spare CPU and dies when GCP reclaims the spot
instance. This script uses IrisClient.submit() directly with
preemptible_constraint(False) so the coord gets a reserved (non-preemptible)
CPU worker that survives indefinitely.

Usage:
    uv run python experiments/scaling_law_sweeps/launch_coord_nonpreemptible.py \
        --methods dclm --child-priority interactive --run-suffix v4 \
        --allowed-regions us-central1 us-central2 us-east5 europe-west4
"""

from __future__ import annotations

import argparse
import os
import sys

from iris.client.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import preemptible_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2


def _connect_client(cluster: str) -> tuple[IrisClient, object]:
    """Establish tunnel + client, mirroring iris CLI's require_controller_url."""
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "lib", "iris", "examples", f"{cluster}.yaml"
    )
    iris_config = IrisConfig.load(config_path)
    bundle = iris_config.provider_bundle()
    controller_address = iris_config.controller_address()
    if not controller_address:
        controller_address = bundle.controller.discover_controller(iris_config.proto.controller)
    tunnel_cm = bundle.controller.tunnel(address=controller_address)
    tunnel_url = tunnel_cm.__enter__()
    from pathlib import Path
    client = IrisClient.remote(tunnel_url, workspace=Path.cwd())
    return client, tunnel_cm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", default=["all"])
    parser.add_argument("--child-priority", default="batch",
                        choices=["production", "interactive", "batch"])
    parser.add_argument("--run-suffix", default="v4")
    parser.add_argument("--allowed-regions", nargs="+", default=None)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--wandb-mode", default="auto")
    parser.add_argument("--cluster", default="marin")
    args = parser.parse_args()

    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        sys.exit("Set WANDB_API_KEY")
    hf_token = os.environ.get("HF_TOKEN", "")

    client, tunnel_cm = _connect_client(args.cluster)

    cmd_parts = [
        "python", "experiments/scaling_law_sweeps/launch_curation_sweep.py",
        "--methods", *args.methods,
        "--experiments", *args.experiments,
        "--child-priority", args.child_priority,
        "--run-suffix", args.run_suffix,
        "--wandb-mode", args.wandb_mode,
    ]
    if args.allowed_regions:
        cmd_parts.extend(["--allowed-regions", *args.allowed_regions])

    env_vars = {
        "WANDB_API_KEY": wandb_key,
        "PYTHONUNBUFFERED": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_parts),
        name=args.job_name,
        resources=ResourceSpec(cpu=4, memory="3GB", disk="5GB"),
        environment=EnvironmentSpec(extras=["tpu"], env_vars=env_vars),
        constraints=[preemptible_constraint(False)],
        max_retries_preemption=1000,
        max_retries_failure=3,
        priority_band=job_pb2.PRIORITY_BAND_PRODUCTION,
    )
    print(f"Submitted: {job.job_id}")
    print(f"  Non-preemptible coord, children at {args.child_priority} priority")


if __name__ == "__main__":
    main()
