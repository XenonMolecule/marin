# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native single-cell smoke launcher for the Levanter logprob-MMLU eval.

Submits ONE TPU child running `run_eval_levanter_standalone.py` on a
specified HF model + domain. Used for smoke-testing the Levanter eval
backend before fanning out to the full re-eval suite.

Parallels `launch_baselines.py` but uses the Levanter standalone and only
asks for `extras=("eval", "tpu")` (no vllm extra needed since Levanter
serves the model directly).
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
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


SCRIPT = "experiments/rephraser/run_eval_levanter_standalone.py"

DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)
DEFAULT_ALLOWED_REGIONS: tuple[str, ...] = ("us-central1", "us-east5", "us-east1")

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-ref", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--domain", default="medical-logprob")
    parser.add_argument("--allowed-regions", nargs="+", default=list(DEFAULT_ALLOWED_REGIONS))
    parser.add_argument("--tpu-variant", default=DEFAULT_TPU_VARIANT)
    parser.add_argument("--tpu-alternatives", nargs="*", default=list(DEFAULT_TPU_ALTERNATIVES))
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="interactive",
    )
    parser.add_argument("--no-keep-alive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    inferred = args.model_ref.split("/")[-1].lower()
    model_name = args.model_name or f"baseline-{inferred}-{args.domain}"

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — must run inside an iris job.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")
    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    cmd_args = [
        "python",
        SCRIPT,
        "--domain",
        args.domain,
        "--model-rel-path",
        args.model_ref,
        "--model-name",
        model_name,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        # HumanEval / MBPP need this; harmless for medical-logprob.
        "HF_ALLOW_CODE_EVAL": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    constraints = [
        preemptible_constraint(True),
        Constraint(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=tuple(args.allowed_regions)),
    ]
    all_variants = (args.tpu_variant, *args.tpu_alternatives)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"levanter-eval-{args.domain}-{model_name}"[:200],
        resources=ResourceSpec(
            cpu=8,
            memory="64GB",  # Levanter loglikelihood doesn't need vLLM's 128GB.
            disk="50GB",
            device=tpu_device(args.tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu", "eval"],  # No vllm extra — Levanter serves directly.
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=100,
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info("Submitted Levanter smoke-eval %s -> %s", model_name, job.job_id)

    if args.no_keep_alive:
        return

    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
