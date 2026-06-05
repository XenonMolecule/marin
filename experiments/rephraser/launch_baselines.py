# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native coordinator for 8B-Base BASELINE evals across all domains.

Submits 3 vLLM-TPU eval children, one per domain (code, math, medical),
all evaluating `Qwen/Qwen3-8B-Base` (no SFT). Writes results to a
region-local `eval_baselines/baseline-qwen3-8b-base-<domain>/` prefix.

WHY THIS FILE EXISTS
--------------------
The 0.6B-Base medical baseline already landed (see
`gs://marin-us-east5/eval_baselines/baseline-qwen3-0.6b-base/`). The 8B
baseline went through `launch_medical_evals.py` for medical only.
The user wants the 8B story to cover all 3 benchmarks symmetrically with
val/test splits — this kicks off code + math + medical in one go, all
using `run_eval_standalone.py` with the per-domain task list.

USAGE
-----
The launcher itself runs inside an iris parent CPU job:

    iris --cluster marin job run --priority interactive --no-wait \\
        --memory 2GB --cpu 2 --job-name baseline-evals-8b \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/launch_baselines.py \\
        --child-priority interactive
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


SCRIPT = "experiments/rephraser/run_eval_standalone.py"

DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)
DEFAULT_ALLOWED_REGIONS: tuple[str, ...] = ("us-central1", "us-east5", "us-east1")

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}

DOMAINS: tuple[str, ...] = ("code", "math", "medical", "medical-logprob")


def submit_one(
    client: IrisClient,
    *,
    domain: str,
    model_ref: str,
    model_name: str,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    allowed_regions: tuple[str, ...],
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...],
) -> str:
    cmd_args = [
        "python",
        SCRIPT,
        "--domain",
        domain,
        "--model-rel-path",
        model_ref,
        "--model-name",
        model_name,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        "MARIN_VLLM_MODE": "native",
        # HumanEval / MBPP execute generated code via the `code_eval` metric;
        # HF gates it behind this env var. Harmless for math/medical (those
        # tasks don't invoke code_eval). Required for the code domain.
        "HF_ALLOW_CODE_EVAL": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    constraints = [
        preemptible_constraint(True),
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.IN, values=tuple(allowed_regions)),
    ]
    all_variants = (tpu_variant, *tpu_alternatives)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"baseline-eval-{domain}-{model_name}"[:200],
        resources=ResourceSpec(
            cpu=8,
            # vLLM-TPU on Iris needs 128GB per feedback_iris_vllm_tpu_standalone.md.
            memory="128GB",
            disk="100GB",
            device=tpu_device(tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu", "vllm", "eval"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=100,
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info("Submitted baseline-eval/%s/%s -> %s", domain, model_name, job.job_id)
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-ref",
        default="Qwen/Qwen3-8B-Base",
        help="HF reference for the baseline model.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the W&B / output name. Default: baseline-qwen3-<size>-base.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=list(DOMAINS),
        default=list(DOMAINS),
    )
    parser.add_argument("--allowed-regions", nargs="+", default=list(DEFAULT_ALLOWED_REGIONS))
    parser.add_argument("--tpu-variant", default=DEFAULT_TPU_VARIANT)
    parser.add_argument("--tpu-alternatives", nargs="*", default=list(DEFAULT_TPU_ALTERNATIVES))
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    # Derive a friendly model-name suffix from the HF ref (e.g.
    # "Qwen/Qwen3-8B-Base" -> "qwen3-8b-base").
    inferred_name = args.model_ref.split("/")[-1].lower()
    base_model_name = args.model_name or f"baseline-{inferred_name}"

    logger.info(
        "Baseline eval coord: model=%s, domains=%s, regions=%s, tpu=%s",
        args.model_ref,
        args.domains,
        args.allowed_regions,
        args.tpu_variant,
    )

    if args.dry_run:
        return

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

    submitted: list[str] = []
    for domain in args.domains:
        # Per-domain model-name keeps wandb runs and GCS output paths separate
        # so we can rerun one domain without colliding with the others.
        model_name = f"{base_model_name}-{domain}"
        try:
            job_id = submit_one(
                client,
                domain=domain,
                model_ref=args.model_ref,
                model_name=model_name,
                child_priority_band=child_priority_band,
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                allowed_regions=tuple(args.allowed_regions),
                tpu_variant=args.tpu_variant,
                tpu_alternatives=tuple(args.tpu_alternatives),
            )
            submitted.append(job_id)
        except Exception as e:
            logger.exception("Failed to submit baseline %s/%s: %s", domain, model_name, e)

    logger.info("Submitted %d/%d baseline eval children.", len(submitted), len(args.domains))

    if args.no_keep_alive:
        return

    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
