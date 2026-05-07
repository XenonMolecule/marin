# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native coordinator for the medical RESILIPARSE 0.6B-Base re-run sweep.

WHY THIS FILE EXISTS
--------------------
Companion to `medical_extraction_sft_v2_base.py` — same bug, different
data branch. The published medical 0.6B numbers in `code_math_medical.md`
claim `Qwen3-0.6B-Base` but every actual script (here, the original was
`medical_resiliparse_sweep.py`) hardcoded `Qwen/Qwen3-0.6B` (the
post-trained instruct variant). This sweep re-runs the medical RESILIPARSE
Phase-1 LRxBS grid on `Qwen/Qwen3-0.6B-Base` so the published row of the
table can be honest.

See `run_medical_sft_standalone.py` for the longer rationale and
`medical_extraction_sft_v2_base.py` for the design discussion of the
Iris-native coordinator/child split.

1:1 PARITY WITH THE ORIGINAL INSTRUCT SWEEP
-------------------------------------------
The Phase-1 grid below is the SAME 9-cell `LR_BS_CONFIGS` from
`medical_resiliparse_sweep.py:68-78`. Same fixed hyperparams (WD=0.01,
warmup=0.03, decay=0.97, cosine, max_grad_norm=1.0, seq_len=4096). Same
tokenized cache reused. Only `--hf-model-name Qwen/Qwen3-0.6B-Base` is
intentionally different.

Phase-2 (WD x warmup) is OMITTED for time, per user direction.

USAGE
-----
The launcher itself must run inside an iris parent CPU job:

    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 4 --job-name medical-resiliparse-base-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/medical_resiliparse_v2_base.py \\
        --child-priority batch
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


# ---------------------------------------------------------------------------
# Sweep definition (1:1 with medical_resiliparse_sweep.py:68-83)
# ---------------------------------------------------------------------------
# Same 9-cell grid as medical_extraction_sft_v2_base.py — the original
# sweeps used identical LRxBS combinations across both data branches.
# Keeping them aligned preserves apples-to-apples comparison both within
# and across branches.
LR_BS_CONFIGS: list[tuple[float, int]] = [
    (1e-6, 64),
    (2e-6, 64),
    (3e-6, 64),
    (5e-6, 64),
    (7e-6, 64),
    (1e-6, 32),
    (2e-6, 32),
    (3e-6, 32),
    (5e-6, 32),
]

P1_WEIGHT_DECAY = 0.01
P1_WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# Cache identity. The hash `2061a9` is the LIVE resiliparse cache, confirmed
# both by `gcloud storage cat <...>/.executor_info` and by the explicit
# comment in `medical_14b_sft.py:113` ("to ensure the hash matches the
# existing cached tokenized data (2061a9)").
CACHE_REL_PATH = "tokenized/medical_resiliparse_qwen3-0.6b_sft-2061a9"
CACHE_STEP_NAME = "medical_resiliparse_qwen3-0.6b_sft"

# The standalone child script. SHARED with the extraction coordinator —
# `--branch resiliparse` is the only thing that changes per-coordinator on
# the child's CLI surface.
SCRIPT = "experiments/rephraser/run_medical_sft_standalone.py"

# Regions where the cache exists — see medical_extraction_sft_v2_base.py
# for the rationale on HARD restriction.
DEFAULT_ALLOWED_REGIONS: tuple[str, ...] = ("us-central1", "us-east5", "us-east1")

# TPU primary + alternatives. Both vm_count=1 → safe to mix under a single
# device_variant_constraint. v6e-4 added for us-east1-d / us-east5-b.
# v4-8 / us-central2 was tried but reverted: checkpoints were copied to
# us-east1 (the v6e-4 region) only, so v4-8 cells in us-central2 would
# start from step 0. Adding the fallback isn't worth the cross-region copy
# of 60GB checkpoints + ~$1 to a slower TPU; v6e-4 is faster anyway.
DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def _config_name(lr: float, bs: int) -> str:
    """Same naming convention as medical_resiliparse_sweep.py:101-102.

    Also matches the extraction coordinator. Stable across both branches so
    the W&B UI lines up cell-by-cell across E and R.
    """
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    return f"lr{lr_str}_bs{bs}"


def submit_one(
    client: IrisClient,
    *,
    lr: float,
    bs: int,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    allowed_regions: tuple[str, ...],
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...] = (),
    run_suffix: str,
) -> str:
    """Submit one (lr, bs) cell as an iris child TPU job. Returns the iris job id."""
    config_name = _config_name(lr, bs)

    cmd_args = [
        "python",
        SCRIPT,
        "--branch",
        "resiliparse",
        "--config-name",
        config_name,
        "--cache-rel-path",
        CACHE_REL_PATH,
        "--cache-step-name",
        CACHE_STEP_NAME,
        # See medical_extraction_sft_v2_base.py for the rationale on swapping the
        # cache_tokenizer to Base. Same situation here: cache tokenized with
        # instruct, BPE identical to Base, no instruct-only special tokens in the
        # extracted text. Loading the Base tokenizer gives len=151665 which pads
        # cleanly to model vocab=151936.
        "--cache-tokenizer",
        "Qwen/Qwen3-0.6B-Base",
        "--hf-model-name",
        "Qwen/Qwen3-0.6B-Base",
        "--learning-rate",
        f"{lr:.0e}",
        "--batch-size",
        str(bs),
        "--weight-decay",
        str(P1_WEIGHT_DECAY),
        "--warmup",
        str(P1_WARMUP),
        "--decay",
        str(DECAY),
        "--lr-schedule",
        LR_SCHEDULE,
        "--max-grad-norm",
        str(MAX_GRAD_NORM),
        "--seq-len",
        str(SEQ_LEN),
        "--run-suffix",
        run_suffix,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    region_constraint = Constraint(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=tuple(allowed_regions),
    )

    constraints = [
        preemptible_constraint(True),
        region_constraint,
    ]

    all_variants = (tpu_variant, *tpu_alternatives)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"medical-res-base-{config_name}"[:200],
        resources=ResourceSpec(
            # Bumped 32→64GB after OOM during HF export, see same comment in
            # medical_extraction_sft_v2_base.py.
            cpu=4,
            memory="64GB",
            disk="50GB",
            device=tpu_device(tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=100,
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info("Submitted resiliparse-%s -> %s", config_name, job.job_id)
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--allowed-regions", nargs="+", default=list(DEFAULT_ALLOWED_REGIONS))
    parser.add_argument("--tpu-variant", default="v5p-8")
    parser.add_argument(
        "--tpu-alternatives",
        nargs="*",
        default=list(DEFAULT_TPU_ALTERNATIVES),
        help="Alternative TPU shapes (default ['v6e-4']). Pass with no args to disable.",
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument("--run-suffix", default="qwen3-0.6b-base-rerun")
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--filter-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    plans = list(LR_BS_CONFIGS)
    if args.filter_config is not None:
        plans = [(lr, bs) for (lr, bs) in plans if args.filter_config in _config_name(lr, bs)]
    if args.max_count is not None:
        plans = plans[: args.max_count]

    logger.info(
        "Medical RESILIPARSE 0.6B-Base re-run sweep: %d cells, "
        "allowed_regions=%s, tpu=%s, child_priority=%s, run_suffix=%s",
        len(plans),
        args.allowed_regions,
        args.tpu_variant,
        args.child_priority,
        args.run_suffix,
    )
    for lr, bs in plans:
        logger.info("  %s", _config_name(lr, bs))

    if args.dry_run:
        logger.info("Dry run — exiting without submitting.")
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — this coordinator must run inside an iris job.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required (set via -e on iris job run).")
    hf_token = os.environ.get("HF_TOKEN")

    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    submitted: list[str] = []
    for lr, bs in plans:
        try:
            job_id = submit_one(
                client,
                lr=lr,
                bs=bs,
                child_priority_band=child_priority_band,
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                allowed_regions=tuple(args.allowed_regions),
                tpu_variant=args.tpu_variant,
                tpu_alternatives=tuple(args.tpu_alternatives),
                run_suffix=args.run_suffix,
            )
            submitted.append(job_id)
        except Exception as e:
            logger.exception("Failed to submit (%g, %d): %s", lr, bs, e)

    logger.info("Submitted %d/%d cells.", len(submitted), len(plans))

    if args.no_keep_alive:
        return

    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
