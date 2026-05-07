# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native coordinator for the medical EXTRACTION 0.6B-Base re-run sweep.

WHY THIS FILE EXISTS
--------------------
The published medical 0.6B numbers in `code_math_medical.md` claim
`Qwen3-0.6B-Base`. The actual scripts that produced those numbers
(`medical_extraction_sft_v2.py`, `medical_extraction_v2_sweep.py`,
`medical_extraction_sft_hpsweep.py`, etc.) ALL hardcode `Qwen/Qwen3-0.6B`
(the post-trained instruct variant). This sweep re-runs the medical
EXTRACTION-V2 Phase-1 LRxBS grid on the correct `Qwen/Qwen3-0.6B-Base`
weights so the size-scan story (0.6B-Base → 14B-Base) becomes internally
consistent. See `run_medical_sft_standalone.py` for the longer rationale.

1:1 PARITY WITH THE ORIGINAL INSTRUCT SWEEP
-------------------------------------------
The Phase-1 grid below is the SAME 9-cell `LR_BS_CONFIGS` from
`medical_extraction_v2_sweep.py:68-83`. Same fixed hyperparams (WD=0.01,
warmup=0.03, decay=0.97, cosine, max_grad_norm=1.0, seq_len=4096).
Same tokenized cache (reused — Qwen3-0.6B and -0.6B-Base share
tokenizer.json). The ONLY intentional divergence is `--hf-model-name
Qwen/Qwen3-0.6B-Base` instead of `Qwen/Qwen3-0.6B`. Tag-side we add
`rerun=base-fix` and `qwen3-0.6b-base` so the W&B UI can clearly
distinguish from the buggy instruct runs.

Phase-2 (WD x warmup) is INTENTIONALLY OMITTED for time. The user
authorized "Phase 1 only" given the time pressure.

PATH HYGIENE
------------
No `gs://` paths are hardcoded for cache reads. The standalone child resolves
the cache via MARIN_PREFIX (or the GCP metadata fallback) → REGION_TO_BUCKET.
This file only hardcodes:
  - the cache REL-PATH (the content-addressed artifact identifier
    `tokenized/medical_extract-v2_qwen3-0.6b_sft-e74e0d`)
  - the tracker prefix and results prefix (small metadata, central home OK)
Both fall through `MARIN_PREFIX` at training time via `region_tracker`.

USAGE
-----
The launcher itself must run inside an iris parent CPU job:

    iris --cluster marin job run --priority production --no-wait \\
        --memory 4GB --cpu 4 --job-name medical-extraction-base-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/medical_extraction_sft_v2_base.py \\
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
# Sweep definition (1:1 with medical_extraction_v2_sweep.py:68-83)
# ---------------------------------------------------------------------------
# 9 cells. Note (7e-6, 32) is intentionally absent — the original sweep skipped
# that combination, presumably because lr=7e-6 was already considered too high
# at bs=32. Keeping the omission preserves apples-to-apples comparability.
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

# Phase-1 fixed hyperparams (1:1 with the original).
P1_WEIGHT_DECAY = 0.01
P1_WARMUP = 0.03
DECAY = 0.97
LR_SCHEDULE = "cosine"
MAX_GRAD_NORM = 1.0
SEQ_LEN = 4096

# Cache identity. The hash `e74e0d` is the LIVE V2-extraction cache
# (`gcloud storage cat gs://marin-us-central1/tokenized/<...>/.executor_info`
# confirms this is the complete 109-shard, 560M-token, 740k-doc artifact).
# The PATH itself is region-relative — the standalone child prepends the
# local region's bucket via MARIN_PREFIX → REGION_TO_BUCKET[region].
CACHE_REL_PATH = "tokenized/medical_extract-v2_qwen3-0.6b_sft-e74e0d"
CACHE_STEP_NAME = "medical_extract-v2_qwen3-0.6b_sft"

# The standalone child script that each iris child runs.
SCRIPT = "experiments/rephraser/run_medical_sft_standalone.py"

# Regions where the tokenized cache exists. Iris MUST schedule children in
# one of these — anywhere else would force a cross-region read of the
# tensorstore cache, which both costs money and would fail the
# `_assert_cache_local` invariant in the standalone child. As of 2026-05-04
# we have us-central1 + us-east5 + us-east1 mirrored. To add more regions
# later, pre-copy the cache and extend this list.
DEFAULT_ALLOWED_REGIONS: tuple[str, ...] = ("us-central1", "us-east5", "us-east1")

# TPU variants this sweep accepts. Both shapes are vm_count=1 (single-host)
# so iris's `device_variant_constraint` can mix them under a single
# resource spec. v5p-8 is the "matches the original 0.6B sweep" choice;
# v6e-4 is added because us-east1-d has 12 idle v6e-4 chips with no quota
# block (per regional capacity survey 2026-05-04). Qwen3-0.6B is tiny
# (~1.2 GB at bf16) so it fits comfortably on either; v6e is roughly 2x
# the FLOPS-per-chip of v5p so v6e-4 ≈ v5p-8 in throughput.
DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _config_name(lr: float, bs: int) -> str:
    """Same naming convention as `medical_extraction_v2_sweep.py:101-102`.

    f"{lr:.0e}".replace("+", "").replace("-0", "-")  -> e.g. 1e-6, 5e-6
    final string                                       -> e.g. lr1e-6_bs32
    Stable across both sweeps so the W&B UI matches up cell-by-cell.
    """
    lr_str = f"{lr:.0e}".replace("+", "").replace("-0", "-")
    return f"lr{lr_str}_bs{bs}"


def _build_run_name(config_name: str, run_suffix: str) -> str:
    """Run-name shape used in checkpoint dirs + W&B run id + tracker keys."""
    base = f"medical-extraction-{config_name}"
    if run_suffix:
        base = f"{base}-{run_suffix}"
    return base


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
    extra_env: dict[str, str] | None = None,
) -> str:
    """Submit one (lr, bs) cell as an iris child TPU job. Returns the iris job id."""
    config_name = _config_name(lr, bs)
    run_name = _build_run_name(config_name, run_suffix)

    cmd_args = [
        "python",
        SCRIPT,
        "--branch",
        "extraction",
        "--config-name",
        config_name,
        "--cache-rel-path",
        CACHE_REL_PATH,
        "--cache-step-name",
        CACHE_STEP_NAME,
        # Override cache_tokenizer from its instruct default to Base. The cache was
        # tokenized with Qwen/Qwen3-0.6B (instruct, len=151669); the model is
        # Qwen/Qwen3-0.6B-Base (vocab=151936). The two tokenizers' BPE merges and
        # vocab[0..151642] are byte-identical (verified via tokenizer.json diff
        # 2026-05-04); the only difference is 4 instruct-only added tokens at IDs
        # 151665..151668 (<tool_response>, </tool_response>, <think>, </think>) that
        # do NOT appear in extracted medical text. Passing the Base tokenizer string
        # here gives Levanter a tokenizer of len=151665 → padded to 151936 by
        # `pad_tokenizer_to_match_model`, matching the model. This sidesteps the
        # in-process pad_tokenizer bug observed in the Iris standalone (medical
        # 14B's Ray-executor path bridged the same len=151669 cache + 151936 model
        # mismatch fine; ours doesn't, and we haven't found the path divergence).
        # Same effective semantics as 14B since the BPE is identical and the 4
        # instruct-only IDs aren't in the data.
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
        # WandB's default init_timeout is 90s, which has timed out on TPU
        # workers with slow egress. Bumped to 5 min — same as the curation
        # sweep (launch_curation_sweep.py:171-173).
        "WANDB_INIT_TIMEOUT": "300",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token
    if extra_env:
        env_vars.update(extra_env)

    # Region constraint: HARD restriction to the regions where we've
    # pre-copied the tokenized cache. Anywhere else would either (a) silently
    # cross-region read the cache (cost money) or (b) fail the standalone
    # child's `_assert_cache_local` invariant. Using HARD here prevents iris
    # from "helpfully" placing a child in europe-west4 or us-west4 because of
    # spare capacity — it should sit pending until central1/east5 frees up.
    region_constraint = Constraint(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=tuple(allowed_regions),
    )  # default mode = CONSTRAINT_MODE_REQUIRED (hard)

    constraints = [
        preemptible_constraint(True),
        region_constraint,
    ]

    # If alternatives are provided, allow iris to land on either the primary
    # or an alternative TPU shape. All variants must share vm_count for
    # multi-host coscheduling correctness — v5p-8 and v6e-4 are both
    # vm_count=1 so this works (same pattern as launch_curation_sweep.py:229-230).
    all_variants = (tpu_variant, *tpu_alternatives)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"medical-ext-base-{config_name}"[:200],
        resources=ResourceSpec(
            # Bumped from 32GB → 64GB on 2026-05-04 after OOM observed during HF
            # export at end of training. Levanter's `save_hf_checkpoint` pulls
            # the model state from TPU HBM into CPU RAM and rounds up buffers
            # to ~6-10x the raw shard size during serialization. 32GB is
            # sufficient for training but tight for the export.
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
        # 0.6B at v5p-8 is single-host (vm_count=1) — no replicas / coscheduling.
        max_retries_preemption=100,
        # Same reasoning as launch_curation_sweep.py:266-272: transient TPU VMs
        # with stale libtpu state can fail multiple retries; 10 gives iris
        # enough attempts to evict bad workers without burning forever on a
        # genuinely broken plan.
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info("Submitted %s -> %s", run_name, job.job_id)
    return str(job.job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--allowed-regions",
        nargs="+",
        default=list(DEFAULT_ALLOWED_REGIONS),
        help=(
            "HARD region restriction. Default us-central1 + us-east5 (where the cache "
            "is mirrored). To add a region, pre-copy the cache first via "
            "`gcloud storage rsync -r ...`."
        ),
    )
    parser.add_argument(
        "--tpu-variant",
        default=DEFAULT_TPU_VARIANT,
        help=("Primary TPU shape per child. Default v5p-8 (matches the original " "0.6B sweep). Single-host only."),
    )
    parser.add_argument(
        "--tpu-alternatives",
        nargs="*",
        default=list(DEFAULT_TPU_ALTERNATIVES),
        help=(
            "Alternative TPU shapes iris can land children on if the primary "
            "is at capacity. Default ['v6e-4'] — us-east1-d had 12 idle v6e-4 "
            "chips (no quota block) per regional capacity survey 2026-05-04. "
            "All alternatives MUST share vm_count with the primary; v5p-8 and "
            "v6e-4 are both vm_count=1 so they mix safely."
        ),
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
        help=(
            "Priority band for child TPU jobs. 'batch' yields to higher-priority work "
            "— appropriate for an HP sweep. The PARENT (this script) should be at "
            "default priority so it doesn't get preempted (per memory: "
            "batch_queue_parents)."
        ),
    )
    parser.add_argument(
        "--run-suffix",
        default="qwen3-0.6b-base-rerun",
        help=(
            "Suffix on every run name. Default 'qwen3-0.6b-base-rerun' so W&B + GCS "
            "keep the buggy instruct runs and the corrected Base runs cleanly "
            "separated. Always include 'base' in this string."
        ),
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
        help=(
            "If set, only submit the first N cells. Useful for smoke testing — e.g. "
            "'--max-count 1' submits just (lr=1e-6, bs=64)."
        ),
    )
    parser.add_argument(
        "--filter-config",
        default=None,
        help=(
            "If set, only submit cells whose config_name CONTAINS this substring. "
            "e.g. '--filter-config lr5e-6_bs32' for the previous instruct-run winner."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted, do not submit.",
    )
    parser.add_argument(
        "--no-keep-alive",
        action="store_true",
        help=(
            "Skip the post-submit `while True: sleep(3600)`. Useful for testing the "
            "coordinator outside iris (when iris's parent-keeps-children-alive "
            "contract doesn't apply)."
        ),
    )
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
        "Medical EXTRACTION 0.6B-Base re-run sweep: %d cells, "
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

    # Connect to iris controller via the parent job's env.
    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError(
            "IRIS_CONTROLLER_ADDRESS not set — this coordinator must run inside an iris job. "
            "Use `iris --cluster marin job run -- python this_script.py ...`."
        )
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required (set via -e on iris job run).")
    hf_token = os.environ.get("HF_TOKEN")  # required for downloading Qwen3-0.6B-Base from HF

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

    # Stay alive — iris's parent-keeps-children-alive contract requires the
    # parent to remain running while children are scheduled. If the parent
    # dies, iris may treat the entire job hierarchy as orphaned.
    # (Same pattern as launch_curation_sweep.py:706-708.)
    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
