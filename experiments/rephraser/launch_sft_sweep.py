# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generalized Iris coordinator for SFT sweeps across (domain, branch, model-size).

Replaces the per-domain/per-branch coordinator sprawl with one parametrized
launcher. Picks the right LR/BS grid + cache rel-path + tracker prefix based
on `--domain {code,math,medical}`, `--branch {extraction,resiliparse}`, and
`--model-size {0.6b,8b,14b}`. Submits one Iris TPU child per cell, all running
`run_sft_standalone.py`.

Sweep design (locked 2026-05-04):

  - **0.6B** (all 3 domains): full Phase-1 LR×BS grid (9 cells per branch,
    matching the original instruct sweep at `medical_extraction_v2_sweep.py:68-83`,
    `code_extraction_sft_v3_sweep.py`, etc.). Default region constraint includes
    {us-central1, us-east5, us-east1}; primary TPU v5p-8 with v6e-4 alternative.

  - **8B**:
    - **code**: full 15-cell Phase-1 grid (5 LRs × 3 BSs), matching the 14B sweep at
      `code_extraction_sft_v3_14b_sweep.py`. This is the only domain that gets a
      fresh sweep at 8B; math + medical inherit from it.
    - **math + medical**: 3 transfer configs (default / best-resili /
      best-extract) borrowed from code 14B, same pattern as `math_14b_top3_sft.py`
      and `medical_14b_sft.py`.
    Default region constraint includes {us-central1, us-east5, us-east1};
    primary TPU v5p-32 with v5p-16 + v6e-16 alternatives.

  - **14B**: not currently used (legacy Ray runs are sufficient), but the
    machinery is here in case we ever need to re-run on Iris.

USAGE
-----
Per launch, one (domain, branch, model-size) combo. Invoke from inside an
iris parent CPU job:

    # 0.6B medical extraction (the original re-run)
    iris --cluster marin job run --priority interactive --no-wait \\
        --memory 2GB --cpu 2 --job-name med-ext-base-0_6b \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/launch_sft_sweep.py \\
        --domain medical --branch extraction --model-size 0.6b

    # 8B code extraction (full 15-cell sweep)
    iris --cluster marin job run --priority interactive --no-wait \\
        --memory 2GB --cpu 2 --job-name code-ext-base-8b \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/launch_sft_sweep.py \\
        --domain code --branch extraction --model-size 8b
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
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    get_tpu_topology,
    tpu_device,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache catalog: (domain, branch) -> (cache_rel_path, cache_step_name)
# ---------------------------------------------------------------------------
# All caches use the Qwen3 Base tokenizer (or are made compatible via
# pad_tokenizer at training time — see the patches in
# `lib/levanter/src/levanter/main/train_lm.py` and
# `lib/levanter/src/levanter/compat/hf_checkpoints.py`). All have been
# mirrored to {us-central1, us-east5, us-east1} per the regional capacity
# survey 2026-05-04.
CACHE_CATALOG: dict[tuple[str, str], tuple[str, str]] = {
    ("code", "extraction"): (
        "tokenized/code_extract-commented_qwen3-0.6b-base_sft-e52621",
        "code_extract-commented_qwen3-0.6b-base_sft",
    ),
    ("code", "resiliparse"): (
        "tokenized/code_resiliparse_qwen3-0.6b-base_sft-c73439",
        "code_resiliparse_qwen3-0.6b-base_sft",
    ),
    ("math", "extraction"): (
        "tokenized/math_mix_top3_qwen3-0.6b-base_sft-a70279",
        "math_mix_top3_qwen3-0.6b-base_sft",
    ),
    ("math", "resiliparse"): (
        "tokenized/math_resili_top3_qwen3-0.6b-base_sft-9a966b",
        "math_resili_top3_qwen3-0.6b-base_sft",
    ),
    # NB: medical caches were originally tokenized with `Qwen/Qwen3-0.6B`
    # (instruct) at tokenize time — see the bug-discovery thread. The cache
    # bytes are still compatible with Base via pad_tokenizer (BPE merges
    # are byte-identical between instruct and Base tokenizers; only 4
    # instruct-only special tokens differ, none of which appear in
    # extracted medical text).
    ("medical", "extraction"): (
        "tokenized/medical_extract-v2_qwen3-0.6b_sft-e74e0d",
        "medical_extract-v2_qwen3-0.6b_sft",
    ),
    ("medical", "resiliparse"): (
        "tokenized/medical_resiliparse_qwen3-0.6b_sft-2061a9",
        "medical_resiliparse_qwen3-0.6b_sft",
    ),
}


# ---------------------------------------------------------------------------
# HP grids
# ---------------------------------------------------------------------------
# Phase-1 LR×BS grid for 0.6B (any domain). 9 cells — note (7e-6, 32) is
# intentionally absent, matching the original instruct sweeps (e.g.
# `medical_extraction_v2_sweep.py:68-83`).
HP_GRID_0_6B: list[dict] = [
    {"lr": lr, "bs": bs, "wd": 0.01, "warmup": 0.03}
    for lr, bs in [
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
]

# 8B code Phase-1 grid (1:1 with code_extraction_sft_v3_14b_sweep.py).
# 15 cells: 5 LRs × 3 BSs, fixed wd=0.01, warmup=0.03.
HP_GRID_8B_CODE: list[dict] = [
    {"lr": lr, "bs": bs, "wd": 0.01, "warmup": 0.03} for lr in [1e-6, 2e-6, 5e-6, 1e-5, 5e-5] for bs in [16, 32, 64]
]

# 8B math + medical: 3 transfer configs from code 14B (matches the pattern
# in `math_14b_top3_sft.py` and `medical_14b_sft.py`). "default" is the
# generic Marin SFT defaults; "best-resili" and "best-extract" come from
# code's 14B Phase-2 sweep winners.
HP_GRID_8B_TRANSFER: list[dict] = [
    {"name": "default", "lr": 2e-5, "bs": 64, "wd": 0.01, "warmup": 0.03},
    {"name": "best-resili", "lr": 1e-6, "bs": 32, "wd": 0.01, "warmup": 0.03},
    {"name": "best-extract", "lr": 2e-6, "bs": 32, "wd": 0.05, "warmup": 0.0},
]


def _resolve_hp_grid(domain: str, model_size: str) -> list[dict]:
    if model_size == "0.6b":
        return HP_GRID_0_6B
    if model_size == "8b":
        if domain == "code":
            return HP_GRID_8B_CODE
        # math + medical
        return HP_GRID_8B_TRANSFER
    if model_size == "14b":
        # Same 3-config transfer for math/medical, full LR×BS for code.
        if domain == "code":
            return HP_GRID_8B_CODE
        return HP_GRID_8B_TRANSFER
    raise ValueError(f"Unknown model_size: {model_size!r}")


# ---------------------------------------------------------------------------
# TPU + region defaults per size
# ---------------------------------------------------------------------------
TPU_DEFAULTS: dict[str, dict] = {
    # CPU memory bumped 2026-05-04 after OOM during HF export at end of
    # training. Levanter's save_hf_checkpoint pulls the model from TPU HBM
    # to CPU and rounds buffers to several multiples of the raw shard size.
    # Rule of thumb: budget ~6-10x the model's bf16 shard size.
    #   0.6B → 1.2GB shard → 64GB OK (~50x headroom)
    #   8B   → 16GB shard → 96GB needed (~6x headroom; tight but works)
    #   14B  → 28GB shard → 128GB safe
    "0.6b": {
        "primary": "v5p-8",
        "alternatives": ("v6e-4",),
        "memory": "64GB",
        "disk": "50GB",
        "cpu": 4,
    },
    "8b": {
        "primary": "v5p-32",
        "alternatives": ("v5p-16", "v6e-16"),
        # Bumped 96→128GB after smoke3 still hit OOM at 96GB. 8B model state
        # spikes transiently during HF download (16GB bf16 weights × ~5x
        # buffer factor in safetensors deserialization).
        "memory": "128GB",
        "disk": "100GB",
        "cpu": 8,
    },
    "14b": {
        "primary": "v5p-32",
        "alternatives": ("v5p-64",),
        "memory": "128GB",
        "disk": "100GB",
        "cpu": 8,
    },
}

DEFAULT_ALLOWED_REGIONS: tuple[str, ...] = ("us-central1", "us-east5", "us-east1")

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}

SCRIPT = "experiments/rephraser/run_sft_standalone.py"


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def _config_name(hp: dict) -> str:
    """Cell identifier. Uses `name` if provided (transfer configs), else lr/bs string.

    Format for LR×BS cells: `lr{1e-6}_bs{32}` (matches the existing
    `medical_extraction_v2_sweep.py:101-102` convention).
    """
    if "name" in hp:
        return hp["name"]
    lr_str = f"{hp['lr']:.0e}".replace("+", "").replace("-0", "-")
    return f"lr{lr_str}_bs{hp['bs']}"


# ---------------------------------------------------------------------------
# Submit one cell
# ---------------------------------------------------------------------------
def submit_one(
    client: IrisClient,
    *,
    domain: str,
    branch: str,
    model_size: str,
    hp: dict,
    cache_rel_path: str,
    cache_step_name: str,
    cache_tokenizer: str,
    hf_model_name: str,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    allowed_regions: tuple[str, ...],
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...],
    memory: str,
    disk: str,
    cpu: int,
    run_suffix: str,
) -> str:
    config_name = _config_name(hp)

    cmd_args = [
        "python",
        SCRIPT,
        "--domain",
        domain,
        "--branch",
        branch,
        "--model-size",
        model_size,
        "--config-name",
        config_name,
        "--cache-rel-path",
        cache_rel_path,
        "--cache-step-name",
        cache_step_name,
        "--cache-tokenizer",
        cache_tokenizer,
        "--hf-model-name",
        hf_model_name,
        "--learning-rate",
        f"{hp['lr']:.0e}",
        "--batch-size",
        str(hp["bs"]),
        "--weight-decay",
        str(hp["wd"]),
        "--warmup",
        str(hp["warmup"]),
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
    # Filter alternatives to those with the same vm_count as the primary —
    # iris's adjust_tpu_replicas auto-scales replicas using the primary's
    # vm_count, so mixing different vm_counts under one constraint causes the
    # scheduler to look for non-existent groups (see
    # launch_curation_sweep.py:140-148 for the canonical comment).
    primary_vm_count = get_tpu_topology(tpu_variant).vm_count
    compatible_alts = tuple(v for v in tpu_alternatives if get_tpu_topology(v).vm_count == primary_vm_count)
    all_variants = (tpu_variant, *compatible_alts)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    # Multi-host TPU support (vm_count > 1, e.g. v5p-16 with vm_count=2 for 8B
    # SFT). iris's adjust_tpu_replicas auto-scales replicas=1 -> vm_count for
    # multi-host topologies; coscheduling=group_by="tpu-name" gang-schedules
    # all replicas onto the same TPU slice so libtpu can wire up multi-host
    # JAX. For single-host (vm_count=1) we OMIT both — see
    # launch_curation_sweep.py:232-249 for the canonical justification.
    submit_kwargs: dict = {}
    if primary_vm_count > 1:
        submit_kwargs["replicas"] = 1
        submit_kwargs["coscheduling"] = CoschedulingConfig(group_by="tpu-name")

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"sft-{model_size}-{domain[:3]}-{branch[:3]}-{config_name}"[:200],
        resources=ResourceSpec(
            cpu=cpu,
            memory=memory,
            disk=disk,
            device=tpu_device(tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        **submit_kwargs,
        max_retries_preemption=100,
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info(
        "Submitted %s/%s/%s/%s -> %s",
        model_size,
        domain,
        branch,
        config_name,
        job.job_id,
    )
    return str(job.job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--domain", choices=["code", "math", "medical"], required=True)
    parser.add_argument("--branch", choices=["extraction", "resiliparse"], required=True)
    parser.add_argument("--model-size", choices=["0.6b", "8b", "14b"], required=True)
    parser.add_argument("--allowed-regions", nargs="+", default=list(DEFAULT_ALLOWED_REGIONS))
    parser.add_argument("--tpu-variant", default=None, help="Override default primary TPU shape.")
    parser.add_argument("--tpu-alternatives", nargs="*", default=None)
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument("--run-suffix", default="qwen3-base-rerun")
    parser.add_argument(
        "--cache-tokenizer",
        default="Qwen/Qwen3-0.6B-Base",
        help=(
            "Tokenizer to use at training time (gets padded by Levanter to match "
            "model's vocab). Default Qwen3-0.6B-Base which is byte-identical to all "
            "Qwen3 *-Base tokenizer.json files."
        ),
    )
    parser.add_argument(
        "--hf-model-name",
        default=None,
        help=(
            "Override the HF reference checkpoint. Default derived from model-size: "
            "0.6b → Qwen/Qwen3-0.6B-Base, 8b → Qwen/Qwen3-8B-Base, "
            "14b → Qwen/Qwen3-14B-Base."
        ),
    )
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument("--filter-config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    cache_rel_path, cache_step_name = CACHE_CATALOG[(args.domain, args.branch)]
    hps = list(_resolve_hp_grid(args.domain, args.model_size))
    hf_model_name = (
        args.hf_model_name
        or {
            "0.6b": "Qwen/Qwen3-0.6B-Base",
            "8b": "Qwen/Qwen3-8B-Base",
            "14b": "Qwen/Qwen3-14B-Base",
        }[args.model_size]
    )

    tpu_defaults = TPU_DEFAULTS[args.model_size]
    tpu_variant = args.tpu_variant or tpu_defaults["primary"]
    tpu_alternatives = (
        tuple(args.tpu_alternatives) if args.tpu_alternatives is not None else tpu_defaults["alternatives"]
    )

    if args.filter_config is not None:
        hps = [h for h in hps if args.filter_config in _config_name(h)]
    if args.max_count is not None:
        hps = hps[: args.max_count]

    logger.info(
        "Sweep: domain=%s branch=%s model=%s. %d cells. tpu_primary=%s alts=%s regions=%s",
        args.domain,
        args.branch,
        args.model_size,
        len(hps),
        tpu_variant,
        tpu_alternatives,
        args.allowed_regions,
    )
    for h in hps:
        logger.info("  %s", _config_name(h))

    if args.dry_run:
        return

    if not hps:
        logger.warning("No HP cells matched filter. Exiting.")
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
    for hp in hps:
        try:
            job_id = submit_one(
                client,
                domain=args.domain,
                branch=args.branch,
                model_size=args.model_size,
                hp=hp,
                cache_rel_path=cache_rel_path,
                cache_step_name=cache_step_name,
                cache_tokenizer=args.cache_tokenizer,
                hf_model_name=hf_model_name,
                child_priority_band=child_priority_band,
                wandb_api_key=wandb_api_key,
                hf_token=hf_token,
                allowed_regions=tuple(args.allowed_regions),
                tpu_variant=tpu_variant,
                tpu_alternatives=tpu_alternatives,
                memory=tpu_defaults["memory"],
                disk=tpu_defaults["disk"],
                cpu=tpu_defaults["cpu"],
                run_suffix=args.run_suffix,
            )
            submitted.append(job_id)
        except Exception as e:
            logger.exception("Failed to submit %s: %s", _config_name(hp), e)

    logger.info("Submitted %d/%d cells.", len(submitted), len(hps))

    if args.no_keep_alive:
        return

    logger.info("Coordinator entering keep-alive (sleep 3600 forever)…")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
