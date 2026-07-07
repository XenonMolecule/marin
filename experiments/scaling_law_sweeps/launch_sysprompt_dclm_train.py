# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot launcher: train a 447M model on the [S][D] sysprompt-DCLM dataset.

Reproduces the EXACT config of the dclm_10k "2e19" natural-epoch run
(`curation-dclm_10k-expFM_natural-2e+19-d1024-L11-B32`, budget 1.8e19 → 447.2M
params, batch=32, lr=0.00322, adam_lr=0.000271, beta2=0.9999, eps=4.483e-8) but
trains on the system-prompt-conditioned cache `sysprompt_dclm` instead of dclm_10k,
for exactly ONE natural epoch of that cache.

The ONLY deltas vs the reference plan:
  - method_name = "sysprompt_dclm" (the [S][D] cache, conditioned_text field).
  - train_steps = ceil(D_obs / (batch * seq_len))  -> one epoch of the new cache
    (7.818B tokens vs the reference's 7.340B). Optimizer HP are kept BYTE-IDENTICAL
    to the reference (recomputing at the +6.5% token count would shift lr ~2%).
  - t_exp = D_obs (the natural epoch token count).

Single child, pinned to us-east5 (cache is there only), interactive priority,
preemptible with durable 15-min checkpoints (resume-safe across preemption).

Usage (CPU coordinator on Iris; submits one TPU child and exits)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-east5 --cpu 2 --memory 3GB \\
        --priority interactive --job-name sysprompt-dclm-train-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/launch_sysprompt_dclm_train.py

    # inspect the plan locally without submitting:
    python experiments/scaling_law_sweeps/launch_sysprompt_dclm_train.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps import curation_plan
from experiments.scaling_law_sweeps.completed_adamh import SEQ_LEN, completed_adamh_heuristic
from experiments.scaling_law_sweeps.curation_plan import _planned_run_from_candidate
from experiments.scaling_law_sweeps.data_curation_math import implicit_target_exp_a
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_one

logger = logging.getLogger(__name__)

METHOD_NAME = "sysprompt_dclm"
REFERENCE_BUDGET = 1.8e19  # the d1024 "2e+19" cell
REFERENCE_HIDDEN_DIM = 1024
EXPERIMENT_TAG = "expFM_natural"  # natural epoching, no Levanter slicing

WANDB_GROUP = "sysprompt-dclm"
TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/sysprompt_dclm/"
RESULTS_PREFIX = "gs://marin-us-central1/metadata/sysprompt_dclm_results/"


def build_plan() -> curation_plan.PlannedRun:
    """Build the single PlannedRun: reference d1024@1.8e19 HP, one-epoch steps."""
    method = curation_plan.METHODS[METHOD_NAME]

    # Reference candidate: the d=1024 cell at the 1.8e19 budget (447.2M params,
    # batch=32, the frozen AdamH HP). Pull it straight from the heuristic so the
    # optimizer config is bit-identical to the dclm_10k "2e19" run.
    candidate = next(
        c
        for c in completed_adamh_heuristic.candidates_for_budget(REFERENCE_BUDGET)
        if c.model_config.hidden_dim == REFERENCE_HIDDEN_DIM
    )

    t_exp = float(method.d_obs_tokens)  # one natural epoch of the [S][D] cache
    target_budget = int(implicit_target_exp_a(method, t_exp))
    plan = _planned_run_from_candidate(
        method, candidate, budget=REFERENCE_BUDGET, target_budget=target_budget, tag=EXPERIMENT_TAG
    )

    # One epoch: walk the whole cache exactly once at the reference batch size.
    one_epoch_steps = math.ceil(method.d_obs_tokens / (plan.batch_size * SEQ_LEN))
    return dataclasses.replace(plan, train_steps=one_epoch_steps, t_exp=t_exp)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--priority", choices=["production", "interactive", "batch", "unspecified"], default="interactive")
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "offline_no_sync"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="Print the plan, do not submit.")
    p.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Exit immediately after submitting (for testing). DANGER: the child is "
        "nested under this coordinator job, so exiting finalizes the parent and "
        "orphan-kills the child. Leave OFF for real launches.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    plan = build_plan()
    tokens_trained = plan.batch_size * plan.seq_len * plan.train_steps
    logger.info(
        "Plan: %s | d=%d L=%d heads=%d int=%d | B=%d steps=%d (%.4fB tokens, %.4f epochs of %.4fB) | "
        "lr=%.6f adam_lr=%.6f beta2=%.5f eps=%.3e | tag=%s",
        plan.run_name_core,
        plan.hidden_dim,
        plan.num_layers,
        plan.num_heads,
        plan.intermediate_dim,
        plan.batch_size,
        plan.train_steps,
        tokens_trained / 1e9,
        tokens_trained / plan.t_exp,
        plan.t_exp / 1e9,
        plan.learning_rate,
        plan.adam_lr,
        plan.beta2,
        plan.epsilon,
        plan.experiment_tag,
    )
    logger.info("TPU variants: v4=%s v5p=%s v6e=%s | tensor_parallel=%d", plan.v4_tpu, plan.v5p_tpu, plan.v6e_tpu, plan.tensor_parallel)

    if args.dry_run:
        logger.info("--dry-run: not submitting.")
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- run this inside an Iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    job_id = submit_one(
        client,
        plan,
        child_priority_band=PRIORITY_BAND_MAP[args.priority],
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group=WANDB_GROUP,
        tracker_prefix=TRACKER_PREFIX,
        wandb_mode=args.wandb_mode,
        results_prefix=RESULTS_PREFIX,
    )
    logger.info("Submitted training child: %s", job_id)

    if args.no_keep_alive:
        return
    # The child is a DESCENDANT of this coordinator job. If this process exits,
    # iris finalizes the parent and orphan-kills the child (observed: child went
    # to "killed / Job finalized" before claiming a TPU). Stay alive for the
    # duration of training -- same keep-alive pattern as launch_10k_natural.
    logger.info("Coordinator entering keep-alive (keeps the training child from being orphan-killed)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
