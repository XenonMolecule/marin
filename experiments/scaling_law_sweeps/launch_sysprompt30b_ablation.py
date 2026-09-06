# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot launcher: the system-prompt ablation arms, all in parallel.

    A  sysprompt      docs with [S] prepended
    B  token-matched  more docs, no [S], same TOKEN count as A
    C  doc-matched    the SAME docs as A, no [S] (fewer tokens)
    D  mix50          the SAME docs as A, [S] on a deterministic doc_id-keyed half

Every arm uses the frozen ``completed_adamh`` cell for its (budget, width) — the
optimizer hyperparameters are byte-identical across arms, so the only difference
is the data. Arms differ ONLY in ``train_steps``, which is set to exactly ONE
epoch of that arm's own cache::

    train_steps = floor(d_obs / (batch_size * seq_len))

That is the "equal epochs" choice: no arm ever sees a document twice. Arm C
therefore runs ~1.7% fewer steps than A and B (its cache lacks the [S] tokens),
so C is deliberately NOT isoFLOP with the others. The alternative — equal steps
with C repeating 1.7% of its data — is a one-line change here.

Scale-generic: ``--budget`` and ``--hidden-dim`` pick the cell, ``--tag`` picks
the caches. The 998M @ 9e19 default is the frozen grid cell
(``PLAN_GRID.md``: 1536/16/12/6144, batch 64, 57,022 steps, 14.9B tokens, v5p-16).

The TPU shape is NOT hardcoded — it comes from the plan for that cell, so it grows
with the budget on its own: 9e19 -> v5p-16, 1.8e20 -> v5p-32, 3e20 -> v5p-64.

Usage (CPU coordinator on Iris; submits three TPU children and stays alive)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 2 --memory 4GB \\
        --priority interactive --job-name sp30b-998m-9e19-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/launch_sysprompt30b_ablation.py

    # inspect the three plans without submitting:
    python experiments/scaling_law_sweeps/launch_sysprompt30b_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps import curation_plan
from experiments.scaling_law_sweeps.completed_adamh import SEQ_LEN
from experiments.scaling_law_sweeps.curation_plan import _planned_run_from_candidate
from experiments.scaling_law_sweeps.data_curation_math import implicit_target_exp_a
from experiments.scaling_law_sweeps.fixed_model_plan import _candidate_for_fixed_model
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_one

logger = logging.getLogger(__name__)

EXPERIMENT_TAG = "expFM_natural"  # natural epoching -> no Levanter slicing
ARMS = ("A", "B", "C", "D")
ARM_DESC = {
    "A": "sysprompt [S][D]",
    "B": "token-matched baseline",
    "C": "doc-matched baseline",
    "D": "mix50: [S][D] on a doc_id-keyed half, [D] on the rest",
}
# Cache lives only in us-central1, and v5p-16 is available in us-central1-a.
PIN_REGION = "us-central1"


def build_plans(tag: str, budget: float, hidden_dim: int, arms: tuple[str, ...]) -> list[curation_plan.PlannedRun]:
    candidate = _candidate_for_fixed_model(hidden_dim, budget, seq_len=SEQ_LEN)
    if candidate is None:
        raise SystemExit(f"no candidate config for hidden_dim={hidden_dim} at budget={budget:.3g}")

    plans = []
    for arm in arms:
        method_name = f"sysprompt30b_{tag}_{arm}"
        if method_name not in curation_plan.METHODS:
            raise SystemExit(
                f"method {method_name!r} is not registered in curation_plan.METHODS.\n"
                "Tokenize the arm first, then add its cache hash to _D_OBS_DEFAULTS "
                "and register the method."
            )
        method = curation_plan.METHODS[method_name]
        t_exp = float(method.d_obs_tokens)
        plan = _planned_run_from_candidate(
            method,
            candidate,
            budget=budget,
            target_budget=int(implicit_target_exp_a(method, t_exp)),
            tag=EXPERIMENT_TAG,
        )
        # Exactly one epoch of THIS arm's cache — floor, so no document is ever
        # seen twice. This is the only field that differs between arms.
        one_epoch = int(method.d_obs_tokens // (plan.batch_size * SEQ_LEN))
        plans.append(dataclasses.replace(plan, train_steps=one_epoch, t_exp=t_exp))
    return plans


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", default="998m_9e19", help="Scale tag used in method/cache names.")
    p.add_argument("--budget", type=float, default=9e19)
    p.add_argument("--hidden-dim", type=int, default=1536)
    p.add_argument("--arms", default=",".join(ARMS))
    # Default None -> use the shape the frozen plan computes for this cell, so
    # the TPU scales with the budget automatically: 9e19 -> v5p-16,
    # 1.8e20 -> v5p-32, 3e20 -> v5p-64. Pass --tpu only to override.
    p.add_argument("--tpu", default=None, help="Override primary TPU shape (default: the plan's v5p shape).")
    # User granted interactive for this ablation. NOTE: while the account is over
    # budget the scheduler demotes the effective band to BATCH anyway — asking for
    # interactive costs nothing and takes effect the moment the budget eases.
    p.add_argument("--priority", choices=["production", "interactive", "batch", "unspecified"], default="interactive")
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "offline_no_sync"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="Print the plans, do not submit.")
    p.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Exit right after submitting (testing only). DANGER: children are nested "
        "under this coordinator, so exiting orphan-kills them before they claim a TPU.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    arms = tuple(x.strip().upper() for x in args.arms.split(",") if x.strip())
    bad = set(arms) - set(ARMS)
    if bad:
        raise SystemExit(f"unknown arms: {sorted(bad)}")
    plans = build_plans(args.tag, args.budget, args.hidden_dim, arms)

    for arm, plan in zip(arms, plans, strict=True):
        tokens = plan.batch_size * plan.seq_len * plan.train_steps
        logger.info(
            "ARM %s (%s): %s | B=%d steps=%d -> %.4fB tokens = %.4f epochs of %.4fB | "
            "lr=%.10g adam_lr=%.10g beta2=%.8g eps=%.4e | tpu=%s tp=%d",
            arm,
            ARM_DESC[arm],
            plan.run_name_core,
            plan.batch_size,
            plan.train_steps,
            tokens / 1e9,
            tokens / plan.t_exp,
            plan.t_exp / 1e9,
            plan.learning_rate,
            plan.adam_lr,
            plan.beta2,
            plan.epsilon,
            args.tpu or plan.v5p_tpu,
            plan.tensor_parallel,
        )

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

    for arm, plan in zip(arms, plans, strict=True):
        job_id = submit_one(
            client,
            plan,
            child_priority_band=PRIORITY_BAND_MAP[args.priority],
            wandb_api_key=wandb_api_key,
            hf_token=hf_token,
            wandb_project="marin",
            wandb_entity="marin-community",
            wandb_group=f"sysprompt30b-{args.tag}",
            tracker_prefix=f"gs://marin-us-central1/metadata/region_locks/sysprompt30b_{args.tag}/",
            results_prefix=f"gs://marin-us-central1/metadata/sysprompt30b_{args.tag}_results/",
            allowed_regions=[PIN_REGION],
            force_primary_tpu=args.tpu or plan.v5p_tpu,
            wandb_mode=args.wandb_mode,
            # Slice-portable compile cache: a preempted run warm-hits on a new
            # slice (~7 min) instead of a full cold recompile.
            extra_env={"LEVANTER_PORTABLE_TPU_CACHE": "1"},
        )
        logger.info("ARM %s submitted: %s", arm, job_id)

    if args.no_keep_alive:
        return
    logger.info("Coordinator entering keep-alive (children are nested under it)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
