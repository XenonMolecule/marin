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

Sweep mode: ``--sweep <name>`` runs every arm at every point of a named
hyperparameter sweep (see ``SWEEPS``). A point may scale the AdamH learning
rate, change the batch size (the other optimizer values are re-derived from the
delphi heuristic for that batch), or override beta1/beta2. Each point gets a run
suffix so its checkpoints, W&B run and result JSON never collide with the plain
one-epoch runs or with each other. ``train_steps`` is still exactly one epoch of
the arm's cache at the point's batch size.

Usage (CPU coordinator on Iris; submits three TPU children and stays alive)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 2 --memory 4GB \\
        --priority interactive --job-name sp30b-998m-9e19-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/launch_sysprompt30b_ablation.py

    # inspect the three plans without submitting:
    python experiments/scaling_law_sweeps/launch_sysprompt30b_ablation.py --dry-run

    # the 447M @ 9e18 hyperparameter sweep, 3 arms x 9 points, batch band:
    python experiments/scaling_law_sweeps/launch_sysprompt30b_ablation.py \\
        --tag 447m_9e18 --budget 9e18 --hidden-dim 1024 --arms A,C,D \\
        --sweep hp1 --memory-gb 128 --priority batch
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps import curation_plan
from experiments.scaling_law_sweeps.completed_adamh import SEQ_LEN, CandidateConfig, completed_adamh_heuristic
from experiments.scaling_law_sweeps.curation_plan import _planned_run_from_candidate
from experiments.scaling_law_sweeps.data_curation_math import implicit_target_exp_a
from experiments.scaling_law_sweeps.fixed_model_plan import _candidate_for_fixed_model
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_one

logger = logging.getLogger(__name__)

EXPERIMENT_TAG = "expFM_natural"  # natural epoching -> no Levanter slicing
ARMS = ("A", "B", "C", "D", "AM", "E")
ARM_DESC = {
    "A": "sysprompt [S][D]",
    "B": "token-matched baseline",
    "C": "doc-matched baseline",
    "D": "mix50: [S][D] on a doc_id-keyed half, [D] on the rest",
    "AM": "sysprompt [S][D], loss masked on the [S] prefix",
    "E": "cooldown: [S][D] on the first 80% of docs, then [D] on the last 20%",
}
# Cache lives only in us-central1, and v5p-16 is available in us-central1-a.
PIN_REGION = "us-central1"
# Seconds between child submits. Sequential, spaced submits keep the controller
# intake responsive; parallel launchers have wedged it before.
SUBMIT_SPACING = 5.0


@dataclasses.dataclass(frozen=True)
class SweepPoint:
    """One hyperparameter setting applied on top of the frozen delphi cell.

    ``lr_mult`` scales the AdamH matrix learning rate only (``adam_lr`` for the
    embedding/norm path stays at its delphi value). ``batch_size`` replaces the
    cell's batch, with ``adam_lr``/``epsilon``/``beta2`` re-derived by the delphi
    heuristic for that batch before ``lr_mult`` is applied. ``beta1``/``beta2``
    override the derived values when set.
    """

    lr_mult: float = 1.0
    batch_size: int | None = None
    beta1: float | None = None
    beta2: float | None = None

    @property
    def suffix(self) -> str:
        parts = [f"lr{_fmt(self.lr_mult)}x"]
        if self.beta1 is not None:
            parts.append(f"b1-{_fmt(self.beta1)}")
        if self.beta2 is not None:
            parts.append(f"b2-{_fmt(self.beta2)}")
        # batch is already part of run_name_core (-B{batch}).
        return "sw-" + "-".join(parts)


def _fmt(x: float) -> str:
    return f"{x:g}".replace(".", "p")


# hp1 (2026-09-25): 447M @ 9e18, one epoch, arms A/C/D. A 3x2 grid over the
# AdamH learning rate and batch size, plus centre-point spokes for beta2 (the
# delphi rule clips to 0.9999 for every batch <= 64, so it was never fit for
# this cell) and beta1 (fixed at 0.9 everywhere in the heuristic).
SWEEPS: dict[str, tuple[SweepPoint, ...]] = {
    "hp1": (
        *(SweepPoint(lr_mult=m, batch_size=b) for b in (16, 32) for m in (0.5, 1.0, 2.0)),
        SweepPoint(beta2=0.95),
        SweepPoint(beta2=0.999),
        SweepPoint(beta1=0.95),
    ),
    # probe1 (2026-09-26): the masked-[S] and cooldown arms at hp1's best A point
    # (batch 32, delphi lr). probe1b repeats it with the short beta2 half-life.
    "probe1": (SweepPoint(batch_size=32),),
    "probe1b": (SweepPoint(batch_size=32, beta2=0.999),),
}


def _candidate_at_batch(candidate: CandidateConfig, batch_size: int) -> CandidateConfig:
    """The same cell at a different batch: same tokens, steps and optimizer re-derived."""
    if batch_size == candidate.batch_size:
        return candidate
    train_steps = round(candidate.tokens / (batch_size * SEQ_LEN))
    return dataclasses.replace(
        candidate,
        batch_size=batch_size,
        train_steps=train_steps,
        optimizer_config=completed_adamh_heuristic.build_optimizer_config(batch_size, candidate.tokens),
    )


def build_plans(
    tag: str,
    budget: float,
    hidden_dim: int,
    arms: tuple[str, ...],
    memory_gb: int | None = None,
    point: SweepPoint = SweepPoint(),
    epochs: int = 1,
) -> list[curation_plan.PlannedRun]:
    candidate = _candidate_for_fixed_model(hidden_dim, budget, seq_len=SEQ_LEN)
    if candidate is None:
        raise SystemExit(f"no candidate config for hidden_dim={hidden_dim} at budget={budget:.3g}")
    if point.batch_size is not None:
        candidate = _candidate_at_batch(candidate, point.batch_size)

    plans = []
    for arm in arms:
        method_name = f"sysprompt30b_{tag}_{arm}" + (f"_ep{epochs}" if epochs > 1 else "")
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
        one_epoch = method.train_steps_for(plan.batch_size, SEQ_LEN)
        plan = dataclasses.replace(
            plan,
            train_steps=one_epoch,
            t_exp=t_exp,
            learning_rate=plan.learning_rate * point.lr_mult,
            beta1=plan.beta1 if point.beta1 is None else point.beta1,
            beta2=plan.beta2 if point.beta2 is None else point.beta2,
        )
        if memory_gb is not None:
            plan = dataclasses.replace(plan, memory_gb=memory_gb)
        plans.append(plan)
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
    # The plan default (256 GB) was sized for d4096 multi-host smokes. On a
    # shared preemptible host, CPU-only jobs land on the same VM and eat RAM, so
    # a 256 GB request can pend on "Insufficient memory" while 4 idle chips sit
    # there. Sub-1B single-host runs peak well under 100 GB.
    p.add_argument("--memory-gb", type=int, default=None, help="Override the child container RAM limit (GB).")
    # User granted interactive for this ablation. NOTE: while the account is over
    # budget the scheduler demotes the effective band to BATCH anyway — asking for
    # interactive costs nothing and takes effect the moment the budget eases.
    p.add_argument("--priority", choices=["production", "interactive", "batch", "unspecified"], default="interactive")
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "offline_no_sync"], default="auto")
    p.add_argument("--sweep", choices=sorted(SWEEPS), default=None, help="Run every arm at every point of this sweep.")
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Passes over each arm's cache (reshuffled per pass). >1 trains the `<arm>_ep<N>` methods; "
        "pass --budget for the processed-token horizon (N x the one-epoch cell) so the optimizer is re-derived.",
    )
    p.add_argument("--seed", type=int, default=0, help="Training seed; non-zero seeds are appended to the run suffix.")
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
    points: tuple[SweepPoint | None, ...] = SWEEPS[args.sweep] if args.sweep else (None,)
    runs: list[tuple[str, str, curation_plan.PlannedRun]] = []
    for point in points:
        plans = build_plans(
            args.tag,
            args.budget,
            args.hidden_dim,
            arms,
            memory_gb=args.memory_gb,
            point=point or SweepPoint(),
            epochs=args.epochs,
        )
        suffix_parts = ([point.suffix] if point else []) + ([f"s{args.seed}"] if args.seed else [])
        suffix = "-".join(suffix_parts)
        for arm, plan in zip(arms, plans, strict=True):
            runs.append((arm, suffix, plan))

    for arm, suffix, plan in runs:
        tokens = plan.batch_size * plan.seq_len * plan.train_steps
        logger.info(
            "ARM %s (%s): %s%s | B=%d steps=%d -> %.4fB tokens = %.4f epochs of %.4fB | "
            "lr=%.10g adam_lr=%.10g beta1=%g beta2=%.8g eps=%.4e | tpu=%s tp=%d mem=%dGB",
            arm,
            ARM_DESC[arm],
            plan.run_name_core,
            f"-{suffix}" if suffix else "",
            plan.batch_size,
            plan.train_steps,
            tokens / 1e9,
            tokens / plan.t_exp,
            plan.t_exp / 1e9,
            plan.learning_rate,
            plan.adam_lr,
            plan.beta1,
            plan.beta2,
            plan.epsilon,
            args.tpu or plan.v5p_tpu,
            plan.tensor_parallel,
            plan.memory_gb,
        )
    logger.info("%d runs planned.", len(runs))

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
    wandb_group = f"sysprompt30b-{args.tag}" + (f"-{args.sweep}" if args.sweep else "")

    for i, (arm, suffix, plan) in enumerate(runs):
        if i:
            time.sleep(SUBMIT_SPACING)
        job_id = submit_one(
            client,
            plan,
            child_priority_band=PRIORITY_BAND_MAP[args.priority],
            wandb_api_key=wandb_api_key,
            hf_token=hf_token,
            wandb_project="marin",
            wandb_entity="marin-community",
            wandb_group=wandb_group,
            tracker_prefix=f"gs://marin-us-central1/metadata/region_locks/sysprompt30b_{args.tag}/",
            results_prefix=f"gs://marin-us-central1/metadata/sysprompt30b_{args.tag}_results/",
            allowed_regions=[PIN_REGION],
            force_primary_tpu=args.tpu or plan.v5p_tpu,
            wandb_mode=args.wandb_mode,
            run_suffix=suffix,
            seed=args.seed,
            # Suffixed sweep points and seed replicates must not overwrite the base run's summary.
            canonical_summary=not (args.sweep or args.seed),
            # Slice-portable compile cache: a preempted run warm-hits on a new
            # slice (~7 min) instead of a full cold recompile.
            extra_env={"LEVANTER_PORTABLE_TPU_CACHE": "1"},
        )
        logger.info("ARM %s %s submitted: %s", arm, suffix or "(base)", job_id)

    if args.no_keep_alive:
        return
    logger.info("Coordinator entering keep-alive (children are nested under it)...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
