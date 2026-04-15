# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure enumeration helper for the data-curation IsoFLOP sweep.

Imported by both the coordinator (`launch_curation_sweep.py`) and the standalone
child (`run_curation_train_standalone.py`). Pure functions, no iris/fray/executor
dependencies — safe to test in isolation and runs fast even without GCS access.

The `PlannedRun` dataclass is the contract between coordinator and child:
the coordinator computes one per (method, experiment, candidate) and serializes
to CLI args; the child rehydrates, region-locks, and runs Levanter training.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass

from experiments.scaling_law_sweeps.completed_adamh import (
    SEQ_LEN,
    completed_adamh_heuristic,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    CurationMethod,
    implicit_target_exp_a,
    load_d_obs_from_stats,
    slice_tokens_for,
    t_exp_ceiling,
)
from marin.scaling_laws import CandidateConfig, pick_v4_type, pick_v5p_type

logger = logging.getLogger(__name__)


# --- v6e (Trillium) TPU picker -----------------------------------------------
# marin.scaling_laws doesn't export a v6e picker yet; keep this local until it's
# upstreamed. v6e-{slice_size} is named by CHIP count (unlike v4/v5p which name
# by core count). HBM per chip = 32 GiB.
#
# Available v6e slice sizes from iris TpuTopologyInfo:
#   v6e-4, v6e-8, v6e-16, v6e-32, v6e-64, v6e-128, v6e-256
# vm_count matters for device_variant_constraint compatibility:
#   v6e-4  → vm_count=1   (matches v4-8 / v5p-8 tier)
#   v6e-8  → vm_count=1   (matches v4-8 / v5p-8 tier, MORE memory)
#   v6e-16 → vm_count=4   (DIFFERENT from v4-16 / v5p-16 which are vm_count=2)
# So v6e can only be included as a constraint alternative at the vm_count=1
# tier — i.e. for plans that would otherwise use v4-8 / v5p-8.
_V6E_HBM_PER_CHIP_GIB = 32
_V6E_SINGLE_VM_SLICES = [4, 8]  # slice sizes with vm_count=1 (compatible alt for v4-8/v5p-8)


def pick_v6e_type_single_vm(estimated_memory_bytes: int) -> str | None:
    """Smallest v6e slice with vm_count=1 that fits the memory.

    Returns None if memory exceeds v6e-8 (i.e. the plan needs vm_count>1 variants,
    where v6e topology diverges from v4/v5p and can't be a compatible alternative).
    """
    chip_bytes = _V6E_HBM_PER_CHIP_GIB * 1024**3
    chips_req = math.ceil(estimated_memory_bytes / chip_bytes)
    for s in _V6E_SINGLE_VM_SLICES:
        if s >= chips_req:
            return f"v6e-{s}"
    return None  # too big — v6e-16+ has vm_count!=1, can't mix with v4/v5p


import math  # re-import for pick_v6e; python allows this

# --- Compute budgets ----------------------------------------------------------
# Matches delphi's `completed_adamh` defaults (7 log-spaced FLOP points).
BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20)


# --- Experiment B target tokens ----------------------------------------------
# Grounded per internal review (20T ≈ 1T-param Chinchilla).
DEFAULT_T_TARGETS: tuple[float, ...] = (20e12,)


# --- Slice safety floor -------------------------------------------------------
MIN_SLICE_TOKENS_DEFAULT: float = 25e6


# --- D_obs constants for the methods registry --------------------------------
# Hardcoded to avoid a slow GCS read at every coordinator/child startup.
# Update from `{path}/train/.stats.json:total_tokens` after re-tokenization.
_D_OBS_DEFAULTS: dict[str, int] = {
    "baseline_dclm-23e9be": 2_663_454_015,
    "baseline_nemotron-c67de9": 1_919_401_016,
    "baseline_nemotron_full-d4e3af": 2_695_507_851,
    "baseline_fineweb_edu-7a3bc5": 817_221_529,
    "baseline_resiliparse-7278c1": 142_652_598_588,
}
_SOURCE_BUCKET: str = "gs://marin-us-central2"


def _method(
    name: str,
    cache_hash: str,
    sampled_warcs: int = 3_000,
    reproduce_per_region: bool = False,
    skip_stats_read: bool = True,
) -> CurationMethod:
    """Build a CurationMethod from a cache hash (the directory under tokenized/)."""
    if cache_hash not in _D_OBS_DEFAULTS:
        raise KeyError(f"No hardcoded D_obs for {cache_hash!r}. Add to _D_OBS_DEFAULTS.")
    d_obs = _D_OBS_DEFAULTS[cache_hash]
    if not skip_stats_read:
        source_stats_path = f"{_SOURCE_BUCKET}/tokenized/{cache_hash}/"
        try:
            live = load_d_obs_from_stats(source_stats_path)
            if live != d_obs:
                logger.warning(
                    "Stale d_obs for %s: hardcoded=%d but live=%d.",
                    cache_hash,
                    d_obs,
                    live,
                )
                d_obs = live
        except Exception as e:
            logger.warning("Could not verify d_obs from %s (%s)", source_stats_path, e)
    return CurationMethod(
        name=name,
        tokenized_rel_path=f"tokenized/{cache_hash}/",
        d_obs_tokens=d_obs,
        sampled_warcs=sampled_warcs,
        reproduce_per_region=reproduce_per_region,
    )


METHODS: dict[str, CurationMethod] = {
    "dclm": _method("dclm", "baseline_dclm-23e9be"),
    "nemotron_org": _method("nemotron_org", "baseline_nemotron-c67de9"),
    "nemotron_full": _method("nemotron_full", "baseline_nemotron_full-d4e3af"),
    "fineweb_edu": _method("fineweb_edu", "baseline_fineweb_edu-7a3bc5"),
    "resiliparse": _method("resiliparse", "baseline_resiliparse-7278c1", reproduce_per_region=True),
    # TODO: add the user's LLM-based method once tokenization completes:
    # "llm_curated": _method("llm_curated", "<TBD>"),
}


# --- Experiment-tag formatter ------------------------------------------------


def experiment_tag(t_target: float | None) -> str:
    if t_target is None:
        return "expA_natural"
    trillions = t_target / 1e12
    if trillions == int(trillions):
        return f"expB_T{int(trillions)}T"
    return f"expB_T{trillions:.1f}T"


# --- PlannedRun: the coordinator → child contract ----------------------------


@dataclass(frozen=True)
class PlannedRun:
    """One planned training run. Serialized to/from CLI args at the coord/child boundary.

    The (model, optimizer) hyperparameters are passed explicitly so the child
    can rebuild a deterministic CandidateConfig without re-invoking the
    heuristic — which would risk drift if the heuristic is later edited.
    """

    method_name: str  # e.g. "dclm"
    experiment_tag: str  # e.g. "expA_natural" or "expB_T20T"
    budget: float  # FLOPs
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    batch_size: int
    train_steps: int
    learning_rate: float
    adam_lr: float
    epsilon: float
    beta1: float
    beta2: float
    t_exp: float  # tokens this run will train on
    t_target: float  # the target regime being simulated
    seq_len: int = SEQ_LEN
    tensor_parallel: int = 1
    z_loss_weight: float = 1e-7
    estimated_memory_bytes: int = 0
    v4_tpu: str = "v4-8"
    v5p_tpu: str = "v5p-8"
    v6e_tpu: str = ""  # empty when plan is too big for v6e single-vm slices
    cpu: float = 32.0
    memory_gb: int = 128
    disk_gb: int = 50

    @property
    def run_name_core(self) -> str:
        # Stable identifier — used for both the iris job name and the output_path.
        return (
            f"curation-{self.method_name}-{self.experiment_tag}"
            f"-{self.budget:.0e}-d{self.hidden_dim}-L{self.num_layers}-B{self.batch_size}"
        )

    @property
    def run_key(self) -> str:
        """Stable key for the region tracker. Matches region_tracker.run_key_for(...) format."""
        return f"{self.method_name}__{self.experiment_tag}__{self.run_name_core}.region"

    def to_cli_args(self) -> list[str]:
        """Serialize to argv list for the standalone child script."""
        return [
            "--method",
            self.method_name,
            "--experiment-tag",
            self.experiment_tag,
            "--budget",
            f"{self.budget:.6e}",
            "--hidden-dim",
            str(self.hidden_dim),
            "--num-layers",
            str(self.num_layers),
            "--num-heads",
            str(self.num_heads),
            "--intermediate-dim",
            str(self.intermediate_dim),
            "--batch-size",
            str(self.batch_size),
            "--train-steps",
            str(self.train_steps),
            "--learning-rate",
            f"{self.learning_rate:.6e}",
            "--adam-lr",
            f"{self.adam_lr:.6e}",
            "--epsilon",
            f"{self.epsilon:.6e}",
            "--beta1",
            f"{self.beta1}",
            "--beta2",
            f"{self.beta2}",
            "--t-exp",
            f"{self.t_exp:.6e}",
            "--t-target",
            f"{self.t_target:.6e}",
            "--seq-len",
            str(self.seq_len),
            "--tensor-parallel",
            str(self.tensor_parallel),
            "--z-loss-weight",
            f"{self.z_loss_weight:.6e}",
        ]

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Register the inverse of `to_cli_args` on a parser."""
        parser.add_argument("--method", required=True)
        parser.add_argument("--experiment-tag", required=True)
        parser.add_argument("--budget", type=float, required=True)
        parser.add_argument("--hidden-dim", type=int, required=True)
        parser.add_argument("--num-layers", type=int, required=True)
        parser.add_argument("--num-heads", type=int, required=True)
        parser.add_argument("--intermediate-dim", type=int, required=True)
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--train-steps", type=int, required=True)
        parser.add_argument("--learning-rate", type=float, required=True)
        parser.add_argument("--adam-lr", type=float, required=True)
        parser.add_argument("--epsilon", type=float, required=True)
        parser.add_argument("--beta1", type=float, required=True)
        parser.add_argument("--beta2", type=float, required=True)
        parser.add_argument("--t-exp", type=float, required=True)
        parser.add_argument("--t-target", type=float, required=True)
        parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
        parser.add_argument("--tensor-parallel", type=int, default=1)
        parser.add_argument("--z-loss-weight", type=float, default=1e-7)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> PlannedRun:
        """Inverse of `to_cli_args` — rebuild from parsed args (defaults set for irrelevant fields)."""
        return cls(
            method_name=ns.method,
            experiment_tag=ns.experiment_tag,
            budget=ns.budget,
            hidden_dim=ns.hidden_dim,
            num_layers=ns.num_layers,
            num_heads=ns.num_heads,
            intermediate_dim=ns.intermediate_dim,
            batch_size=ns.batch_size,
            train_steps=ns.train_steps,
            learning_rate=ns.learning_rate,
            adam_lr=ns.adam_lr,
            epsilon=ns.epsilon,
            beta1=ns.beta1,
            beta2=ns.beta2,
            t_exp=ns.t_exp,
            t_target=ns.t_target,
            seq_len=ns.seq_len,
            tensor_parallel=ns.tensor_parallel,
            z_loss_weight=ns.z_loss_weight,
            # Coordinator-side fields not needed in the child:
            estimated_memory_bytes=0,
            v4_tpu="",
            v5p_tpu="",
        )


# --- Enumeration -------------------------------------------------------------


def _iter_valid_candidates(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> Iterator[tuple[float, CandidateConfig, int]]:
    """Yield (budget, candidate, target_budget) for each candidate that survives filtering.

    - Experiment A (t_target=None): no ceiling, implicit target = T_exp * s.
    - Experiment B (t_target set):  reject T_exp > ceiling; reject slice < floor.
    """
    for budget in budgets:
        for cand in completed_adamh_heuristic.candidates_for_budget(budget, seq_len=seq_len):
            t_exp = cand.tokens
            if t_target is not None:
                if t_exp > t_exp_ceiling(method, t_target):
                    continue
                if slice_tokens_for(method, t_exp, t_target) < min_slice_tokens:
                    continue
                target_budget = int(t_target)
            else:
                target_budget = int(implicit_target_exp_a(method, t_exp))
            yield budget, cand, target_budget


def _planned_run_from_candidate(
    method: CurationMethod,
    candidate: CandidateConfig,
    budget: float,
    target_budget: int,
    tag: str,
    seq_len: int = SEQ_LEN,
) -> PlannedRun:
    model = candidate.model_config
    opt = candidate.optimizer_config
    mem = completed_adamh_heuristic.estimate_memory_bytes(candidate)
    return PlannedRun(
        method_name=method.name,
        experiment_tag=tag,
        budget=budget,
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        intermediate_dim=model.intermediate_dim,
        batch_size=candidate.batch_size,
        train_steps=candidate.train_steps,
        learning_rate=opt.learning_rate,
        adam_lr=opt.adam_lr,
        epsilon=opt.epsilon,
        beta1=opt.beta1,
        beta2=opt.beta2,
        t_exp=candidate.tokens,
        t_target=float(target_budget),
        seq_len=seq_len,
        tensor_parallel=1,
        z_loss_weight=completed_adamh_heuristic.z_loss_weight,
        estimated_memory_bytes=mem,
        v4_tpu=pick_v4_type(mem),
        v5p_tpu=pick_v5p_type(mem),
        v6e_tpu=pick_v6e_type_single_vm(mem) or "",
    )


def enumerate_plans(
    methods: list[CurationMethod],
    experiments: list[float | None],
    *,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> list[PlannedRun]:
    """Build the full list of `PlannedRun`s for the cartesian product of methods × experiments."""
    plans: list[PlannedRun] = []
    for method in methods:
        for t_target in experiments:
            tag = experiment_tag(t_target)
            for budget, cand, target_budget in _iter_valid_candidates(
                method,
                t_target=t_target,
                budgets=budgets,
                min_slice_tokens=min_slice_tokens,
                seq_len=seq_len,
            ):
                plans.append(_planned_run_from_candidate(method, cand, budget, target_budget, tag, seq_len))
    return plans


# --- Pure enumeration helpers (no submission) --------------------------------


def count_valid(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> int:
    return sum(
        1
        for _ in _iter_valid_candidates(
            method,
            t_target=t_target,
            budgets=budgets,
            min_slice_tokens=min_slice_tokens,
            seq_len=seq_len,
        )
    )


def per_budget_counts(
    method: CurationMethod,
    *,
    t_target: float | None,
    budgets: tuple[float, ...] = BUDGETS,
    min_slice_tokens: float = MIN_SLICE_TOKENS_DEFAULT,
    seq_len: int = SEQ_LEN,
) -> dict[float, int]:
    counts: dict[float, int] = {b: 0 for b in budgets}
    for budget, _, _ in _iter_valid_candidates(
        method,
        t_target=t_target,
        budgets=budgets,
        min_slice_tokens=min_slice_tokens,
        seq_len=seq_len,
    ):
        counts[budget] += 1
    return counts


# --- CLI helpers (shared between coordinator and dry-run) --------------------


def resolve_methods(method_names: list[str]) -> list[CurationMethod]:
    if "all" in method_names:
        return list(METHODS.values())
    missing = [m for m in method_names if m not in METHODS]
    if missing:
        raise ValueError(f"Unknown method(s): {missing}. Available: {list(METHODS)}")
    return [METHODS[m] for m in method_names]


def resolve_experiments(experiments: list[str], t_targets: list[float]) -> list[float | None]:
    resolved: list[float | None] = []
    want_a = "A" in experiments or "all" in experiments
    want_b = "B" in experiments or "all" in experiments
    if want_a:
        resolved.append(None)
    if want_b:
        resolved.extend(t_targets)
    return resolved


def print_dry_run(plans: list[PlannedRun]) -> None:
    """Pretty-print per-(method, experiment) counts and TPU pair distribution."""
    by_group: dict[tuple[str, str], list[PlannedRun]] = {}
    for p in plans:
        by_group.setdefault((p.method_name, p.experiment_tag), []).append(p)

    for (method, tag), group in sorted(by_group.items()):
        n = len(group)
        v4 = Counter(p.v4_tpu for p in group)
        v5p = Counter(p.v5p_tpu for p in group)
        v4_str = ", ".join(f"{k}:{v}" for k, v in sorted(v4.items()))
        v5p_str = ", ".join(f"{k}:{v}" for k, v in sorted(v5p.items()))
        print(f"{method:>15} | {tag:>14} | runs={n:>3} | v4=[{v4_str}]  v5p=[{v5p_str}]")
    print(f"\n  TOTAL: {len(plans)} runs")
