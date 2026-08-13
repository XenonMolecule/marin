# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mixture optimization: turn fitted per-task laws into the proposed mixture.

Step 3 of OlmixBase (Algorithm 1). Ported from
``olmix/fit/utils.py::LogLinearExactProposer``:

    minimize_x  sum_i w_i * exp(t_i . x)  +  lambda * sum_j rel_entr(x_j, p0_j)
    subject to  x >= 0,  sum(x) == 1,  x_j <= k * N_j / R

with ``w_i = 1/n`` (flat mean of per-task predictions -- OlmixBase does not weight
tasks; the ``obj_weights`` path in the reference belongs to the per-*family*
ablation, Table 3's middle row), ``lambda = 0.05``, ``k = 4``, and ECOS as the
solver. The fitted ``exp(log c_i)`` offsets are dropped because they are additive
constants that cannot change the argmin -- the reference drops them too.

Two facts about this objective that shape how results should be read:

* the KL term is what makes the exact solver work. Paper Figure 8: the unregularized
  exact solver minimizes *predicted* BPB best but lands on worse *downstream* BPB
  than lambda=0.05, because the regression is imperfect and the proxy-to-target
  transfer is noisy. So lambda is not cosmetic.
* it also means the proposed mixture is **dense**. ``d/dx [x log(x/q)] -> -inf`` as
  ``x -> 0+``, so the penalty actively pushes weights off zero. Sparsity in OlmixBase
  comes from the *swarm* (see ``olmix_sample.minimum_weight_for_m``), which lets the
  per-task fits learn that a domain can be excluded; it does not come from here.
  :func:`mixture_density` reports how dense the answer actually is.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

logger = logging.getLogger(__name__)

# OlmixBase defaults: every fit config in the olmix repo uses these.
DEFAULT_KL_REG = 0.05
DEFAULT_REPETITION_FACTOR = 4.0

# cvxpy needs q > 0 for rel_entr; the reference floors it at 1e-12 then renormalizes.
PRIOR_FLOOR = 1e-12

SOLVER = "ECOS"


def availability_caps(tokens: np.ndarray, *, requested_tokens: float, repetition_factor: float) -> np.ndarray:
    """``k * N_j / R``: the most of domain ``j`` a mixture may draw.

    Not clipped to 1.0 -- the reference does not clip here either (unlike its
    sampler, which does). A cap above 1 is simply non-binding.
    """
    if requested_tokens <= 0:
        raise ValueError(f"requested_tokens must be positive, got {requested_tokens}")
    if repetition_factor <= 0:
        raise ValueError(f"repetition_factor must be positive, got {repetition_factor}")
    return tokens * repetition_factor / requested_tokens


def feasibility_slack(tokens: np.ndarray, *, requested_tokens: float, repetition_factor: float) -> float:
    """``k * sum(N) / R``. The constraint set is non-empty iff this is >= 1.

    Below 1 the request exceeds ``k`` epochs of the entire corpus, so no mixture on
    the simplex can satisfy the caps and the solve is infeasible -- worth reporting
    as a number rather than discovering as a solver status.
    """
    return float(tokens.sum() * repetition_factor / requested_tokens)


def solve_mixture(
    *,
    params: np.ndarray,
    prior: np.ndarray,
    tokens: np.ndarray | None = None,
    requested_tokens: float | None = None,
    repetition_factor: float = DEFAULT_REPETITION_FACTOR,
    kl_reg: float = DEFAULT_KL_REG,
    task_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Solve for the proposed mixture.

    Args:
        params: ``(n_tasks, m + 1)`` fitted ``[log_c, t...]`` rows from
            :func:`~experiments.data_mixing.olmix_law.fit_log_linear`. Column order
            must match ``prior`` and ``tokens``.
        prior: ``(m,)`` natural distribution p0, the KL reference point.
        tokens: ``(m,)`` available tokens per domain. Required to constrain.
        requested_tokens: R, the target run's token budget. ``None`` solves
            unconstrained (infinite-data assumption), which OlmixBase does not
            recommend but is useful as a comparison point.
        repetition_factor: k.
        kl_reg: lambda.
        task_weights: ``(n_tasks,)``. ``None`` gives the flat ``1/n`` of OlmixBase.

    Returns:
        ``(m,)`` mixture summing to 1.
    """
    params = np.asarray(params, dtype=float)
    if params.ndim != 2:
        raise ValueError(f"params must be (n_tasks, m+1), got {params.shape}")
    # params[:, 0] is log_c: an additive constant per task, so it cannot move the argmin.
    t = params[:, 1:]
    n_tasks, m = t.shape
    if prior.shape != (m,):
        raise ValueError(f"prior must be ({m},) to match params, got {prior.shape}")

    weights = np.ones(n_tasks) / n_tasks if task_weights is None else np.asarray(task_weights, dtype=float)
    if weights.shape != (n_tasks,):
        raise ValueError(f"task_weights must be ({n_tasks},), got {weights.shape}")

    q = np.maximum(np.asarray(prior, dtype=float), PRIOR_FLOOR)
    q = q / q.sum()

    caps = None
    if requested_tokens is not None:
        if tokens is None:
            raise ValueError("requested_tokens given without tokens; cannot build availability caps")
        tokens = np.asarray(tokens, dtype=float)
        if tokens.shape != (m,):
            raise ValueError(f"tokens must be ({m},) to match params, got {tokens.shape}")
        slack = feasibility_slack(tokens, requested_tokens=requested_tokens, repetition_factor=repetition_factor)
        if slack < 1.0:
            raise ValueError(
                f"infeasible: k*sum(N)/R = {slack:.3f} < 1. Requesting {requested_tokens:,.0f} tokens at "
                f"k={repetition_factor} needs more than {repetition_factor} epochs of the whole "
                f"{tokens.sum():,.0f}-token corpus. Lower R to <= {tokens.sum() * repetition_factor:,.0f} "
                f"or raise k."
            )
        caps = availability_caps(tokens, requested_tokens=requested_tokens, repetition_factor=repetition_factor)

    x = cp.Variable(m)
    objective = cp.sum(cp.multiply(weights, cp.exp(t @ x))) + kl_reg * cp.sum(cp.rel_entr(x, q))
    constraints = [x >= 0, cp.sum(x) == 1]
    if caps is not None:
        constraints.append(x <= caps)

    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=SOLVER)
    if x.value is None:
        raise RuntimeError(f"{SOLVER} failed to solve the mixture problem: status={problem.status}")
    logger.info("solved mixture: objective=%.6g status=%s", problem.value, problem.status)

    solution = np.asarray(x.value, dtype=float)
    # ECOS returns tiny negatives and a sum a hair off 1; clean up without reshaping
    # the answer, then verify we did not paper over a real violation.
    solution = np.clip(solution, 0.0, None)
    solution = solution / solution.sum()
    if caps is not None and np.any(solution > caps + 1e-6):
        worst = int(np.argmax(solution - caps))
        raise RuntimeError(
            f"solution violates its availability cap at domain {worst}: " f"{solution[worst]:.6f} > {caps[worst]:.6f}"
        )
    return solution


def mixture_density(mixture: np.ndarray, *, threshold: float) -> dict[str, float]:
    """Summarize how concentrated a proposed mixture is.

    OlmixBase's exact+KL solve returns a dense vector, so "how many domains does
    this actually use" needs a threshold to be a meaningful question. Use the swarm's
    ``minimum_weight`` for comparability with the sampled mixtures.
    """
    mixture = np.asarray(mixture, dtype=float)
    above = mixture >= threshold
    return {
        "n_domains": int(mixture.size),
        "n_exact_zero": int((mixture == 0).sum()),
        "n_above_threshold": int(above.sum()),
        "mass_above_threshold": float(mixture[above].sum()),
        "threshold": float(threshold),
        "max_weight": float(mixture.max()),
        # exp(entropy) -- the effective number of domains actually being mixed.
        "effective_domains": float(
            np.exp(-np.sum(np.where(mixture > 0, mixture * np.log(mixture, where=mixture > 0), 0.0)))
        ),
    }
