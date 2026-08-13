# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2 and 3 of OlmixBase: fit one law per task, then solve for the mixture.

Assembles the swarm's ``(mixture, BPB)`` dataset, fits a log-linear law per task, and
solves the constrained convex program. The pieces are ported in ``olmix_law`` and
``olmix_solve``; this is the driver that wires them to real data and handles the two
things the reference implementation does not have to.

**Never-sampled domains must be excluded, not fitted.** A domain that appears in no swarm
mixture has an all-zero design column, so the loss is exactly flat in its coefficient and
LBFGS leaves it wherever the restart initialised it. Measured on a synthetic dead column:
the fitted ``t`` stayed at its raw ``U(0,1)*0.1`` initialisation while live domains
recovered their true coefficients to 3 decimals, and the solver then drove that domain to
``x = 0.00028`` against a natural share of ``0.167`` -- i.e. spuriously *zeroed*, on
information it never had. With the natural prior roughly 43 of dclm's 118 cells are never
sampled, so this is the common case, not an edge case. They are dropped from the fit and
the solve, then re-inserted at their natural weight and the result renormalised.

**Report the objective's two terms separately.** ``lambda = 0.05`` is an olmix literal
tuned at m=24, but ``KL(p||q)`` grows roughly like ``log m`` for concentrated mixtures, so
at m=118 the penalty does ~1.5x less relative work than it did for them. olmix never
studied lambda's m-dependence. We cannot fix what we have not measured, so the solve
reports ``loss`` and ``lambda*KL`` separately and their ratio; if the penalty turns out
negligible that is the evidence for raising it.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from experiments.data_mixing.olmix_law import fit_log_linear, predict_log_linear
from experiments.data_mixing.olmix_solve import (
    DEFAULT_KL_REG,
    DEFAULT_REPETITION_FACTOR,
    availability_caps,
    feasibility_slack,
    solve_mixture,
)

logger = logging.getLogger(__name__)


def split_sampled_domains(
    domains: list[str], weights: np.ndarray, min_appearances: int = 1
) -> tuple[list[int], list[int]]:
    """Indices of (identifiable, dropped) domains for a swarm design matrix.

    ``min_appearances`` is the number of mixtures a domain must appear in to be fitted.
    The default of 1 is the production rule: only never-sampled domains are dropped, since
    an all-zero column leaves its coefficient at whatever the restart initialised it to.

    Raising it is how a PRELIMINARY fit is done honestly before the swarm is complete.
    With fewer runs than live domains the system is underdetermined, and the fix is to
    shrink the support to what the available rows can actually identify rather than to
    relax the guard -- a domain appearing in 2 of 41 mixtures is barely better determined
    than one appearing in none. Dropped domains are re-inserted at their natural weight,
    exactly as never-sampled ones are, so the result is a fit over a restricted support
    rather than a fit on noise.
    """
    if weights.ndim != 2 or weights.shape[1] != len(domains):
        raise ValueError(f"weights {weights.shape} does not match {len(domains)} domains")
    if min_appearances < 1:
        raise ValueError(f"min_appearances must be >= 1, got {min_appearances}")
    appeared = (weights != 0).sum(axis=0)
    live = [i for i, a in enumerate(appeared) if a >= min_appearances]
    dead = [i for i, a in enumerate(appeared) if a < min_appearances]
    return live, dead


def _fit_one_task(args: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    """Worker for the process pool. Module-level so it pickles."""
    weights, bpb, idx = args
    # One thread per worker: these are 300 restarts of LBFGS on a (K, m_live) design, so the
    # cost is per-op overhead, not FLOPs. Letting each worker spawn its own torch thread pool
    # oversubscribes the cores and makes the fan-out slower than serial.
    torch.set_num_threads(1)
    return fit_log_linear(weights, bpb, idx=idx)


def fit_all_tasks(weights: np.ndarray, bpb: np.ndarray, task_names: list[str], num_workers: int = 1) -> np.ndarray:
    """Fit one log-linear law per task over the *live* design matrix.

    Args:
        weights: ``(K, m_live)`` swarm mixtures, already restricted to sampled domains.
        bpb: ``(K, n_tasks)`` measured BPB, rows aligned with ``weights``.
        num_workers: processes to fan the per-task fits across. 1 runs serially.

            Measured on the real dclm swarm (318 runs, 75 live domains): **236 s for a single
            task**, so all 42 take ~2.75 h serially -- the fit, not the sweep, becomes the tail.
            The fan-out is exact rather than approximate: ``fit_log_linear`` reseeds
            ``np.random``/``random`` to 42 on entry, so a task's 300 restart draws depend only
            on that task, never on evaluation order. Results are bit-identical to serial, which
            ``tests/data_mixing/test_olmix_fit_parallel.py`` asserts. olmix itself fans these
            out with ``mp.Pool``, so this restores its behaviour rather than departing from it.

    Returns:
        ``(n_tasks, m_live + 1)`` array of ``[log_c, t...]``.
    """
    if weights.shape[0] != bpb.shape[0]:
        raise ValueError(f"{weights.shape[0]} mixtures but {bpb.shape[0]} BPB rows")
    if bpb.shape[1] != len(task_names):
        raise ValueError(f"{bpb.shape[1]} BPB columns but {len(task_names)} task names")
    if not np.isfinite(bpb).all():
        bad = [task_names[j] for j in range(bpb.shape[1]) if not np.isfinite(bpb[:, j]).all()]
        raise ValueError(f"non-finite BPB in tasks: {bad}")

    if num_workers <= 1:
        params = []
        for j, name in enumerate(task_names):
            p = fit_log_linear(weights, bpb, idx=j)
            params.append(p)
            logger.info("fitted task %d/%d %s", j + 1, len(task_names), name)
        return np.stack(params)

    logger.info("fitting %d tasks across %d processes", len(task_names), num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        # `map` preserves input order, so row j stays task_names[j].
        params = list(pool.map(_fit_one_task, [(weights, bpb, j) for j in range(len(task_names))]))
    return np.stack(params)


def regression_fit_quality(params: np.ndarray, weights: np.ndarray, bpb: np.ndarray) -> dict[str, float]:
    """Pearson correlation between predicted and measured BPB, per task and pooled.

    This is olmix's "regression fit" diagnostic. In their default configuration it is
    measured in-sample (``n_test: 0`` makes the test set a copy of the training set), so
    treat it as a sanity check on the fit rather than as generalisation evidence.
    """
    preds = np.stack([predict_log_linear(params[j], weights) for j in range(params.shape[0])], axis=1)
    per_task = []
    for j in range(preds.shape[1]):
        if np.std(preds[:, j]) < 1e-12 or np.std(bpb[:, j]) < 1e-12:
            per_task.append(float("nan"))
        else:
            per_task.append(float(np.corrcoef(preds[:, j], bpb[:, j])[0, 1]))
    finite = [r for r in per_task if np.isfinite(r)]
    avg_pred, avg_true = preds.mean(axis=1), bpb.mean(axis=1)
    return {
        "per_task_mean": float(np.mean(finite)) if finite else float("nan"),
        "per_task_min": float(np.min(finite)) if finite else float("nan"),
        "n_degenerate_tasks": int(len(per_task) - len(finite)),
        "average_bpb": float(np.corrcoef(avg_pred, avg_true)[0, 1]),
    }


def solve_with_natural_reinsertion(
    params: np.ndarray,
    domains: list[str],
    live: list[int],
    tokens: np.ndarray,
    natural: np.ndarray,
    requested_tokens: float | None,
    repetition_factor: float = DEFAULT_REPETITION_FACTOR,
    kl_reg: float = DEFAULT_KL_REG,
) -> dict:
    """Solve over the live support, then re-insert dead domains at their natural weight.

    Returns the full-length mixture plus the diagnostics needed to judge the solve:
    the two objective terms separately, and how much mass the dead domains hold.
    """
    live_prior = natural[live]
    live_prior = live_prior / live_prior.sum()
    live_tokens = tokens[live]

    solution_live = solve_mixture(
        params=params,
        prior=live_prior,
        tokens=live_tokens if requested_tokens is not None else None,
        requested_tokens=requested_tokens,
        repetition_factor=repetition_factor,
        kl_reg=kl_reg,
    )

    # Dead domains keep their natural share; the live solution is scaled into what is
    # left. This is "leave it at baseline", the only honest thing to do with a domain we
    # measured nothing about -- as opposed to letting a data-free coefficient move it.
    dead_mass = float(natural.sum() - natural[live].sum())
    full = np.zeros(len(domains))
    for pos, idx in enumerate(live):
        full[idx] = solution_live[pos] * (1.0 - dead_mass)
    for idx in range(len(domains)):
        if idx not in set(live):
            full[idx] = natural[idx]
    full = full / full.sum()

    t = params[:, 1:]
    n_tasks = t.shape[0]
    loss_term = float(np.mean(np.exp(t @ solution_live)))
    q = np.maximum(live_prior, 1e-12)
    q = q / q.sum()
    x = np.maximum(solution_live, 0.0)
    kl = float(np.sum(np.where(x > 0, x * np.log(x / q), 0.0)))
    return {
        "mixture": full,
        "mixture_live": solution_live,
        "live_indices": live,
        "dead_natural_mass": dead_mass,
        "objective_loss_term": loss_term,
        "objective_kl_term": kl,
        "objective_kl_penalty": kl_reg * kl,
        # If this is negligible the KL regularizer is not doing the work Figure 8 relied
        # on, which is the evidence for raising lambda at our m. See module docstring.
        "kl_penalty_share": float(kl_reg * kl / (loss_term + kl_reg * kl)) if loss_term else float("nan"),
        "kl_reg": kl_reg,
        "n_tasks": n_tasks,
        "requested_tokens": requested_tokens,
        "repetition_factor": repetition_factor,
    }


def sweep_constraints(
    params: np.ndarray,
    domains: list[str],
    live: list[int],
    tokens: np.ndarray,
    natural: np.ndarray,
    requested_tokens_grid: list[float],
    repetition_grid: list[float],
    kl_reg: float = DEFAULT_KL_REG,
) -> list[dict]:
    """Solve across (R, k), skipping infeasible combinations with a reason.

    `k` and `R` enter only here, never the swarm, so one swarm supports the whole sweep --
    this is olmix's Figure 7 analysis and it costs no training.
    """
    out = []
    for r in requested_tokens_grid:
        for k in repetition_grid:
            slack = feasibility_slack(tokens[live], requested_tokens=r, repetition_factor=k)
            if slack < 1.0:
                out.append({"requested_tokens": r, "repetition_factor": k, "feasible": False, "slack": slack})
                logger.info("R=%.2e k=%s infeasible (slack %.3f)", r, k, slack)
                continue
            res = solve_with_natural_reinsertion(
                params, domains, live, tokens, natural, requested_tokens=r, repetition_factor=k, kl_reg=kl_reg
            )
            caps = availability_caps(tokens[live], requested_tokens=r, repetition_factor=k)
            res.update(
                {
                    "feasible": True,
                    "slack": slack,
                    "n_at_cap": int(np.sum(np.isclose(res["mixture_live"], caps, rtol=1e-3))),
                }
            )
            out.append(res)
    return out
