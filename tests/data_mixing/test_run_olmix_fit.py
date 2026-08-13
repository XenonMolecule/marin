# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fit/solve driver, with the identifiability guard as the main subject.

The failure this file exists to prevent is a confident-looking mixture solved from an
underdetermined regression. With fewer runs than live domains the loss is flat in some
coefficients, LBFGS leaves them at their initialisation, and the solver acts on that noise
-- producing a full, plausible-looking mixture with no error raised anywhere.

The guard counts LIVE domains, not grid cells, because the fit runs over the sampled support
only. That is a real distinction: dclm has 118 cells but ~75 are ever sampled early on, so
the bar is ~76 runs rather than 119 -- and it RISES as more runs land and more cells become
live, which is why it is recomputed per collection instead of hard-coded.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.data_mixing.collect_olmix_swarm import SwarmRow
from experiments.data_mixing.olmix_fit import fit_all_tasks
from experiments.data_mixing.olmix_plan import SwarmManifest
from experiments.data_mixing.run_olmix_fit import fit_and_solve, swarm_design

DOMAINS = ("c00_q0", "c00_q1", "c01_q0", "c01_q1")
TASKS = ["gsm8k/gold_bpb_5shot", "drop/bpb_5shot"]


def _manifest(weights) -> SwarmManifest:
    return SwarmManifest(
        corpus="dclm_10k",
        region="us-east5",
        seed=42,
        domains=DOMAINS,
        weights=tuple(tuple(float(x) for x in row) for row in weights),
        tokens={d: 1_000_000_000 for d in DOMAINS},
        cache_dirs={d: f"gs://marin-us-east5/x/{d}" for d in DOMAINS},
    )


def _rows(weights, bpb) -> list[SwarmRow]:
    return [
        SwarmRow(
            run_name=f"run-{i}",
            index=i,
            weights=dict(zip(DOMAINS, w, strict=True)),
            bpb=dict(zip(TASKS, b, strict=True)),
        )
        for i, (w, b) in enumerate(zip(weights, bpb, strict=True))
    ]


def _synthetic(n_runs: int, seed: int = 0, dead_last: bool = False):
    """Mixtures plus BPB generated from a known log-linear law."""
    rng = np.random.default_rng(seed)
    m = len(DOMAINS)
    w = rng.dirichlet(np.ones(m), n_runs)
    if dead_last:
        w[:, -1] = 0.0
        w = w / w.sum(axis=1, keepdims=True)
    true_t = np.array([[-1.5, -0.3, 0.1, -0.7], [-0.2, -1.1, -0.4, 0.05]])
    bpb = np.stack([np.exp(-0.6) + np.exp(w @ true_t[i]) for i in range(len(TASKS))], axis=1)
    return w, bpb


def test_design_matrix_is_aligned_with_manifest_domain_order():
    """Column order is load-bearing: it indexes the priors, the caps, and every t_i."""
    w, bpb = _synthetic(6)
    mf = _manifest(w)
    W, B = swarm_design(_rows(w, bpb), mf, TASKS)
    assert W.shape == (6, len(DOMAINS))
    assert B.shape == (6, len(TASKS))
    np.testing.assert_allclose(W, w)


def test_refuses_to_solve_when_underdetermined():
    """4 live domains need 5 runs; 4 must refuse rather than return a mixture."""
    w, bpb = _synthetic(4)
    with pytest.raises(RuntimeError, match="underdetermined"):
        fit_and_solve(
            manifest=_manifest(w),
            rows=_rows(w, bpb),
            task_names=TASKS,
            requested_tokens=None,
            repetition_factor=4.0,
            kl_reg=0.05,
        )


def test_the_law_layer_refuses_independently():
    """Defense in depth, and the reason there is no override flag: even if the driver's
    guard were bypassed, fit_log_linear rejects the same case. An escape hatch here could
    never have produced a fit -- it would only have failed later with a worse message."""
    w, bpb = _synthetic(4)
    with pytest.raises(ValueError, match="cannot identify"):
        fit_all_tasks(w, bpb, TASKS)


def test_threshold_counts_live_domains_not_grid_cells():
    """A never-sampled cell is dropped from the fit, so it must not raise the bar. Here one
    of four domains is dead, so 4 runs suffice where 5 would be needed if all were live."""
    w, bpb = _synthetic(4, dead_last=True)
    res = fit_and_solve(
        manifest=_manifest(w),
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=None,
        repetition_factor=4.0,
        kl_reg=0.05,
    )
    assert res["live_domains"] == 3
    assert res["required_runs_for_unique_fit"] == 4
    assert res["dead_domains"] == [DOMAINS[-1]]


def test_dead_domain_keeps_its_natural_weight():
    """The measured failure mode: a data-free coefficient drove a dead cell to ~0."""
    w, bpb = _synthetic(8, dead_last=True)
    res = fit_and_solve(
        manifest=_manifest(w),
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=None,
        repetition_factor=4.0,
        kl_reg=0.05,
    )
    assert res["mixture"][DOMAINS[-1]] == pytest.approx(res["natural"][DOMAINS[-1]], rel=1e-6)


def test_proposed_mixture_is_a_valid_distribution():
    w, bpb = _synthetic(12)
    res = fit_and_solve(
        manifest=_manifest(w),
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=None,
        repetition_factor=4.0,
        kl_reg=0.05,
    )
    vals = np.array(list(res["mixture"].values()))
    assert vals.sum() == pytest.approx(1.0)
    assert (vals >= 0).all()
    assert set(res["mixture"]) == set(DOMAINS)


def test_availability_caps_bind_when_a_budget_is_given():
    """R enters only here, never the swarm -- so caps must actually constrain the solution."""
    w, bpb = _synthetic(12)
    mf = _manifest(w)
    # R must sit in the window where caps BIND but the problem stays feasible: the corpus
    # holds 4e9 tokens, so k*sum(N)/R >= 1 needs R <= 1.6e10, and a per-cell cap below 1
    # needs R > tokens*k = 4e9.
    R, k = 8.0e9, 4.0
    res = fit_and_solve(
        manifest=mf,
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=R,
        repetition_factor=k,
        kl_reg=0.05,
    )
    cap = 1e9 * k / R  # every cell holds 1e9 tokens here
    for d in DOMAINS:
        assert res["mixture"][d] <= cap + 1e-6


def test_objective_terms_are_reported_separately():
    """lambda=0.05 is an olmix literal tuned at m=24; we must be able to see whether the KL
    penalty still does meaningful work at our m."""
    w, bpb = _synthetic(12)
    res = fit_and_solve(
        manifest=_manifest(w),
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=None,
        repetition_factor=4.0,
        kl_reg=0.05,
    )
    assert res["objective_loss_term"] > 0
    assert 0.0 <= res["kl_penalty_share"] <= 1.0
    assert res["objective_kl_penalty"] == pytest.approx(0.05 * res["objective_kl_term"])


def test_interaction_matrix_covers_every_task_over_live_support():
    w, bpb = _synthetic(8, dead_last=True)
    res = fit_and_solve(
        manifest=_manifest(w),
        rows=_rows(w, bpb),
        task_names=TASKS,
        requested_tokens=None,
        repetition_factor=4.0,
        kl_reg=0.05,
    )
    assert set(res["interaction_matrix"]) == set(TASKS)
    for task in TASKS:
        assert set(res["interaction_matrix"][task]) == set(DOMAINS[:-1])
    assert set(res["log_c"]) == set(TASKS)
