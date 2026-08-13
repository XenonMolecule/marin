# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fit/solve driver, especially its handling of never-sampled domains.

With the natural prior roughly 43 of dclm's 118 cells never appear in any swarm mixture.
Their coefficients are data-free, and a measured experiment showed the solver acts on that
garbage and drives them to ~0 rather than leaving them at baseline. Getting this wrong
would silently delete a chunk of the corpus from the proposed mixture, so it is tested
harder than anything else here.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.data_mixing.olmix_fit import (
    fit_all_tasks,
    regression_fit_quality,
    solve_with_natural_reinsertion,
    split_sampled_domains,
    sweep_constraints,
)

DOMAINS = ["c00_q0", "c00_q1", "c00_q2", "c01_q0"]


def test_split_identifies_never_sampled_domains():
    w = np.array([[0.5, 0.5, 0.0, 0.0], [0.25, 0.0, 0.75, 0.0]])
    live, dead = split_sampled_domains(DOMAINS, w)
    assert live == [0, 1, 2]
    assert dead == [3]


def test_split_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        split_sampled_domains(DOMAINS, np.zeros((3, 2)))


def test_dead_domain_keeps_its_natural_weight_not_zero():
    """The measured failure: a data-free coefficient drove a dead domain to 0.00028
    against a natural share of 0.167. Re-insertion must restore the baseline exactly."""
    tokens = np.array([1e9, 1e9, 1e9, 1e9])
    natural = tokens / tokens.sum()
    live = [0, 1, 2]
    params = np.array([[0.0, -1.0, -0.5, -0.2]])  # one task, live support only
    res = solve_with_natural_reinsertion(params, DOMAINS, live, tokens, natural, requested_tokens=None)
    mixture = res["mixture"]
    assert mixture[3] == pytest.approx(natural[3], rel=1e-6)
    assert res["dead_natural_mass"] == pytest.approx(0.25)
    assert mixture.sum() == pytest.approx(1.0)
    assert (mixture >= 0).all()


def test_live_domains_share_the_remaining_mass():
    tokens = np.array([1e9, 1e9, 1e9, 3e9])
    natural = tokens / tokens.sum()
    live = [0, 1, 2]
    params = np.array([[0.0, -2.0, -0.1, -0.1]])
    res = solve_with_natural_reinsertion(params, DOMAINS, live, tokens, natural, requested_tokens=None)
    assert res["mixture"][:3].sum() == pytest.approx(1.0 - natural[3], rel=1e-6)


def test_objective_terms_are_reported_separately():
    """lambda=0.05 is an olmix literal tuned at m=24, and KL grows ~log m, so we must be
    able to see whether the penalty is still doing meaningful work at our m."""
    tokens = np.full(4, 1e9)
    natural = tokens / tokens.sum()
    params = np.array([[0.0, -1.0, -0.5, -0.2]])
    res = solve_with_natural_reinsertion(params, DOMAINS, [0, 1, 2], tokens, natural, requested_tokens=None)
    assert res["objective_loss_term"] > 0
    assert res["objective_kl_term"] >= 0
    assert res["objective_kl_penalty"] == pytest.approx(res["kl_reg"] * res["objective_kl_term"])
    assert 0.0 <= res["kl_penalty_share"] <= 1.0


def test_caps_are_respected_after_reinsertion():
    tokens = np.array([2e9, 2e9, 2e9, 2e9])
    natural = tokens / tokens.sum()
    live = [0, 1, 2]
    params = np.array([[0.0, -8.0, 0.0, 0.0]])  # strongly favours domain 0
    R, k = 2e10, 4.0
    res = solve_with_natural_reinsertion(params, DOMAINS, live, tokens, natural, requested_tokens=R, repetition_factor=k)
    caps = tokens[live] * k / R
    assert (res["mixture_live"] <= caps + 1e-6).all()


def test_sweep_marks_infeasible_combinations_instead_of_raising():
    """A bigger target run shrinks every cap, so parts of the (R, k) grid have no feasible
    mixture at all. The sweep must report that, not abort the whole analysis."""
    tokens = np.full(4, 1e9)
    natural = tokens / tokens.sum()
    params = np.array([[0.0, -1.0, -0.5, -0.2]])
    rows = sweep_constraints(
        params,
        DOMAINS,
        [0, 1, 2],
        tokens,
        natural,
        requested_tokens_grid=[1e9, 1e12],
        repetition_grid=[4.0],
    )
    by_r = {r["requested_tokens"]: r for r in rows}
    assert by_r[1e9]["feasible"] is True
    assert by_r[1e12]["feasible"] is False
    assert by_r[1e12]["slack"] < 1.0


def test_fit_recovers_a_known_law_and_reports_fit_quality():
    """End to end on the driver: synthesize BPB from a known law, fit, and check both the
    coefficients and the reported diagnostic."""
    rng = np.random.default_rng(3)
    m, n_tasks, k = 4, 2, 20
    w = rng.dirichlet(np.ones(m), k)
    true_t = np.array([[-1.5, -0.3, 0.1, -0.7], [-0.2, -1.1, -0.4, 0.05]])
    bpb = np.stack([np.exp(-0.6) + np.exp(w @ true_t[i]) for i in range(n_tasks)], axis=1)

    params = fit_all_tasks(w, bpb, ["task_a", "task_b"])
    assert params.shape == (n_tasks, m + 1)
    q = regression_fit_quality(params, w, bpb)
    assert q["per_task_mean"] > 0.99
    assert q["n_degenerate_tasks"] == 0
    assert q["average_bpb"] > 0.99


def test_fit_rejects_non_finite_bpb():
    w = np.random.default_rng(0).dirichlet(np.ones(3), 10)
    bpb = np.ones((10, 2))
    bpb[3, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite BPB"):
        fit_all_tasks(w, bpb, ["a", "b"])


def test_fit_rejects_misaligned_inputs():
    w = np.random.default_rng(0).dirichlet(np.ones(3), 10)
    with pytest.raises(ValueError, match="BPB rows"):
        fit_all_tasks(w, np.ones((9, 1)), ["a"])
    with pytest.raises(ValueError, match="task names"):
        fit_all_tasks(w, np.ones((10, 2)), ["a"])
