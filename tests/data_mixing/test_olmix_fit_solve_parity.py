# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin the log-linear fit and the mixture solver against allenai/olmix's source.

Goldens are captured by ``tests/data_mixing/make_olmix_goldens.py``:

* ``fit_log_linear.json`` -- ``LogLinearRegressor.fit`` on a fixed Dirichlet design
  whose targets come from a known ``(log_c, t)``, so the fixture pins both agreement
  with the reference *and* that the fit recovers a law it should be able to recover.
* ``solve_exact.json`` -- ``LogLinearExactProposer.propose`` unconstrained and with
  availability caps where one cap binds exactly.

Tolerances are loose enough for platform-dependent LBFGS/ECOS arithmetic and tight
enough that a wrong loss, delta, restart grid, objective, or regularizer shows up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.data_mixing.olmix_law import fit_log_linear, predict_log_linear
from experiments.data_mixing.olmix_solve import (
    DEFAULT_KL_REG,
    availability_caps,
    feasibility_slack,
    mixture_density,
    solve_mixture,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _load(name: str) -> dict:
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(f"golden {path} missing; regenerate with make_olmix_goldens.py")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def fit_golden() -> dict:
    return _load("fit_log_linear.json")


@pytest.fixture(scope="module")
def solve_golden() -> dict:
    return _load("solve_exact.json")


def test_fit_matches_olmix_parameters(fit_golden):
    x = np.array(fit_golden["x"])
    y = np.array(fit_golden["y"])
    params = fit_log_linear(x, y, idx=0)
    np.testing.assert_allclose(params, np.array(fit_golden["params"]), rtol=1e-4, atol=1e-5)


def test_fit_predictions_match_olmix(fit_golden):
    """Parameters could agree while evaluation disagrees, so pin the curve too."""
    x = np.array(fit_golden["x"])
    y = np.array(fit_golden["y"])
    params = fit_log_linear(x, y, idx=0)
    np.testing.assert_allclose(predict_log_linear(params, x), np.array(fit_golden["predictions"]), rtol=1e-5, atol=1e-6)


def test_fit_recovers_the_generating_law(fit_golden):
    """Not a parity check: the fixture's targets were generated from a known law, so a
    correct fit must land on it. This is what fails if the loss, delta, or restart
    initialization is wrong in a way that happens to match a broken reference call."""
    x = np.array(fit_golden["x"])
    y = np.array(fit_golden["y"])
    params = fit_log_linear(x, y, idx=0)
    np.testing.assert_allclose(params[1:], np.array(fit_golden["true_t"]), atol=5e-3)
    assert params[0] == pytest.approx(-0.7, abs=5e-3)


def test_fit_rejects_underdetermined_swarms():
    """The log-linear law needs m+1 runs for a unique solution (paper RQ2); fewer
    should fail loudly rather than return an arbitrary local optimum."""
    x = np.random.default_rng(0).dirichlet(np.ones(6), 4)
    y = np.ones((4, 1))
    with pytest.raises(ValueError, match="cannot identify"):
        fit_log_linear(x, y, idx=0)


@pytest.mark.parametrize("case_idx", [0, 1])
def test_solver_matches_olmix(solve_golden, case_idx):
    case = solve_golden["cases"][case_idx]
    params = np.column_stack([np.array(solve_golden["log_c"]), np.array(solve_golden["t"])])
    domains = list(solve_golden["token_counts"])
    tokens = np.array([solve_golden["token_counts"][d] for d in domains], dtype=float)
    prior = np.array([solve_golden["prior"][d] for d in domains], dtype=float)

    solution = solve_mixture(
        params=params,
        prior=prior,
        tokens=tokens,
        requested_tokens=case["target_tokens"],
        repetition_factor=case["repetition_factor"],
        kl_reg=solve_golden["kl_reg"],
    )
    np.testing.assert_allclose(solution, np.array(case["x"]), rtol=1e-4, atol=1e-6)


def test_solution_is_on_the_simplex_and_respects_caps(solve_golden):
    case = solve_golden["cases"][1]
    params = np.column_stack([np.array(solve_golden["log_c"]), np.array(solve_golden["t"])])
    domains = list(solve_golden["token_counts"])
    tokens = np.array([solve_golden["token_counts"][d] for d in domains], dtype=float)
    prior = np.array([solve_golden["prior"][d] for d in domains], dtype=float)

    solution = solve_mixture(
        params=params,
        prior=prior,
        tokens=tokens,
        requested_tokens=case["target_tokens"],
        repetition_factor=case["repetition_factor"],
    )
    caps = availability_caps(tokens, requested_tokens=case["target_tokens"], repetition_factor=case["repetition_factor"])
    assert solution.sum() == pytest.approx(1.0)
    assert (solution >= 0).all()
    assert (solution <= caps + 1e-6).all()
    # The fixture is only interesting if a cap actually binds.
    assert np.isclose(solution, caps, atol=1e-4).any(), "no cap binds; fixture is effectively unconstrained"


def test_log_c_does_not_move_the_argmin(solve_golden):
    """params[:, 0] enters the objective as an additive constant, so perturbing it must
    not change the solution. The reference drops it entirely; this pins that they agree."""
    params = np.column_stack([np.array(solve_golden["log_c"]), np.array(solve_golden["t"])])
    domains = list(solve_golden["token_counts"])
    prior = np.array([solve_golden["prior"][d] for d in domains], dtype=float)

    baseline = solve_mixture(params=params, prior=prior)
    shifted = params.copy()
    shifted[:, 0] += 3.0
    np.testing.assert_allclose(solve_mixture(params=shifted, prior=prior), baseline, atol=1e-6)


def test_two_domain_closed_form():
    """A hand-checkable case. With one task, t = [t0, t1] and no caps, the objective
    on x = [w, 1-w] is exp(t0*w + t1*(1-w)) + lambda*KL. With t0 << t1 the exponential
    pushes all mass to domain 0 and only the KL pull keeps domain 1 alive, so the
    solution must sit strictly inside (0.5, 1) for a uniform prior."""
    params = np.array([[0.0, -8.0, 0.0]])
    prior = np.array([0.5, 0.5])
    solution = solve_mixture(params=params, prior=prior, kl_reg=DEFAULT_KL_REG)
    assert solution.sum() == pytest.approx(1.0)
    assert 0.5 < solution[0] < 1.0
    assert solution[1] > 0.0, "the KL term must keep the dominated domain off zero"
    # Stronger regularization must pull back toward the prior.
    weaker = solve_mixture(params=params, prior=prior, kl_reg=0.5)
    assert weaker[0] < solution[0]


def test_infeasible_request_is_reported_as_slack_not_a_solver_status():
    tokens = np.array([1e9, 2e9])
    params = np.zeros((1, 3))
    prior = np.array([0.5, 0.5])
    slack = feasibility_slack(tokens, requested_tokens=1e11, repetition_factor=4.0)
    assert slack < 1.0
    with pytest.raises(ValueError, match=r"infeasible: k\*sum\(N\)/R"):
        solve_mixture(params=params, prior=prior, tokens=tokens, requested_tokens=1e11, repetition_factor=4.0)


def test_requested_tokens_without_token_counts_raises():
    with pytest.raises(ValueError, match="cannot build availability caps"):
        solve_mixture(params=np.zeros((1, 3)), prior=np.array([0.5, 0.5]), requested_tokens=1e9)


def test_mixture_density_reports_a_dense_solution_honestly(solve_golden):
    """OlmixBase's exact+KL solve is dense by construction; the reporting helper has to
    show that rather than imply sparsity."""
    params = np.column_stack([np.array(solve_golden["log_c"]), np.array(solve_golden["t"])])
    domains = list(solve_golden["token_counts"])
    prior = np.array([solve_golden["prior"][d] for d in domains], dtype=float)
    solution = solve_mixture(params=params, prior=prior)

    stats = mixture_density(solution, threshold=1.2 / len(domains))
    assert stats["n_domains"] == len(domains)
    assert stats["n_exact_zero"] == 0, "KL regularization should not produce exact zeros"
    assert 1.0 <= stats["effective_domains"] <= len(domains)
    assert stats["mass_above_threshold"] <= 1.0 + 1e-9
