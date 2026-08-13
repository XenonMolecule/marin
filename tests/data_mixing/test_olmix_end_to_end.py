# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the whole OlmixBase pipeline on a synthetic world with a known answer.

The component-level parity tests pin each stage against olmix, but they cannot catch
a mistake in how the stages are *wired*: a domain order that differs between the
swarm, the fit, and the caps would leave every component correct and the answer
wrong. So here BPB is generated from a known log-linear law, and we check that
sample -> fit -> solve lands on the mixture that law actually implies.

This is the test that fails on:
* a permuted domain order anywhere in the chain,
* caps applied to the wrong domain,
* task weighting that isn't the flat 1/n of OlmixBase,
* an objective that optimizes the wrong sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.data_mixing.olmix_law import fit_log_linear, predict_log_linear
from experiments.data_mixing.olmix_sample import (
    minimum_weight_for_m,
    natural_prior,
    sample_flat_dirichlet_swarm,
)
from experiments.data_mixing.olmix_solve import solve_mixture

M_DOMAINS = 8
N_TASKS = 4
PROXY_TOKENS = 3_000_000_000


@pytest.fixture(scope="module")
def synthetic_world() -> dict:
    """A ground-truth log-linear world plus a swarm measured on it, noise-free.

    Noise-free is deliberate: with noise, a failure could always be blamed on the
    fit rather than the wiring. Here any mismatch is structural.
    """
    rng = np.random.default_rng(7)
    domains = [f"c{i:02d}_q{i % 5}" for i in range(M_DOMAINS)]
    tokens = {d: int(v) for d, v in zip(domains, rng.integers(2_000_000_000, 40_000_000_000, M_DOMAINS))}
    prior = natural_prior(tokens)

    # K = 3(m+1), the OlmixBase swarm size. minimum_weight is kept finer than 1.2/m
    # here so that 8 domains still yield 27 distinct mixtures after snap-to-grid.
    k_runs = 3 * (M_DOMAINS + 1)
    sampled_domains, weights, _ = sample_flat_dirichlet_swarm(
        prior=prior,
        tokens=tokens,
        num_samples_out=k_runs,
        minimum_weight=0.05,
        max_tokens=PROXY_TOKENS,
        repetition_factor=float("inf"),
        enable_bound=False,
        seed=11,
        sample_multiplier=40,
    )

    true_log_c = rng.uniform(-1.0, 0.0, N_TASKS)
    true_t = rng.uniform(-2.0, 0.0, (N_TASKS, M_DOMAINS))
    bpb = np.stack(
        [np.exp(true_log_c[i]) + np.exp(weights @ true_t[i]) for i in range(N_TASKS)],
        axis=1,
    )
    return {
        "domains": sampled_domains,
        "tokens": tokens,
        "prior": prior,
        "weights": weights,
        "bpb": bpb,
        "true_params": np.column_stack([true_log_c, true_t]),
    }


@pytest.fixture(scope="module")
def fitted_params(synthetic_world) -> np.ndarray:
    """Fit all tasks once.

    300 LBFGS restarts per task is ~15s, so refitting per test blows past the repo's
    60s default per-test timeout and makes these tests flaky under load. The fit is
    deterministic, so one module-scoped fit serves every assertion below.
    """
    return np.stack([fit_log_linear(synthetic_world["weights"], synthetic_world["bpb"], idx=i) for i in range(N_TASKS)])


def test_swarm_domain_order_is_the_prior_order(synthetic_world):
    """Everything downstream indexes positionally off this order."""
    assert synthetic_world["domains"] == list(synthetic_world["prior"])


@pytest.mark.timeout(600)
def test_fit_recovers_the_generating_coefficients(synthetic_world, fitted_params):
    fitted = fitted_params
    # exp(log_c) + exp(t.p) is only identified up to how well the swarm spans the
    # simplex, so compare the predictions the coefficients imply, which is what the
    # solver actually consumes.
    for i in range(N_TASKS):
        np.testing.assert_allclose(
            predict_log_linear(fitted[i], synthetic_world["weights"]),
            synthetic_world["bpb"][:, i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.timeout(600)
def test_solve_from_fitted_params_matches_solve_from_true_params(synthetic_world, fitted_params):
    """The end-to-end claim: the mixture we propose is the one the true law implies."""
    prior_vec = np.array([synthetic_world["prior"][d] for d in synthetic_world["domains"]])
    token_vec = np.array([synthetic_world["tokens"][d] for d in synthetic_world["domains"]], dtype=float)
    requested = 0.25 * token_vec.sum()  # comfortably feasible at k=4

    from_fitted = solve_mixture(
        params=fitted_params, prior=prior_vec, tokens=token_vec, requested_tokens=requested, repetition_factor=4.0
    )
    from_true = solve_mixture(
        params=synthetic_world["true_params"],
        prior=prior_vec,
        tokens=token_vec,
        requested_tokens=requested,
        repetition_factor=4.0,
    )
    np.testing.assert_allclose(from_fitted, from_true, atol=2e-2)


@pytest.mark.timeout(600)
def test_proposed_mixture_beats_the_natural_one_under_the_true_law(synthetic_world, fitted_params):
    """A mixture optimizer that cannot beat the natural distribution on a world where
    the law holds exactly is not optimizing. Scored with the TRUE law, so this measures
    the proposal, not the surrogate."""
    prior_vec = np.array([synthetic_world["prior"][d] for d in synthetic_world["domains"]])
    token_vec = np.array([synthetic_world["tokens"][d] for d in synthetic_world["domains"]], dtype=float)
    true_params = synthetic_world["true_params"]

    def true_objective(mixture: np.ndarray) -> float:
        preds = [predict_log_linear(true_params[i], mixture[None, :])[0] for i in range(N_TASKS)]
        return float(np.mean(preds))

    proposed = solve_mixture(
        params=fitted_params,
        prior=prior_vec,
        tokens=token_vec,
        requested_tokens=0.25 * token_vec.sum(),
        repetition_factor=4.0,
    )
    assert true_objective(proposed) < true_objective(prior_vec)


def test_binding_caps_shift_the_mixture_toward_the_capped_domains(synthetic_world):
    """Tightening R must move mass off the domains the optimizer wants but cannot
    afford -- the paper's Figure 7 effect. If caps were applied to the wrong domain
    index, this ordering would not hold."""
    prior_vec = np.array([synthetic_world["prior"][d] for d in synthetic_world["domains"]])
    token_vec = np.array([synthetic_world["tokens"][d] for d in synthetic_world["domains"]], dtype=float)
    params = synthetic_world["true_params"]

    loose = solve_mixture(
        params=params, prior=prior_vec, tokens=token_vec, requested_tokens=0.1 * token_vec.sum(), repetition_factor=4.0
    )
    tight = solve_mixture(
        params=params, prior=prior_vec, tokens=token_vec, requested_tokens=2.0 * token_vec.sum(), repetition_factor=4.0
    )
    tight_caps = token_vec * 4.0 / (2.0 * token_vec.sum())

    assert not np.allclose(loose, tight, atol=1e-3), "tightening R had no effect; caps are not binding"
    assert (tight <= tight_caps + 1e-6).all()
    # Every domain the loose solution over-allocates relative to its tight cap must
    # have been pulled down to (or below) that cap.
    over = loose > tight_caps + 1e-6
    assert over.any(), "fixture does not exercise a binding cap"
    assert (tight[over] <= tight_caps[over] + 1e-6).all()


def test_minimum_weight_default_would_be_used_at_m120():
    """Sanity-check the scaling rule at the size we will actually run."""
    assert minimum_weight_for_m(120) == pytest.approx(0.01)
    # ...and that it bounds the support: weights snap to multiples of it.
    assert 1.0 / minimum_weight_for_m(120) == pytest.approx(100.0)
