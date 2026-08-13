# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pin ``olmix_sample`` against outputs captured from allenai/olmix's own source.

The golden fixtures come from running the reference ``generate_weights_dirichlet``
over olmix's published 24 DCLM topic priors -- see
``tests/data_mixing/make_olmix_goldens.py`` for how, and why the package cannot
simply be installed. Two cases are pinned:

* ``dclm24_proxy_budget_caps_inert`` -- olmix's own proxy budget
  (``max_tokens=2_910_233_600``), where every availability cap saturates at 1.0 and
  the sampler is exercising only the Dirichlet draw, clip, and snap-to-grid.
* ``dclm24_caps_bind_k4`` -- a budget where caps land in [0.11, 1.0] with k=4, so
  the bounds filter, the post-renormalization re-check, and the repetition
  bookkeeping all actually fire.

Bit-parity (not just distributional agreement) is the bar, because it is the only
check that catches a desynchronised RNG stream -- e.g. dropping the reference's
degenerate per-domain Dirichlet draws, which look like dead code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.data_mixing.olmix_sample import (
    minimum_weight_for_m,
    natural_prior,
    sample_flat_dirichlet_swarm,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def _load(name: str) -> dict:
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(f"golden {path} missing; regenerate with make_olmix_goldens.py")
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def swarm_goldens() -> dict[str, dict]:
    payload = _load("swarm_dclm24.json")
    return {s["label"]: s for s in payload["swarms"]}


@pytest.mark.parametrize("label", ["dclm24_proxy_budget_caps_inert", "dclm24_caps_bind_k4"])
def test_sampler_reproduces_olmix_bit_for_bit(swarm_goldens, label):
    golden = swarm_goldens[label]
    params = golden["params"]
    tokens = {k: int(v) for k, v in golden["leaf_tokens"].items()}
    # Domain order is load-bearing: the reference indexes caps and the design matrix
    # positionally off the prior dict's iteration order.
    prior = {d: tokens[d] / sum(tokens.values()) for d in golden["domains"]}

    domains, weights, repetitions = sample_flat_dirichlet_swarm(
        prior=prior,
        tokens=tokens,
        num_samples_out=params["num_samples_out"],
        minimum_weight=params["minimum_weight"],
        max_tokens=params["max_tokens"],
        repetition_factor=params["repetition_factor"],
        seed=params["seed"],
        min_strength=params["min_strength"],
        max_strength=params["max_strength"],
        temperature=params["temperature"],
        sample_multiplier=params["sample_multiplier"],
        enable_bound=params["enable_bound"],
    )

    assert domains == golden["domains"]
    np.testing.assert_allclose(weights, np.array(golden["weights"]), rtol=0, atol=0)
    np.testing.assert_allclose(repetitions, np.array(golden["repetitions"]), rtol=0, atol=0)


def test_goldens_are_well_formed_mixtures(swarm_goldens):
    """Guard the fixtures themselves: a silently-degenerate golden would make the
    parity test above pass while pinning nothing."""
    for label, golden in swarm_goldens.items():
        weights = np.array(golden["weights"])
        assert weights.shape == (golden["params"]["num_samples_out"], len(golden["domains"])), label
        assert not np.isnan(weights).any(), label
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
        assert (weights >= 0).all(), label
        # The swarm must actually explore: not every mix one-hot, not all identical.
        nnz = (weights != 0).sum(axis=1)
        assert nnz.max() > 1, label
        assert len({tuple(row) for row in weights.tolist()}) > 1, label


def test_caps_bind_case_actually_binds(swarm_goldens):
    """The second fixture is only worth having if its caps are reachable."""
    golden = swarm_goldens["dclm24_caps_bind_k4"]
    tokens = np.array([golden["leaf_tokens"][d] for d in golden["domains"]], dtype=float)
    params = golden["params"]
    caps = np.minimum(tokens * params["repetition_factor"] / params["max_tokens"], 1.0)
    assert (caps < 1.0).any(), "no cap is binding, fixture is equivalent to the inert case"
    assert caps.min() > params["minimum_weight"], "caps below the weight grid make every draw infeasible"
    # And the sampled weights respect them.
    assert (np.array(golden["weights"]) <= caps + 1e-12).all()


def test_minimum_weight_reproduces_the_paper_at_m24_and_scales():
    """m=24 must give back olmix's literal 0.05, or the rescaling is not a rescaling."""
    assert minimum_weight_for_m(24) == pytest.approx(0.05)
    assert minimum_weight_for_m(120) == pytest.approx(0.01)
    # Clip stays a fixed multiple of the mean weight 1/m.
    for m in (6, 24, 64, 120):
        assert minimum_weight_for_m(m) * m == pytest.approx(1.2)


def test_natural_prior_is_token_proportional():
    prior = natural_prior({"a": 1, "b": 3})
    assert prior == {"a": 0.25, "b": 0.75}
    assert sum(prior.values()) == pytest.approx(1.0)


def test_infeasible_caps_raise_rather_than_looping_forever():
    """A budget larger than k * total tokens has no feasible mixture at all; the
    sampler must say so instead of rejecting every draw and reporting an opaque
    empty-pool error."""
    tokens = {"a": 1_000, "b": 2_000}
    with pytest.raises(ValueError, match="availability caps sum to"):
        sample_flat_dirichlet_swarm(
            prior=natural_prior(tokens),
            tokens=tokens,
            num_samples_out=4,
            minimum_weight=0.05,
            max_tokens=1_000_000,
            repetition_factor=4.0,
            seed=0,
        )


def test_mismatched_prior_and_tokens_raise():
    with pytest.raises(ValueError, match="same domains"):
        sample_flat_dirichlet_swarm(
            prior={"a": 1.0},
            tokens={"b": 10},
            num_samples_out=1,
            minimum_weight=0.05,
            max_tokens=10,
            repetition_factor=4.0,
            seed=0,
        )


def test_fully_clipped_draw_is_rejected_not_turned_into_nan():
    """A clip above the mean weight 1/m zeroes an entire well-spread draw.

    The reference then divides by zero, builds an all-NaN mixture, passes its own
    bounds re-check (every NaN comparison is False), and dies on
    ``int(weight * max_tokens)`` with "cannot convert float NaN to integer". It never
    fires at m=24 with strength 0.1-5 because those draws are spiky enough that
    something clears 0.05 -- but the hazard grows with m AND with strength, which is
    exactly where a high-concentration setting at m>=120 lives.

    Here the clip (0.4) sits far above the mean weight (1/8 = 0.125) and the
    concentration is high enough to keep draws near-uniform, so most candidates clip to
    nothing. The sampler must either return clean mixtures or raise -- never emit NaN.
    """
    tokens = {f"d{i}": 10_000_000_000 for i in range(8)}
    try:
        _domains, weights, repetitions = sample_flat_dirichlet_swarm(
            prior=natural_prior(tokens),
            tokens=tokens,
            num_samples_out=4,
            minimum_weight=0.4,
            max_tokens=1_000_000_000,
            repetition_factor=float("inf"),
            enable_bound=False,
            seed=3,
            min_strength=200.0,
            max_strength=2000.0,
            sample_multiplier=60,
            rng_parity=False,
        )
    except ValueError as exc:
        # Acceptable outcome: too few survivors. But it must be our diagnostic, not a
        # NaN conversion blowing up inside the repetition bookkeeping.
        assert "NaN" not in str(exc), exc
        return
    assert not np.isnan(weights).any(), "emitted a NaN mixture"
    assert not np.isnan(repetitions).any()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-9)
    assert (weights >= 0).all()
