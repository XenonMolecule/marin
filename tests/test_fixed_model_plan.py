# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixed-model data-curation sweep enumerator.

Verifies the 4-method x 3-size x 7-budget grid produces 84 valid PlannedRuns,
that token and FLOP accounting hold within tolerance, and that hparams stay
below their heuristic clamps.
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps.completed_adamh import (
    SEQ_LEN,
    completed_adamh_heuristic,
)
from experiments.scaling_law_sweeps.curation_plan import BUDGETS, METHODS
from experiments.scaling_law_sweeps.fixed_model_plan import (
    EXPERIMENT_TAG,
    TARGET_HIDDEN_SIZES,
    enumerate_fixed_model_plans,
    resolve_hidden_sizes,
    resolve_methods,
)

_FOUR_METHODS = [
    METHODS["dclm"],
    METHODS["nemotron_org"],
    METHODS["nemotron_full_bos_fixed"],
    METHODS["fineweb_edu"],
]


# =============================================================================
# Grid coverage
# =============================================================================


def test_full_grid_is_n_methods_x_hidden_sizes_x_budgets():
    """4 methods x len(TARGET_HIDDEN_SIZES) x 7 budgets PlannedRuns, no drops."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    assert len(plans) == len(_FOUR_METHODS) * len(TARGET_HIDDEN_SIZES) * len(BUDGETS)


def test_every_method_size_budget_triple_present():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    triples = {(p.method_name, p.hidden_dim, p.budget) for p in plans}
    expected = {(m.name, h, b) for m in _FOUR_METHODS for h in TARGET_HIDDEN_SIZES for b in BUDGETS}
    assert triples == expected


def test_experiment_tag_is_expfm_natural_for_every_plan():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    assert {p.experiment_tag for p in plans} == {EXPERIMENT_TAG}
    # Must not start with "expB" -- otherwise `_build_summary` would route
    # through the simulated-epoching branch and report wrong slice_tokens.
    assert not EXPERIMENT_TAG.startswith("expB")


# =============================================================================
# Accounting
# =============================================================================


def test_token_accounting_within_one_percent():
    """batch_size * train_steps * seq_len ~= target tokens (from AdamH heuristic)."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    for p in plans:
        actual_tokens = p.batch_size * p.train_steps * p.seq_len
        assert abs(actual_tokens - p.t_exp) / p.t_exp < 0.01, (
            f"token accounting off for {p.run_name_core}: " f"actual={actual_tokens:.3e} vs t_exp={p.t_exp:.3e}"
        )


def test_flop_accounting_within_ten_percent():
    """6 * N * tokens ~= budget. Uses the Chinchilla 6N approximation."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    vocab_size = completed_adamh_heuristic.vocab_size
    # Chinchilla 6N is approximate; the heuristic itself uses model.flops_per_token
    # directly (typically returning ~6N per token for dense transformers), so we
    # check against that same flops_per_token.
    for p in plans:
        model_config = completed_adamh_heuristic._build_model_config(p.hidden_dim)
        fpt = model_config.flops_per_token(vocab_size, p.seq_len)
        # Heuristic uses: tokens = budget / (3 * flops_per_token). So effective
        # total FLOPs per run = 3 * fpt * tokens ~= budget.
        effective_budget = 3 * fpt * p.t_exp
        assert abs(effective_budget - p.budget) / p.budget < 0.10, (
            f"flop accounting off for {p.run_name_core}: " f"effective={effective_budget:.3e} vs budget={p.budget:.3e}"
        )


def test_params_match_three_target_sizes():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    vocab_size = completed_adamh_heuristic.vocab_size
    params_by_hidden: dict[int, float] = {}
    for p in plans:
        model_config = completed_adamh_heuristic._build_model_config(p.hidden_dim)
        params = model_config.total_trainable_params(vocab_size)
        if p.hidden_dim in params_by_hidden:
            assert params_by_hidden[p.hidden_dim] == params
        else:
            params_by_hidden[p.hidden_dim] = params
    # Expected from PLAN_GRID.md: 157M / 998M / 8.11B. Ranges allow for small
    # vocab_size variations between tokenizer runs.
    assert 150e6 < params_by_hidden[512] < 180e6, params_by_hidden[512]
    assert 950e6 < params_by_hidden[1536] < 1.10e9, params_by_hidden[1536]
    assert 7.5e9 < params_by_hidden[3584] < 8.5e9, params_by_hidden[3584]


# =============================================================================
# Hparam clamps
# =============================================================================


def test_hparams_below_clamps():
    """lr / adam_lr must stay below max_learning_rate; beta2 in [min_beta2, max_beta2]."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    h = completed_adamh_heuristic
    for p in plans:
        assert 0 < p.learning_rate <= h.max_learning_rate, p
        assert 0 < p.adam_lr <= h.max_learning_rate, p
        assert h.min_beta2 <= p.beta2 <= h.max_beta2, p
        assert 0 < p.epsilon, p
        assert p.beta1 == 0.9


def test_batch_size_is_power_of_two_in_valid_range():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    h = completed_adamh_heuristic
    for p in plans:
        assert h.min_batch_size <= p.batch_size <= h.max_batch_size
        assert (p.batch_size & (p.batch_size - 1)) == 0, f"batch_size {p.batch_size} not pow2"


def test_train_steps_positive():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    for p in plans:
        # Even 8.11B @ 3e18 (the thinnest slice) should have >0 steps.
        assert p.train_steps > 0, p


# =============================================================================
# Filters
# =============================================================================


def test_single_hidden_size_tier_1():
    """Tier 1 rollout: --hidden-sizes 1536 yields 4 methods x 1 size x 7 budgets = 28 plans."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS, hidden_sizes=(1536,))
    assert len(plans) == 28
    assert {p.hidden_dim for p in plans} == {1536}


def test_single_budget_subset():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS, budgets=(3e20,))
    assert len(plans) == len(_FOUR_METHODS) * len(TARGET_HIDDEN_SIZES)
    assert {p.budget for p in plans} == {3e20}


# =============================================================================
# CLI resolvers
# =============================================================================


def test_resolve_methods_all_excludes_resiliparse():
    """'all' must omit the deferred resiliparse method and ExpC placeholders."""
    methods = resolve_methods(["all"])
    names = {m.name for m in methods}
    # Hard exclusions:
    assert "resiliparse" not in names, "resiliparse is deferred (large 571 GB cache)"
    assert "dclm_10k" not in names, "dclm_10k is an ExpC placeholder; must not enter fixed_model"
    assert "nemotron_10k" not in names, "nemotron_10k is an ExpC placeholder; must not enter fixed_model"
    # Original 3k methods that fixed_model knows how to handle:
    assert "dclm" in names
    assert "nemotron_org" in names
    assert "fineweb_edu" in names


def test_resolve_methods_explicit_resiliparse_still_allowed():
    """Explicit listing is allowed (user chooses to include it)."""
    methods = resolve_methods(["resiliparse"])
    assert [m.name for m in methods] == ["resiliparse"]


def test_resolve_methods_unknown_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_methods(["no_such_method"])


def test_resolve_hidden_sizes_default_is_all_three():
    assert resolve_hidden_sizes(None) == TARGET_HIDDEN_SIZES
    assert resolve_hidden_sizes([]) == TARGET_HIDDEN_SIZES


def test_resolve_hidden_sizes_unknown_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_hidden_sizes([768])


# =============================================================================
# Run name distinctness from older sweep
# =============================================================================


def test_run_names_distinct_from_expa_natural():
    """Fixed-model run names must not collide with old ExpA sweep run names."""
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    for p in plans:
        assert EXPERIMENT_TAG in p.run_name_core
        assert "expA_natural" not in p.run_name_core
        assert "expB_" not in p.run_name_core


def test_seq_len_matches_heuristic_default():
    plans = enumerate_fixed_model_plans(_FOUR_METHODS)
    assert all(p.seq_len == SEQ_LEN for p in plans)
