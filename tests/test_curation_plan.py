# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure-enumeration `curation_plan` module.

Verifies expected run counts (the v1 audit values), TPU pair assignments,
and CLI-args round-trip for the PlannedRun → standalone-child contract.
"""

from __future__ import annotations

import argparse

import pytest

from experiments.scaling_law_sweeps.curation_plan import (
    METHODS,
    PlannedRun,
    count_valid,
    enumerate_plans,
    experiment_tag,
    per_budget_counts,
    print_dry_run,
    resolve_experiments,
    resolve_methods,
)

# =============================================================================
# Run counts (golden values from v1 audit — must stay stable)
# =============================================================================


def test_dclm_count_experiment_a_is_78():
    assert count_valid(METHODS["dclm"], t_target=None) == 78


def test_dclm_count_experiment_b_at_20T_is_49():
    assert count_valid(METHODS["dclm"], t_target=20e12) == 49


def test_per_budget_b_distribution_matches_v1():
    counts = list(per_budget_counts(METHODS["dclm"], t_target=20e12).values())
    assert counts == [5, 8, 11, 6, 9, 6, 4]


def test_count_independent_of_method_for_same_s():
    """All methods sampled on 3000 WARCs share s, so they share count distribution."""
    for name in ["dclm", "nemotron_org", "nemotron_full", "fineweb_edu", "resiliparse"]:
        m = METHODS[name]
        assert count_valid(m, t_target=None) == 78, f"{name} A != 78"
        assert count_valid(m, t_target=20e12) == 49, f"{name} B != 49"


def test_total_plans_dclm_only_is_127():
    plans = enumerate_plans([METHODS["dclm"]], [None, 20e12])
    assert len(plans) == 127


def test_total_plans_all_methods_is_5_x_127():
    plans = enumerate_plans(list(METHODS.values()), [None, 20e12])
    # 5 methods × (78 A + 49 B) = 5 × 127 = 635
    assert len(plans) == 5 * 127


# =============================================================================
# experiment_tag formatter
# =============================================================================


def test_experiment_tag_natural():
    assert experiment_tag(None) == "expA_natural"


def test_experiment_tag_integer_t_target():
    assert experiment_tag(20e12) == "expB_T20T"
    assert experiment_tag(200e12) == "expB_T200T"


def test_experiment_tag_decimal_t_target():
    assert experiment_tag(36.5e12) == "expB_T36.5T"


# =============================================================================
# PlannedRun: stable run_name_core, run_key, CLI roundtrip
# =============================================================================


def _sample_plan() -> PlannedRun:
    plans = enumerate_plans([METHODS["dclm"]], [None])
    return plans[0]


def test_planned_run_run_name_core_format():
    p = _sample_plan()
    assert p.method_name in p.run_name_core
    assert p.experiment_tag in p.run_name_core
    assert f"d{p.hidden_dim}" in p.run_name_core
    assert f"L{p.num_layers}" in p.run_name_core
    assert f"B{p.batch_size}" in p.run_name_core


def test_planned_run_key_matches_region_tracker_format():
    """PlannedRun.run_key must match what region_tracker.run_key_for() produces."""
    from experiments.scaling_law_sweeps import region_tracker

    p = _sample_plan()
    expected = region_tracker.run_key_for(p.method_name, p.experiment_tag, p.run_name_core)
    assert p.run_key == expected


def test_cli_args_roundtrip():
    """PlannedRun.to_cli_args() ↔ from_namespace must be a faithful inverse for primitive fields."""
    p = _sample_plan()
    args = p.to_cli_args()

    parser = argparse.ArgumentParser()
    PlannedRun.add_cli_args(parser)
    parsed = parser.parse_args(args)
    rebuilt = PlannedRun.from_namespace(parsed)

    # All primitive fields must match (exact for ints, approx for floats)
    assert rebuilt.method_name == p.method_name
    assert rebuilt.experiment_tag == p.experiment_tag
    assert rebuilt.budget == pytest.approx(p.budget, rel=1e-5)
    assert rebuilt.hidden_dim == p.hidden_dim
    assert rebuilt.num_layers == p.num_layers
    assert rebuilt.num_heads == p.num_heads
    assert rebuilt.intermediate_dim == p.intermediate_dim
    assert rebuilt.batch_size == p.batch_size
    assert rebuilt.train_steps == p.train_steps
    assert rebuilt.learning_rate == pytest.approx(p.learning_rate, rel=1e-5)
    assert rebuilt.adam_lr == pytest.approx(p.adam_lr, rel=1e-5)
    assert rebuilt.t_exp == pytest.approx(p.t_exp, rel=1e-5)
    assert rebuilt.t_target == pytest.approx(p.t_target, rel=1e-5)
    assert rebuilt.seq_len == p.seq_len


# =============================================================================
# TPU pair assignment
# =============================================================================


def test_every_plan_has_v4_and_v5p_tpu():
    plans = enumerate_plans([METHODS["dclm"]], [None, 20e12])
    for p in plans:
        assert p.v4_tpu.startswith("v4-"), f"{p.run_name_core}: bad v4_tpu {p.v4_tpu}"
        assert p.v5p_tpu.startswith("v5p-"), f"{p.run_name_core}: bad v5p_tpu {p.v5p_tpu}"


def test_tpu_pair_consistent_across_methods():
    """A given (budget, hidden_dim) candidate gets the same TPU pair regardless of method."""
    plans_dclm = enumerate_plans([METHODS["dclm"]], [None])
    plans_fw = enumerate_plans([METHODS["fineweb_edu"]], [None])
    by_key_dclm = {(p.budget, p.hidden_dim, p.batch_size): (p.v4_tpu, p.v5p_tpu) for p in plans_dclm}
    by_key_fw = {(p.budget, p.hidden_dim, p.batch_size): (p.v4_tpu, p.v5p_tpu) for p in plans_fw}
    for key, pair in by_key_dclm.items():
        assert by_key_fw.get(key) == pair


# =============================================================================
# CLI helpers
# =============================================================================


def test_resolve_methods_all_includes_resiliparse():
    """The user explicitly wants 'all' to include resiliparse (despite its size)."""
    resolved = resolve_methods(["all"])
    names = {m.name for m in resolved}
    assert "resiliparse" in names


def test_resolve_methods_subset():
    resolved = resolve_methods(["dclm", "fineweb_edu"])
    assert {m.name for m in resolved} == {"dclm", "fineweb_edu"}


def test_resolve_methods_unknown_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        resolve_methods(["nonexistent"])


def test_resolve_experiments_a_only():
    assert resolve_experiments(["A"], [20e12]) == [None]


def test_resolve_experiments_b_only():
    assert resolve_experiments(["B"], [20e12]) == [20e12]


def test_resolve_experiments_all_default():
    out = resolve_experiments(["all"], [20e12])
    assert None in out
    assert 20e12 in out


def test_resolve_experiments_multiple_t_targets():
    assert resolve_experiments(["B"], [20e12, 200e12]) == [20e12, 200e12]


# =============================================================================
# Dry-run output
# =============================================================================


def test_print_dry_run_does_not_crash(capsys):
    plans = enumerate_plans([METHODS["dclm"]], [None, 20e12])
    print_dry_run(plans)
    captured = capsys.readouterr().out
    assert "TOTAL: 127 runs" in captured
    assert "expA_natural" in captured
    assert "expB_T20T" in captured
    assert "v4=" in captured
    assert "v5p=" in captured
