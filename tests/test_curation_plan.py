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
    for name in ["dclm", "nemotron_org", "nemotron_full_bos_fixed", "fineweb_edu", "resiliparse"]:
        m = METHODS[name]
        assert m.sampled_warcs == 3_000, f"{name} sampled_warcs != 3000 — test premise broken"
        assert count_valid(m, t_target=None) == 78, f"{name} A != 78"
        assert count_valid(m, t_target=20e12) == 49, f"{name} B != 49"


def test_total_plans_dclm_only_is_127():
    plans = enumerate_plans([METHODS["dclm"]], [None, 20e12])
    assert len(plans) == 127


def test_total_plans_3k_methods_is_127_each():
    """The five canonical 3k methods each produce 78 (ExpA) + 49 (ExpB at 20T) = 127."""
    methods_3k = [METHODS[n] for n in ("dclm", "nemotron_org", "nemotron_full_bos_fixed", "fineweb_edu", "resiliparse")]
    plans = enumerate_plans(methods_3k, [None, 20e12])
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
    assert resolve_experiments(["A"], [20e12]) == [("A", None)]


def test_resolve_experiments_b_only():
    assert resolve_experiments(["B"], [20e12]) == [("B", 20e12)]


def test_resolve_experiments_c_only():
    """ExpC must be opted into explicitly (not via 'all') and uses DEFAULT_T_TARGET_C."""
    out = resolve_experiments(["C"], [20e12])  # t_targets is ignored for C
    assert out == [("C", 33e12)]


def test_resolve_experiments_c_with_explicit_t_target():
    out = resolve_experiments(["C"], [20e12], t_target_c=50e12)
    assert out == [("C", 50e12)]


def test_resolve_experiments_all_default():
    out = resolve_experiments(["all"], [20e12])
    kinds = {spec[0] for spec in out}
    # 'all' covers A and B but NOT C (C requires explicit opt-in to avoid
    # accidentally launching ExpC's 4-method set on every sweep).
    assert kinds == {"A", "B"}
    assert ("A", None) in out
    assert ("B", 20e12) in out


def test_resolve_experiments_multiple_t_targets():
    assert resolve_experiments(["B"], [20e12, 200e12]) == [("B", 20e12), ("B", 200e12)]


def test_resolve_experiments_b_and_c_combined():
    """User can request both B and C in a single sweep (both fire ExpB at t_targets and one ExpC)."""
    out = resolve_experiments(["B", "C"], [20e12])
    assert ("B", 20e12) in out
    assert ("C", 33e12) in out


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


# =============================================================================
# Experiment C: 10k WARC re-extraction + 33T target + uniform cross-method cap
# =============================================================================
#
# These tests pin down the math for ExpC's two regimes (data-rich + sliced)
# and the uniform 43.1B cap that's applied across all methods. Use synthetic
# CurationMethod fixtures so we don't depend on the (still-pending) 10k cache
# hashes for dclm/nemotron.

from experiments.scaling_law_sweeps.curation_plan import (
    EXPC_METHOD_NAMES,
    EXPC_SAMPLED_WARCS,
    _iter_valid_candidates,
    _normalize_experiment_spec,
    expc_uniform_t_exp_cap,
)
from experiments.scaling_law_sweeps.data_curation_math import (
    TOTAL_WARCS_CC,
    CurationMethod,
)


# Real cache D_obs values (3k WARCs); extrapolated to 10k under uniformity.
_DCLM_3K = 2_663_454_015
_NEMOTRON_3K = 1_919_401_016
_LC_3K = 56_008_357_279
_RES_3K = 142_652_598_588


def _synthetic_method(name: str, d_obs: int, sampled_warcs: int) -> CurationMethod:
    return CurationMethod(
        name=name,
        tokenized_rel_path=f"tokenized/test_{name}/",
        d_obs_tokens=d_obs,
        sampled_warcs=sampled_warcs,
    )


@pytest.fixture
def dclm_10k_synthetic() -> CurationMethod:
    """DCLM at 10,364 WARCs under uniformity: D_obs scales linearly."""
    return _synthetic_method(
        "dclm_10k",
        d_obs=int(_DCLM_3K * EXPC_SAMPLED_WARCS / 3_000),
        sampled_warcs=EXPC_SAMPLED_WARCS,
    )


@pytest.fixture
def nemotron_10k_synthetic() -> CurationMethod:
    return _synthetic_method(
        "nemotron_10k",
        d_obs=int(_NEMOTRON_3K * EXPC_SAMPLED_WARCS / 3_000),
        sampled_warcs=EXPC_SAMPLED_WARCS,
    )


@pytest.fixture
def llm_curated_3k_synthetic() -> CurationMethod:
    """LC stays at 3k WARCs in ExpC — its D_obs already exceeds the uniform cap."""
    return _synthetic_method("lc", d_obs=_LC_3K, sampled_warcs=3_000)


@pytest.fixture
def resiliparse_3k_synthetic() -> CurationMethod:
    return _synthetic_method("res", d_obs=_RES_3K, sampled_warcs=3_000)


# ---- Uniform cap math ----


def test_expc_uniform_t_exp_cap_at_33T_is_43_14B():
    """T=33T × 10,364 / 7,925,398 = 43.1374...B. Pin the exact value."""
    cap = expc_uniform_t_exp_cap(33e12)
    expected = 33e12 * 10_364 / TOTAL_WARCS_CC
    assert cap == pytest.approx(expected)
    assert cap == pytest.approx(43.137e9, rel=1e-3)


def test_expc_uniform_t_exp_cap_linear_in_t_target():
    """Cap scales linearly with T_target."""
    c_20 = expc_uniform_t_exp_cap(20e12)
    c_33 = expc_uniform_t_exp_cap(33e12)
    assert c_33 / c_20 == pytest.approx(33 / 20)


def test_expc_uniform_t_exp_cap_uses_total_warcs_cc():
    """Sanity: denominator is the canonical TOTAL_WARCS_CC, not a stale literal."""
    cap = expc_uniform_t_exp_cap(20e12, sampled_warcs=10_364)
    assert cap == pytest.approx(20e12 * 10_364 / TOTAL_WARCS_CC)


# ---- Cross-method ceiling uniformity ----


def test_expc_ceiling_uniform_across_all_four_methods(
    dclm_10k_synthetic,
    nemotron_10k_synthetic,
    llm_curated_3k_synthetic,
    resiliparse_3k_synthetic,
):
    """The whole point of the uniform cap: every ExpC method tops out at 43.14B.

    For the two 10k sliced methods, ceiling = T/s_10k = 43.14B (slicing regime).
    For the two 3k data-rich methods, ceiling = D_obs (>> 43.14B), capped to 43.14B.
    All four converge on the same ceiling — required for cross-method comparability.
    """
    cap = expc_uniform_t_exp_cap(33e12)
    methods = [
        dclm_10k_synthetic,
        nemotron_10k_synthetic,
        llm_curated_3k_synthetic,
        resiliparse_3k_synthetic,
    ]
    seen_t_exps_by_method = []
    for m in methods:
        t_exps = sorted(cand.tokens for _, cand, _ in _iter_valid_candidates(m, t_target=33e12, kind="C", uniform_t_exp_cap=cap))
        seen_t_exps_by_method.append(t_exps)
        # Every emitted candidate respects the cap:
        assert all(t <= cap for t in t_exps), f"{m.name}: candidate exceeds cap"
    # All four methods see identical candidate sets:
    assert seen_t_exps_by_method[0] == seen_t_exps_by_method[1] == seen_t_exps_by_method[2] == seen_t_exps_by_method[3]


def test_expc_ceiling_caps_data_rich_methods_below_d_obs(llm_curated_3k_synthetic):
    """Without the uniform cap, LC would admit candidates up to D_obs=56B.
    With the cap, it's pinned at 43.14B. Confirm the cap actually bites.
    """
    cap = expc_uniform_t_exp_cap(33e12)
    with_cap = list(_iter_valid_candidates(llm_curated_3k_synthetic, t_target=33e12, kind="C", uniform_t_exp_cap=cap))
    no_cap = list(_iter_valid_candidates(llm_curated_3k_synthetic, t_target=33e12, kind="C", uniform_t_exp_cap=None))
    # No-cap must admit STRICTLY MORE candidates (the ones in (43.1B, 56B]).
    assert len(no_cap) > len(with_cap)
    # And the extra ones are precisely those above the cap.
    cap_t_exps = {cand.tokens for _, cand, _ in with_cap}
    extra = [cand.tokens for _, cand, _ in no_cap if cand.tokens not in cap_t_exps]
    assert all(t > cap for t in extra)


def test_expc_ceiling_caps_sliced_methods_at_their_natural_ceiling(dclm_10k_synthetic):
    """For 10k sliced methods, T/s_10k ≈ 43.14B, which equals the uniform cap.
    So with-cap and no-cap should produce identical candidate sets (cap doesn't bite).
    """
    cap = expc_uniform_t_exp_cap(33e12)
    with_cap = sorted(cand.tokens for _, cand, _ in _iter_valid_candidates(dclm_10k_synthetic, t_target=33e12, kind="C", uniform_t_exp_cap=cap))
    no_cap = sorted(cand.tokens for _, cand, _ in _iter_valid_candidates(dclm_10k_synthetic, t_target=33e12, kind="C", uniform_t_exp_cap=None))
    assert with_cap == no_cap


# ---- Slicing math correctness for ExpC sliced methods ----


def test_expc_sliced_methods_satisfy_epoch_match(dclm_10k_synthetic):
    """For data-constrained ExpC methods, every candidate's slice must give
    sim_epochs == target_epochs (the central invariant of simulated epoching).
    """
    from experiments.scaling_law_sweeps.data_curation_math import slice_tokens_for

    t_target = 33e12
    target_epochs = t_target / dclm_10k_synthetic.d_proj
    cap = expc_uniform_t_exp_cap(t_target)
    for _, cand, _ in _iter_valid_candidates(dclm_10k_synthetic, t_target=t_target, kind="C", uniform_t_exp_cap=cap):
        slice_size = slice_tokens_for(dclm_10k_synthetic, cand.tokens, t_target)
        sim_epochs = cand.tokens / slice_size
        assert sim_epochs == pytest.approx(target_epochs, rel=1e-12)
        # And the slice fits in D_obs (the slicing constraint binds at the cap):
        assert slice_size <= dclm_10k_synthetic.d_obs_tokens


def test_expc_data_rich_methods_have_t_exp_below_d_obs_at_cap(llm_curated_3k_synthetic, resiliparse_3k_synthetic):
    """For LC and Res, every accepted ExpC candidate must satisfy T_exp ≤ D_obs.
    The runner trains naturally on D_obs — over-epoching would break the i.i.d.
    argument that justifies dropping the slicing.
    """
    cap = expc_uniform_t_exp_cap(33e12)
    for m in (llm_curated_3k_synthetic, resiliparse_3k_synthetic):
        for _, cand, _ in _iter_valid_candidates(m, t_target=33e12, kind="C", uniform_t_exp_cap=cap):
            assert cand.tokens <= m.d_obs_tokens, f"{m.name}: T_exp={cand.tokens} > D_obs={m.d_obs_tokens}"


# ---- Placeholder safety (refuse to enumerate ExpC with d_obs=0) ----


def test_expc_placeholder_method_raises_on_enumeration():
    """dclm_10k / nemotron_10k registered as placeholders with d_obs=0;
    enumerating them with kind='C' must raise rather than emit nonsense.
    """
    placeholder = CurationMethod(
        name="dclm_10k",
        tokenized_rel_path="tokenized/TBD_dclm_10k_HASH_PENDING/",
        d_obs_tokens=0,
        sampled_warcs=EXPC_SAMPLED_WARCS,
    )
    with pytest.raises(ValueError, match="d_obs_tokens=0"):
        list(_iter_valid_candidates(placeholder, t_target=33e12, kind="C"))


def test_expc_placeholder_method_does_not_raise_for_other_kinds():
    """Placeholder check only kicks in for ExpC. ExpA/ExpB enumeration shouldn't
    blow up on a 0-d_obs method (they wouldn't usually call it, but the guard
    should be narrow)."""
    placeholder = CurationMethod(
        name="x",
        tokenized_rel_path="tokenized/TBD/",
        d_obs_tokens=0,
        sampled_warcs=EXPC_SAMPLED_WARCS,
    )
    # ExpA: should not raise (slicing isn't applied; d_obs is only used for
    # implicit_target_exp_a, which is degenerate but doesn't fail).
    list(_iter_valid_candidates(placeholder, t_target=None, kind="A"))


# ---- Backward compat: ExpA and ExpB unchanged ----


def test_expa_count_dclm_unchanged_at_78():
    """ExpA candidate count must not regress."""
    assert count_valid(METHODS["dclm"], t_target=None) == 78


def test_expb_count_dclm_unchanged_at_49():
    """ExpB candidate count for DCLM must not regress."""
    assert count_valid(METHODS["dclm"], t_target=20e12) == 49


def test_expb_lc_count_unchanged_with_default_kind():
    """LC ExpB candidate count under the original (slicing-regime) ceiling.

    Even though LC is data-rich, the DEFAULT kind="B" preserves the old
    (slicing) ceiling — so this count should match the prior baseline.
    """
    n_lc = count_valid(METHODS["llm_curated_bos_fixed"], t_target=20e12)
    n_dclm = count_valid(METHODS["dclm"], t_target=20e12)
    assert n_lc == n_dclm  # same s, same ceiling, same candidate set


# ---- enumerate_plans accepts ExperimentSpec tuples and legacy float|None ----


def test_normalize_experiment_spec_legacy_none():
    assert _normalize_experiment_spec(None) == ("A", None)


def test_normalize_experiment_spec_legacy_float():
    assert _normalize_experiment_spec(20e12) == ("B", 20e12)


def test_normalize_experiment_spec_explicit_c():
    assert _normalize_experiment_spec(("C", 33e12)) == ("C", 33e12)


def test_normalize_experiment_spec_rejects_bad_kind():
    with pytest.raises(ValueError, match="Unknown experiment kind"):
        _normalize_experiment_spec(("D", 1e12))


def test_normalize_experiment_spec_rejects_a_with_t_target():
    with pytest.raises(ValueError, match="A must have t_target=None"):
        _normalize_experiment_spec(("A", 20e12))


def test_normalize_experiment_spec_rejects_bc_without_t_target():
    with pytest.raises(ValueError, match="t_target to be set"):
        _normalize_experiment_spec(("B", None))
    with pytest.raises(ValueError, match="t_target to be set"):
        _normalize_experiment_spec(("C", None))


def test_enumerate_plans_legacy_form_unchanged():
    """Calling enumerate_plans with the old `[None, 20e12]` form keeps producing
    expA_natural + expB_T20T plans — no silent shift to ExpC."""
    plans = enumerate_plans([METHODS["dclm"]], [None, 20e12])
    tags = {p.experiment_tag for p in plans}
    assert tags == {"expA_natural", "expB_T20T"}


def test_enumerate_plans_with_expc_tuples_emits_expc_tag():
    """With explicit ('C', 33e12) spec, enumerate emits expC_T33T runs."""
    # Use a synthetic LC method to avoid the placeholder-d_obs guard.
    method = _synthetic_method("lc_test", d_obs=_LC_3K, sampled_warcs=3_000)
    plans = enumerate_plans([method], [("C", 33e12)])
    assert len(plans) > 0
    assert all(p.experiment_tag == "expC_T33T" for p in plans)


def test_enumerate_plans_expc_uniform_cap_applied_via_top_level():
    """End-to-end: enumerate_plans for an ExpC spec produces only candidates ≤ cap,
    even when D_obs would allow more (data-rich case).
    """
    method = _synthetic_method("res_test", d_obs=_RES_3K, sampled_warcs=3_000)
    plans = enumerate_plans([method], [("C", 33e12)])
    cap = expc_uniform_t_exp_cap(33e12)
    for p in plans:
        assert p.t_exp <= cap


# ---- ExpC method registry ----


def test_expc_method_names_are_the_four_canonical_methods():
    """The 4 ExpC methods: dclm_10k, nemotron_10k, llm_curated_bos_fixed, resiliparse.
    Fineweb deliberately excluded (per upstream review)."""
    assert set(EXPC_METHOD_NAMES) == {"dclm_10k", "nemotron_10k", "llm_curated_bos_fixed", "resiliparse"}


def test_expc_method_names_all_resolvable_in_methods_registry():
    for name in EXPC_METHOD_NAMES:
        assert name in METHODS, f"ExpC method {name} missing from METHODS registry"


def test_expc_data_rich_methods_pinned_to_us_central1():
    """LC and Resiliparse must be region-locked to us-central1 in ExpC
    (BOS-fixed cache hard requirement + cross-experiment dedup consistency)."""
    assert METHODS["llm_curated_bos_fixed"].pin_region == "us-central1"
    assert METHODS["resiliparse"].pin_region == "us-central1"
    assert METHODS["nemotron_full_bos_fixed"].pin_region == "us-central1"


def test_expc_10k_methods_have_correct_sampled_warcs():
    assert METHODS["dclm_10k"].sampled_warcs == EXPC_SAMPLED_WARCS
    assert METHODS["nemotron_10k"].sampled_warcs == EXPC_SAMPLED_WARCS
    assert EXPC_SAMPLED_WARCS == 10_364
