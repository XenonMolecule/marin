# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the data curation sweep's slicing/ceiling/projection math.

The user is nervous that complex simulated-epoching plumbing will silently
misbehave and pollute scaling-law fits. These tests pin down the invariants
the sweep relies on, so any drift (in the heuristic, in levanter's slicing,
in our own helpers) lights a red test immediately.
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps import data_curation_isoflop as dc
from experiments.scaling_law_sweeps.data_curation_math import (
    TOTAL_WARCS_CC,
    CurationMethod,
    implicit_target_exp_a,
    slice_tokens_for,
    t_exp_ceiling,
)

# --- D_obs values from .stats.json at the time of design (stable constants). ---
# Methods use the same 3000-WARC sample; hardcoded here so tests don't need GCS.
DCLM_D_OBS = 2_663_454_015
FINEWEB_EDU_D_OBS = 817_221_529
NEMOTRON_ORG_D_OBS = 1_919_401_016
NEMOTRON_FULL_D_OBS = 2_695_507_851
RESILIPARSE_D_OBS = 142_652_598_588
# User's method, approximate (not yet tokenized):
USER_METHOD_D_OBS = int(58.4e9)


@pytest.fixture
def dclm() -> CurationMethod:
    return CurationMethod(
        name="dclm",
        tokenized_rel_path="tokenized/test/",
        d_obs_tokens=DCLM_D_OBS,
        sampled_warcs=3_000,
    )


@pytest.fixture
def fineweb_edu() -> CurationMethod:
    return CurationMethod(
        name="fw_edu",
        tokenized_rel_path="tokenized/test/",
        d_obs_tokens=FINEWEB_EDU_D_OBS,
        sampled_warcs=3_000,
    )


@pytest.fixture
def resiliparse() -> CurationMethod:
    return CurationMethod(
        name="resili",
        tokenized_rel_path="tokenized/test/",
        d_obs_tokens=RESILIPARSE_D_OBS,
        sampled_warcs=3_000,
    )


@pytest.fixture
def user_method() -> CurationMethod:
    return CurationMethod(
        name="user",
        tokenized_rel_path="tokenized/test/",
        d_obs_tokens=USER_METHOD_D_OBS,
        sampled_warcs=3_000,
    )


# =============================================================================
# 7.1  Pure math tests
# =============================================================================


def test_s_from_sampling(dclm):
    """s = total_warcs / sampled_warcs, using the user-confirmed total."""
    assert dclm.s == pytest.approx(TOTAL_WARCS_CC / 3_000)
    assert dclm.s == pytest.approx(2641.8, rel=1e-3)


def test_d_proj_invariant_to_sampled_warcs():
    """Doubling sampled WARCs doubles D_obs and halves s, so D_proj unchanged."""
    m1 = CurationMethod("x", "tokenized/test/", d_obs_tokens=1_000_000_000, sampled_warcs=3_000)
    m2 = CurationMethod("x", "tokenized/test/", d_obs_tokens=2_000_000_000, sampled_warcs=6_000)
    m3 = CurationMethod("x", "tokenized/test/", d_obs_tokens=10_000_000_000, sampled_warcs=30_000)
    assert m1.d_proj == pytest.approx(m2.d_proj, rel=1e-12)
    assert m1.d_proj == pytest.approx(m3.d_proj, rel=1e-12)


def test_t_exp_ceiling_expected_value(dclm):
    """Ceiling = T_target / s. With s≈2642, T_target=20T: ceiling≈7.57B."""
    ceiling = t_exp_ceiling(dclm, t_target=20e12)
    assert ceiling == pytest.approx(20e12 / dclm.s)
    assert ceiling == pytest.approx(7.57e9, rel=1e-2)


def test_t_exp_ceiling_shared_across_methods_with_same_s(dclm, fineweb_edu, resiliparse):
    """All methods sampled on the same 3000 WARCs share the same ceiling."""
    c_dclm = t_exp_ceiling(dclm, 20e12)
    c_fw = t_exp_ceiling(fineweb_edu, 20e12)
    c_res = t_exp_ceiling(resiliparse, 20e12)
    assert c_dclm == pytest.approx(c_fw)
    assert c_dclm == pytest.approx(c_res)


def test_ceiling_raises_with_more_warcs():
    """Lever #2: more sampled WARCs → smaller s → higher ceiling."""
    base = CurationMethod("x", "tokenized/test/", d_obs_tokens=1_000_000_000, sampled_warcs=3_000)
    scaled = CurationMethod("x", "tokenized/test/", d_obs_tokens=1_000_000_000, sampled_warcs=10_000)
    assert t_exp_ceiling(scaled, 20e12) == pytest.approx((10 / 3) * t_exp_ceiling(base, 20e12))


def test_ceiling_raises_with_larger_t_target(dclm):
    """Lever #1: larger T_target → higher ceiling (linear)."""
    c_20 = t_exp_ceiling(dclm, 20e12)
    c_200 = t_exp_ceiling(dclm, 200e12)
    assert c_200 == pytest.approx(10 * c_20)


def test_slice_formula_reproduces_target_epoch_count(dclm):
    """By construction: experiment epoch count = target epoch count.

    slice = T_exp * D_proj / T_target  =>  T_exp/slice = T_target/D_proj.
    This is the whole point of simulated epoching.
    """
    t_target = 20e12
    target_epochs = t_target / dclm.d_proj
    for t_exp in [1e9, 3e9, 7e9]:
        sl = slice_tokens_for(dclm, t_exp, t_target)
        sim_epochs = t_exp / sl
        assert sim_epochs == pytest.approx(target_epochs, rel=1e-12)


def test_implicit_target_is_t_exp_times_s(dclm):
    """Experiment A's implicit target scales linearly with T_exp."""
    for t_exp in [1e9, 5e9, 20e9]:
        assert implicit_target_exp_a(dclm, t_exp) == pytest.approx(t_exp * dclm.s)


def test_known_epoch_values_at_20T(dclm, fineweb_edu, resiliparse, user_method):
    """Sanity check against the hand computations we used for the design."""
    # DCLM: 20T / 7.03T ≈ 2.85 epochs
    assert 20e12 / dclm.d_proj == pytest.approx(2.85, abs=0.02)
    # FineWeb-Edu: 20T / 2.16T ≈ 9.26 — deep past the Muennighoff cliff
    fw_epochs = 20e12 / fineweb_edu.d_proj
    assert 9.0 < fw_epochs < 9.5
    # Resiliparse: 20T / 377T ≈ 0.053 — deeply sub-1-epoch (data-abundant)
    assert 20e12 / resiliparse.d_proj == pytest.approx(0.053, abs=0.005)
    # User's method: 20T / 154T ≈ 0.13 — sub-1-epoch
    assert 20e12 / user_method.d_proj == pytest.approx(0.13, abs=0.01)


def test_slice_monotone_in_t_exp(dclm):
    """slice scales linearly with T_exp at fixed T_target."""
    prev = 0.0
    for t_exp in [1e8, 1e9, 3e9, 7e9]:
        cur = slice_tokens_for(dclm, t_exp, t_target=20e12)
        assert cur > prev
        prev = cur


# =============================================================================
# 7.2  Enumeration / run-count tests
# =============================================================================


def test_experiment_b_rejects_above_ceiling(dclm):
    """Experiment B must have fewer valid candidates than A (ceiling rejects some)."""
    a = dc._count_valid(dclm, t_target=None)
    b = dc._count_valid(dclm, t_target=20e12)
    assert b < a
    # Every Experiment B candidate must satisfy the ceiling.
    ceiling = t_exp_ceiling(dclm, 20e12)
    for _, cand, _ in dc._iter_valid_candidates(dclm, t_target=20e12):
        assert cand.tokens <= ceiling, f"Experiment B candidate violates ceiling: {cand.tokens} > {ceiling}"


def test_expected_run_counts_match_design():
    """Golden: 78 runs in Experiment A, 49 in Experiment B, per-budget distribution locked."""
    # Any method works — counts only depend on ceiling (shared s), not D_obs.
    method = CurationMethod("x", "tokenized/test/", d_obs_tokens=DCLM_D_OBS, sampled_warcs=3_000)
    assert dc._count_valid(method, t_target=None) == 78
    assert dc._count_valid(method, t_target=20e12) == 49
    per_budget_b = list(dc._per_budget_counts(method, t_target=20e12).values())
    assert per_budget_b == [5, 8, 11, 6, 9, 6, 4]


def test_experiment_a_candidates_identical_across_methods(dclm, fineweb_edu, resiliparse):
    """In Experiment A, every method sees the same candidate set (no ceiling).
    Different D_obs → different implicit T_target, same T_exp values.
    """
    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    t_exps_dclm = sorted(cand.tokens for _, cand, _ in dc._iter_valid_candidates(dclm, t_target=None))
    t_exps_fw = sorted(cand.tokens for _, cand, _ in dc._iter_valid_candidates(fineweb_edu, t_target=None))
    t_exps_res = sorted(cand.tokens for _, cand, _ in dc._iter_valid_candidates(resiliparse, t_target=None))
    assert t_exps_dclm == t_exps_fw == t_exps_res
    # Sanity: total candidates match the heuristic enumeration.
    total = sum(1 for b in dc.BUDGETS for _ in completed_adamh_heuristic.candidates_for_budget(b, seq_len=dc.SEQ_LEN))
    assert len(t_exps_dclm) == total == 78


def test_experiment_b_candidate_set_identical_across_same_s_methods(dclm, fineweb_edu, resiliparse):
    """With same `s`, ceiling is the same, so ceiling-rejection gives the same T_exp set."""
    sets = []
    for m in [dclm, fineweb_edu, resiliparse]:
        sets.append(sorted(cand.tokens for _, cand, _ in dc._iter_valid_candidates(m, t_target=20e12)))
    assert sets[0] == sets[1] == sets[2]


def test_slice_floor_non_binding_with_default_config():
    """The 25M slice floor should not reject any candidate with the default BUDGETS/T_target.

    If the heuristic adds smaller-T_exp candidates (e.g. new smaller budgets), the floor
    would start binding — we'd want to notice via this test, then decide whether to
    lower the floor or accept the rejection.
    """
    for m in [
        CurationMethod("x", "tokenized/test/", d_obs_tokens=DCLM_D_OBS),
        CurationMethod("x", "tokenized/test/", d_obs_tokens=FINEWEB_EDU_D_OBS),
        CurationMethod("x", "tokenized/test/", d_obs_tokens=NEMOTRON_ORG_D_OBS),
        CurationMethod("x", "tokenized/test/", d_obs_tokens=NEMOTRON_FULL_D_OBS),
    ]:
        zero_floor = dc._count_valid(m, t_target=20e12, min_slice_tokens=0)
        with_floor = dc._count_valid(m, t_target=20e12, min_slice_tokens=25e6)
        assert zero_floor == with_floor, f"Floor binds unexpectedly for D_obs={m.d_obs_tokens}"


def test_target_budget_matches_implicit_in_exp_a(dclm):
    """In Experiment A, the reported target_budget should equal T_exp * s (integerized)."""
    for _, cand, target_budget in dc._iter_valid_candidates(dclm, t_target=None):
        expected = int(cand.tokens * dclm.s)
        assert target_budget == expected


def test_target_budget_is_t_target_in_exp_b(dclm):
    """In Experiment B, every survivor's target_budget should equal the requested T_target."""
    t_target = 20e12
    for _, _, target_budget in dc._iter_valid_candidates(dclm, t_target=t_target):
        assert target_budget == int(t_target)


# =============================================================================
# CLI plumbing tests
# =============================================================================


def test_cli_resolve_experiments_a_only():
    resolved = dc._resolve_experiments(["A"], [20e12])
    assert resolved == [None]


def test_cli_resolve_experiments_b_only():
    resolved = dc._resolve_experiments(["B"], [20e12])
    assert resolved == [20e12]


def test_cli_resolve_experiments_all_by_default():
    resolved = dc._resolve_experiments(["all"], [20e12])
    assert None in resolved
    assert 20e12 in resolved


def test_cli_resolve_experiments_multiple_t_targets():
    resolved = dc._resolve_experiments(["B"], [20e12, 200e12])
    assert resolved == [20e12, 200e12]


def test_cli_resolve_methods_all():
    registry = {"a": CurationMethod("a", "tokenized/test/", 1), "b": CurationMethod("b", "tokenized/test/", 2)}
    result = dc._resolve_methods(["all"], registry)
    assert {m.name for m in result} == {"a", "b"}


def test_cli_resolve_methods_subset():
    registry = {"a": CurationMethod("a", "tokenized/test/", 1), "b": CurationMethod("b", "tokenized/test/", 2)}
    result = dc._resolve_methods(["a"], registry)
    assert [m.name for m in result] == ["a"]


def test_cli_resolve_methods_rejects_unknown():
    registry = {"a": CurationMethod("a", "tokenized/test/", 1)}
    with pytest.raises(ValueError, match="Unknown method"):
        dc._resolve_methods(["nonexistent"], registry)


def test_experiment_tag_format():
    assert dc._experiment_tag(None) == "expA_natural"
    assert dc._experiment_tag(20e12) == "expB_T20T"
    assert dc._experiment_tag(200e12) == "expB_T200T"
    assert dc._experiment_tag(36e12) == "expB_T36T"


# =============================================================================
# Stats.json loader (mocked I/O)
# =============================================================================


def test_load_d_obs_from_stats(tmp_path, monkeypatch):
    """Loader reads `total_tokens` from `{path}/train/.stats.json`."""
    from experiments.scaling_law_sweeps.data_curation_math import load_d_obs_from_stats

    stats_dir = tmp_path / "train"
    stats_dir.mkdir()
    (stats_dir / ".stats.json").write_text('{"total_tokens": 1234567890, "total_elements": 42}')
    assert load_d_obs_from_stats(str(tmp_path)) == 1234567890
    # Trailing slash is tolerated
    assert load_d_obs_from_stats(f"{tmp_path}/") == 1234567890
