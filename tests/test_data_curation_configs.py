# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Config-generation tests for the data curation sweep.

These tests exercise `build_curation_sweep()` end-to-end and inspect the
generated `ExecutorStep` + nested `LMMixtureDatasetConfig` to verify that:

- Simulated epoching is actually wired into the `LmDataConfig` fields
  levanter consumes (`target_budget`, `experiment_budget`).
- Experiment A and Experiment B produce configs with the expected relationship
  (target_budget = 20T vs target_budget = T_exp * s).
- Per-method and per-experiment metadata (tags, output paths) are distinct and
  informative, not silently overwritten.

Tests do NOT hit GCS — they use the hardcoded GCS path on `CurationMethod` but
never actually read from it (we inspect the *config*, not run the training).
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps import data_curation_isoflop as dc
from experiments.scaling_law_sweeps.data_curation_math import CurationMethod

# Deterministic fixtures — ecxat tokens values match .stats.json.
DCLM_PATH = "mirror://tokenized/baseline_dclm-23e9be/"
FINEWEB_PATH = "mirror://tokenized/baseline_fineweb_edu-7a3bc5/"


@pytest.fixture
def dclm() -> CurationMethod:
    return CurationMethod(
        name="dclm",
        tokenized_path=DCLM_PATH,
        d_obs_tokens=2_663_454_015,
        sampled_warcs=3_000,
    )


@pytest.fixture
def fineweb() -> CurationMethod:
    return CurationMethod(
        name="fw_edu",
        tokenized_path=FINEWEB_PATH,
        d_obs_tokens=817_221_529,
        sampled_warcs=3_000,
    )


@pytest.fixture
def tiny_sweep_dclm_b(dclm) -> list:
    """Build a minimal sweep for DCLM, Experiment B, single budget."""
    return dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,))


@pytest.fixture
def tiny_sweep_dclm_a(dclm) -> list:
    """Build a minimal sweep for DCLM, Experiment A, single budget."""
    return dc.build_curation_sweep(dclm, t_target=None, budgets=(3e18,))


# =============================================================================
# Basic sweep shape
# =============================================================================


def test_build_sweep_returns_non_empty(tiny_sweep_dclm_b):
    assert len(tiny_sweep_dclm_b) > 0


def test_experiment_b_step_count_matches_enumeration(dclm, tiny_sweep_dclm_b):
    expected = dc._count_valid(dclm, t_target=20e12, budgets=(3e18,))
    assert len(tiny_sweep_dclm_b) == expected


def test_experiment_a_step_count_matches_enumeration(dclm, tiny_sweep_dclm_a):
    expected = dc._count_valid(dclm, t_target=None, budgets=(3e18,))
    assert len(tiny_sweep_dclm_a) == expected


def test_build_sweep_all_methods_non_empty():
    """Every method registered in the registry builds at least one step."""
    # Use hardcoded D_obs so we don't hit GCS in tests.
    methods = {
        "dclm": CurationMethod("dclm", DCLM_PATH, d_obs_tokens=2_663_454_015),
        "fw": CurationMethod("fw", FINEWEB_PATH, d_obs_tokens=817_221_529),
    }
    for m in methods.values():
        assert len(dc.build_curation_sweep(m, t_target=20e12, budgets=(3e18,))) > 0
        assert len(dc.build_curation_sweep(m, t_target=None, budgets=(3e18,))) > 0


# =============================================================================
# LmDataConfig — simulated epoching actually applied
# =============================================================================


def test_experiment_b_target_budget_is_t_target(tiny_sweep_dclm_b):
    """Every Experiment B step must have target_budget = T_target (20T)."""
    for step in tiny_sweep_dclm_b:
        data = step.config.train_config.data
        assert data.target_budget == 20_000_000_000_000


def test_experiment_b_experiment_budget_within_ceiling(dclm, tiny_sweep_dclm_b):
    """Every Experiment B step's experiment_budget must be ≤ the T_exp ceiling."""
    from experiments.scaling_law_sweeps.data_curation_math import t_exp_ceiling

    ceiling = t_exp_ceiling(dclm, 20e12)
    for step in tiny_sweep_dclm_b:
        data = step.config.train_config.data
        assert data.experiment_budget is not None
        # simulated_epoching_train computes experiment_budget = batch * steps * seq_len,
        # which should match the candidate's `tokens` closely (±1 due to rounding).
        assert data.experiment_budget <= int(ceiling) + 1


def test_experiment_b_experiment_budget_positive(tiny_sweep_dclm_b):
    for step in tiny_sweep_dclm_b:
        assert step.config.train_config.data.experiment_budget > 0


def test_experiment_b_epoch_count_matches_target(dclm, tiny_sweep_dclm_b):
    """Critical invariant: T_exp / D_proj_sliced == T_target / D_proj.

    Equivalently, target_budget / experiment_budget == D_proj / slice.
    By construction of the slicing formula, slice is implied by these values.
    We assert that T_target / experiment_budget matches what we'd compute via
    `slice_tokens_for` and the implied epoch count.
    """
    from experiments.scaling_law_sweeps.data_curation_math import slice_tokens_for

    for step in tiny_sweep_dclm_b:
        data = step.config.train_config.data
        t_exp = data.experiment_budget
        t_target = data.target_budget
        expected_slice = slice_tokens_for(dclm, t_exp, t_target)
        expected_epochs = t_exp / expected_slice
        target_epochs = t_target / dclm.d_proj
        assert expected_epochs == pytest.approx(target_epochs, rel=1e-6)


def test_experiment_a_target_budget_is_implicit(dclm, tiny_sweep_dclm_a):
    """Experiment A: target_budget should equal experiment_budget × s (integerized)."""
    for step in tiny_sweep_dclm_a:
        data = step.config.train_config.data
        expected = int(data.experiment_budget * dclm.s)
        assert data.target_budget == expected


def test_experiment_a_and_b_differ_on_target_budget(dclm):
    """Experiment A and B should produce different target_budget values for the same candidate."""
    a = dc.build_curation_sweep(dclm, t_target=None, budgets=(3e18,))
    b = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,))
    # Match candidates by their experiment_budget (which is the same T_exp).
    a_map = {s.config.train_config.data.experiment_budget: s.config.train_config.data.target_budget for s in a}
    b_map = {s.config.train_config.data.experiment_budget: s.config.train_config.data.target_budget for s in b}
    # Overlap on at least one T_exp (Experiment B strictly ⊆ Experiment A below ceiling).
    common = set(a_map) & set(b_map)
    assert common, "Expected at least one shared T_exp between A and B"
    for t_exp in common:
        assert a_map[t_exp] != b_map[t_exp], f"A and B target_budgets collide at T_exp={t_exp}"


# =============================================================================
# Tokenized cache wiring
# =============================================================================


def test_config_cache_dir_matches_method(dclm, tiny_sweep_dclm_b):
    """The LMMixtureDatasetConfig must point at the method's mirror:// path."""
    for step in tiny_sweep_dclm_b:
        data = step.config.train_config.data
        assert dclm.name in data.components
        component = data.components[dclm.name]
        # cache_dir is now a raw mirror:// string (Levanter expects str).
        # MirrorFileSystem in rigging handles cross-region resolution at read time.
        assert isinstance(component.cache_dir, str)
        assert component.cache_dir == dclm.tokenized_path
        assert dclm.tokenized_path.startswith("mirror://"), "must never use raw gs:// for cross-region safety"


def test_config_cache_dir_uses_mirror_prefix(tiny_sweep_dclm_b):
    """Critical safety invariant: training configs must use mirror:// strings.

    If this test fails, a run might do cross-region reads during training.
    """
    for step in tiny_sweep_dclm_b:
        for name, component in step.config.train_config.data.components.items():
            if name == "dclm":  # only the training component — validation sets route via other steps
                assert isinstance(component.cache_dir, str)
                assert component.cache_dir.startswith("mirror://")
                assert "gs://" not in component.cache_dir


def test_raw_gs_path_rejected_at_construction():
    """CurationMethod must reject raw gs:// paths — fail fast, not at training time."""
    from experiments.scaling_law_sweeps.data_curation_math import CurationMethod

    m = CurationMethod("bad", "gs://marin-us-central2/tokenized/bad/", d_obs_tokens=1_000_000_000)
    with pytest.raises(ValueError, match="mirror://"):
        m.as_lm_mixture_config()
    with pytest.raises(ValueError, match="mirror://"):
        m.tokenized_input()


def test_config_tokenizer_is_llama3(tiny_sweep_dclm_b):
    for step in tiny_sweep_dclm_b:
        assert step.config.train_config.data.tokenizer == "meta-llama/Meta-Llama-3.1-8B"


def test_train_weights_sum_to_one_on_training_component(dclm, tiny_sweep_dclm_b):
    """Training weight on the method's component should be 1.0; validation sets at 0.0."""
    for step in tiny_sweep_dclm_b:
        weights = step.config.train_config.data.train_weights
        assert weights[dclm.name] == 1.0
        for name, w in weights.items():
            if name != dclm.name:
                assert w == 0.0, f"Non-training component {name} should have weight 0 (was {w})"


# =============================================================================
# Naming, tags, output paths
# =============================================================================


def test_step_name_includes_method_and_experiment(dclm, tiny_sweep_dclm_b):
    for step in tiny_sweep_dclm_b:
        assert "curation-dclm" in step.name
        assert "expB_T20T" in step.name


def test_step_name_differs_between_experiments(dclm):
    a = dc.build_curation_sweep(dclm, t_target=None, budgets=(3e18,))
    b = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,))
    a_names = {s.name for s in a}
    b_names = {s.name for s in b}
    assert a_names != b_names
    assert a_names.isdisjoint(b_names), "A and B step names must not collide"


def test_step_name_differs_between_methods(dclm, fineweb):
    d = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,))
    f = dc.build_curation_sweep(fineweb, t_target=20e12, budgets=(3e18,))
    d_names = {s.name for s in d}
    f_names = {s.name for s in f}
    assert d_names.isdisjoint(f_names), "Method step names must not collide"


def test_output_path_embedded_in_override(tiny_sweep_dclm_b):
    """We set `with_output_path("checkpoints/isoflop-curation/{run_name}")`."""
    for step in tiny_sweep_dclm_b:
        assert "isoflop-curation/" in step.override_output_path


def test_all_step_names_unique_within_sweep(tiny_sweep_dclm_b):
    names = [s.name for s in tiny_sweep_dclm_b]
    assert len(names) == len(set(names)), f"Duplicate step names in sweep: {names}"


# =============================================================================
# TPU resource selection
# =============================================================================


def test_v4_generation_picks_v4_resources(dclm):
    steps = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,), tpu_generation="v4")
    for step in steps:
        resources = step.config.resources
        # Inspect the underlying TPU spec
        tpu_str = str(resources)
        assert "v4-" in tpu_str, f"Expected v4 TPU resource, got {tpu_str}"


def test_v5p_generation_picks_v5p_resources(dclm):
    steps = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,), tpu_generation="v5p")
    for step in steps:
        tpu_str = str(step.config.resources)
        assert "v5p-" in tpu_str, f"Expected v5p TPU resource, got {tpu_str}"


def test_unknown_tpu_generation_raises(dclm):
    with pytest.raises(ValueError, match="Unknown tpu generation"):
        dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,), tpu_generation="v6")


# =============================================================================
# Edge cases
# =============================================================================


def test_experiment_b_with_small_t_target_rejects_all_candidates():
    """If T_target is impossibly small, every candidate exceeds the ceiling → empty sweep."""
    m = CurationMethod("x", DCLM_PATH, d_obs_tokens=2_663_454_015)
    # T_target = 1T with s=2642 means ceiling ≈ 378M. All candidates are > 1B.
    steps = dc.build_curation_sweep(m, t_target=1e12)
    assert len(steps) == 0


def test_experiment_b_with_huge_t_target_accepts_everything_when_floor_disabled():
    """If T_target is huge, the ceiling exceeds every candidate. With floor disabled, all 78 survive.

    (Huge T_target also shrinks the slice formula's output, so the floor would otherwise reject
    most candidates — that's a correct behavior that we disable here to isolate the ceiling test.)
    """
    m = CurationMethod("x", DCLM_PATH, d_obs_tokens=2_663_454_015)
    steps = dc.build_curation_sweep(m, t_target=10_000e12, min_slice_tokens=0)
    assert len(steps) == 78


def test_experiment_a_covers_full_compute_range():
    """Experiment A doesn't filter — should match heuristic's full candidate count."""
    m = CurationMethod("x", DCLM_PATH, d_obs_tokens=2_663_454_015)
    assert len(dc.build_curation_sweep(m, t_target=None)) == 78


def test_experiment_b_at_t_target_20t_expected_count():
    """Golden: 49 runs at T_target=20T."""
    m = CurationMethod("x", DCLM_PATH, d_obs_tokens=2_663_454_015)
    assert len(dc.build_curation_sweep(m, t_target=20e12)) == 49
