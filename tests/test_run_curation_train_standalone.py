# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone training child.

Verifies CLI parsing, model/optimizer config rebuild, tag construction,
TPU type detection, and the region-lock + output_path flow.

Does NOT actually invoke training — `run_levanter_train_lm` is mocked.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from experiments.scaling_law_sweeps import (
    curation_plan,
    region_tracker,
)
from experiments.scaling_law_sweeps import (
    run_curation_train_standalone as standalone,
)
from experiments.scaling_law_sweeps.curation_plan import METHODS, PlannedRun


def _sample_plan() -> PlannedRun:
    return curation_plan.enumerate_plans([METHODS["dclm"]], [None])[0]


def _sample_plan_exp_b(t_target: float = 20e12) -> PlannedRun:
    return curation_plan.enumerate_plans([METHODS["dclm"]], [t_target])[0]


# =============================================================================
# Arg parsing
# =============================================================================


def test_parse_args_with_full_planned_run():
    p = _sample_plan()
    argv = p.to_cli_args() + ["--tracker-prefix", "file:///tmp/test_tracker/"]
    ns = standalone._parse_args(argv)
    assert ns.method == p.method_name
    assert ns.experiment_tag == p.experiment_tag
    assert ns.budget == pytest.approx(p.budget, rel=1e-5)


def test_parse_args_default_tracker_prefix():
    p = _sample_plan()
    ns = standalone._parse_args(p.to_cli_args())
    assert ns.tracker_prefix == standalone.DEFAULT_TRACKER_PREFIX


def test_parse_args_default_wandb_group():
    p = _sample_plan()
    ns = standalone._parse_args(p.to_cli_args())
    assert ns.wandb_group == "data-curation-isoflop"


# =============================================================================
# Model / optimizer config rebuild
# =============================================================================


def test_build_model_config_matches_heuristic():
    """The standalone child's reconstructed model must match what the heuristic produces."""
    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    p = _sample_plan()
    rebuilt = standalone._build_model_config(p)

    candidates = list(completed_adamh_heuristic.candidates_for_budget(p.budget, seq_len=p.seq_len))
    canonical = next(c for c in candidates if c.model_config.hidden_dim == p.hidden_dim)
    cm = canonical.model_config

    assert rebuilt.hidden_dim == cm.hidden_dim
    assert rebuilt.num_layers == cm.num_layers
    assert rebuilt.num_heads == cm.num_heads
    assert rebuilt.intermediate_dim == cm.intermediate_dim
    assert rebuilt.max_seq_len == cm.max_seq_len


def test_build_optimizer_config_matches_heuristic():
    from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

    p = _sample_plan()
    rebuilt = standalone._build_optimizer_config(p)

    candidates = list(completed_adamh_heuristic.candidates_for_budget(p.budget, seq_len=p.seq_len))
    canonical = next(c for c in candidates if c.model_config.hidden_dim == p.hidden_dim)
    co = canonical.optimizer_config

    assert rebuilt.learning_rate == pytest.approx(co.learning_rate, rel=1e-5)
    assert rebuilt.adam_lr == pytest.approx(co.adam_lr, rel=1e-5)
    assert rebuilt.beta1 == pytest.approx(co.beta1)
    assert rebuilt.beta2 == pytest.approx(co.beta2)
    assert rebuilt.epsilon == pytest.approx(co.epsilon)


# =============================================================================
# Tags
# =============================================================================


def test_build_tags_includes_all_expected_fields():
    """Tags are what WandB's sidebar surfaces as one-click filters. They
    must cover: experiment type, method, budget/scale, architecture, and
    method-specific values, so grouping/filtering is trivial on the dashboard.
    """
    p = _sample_plan()
    method = METHODS[p.method_name]
    tags = standalone._build_tags(p, method)
    tag_str = "|".join(tags)
    for needle in [
        # Coarse grouping tags
        "experiment=A",  # ExpA sample → "A"
        f"method={p.method_name}",
        f"exp_full={p.experiment_tag}",
        # Budget / scale
        f"budget={p.budget:.0e}",
        # Architecture
        f"d_model={p.hidden_dim}",
        f"num_layers={p.num_layers}",
        f"batch_size={p.batch_size}",
        f"train_steps={p.train_steps}",
        # Method details
        f"sampled_warcs={method.sampled_warcs}",
        # Optimizer
        "optimizer=completed-adamh",
    ]:
        assert needle in tag_str, f"missing {needle!r} in tags; got {tags}"


def test_build_tags_exp_b_gets_B_label():
    """ExpB plans must tag experiment=B (short form for WandB filtering)."""
    p = _sample_plan_exp_b()
    method = METHODS[p.method_name]
    tags = standalone._build_tags(p, method)
    assert "experiment=B" in tags
    assert f"exp_full={p.experiment_tag}" in tags


def test_build_tags_includes_params_bucket():
    """Params tag must be in Mx/Bx form so it's readable + filterable in the UI."""
    p = _sample_plan()
    method = METHODS[p.method_name]
    tags = standalone._build_tags(p, method)
    params_tags = [t for t in tags if t.startswith("params=")]
    assert len(params_tags) == 1
    # For d512/L6 the param count is ~156M so the tag should end in "M".
    assert params_tags[0].endswith("M") or params_tags[0].endswith("B")


# =============================================================================
# TPU type detection
# =============================================================================


def test_detect_local_tpu_type_override():
    assert standalone._detect_local_tpu_type(override="v5p-32") == "v5p-32"


def test_detect_local_tpu_type_from_env(monkeypatch):
    monkeypatch.setenv("IRIS_DEVICE_VARIANT", "v5p-16")
    assert standalone._detect_local_tpu_type() == "v5p-16"


def test_detect_local_tpu_type_fallback(monkeypatch):
    for v in ("IRIS_DEVICE_VARIANT", "TPU_TYPE", "ACCELERATOR_TYPE"):
        monkeypatch.delenv(v, raising=False)
    assert standalone._detect_local_tpu_type() == "v4-8"


# =============================================================================
# Region-lock + output_path integration
# =============================================================================


def test_main_writes_output_path_in_pinned_region(tmp_path, monkeypatch):
    """End-to-end: parsed args → region detection → tracker claim → output_path."""
    p = _sample_plan()
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()

    monkeypatch.setenv("MARIN_REGION", "us-central1")
    monkeypatch.delenv("MARIN_PREFIX", raising=False)

    captured: dict = {}

    def fake_run(pod_config):
        captured["output_path"] = pod_config.output_path
        captured["resources"] = pod_config.resources
        captured["data_target_budget"] = pod_config.train_config.data.target_budget
        captured["data_experiment_budget"] = pod_config.train_config.data.experiment_budget
        captured["components"] = list(pod_config.train_config.data.components.keys())
        captured["weights"] = dict(pod_config.train_config.data.train_weights)

    # Patch the in-process levanter call (no nested iris submit anymore).
    fake_train_lm_module = MagicMock()
    fake_train_lm_module.main = MagicMock()

    def fake_prepare(pod_config):
        fake_run(pod_config)
        return pod_config, pod_config.train_config, {}, []

    real_import = importlib.import_module

    def fake_import(name, *a, **kw):
        return fake_train_lm_module if name == "levanter.main.train_lm" else real_import(name, *a, **kw)

    with (
        patch.object(standalone, "_prepare_training_run", fake_prepare),
        patch.object(standalone, "_decide_wandb_mode", return_value="online"),
        patch.object(standalone.importlib, "import_module", fake_import),
        patch.object(standalone, "_read_last_eval_metrics", return_value=None),
        patch.object(standalone, "_write_summary", return_value=None),
    ):
        standalone.main(p.to_cli_args() + ["--tracker-prefix", tracker_prefix, "--tpu-type", "v4-8"])

    # output_path landed in the pinned region's bucket
    assert captured["output_path"].startswith("gs://marin-us-central1/checkpoints/isoflop-curation/")
    assert p.run_name_core in captured["output_path"]

    # Experiment A: NO simulated-epoching budgets. Levanter must not slice
    # the training pool. (Bugfix: previously we set target_budget=T_exp*s,
    # experiment_budget=T_exp, which made Levanter slice DCLM to ~1M tokens
    # and the model over-epoched ~s times, memorizing the slice.)
    assert captured["data_target_budget"] is None, (
        f"ExpA must not set target_budget (got {captured['data_target_budget']}). "
        f"Levanter would slice D_obs by T_exp/T_target."
    )
    assert (
        captured["data_experiment_budget"] is None
    ), f"ExpA must not set experiment_budget (got {captured['data_experiment_budget']})."

    # mixture has training data + paloma + uncheatable_eval components
    assert p.method_name in captured["components"]
    paloma_components = [c for c in captured["components"] if c.startswith("paloma/")]
    uncheat_components = [c for c in captured["components"] if c.startswith("uncheatable_eval/")]
    assert len(paloma_components) == 16
    assert len(uncheat_components) == 7

    # ALL validation components have weight=0 (training-safety invariant)
    for k, v in captured["weights"].items():
        if k != p.method_name:
            assert v == 0.0, f"validation component {k} has nonzero training weight {v}"


def test_main_migrates_on_region_mismatch(tmp_path, monkeypatch):
    """Region mismatch triggers a migration to the local region by default.

    Previously this test expected `RegionMismatch` to be raised, but iris
    doesn't respect per-task region affinity on retry — runs that get preempted
    and re-scheduled in a different region would hit this and die terminally.
    The new default (`allow_region_migration=True`) lets the worker proceed in
    its own region, orphaning the old checkpoint and restarting at step 0.
    """
    p = _sample_plan()
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()

    # Pre-populate tracker as if a us-east5 worker had claimed this run.
    region_tracker.claim_or_read_region(
        p.run_key,
        "us-east5",
        tracker_prefix=tracker_prefix,
    )

    monkeypatch.setenv("MARIN_REGION", "us-central1")
    monkeypatch.delenv("MARIN_PREFIX", raising=False)

    captured: dict = {}

    def fake_prepare(pod_config):
        captured["output_path"] = pod_config.output_path
        return pod_config, pod_config.train_config, {}, []

    fake_train_lm_module = MagicMock()
    fake_train_lm_module.main = MagicMock()
    with (
        patch.object(standalone, "_prepare_training_run", fake_prepare),
        patch.object(standalone, "_decide_wandb_mode", return_value="online"),
        patch.object(standalone, "_read_last_eval_metrics", return_value=None),
        patch.object(standalone, "_write_summary", return_value=None),
        patch.object(
            standalone.importlib,
            "import_module",
            lambda name, *a, **kw: (
                fake_train_lm_module if name == "levanter.main.train_lm" else importlib.import_module(name, *a, **kw)
            ),
        ),
    ):
        # Must NOT raise — migration kicks in.
        standalone.main(p.to_cli_args() + ["--tracker-prefix", tracker_prefix, "--tpu-type", "v4-8"])

    # Output path uses the LOCAL region (us-central1), not the original claim (us-east5).
    assert captured["output_path"].startswith(
        "gs://marin-us-central1/"
    ), f"After migration output_path must live in the local region, got {captured['output_path']}"


# =============================================================================
# Simulated-epoching semantics (regression tests for the ExpA over-epoching bug)
# =============================================================================


def _run_main_capture_budgets(plan: PlannedRun, tmp_path, monkeypatch) -> dict:
    """Invoke standalone.main with a mocked Levanter and return the budgets/method."""
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir(exist_ok=True)
    monkeypatch.setenv("MARIN_REGION", "us-central1")
    monkeypatch.delenv("MARIN_PREFIX", raising=False)

    captured: dict = {}

    def fake_prepare(pod_config):
        data = pod_config.train_config.data
        captured["target_budget"] = data.target_budget
        captured["experiment_budget"] = data.experiment_budget
        captured["cache_dirs"] = {name: c.cache_dir for name, c in data.components.items()}
        return pod_config, pod_config.train_config, {}, []

    fake_train_lm_module = MagicMock()
    fake_train_lm_module.main = MagicMock()
    real_import = importlib.import_module

    def fake_import(name, *a, **kw):
        return fake_train_lm_module if name == "levanter.main.train_lm" else real_import(name, *a, **kw)

    with (
        patch.object(standalone, "_prepare_training_run", fake_prepare),
        patch.object(standalone, "_decide_wandb_mode", return_value="online"),
        patch.object(standalone.importlib, "import_module", fake_import),
        patch.object(standalone, "_read_last_eval_metrics", return_value=None),
        patch.object(standalone, "_write_summary", return_value=None),
    ):
        standalone.main(plan.to_cli_args() + ["--tracker-prefix", tracker_prefix, "--tpu-type", "v4-8"])

    return captured


def test_exp_a_leaves_budgets_unset(tmp_path, monkeypatch):
    """Regression: Experiment A must NOT set target_budget / experiment_budget.

    If set, Levanter slices the training pool to D_obs * (T_exp / T_target)
    = D_obs / s ≈ 1M tokens for DCLM, and the model memorizes the slice.
    This was the cause of the observed divergent train-vs-val loss in the
    first smoke run.
    """
    plan = _sample_plan()
    assert plan.experiment_tag.startswith("expA"), "fixture must be ExpA"
    captured = _run_main_capture_budgets(plan, tmp_path, monkeypatch)
    assert captured["target_budget"] is None
    assert captured["experiment_budget"] is None


def test_exp_b_scales_target_budget_by_s(tmp_path, monkeypatch):
    """Experiment B must pass target_budget = T_target / s so Levanter's
    D_obs-based slice formula yields the intended D_proj-based slice.

    Levanter: slice = D_obs * (experiment_budget / target_budget)
    Intended: slice = D_proj * T_exp / T_target = D_obs * s * T_exp / T_target
    Equating: target_budget = T_target / s, experiment_budget = T_exp.
    """
    plan = _sample_plan_exp_b(t_target=20e12)
    assert plan.experiment_tag.startswith("expB"), "fixture must be ExpB"
    method = METHODS[plan.method_name]
    captured = _run_main_capture_budgets(plan, tmp_path, monkeypatch)
    assert captured["target_budget"] == pytest.approx(int(plan.t_target / method.s), rel=1e-5)
    assert captured["experiment_budget"] == pytest.approx(int(plan.t_exp), rel=1e-5)


def test_exp_b_budget_inequality_satisfies_levanter_check():
    """Levanter rejects experiment_budget > target_budget.

    For ExpB our plan ceiling guarantees T_exp <= T_target / s (the
    t_exp_ceiling), so the scaled budgets always satisfy Levanter's
    invariant: experiment_budget (= T_exp) <= target_budget (= T_target / s).
    """
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [20e12])
    method = METHODS["dclm"]
    assert plans, "no ExpB plans enumerated"
    for plan in plans:
        target_budget = int(plan.t_target / method.s)
        experiment_budget = int(plan.t_exp)
        assert experiment_budget <= target_budget, (
            f"{plan.run_name_core}: T_exp={experiment_budget:.2e} > "
            f"target_budget={target_budget:.2e}; Levanter will reject."
        )


def test_levanter_slice_matches_intended_formula():
    """Simulate Levanter's slice computation for ExpB and verify it matches
    the plan doc's intended formula slice = T_exp * D_proj / T_target.
    """
    method = METHODS["dclm"]
    d_obs = method.d_obs_tokens
    d_proj = method.d_proj
    s = method.s

    plans = curation_plan.enumerate_plans([method], [20e12])
    assert plans, "no ExpB plans enumerated"

    for plan in plans[:5]:
        # Values the standalone passes to Levanter:
        target_budget_passed = int(plan.t_target / s)
        experiment_budget_passed = int(plan.t_exp)
        # Levanter's slice formula (in Levanter datasets.py:764-769):
        levanter_ratio = experiment_budget_passed / target_budget_passed
        levanter_slice = int(d_obs * levanter_ratio)
        # Intended slice per the experiment's math:
        intended_slice = plan.t_exp * d_proj / plan.t_target
        assert levanter_slice == pytest.approx(intended_slice, rel=1e-3), (
            f"{plan.run_name_core}: Levanter slice {levanter_slice:.3e} " f"≠ intended {intended_slice:.3e}"
        )


def test_exp_a_no_slice_means_full_data_pool():
    """For ExpA, with no budgets set, Levanter never slices the dataset
    (its `if experiment_budget is not None and target_budget is not None`
    branch is skipped), so the model trains on the full D_obs pool.
    """
    plan = _sample_plan()
    assert plan.experiment_tag.startswith("expA")
    # In ExpA we want the model to see D_obs; over the full training run
    # it loops naturally based on the trainer's step schedule.
    # The number of epochs = T_exp / D_obs.
    method = METHODS[plan.method_name]
    epochs_over_d_obs = plan.t_exp / method.d_obs_tokens
    # Sanity: ExpA at budget 3e18 should do ~1-2 epochs over DCLM (D_obs ≈ 2.66B).
    assert 0.1 <= epochs_over_d_obs <= 10, (
        f"ExpA epoch count {epochs_over_d_obs:.2f} outside plausible range — "
        f"if this fails, either the heuristic or the plan enumeration changed."
    )


# =============================================================================
# Concrete predictions: tokens trained, slice size, effective epochs
#
# These mirror what actually happens inside Levanter given the budgets we pass.
# If they stop matching our intended math, either the plan generation changed
# or Levanter's slicing semantics shifted — either way we want to know before
# launching 500+ training runs.
# =============================================================================


def _predicted_tokens_trained(plan: PlannedRun) -> int:
    """Total training tokens the model will process (batch × seq × steps)."""
    return plan.batch_size * plan.seq_len * plan.train_steps


def _predicted_levanter_slice(plan: PlannedRun, method) -> int:
    """Mimic Levanter's slice computation under the budgets we pass.

    ExpA: we pass None/None → Levanter skips slicing, slice = D_obs.
    ExpB: we pass target_budget=T_target/s, experiment_budget=T_exp
          → Levanter's `D_obs * (experiment_budget / target_budget)` formula
            reproduces the intended `D_proj * T_exp / T_target`.
    """
    if plan.experiment_tag.startswith("expA"):
        return method.d_obs_tokens
    target_budget_l = int(plan.t_target / method.s)
    experiment_budget_l = int(plan.t_exp)
    return int(method.d_obs_tokens * experiment_budget_l / target_budget_l)


def _predicted_epochs(plan: PlannedRun, method) -> float:
    """How many full passes of the (sliced) training pool the model makes."""
    return _predicted_tokens_trained(plan) / _predicted_levanter_slice(plan, method)


def test_dclm_expa_3e18_predicts_roughly_1_6_epochs_of_2_66b_tokens():
    """Golden: the exact smoke plan we ran should train on ~1.6 epochs of
    DCLM's 2.66 B tokens — not 4000 epochs of a 1M-token slice.

    This is the specific regression of the bug the user found: with the
    old code we sliced to D_obs/s ≈ 1M tokens and epoched ~4000 times.
    """
    plan = _sample_plan()
    method = METHODS[plan.method_name]

    tokens = _predicted_tokens_trained(plan)
    slice_size = _predicted_levanter_slice(plan, method)
    epochs = _predicted_epochs(plan, method)

    assert method.name == "dclm"
    assert method.d_obs_tokens == 2_663_454_015
    # Tokens trained is batch*seq*steps ≈ the plan's T_exp (≈4.3B for 3e18).
    assert tokens == pytest.approx(plan.t_exp, rel=1e-6)
    # Slice is the full pool (no Levanter slicing in ExpA).
    assert slice_size == 2_663_454_015
    # Effective epochs: 1.6, not 4000.
    assert 1.0 <= epochs <= 3.0, (
        f"DCLM ExpA 3e18 should train for ~1-2 epochs of D_obs, got {epochs:.2f}. "
        f"If this asserts > 100, we've regressed to the D_obs/s slicing bug."
    )
    # Tight bound for the specific smoke plan.
    assert epochs == pytest.approx(1.62, abs=0.05)


def test_all_exp_a_plans_are_never_extreme_epoched():
    """Across every ExpA plan, no plan should end up looping 100+ epochs."""
    for method in [METHODS["dclm"], METHODS["fineweb_edu"]]:
        plans = curation_plan.enumerate_plans([method], [None])
        assert plans, f"no ExpA plans for {method.name}"
        for plan in plans:
            epochs = _predicted_epochs(plan, method)
            assert epochs < 100, (
                f"{plan.run_name_core}: ExpA epoch count {epochs:.1f} > 100 — "
                f"likely regressed to the D_obs/s slicing bug."
            )
            # And epochs should be > 0 (we do train on SOMETHING).
            assert epochs > 0


def test_exp_b_effective_epochs_match_target_regime_epochs():
    """For every ExpB plan, the model's epochs over the slice should equal
    the target-regime epoch count T_target / D_proj — this is the whole
    point of simulated epoching (faithful epoch-count reproduction).

    effective_epochs_in_experiment = T_exp / slice
                                   = T_exp / (T_exp * D_proj / T_target)
                                   = T_target / D_proj
    """
    method = METHODS["dclm"]
    t_target = 20e12
    plans = curation_plan.enumerate_plans([method], [t_target])
    assert plans, "no ExpB plans for dclm"
    target_regime_epochs = t_target / method.d_proj
    for plan in plans:
        epochs = _predicted_epochs(plan, method)
        assert epochs == pytest.approx(target_regime_epochs, rel=1e-3), (
            f"{plan.run_name_core}: effective epochs {epochs:.4f} " f"≠ target-regime epochs {target_regime_epochs:.4f}"
        )


def test_exp_b_slice_never_exceeds_d_obs():
    """By the T_exp ceiling (T_exp ≤ T_target / s), the slice is always ≤ D_obs
    — i.e. we never need more observed tokens than we have. If this fails,
    the plan's ceiling enforcement regressed.
    """
    method = METHODS["dclm"]
    plans = curation_plan.enumerate_plans([method], [20e12])
    for plan in plans:
        slice_size = _predicted_levanter_slice(plan, method)
        assert (
            slice_size <= method.d_obs_tokens
        ), f"{plan.run_name_core}: slice {slice_size:,} > D_obs {method.d_obs_tokens:,}"


# =============================================================================
# Sweep-level guarantees: ExpA never triggers simulated epoching;
# ExpB has constant target-regime epoch count across candidates.
# =============================================================================

_ALL_REGISTERED_METHODS = ["dclm", "nemotron_org", "nemotron_full_bos_fixed", "fineweb_edu"]


def test_every_plan_experiment_tag_correctly_partitions_a_vs_b():
    """The lone dispatch in `run_curation_train_standalone.main()` is
    `plan.experiment_tag.startswith('expB')`. Every ExpA plan MUST have a tag
    that fails this check, and every ExpB plan MUST have a tag that passes.
    """
    for method_name in _ALL_REGISTERED_METHODS:
        method = METHODS[method_name]
        # ExpA: tags must never start with 'expB'.
        plans_a = curation_plan.enumerate_plans([method], [None])
        assert plans_a, f"no ExpA plans for {method.name}"
        for plan in plans_a:
            assert not plan.experiment_tag.startswith("expB"), (
                f"ExpA plan {plan.run_name_core}: tag={plan.experiment_tag!r} "
                "would incorrectly activate the ExpB simulated-epoching branch."
            )
            assert plan.experiment_tag == "expA_natural"
        # ExpB: tags must always start with 'expB'.
        plans_b = curation_plan.enumerate_plans([method], [20e12])
        assert plans_b, f"no ExpB plans for {method.name}"
        for plan in plans_b:
            assert plan.experiment_tag.startswith("expB"), (
                f"ExpB plan {plan.run_name_core}: tag={plan.experiment_tag!r} "
                "would incorrectly skip the ExpB simulated-epoching branch."
            )


def test_every_exp_a_plan_slice_equals_d_obs_across_all_methods():
    """Sweep-wide guarantee: for ANY ExpA plan across any method, the Levanter
    slice is exactly D_obs — Levanter never slices under ExpA. If this ever
    fails, we've regressed to the over-epoching bug.
    """
    for method_name in _ALL_REGISTERED_METHODS:
        method = METHODS[method_name]
        plans = curation_plan.enumerate_plans([method], [None])
        for plan in plans:
            slice_tokens = _predicted_levanter_slice(plan, method)
            assert slice_tokens == method.d_obs_tokens, (
                f"{method.name} {plan.run_name_core}: slice {slice_tokens} " f"!= D_obs {method.d_obs_tokens}"
            )


def test_every_exp_a_plan_epoch_count_is_reasonable_across_all_methods():
    """ExpA epoch count = T_exp / D_obs. Across the sweep, should be in a
    plausible range (0.1x to 100x). If this fails, either the heuristic is
    producing absurd step counts OR simulated epoching snuck back in.
    """
    for method_name in _ALL_REGISTERED_METHODS:
        method = METHODS[method_name]
        plans = curation_plan.enumerate_plans([method], [None])
        for plan in plans:
            epochs = _predicted_epochs(plan, method)
            assert 0.1 <= epochs <= 100, (
                f"{method.name} {plan.run_name_core}: ExpA epochs={epochs:.2f} " f"outside plausible range"
            )


def test_exp_a_epoch_counts_vary_across_compute_budgets():
    """Sanity: an ExpA isoFLOP sweep should train for *different* epoch counts
    at different FLOP budgets (higher budget → more T_exp → more epochs). If
    every candidate had the same epoch count, something nuked T_exp variety.
    """
    method = METHODS["dclm"]
    plans = curation_plan.enumerate_plans([method], [None])
    epochs_list = [_predicted_epochs(plan, method) for plan in plans]
    assert min(epochs_list) < max(epochs_list), (
        f"Expected a range of epoch counts across ExpA DCLM candidates; " f"got all the same: {epochs_list[:5]}..."
    )


def test_every_exp_b_plan_has_identical_target_regime_epoch_count():
    """ExpB sweep invariant: every candidate epochs the same number of times
    over its slice (the target-regime epoch count T_target / D_proj). Only the
    slice size varies between candidates within a method.
    """
    t_target = 20e12
    for method_name in _ALL_REGISTERED_METHODS:
        method = METHODS[method_name]
        plans = curation_plan.enumerate_plans([method], [t_target])
        target_epochs = t_target / method.d_proj
        for plan in plans:
            epochs = _predicted_epochs(plan, method)
            assert epochs == pytest.approx(target_epochs, rel=1e-3), (
                f"{method.name} {plan.run_name_core}: epochs={epochs:.4f} " f"!= target-regime {target_epochs:.4f}"
            )


def test_build_summary_contains_all_scaling_law_fields():
    """The per-run summary JSON must carry every field a scaling-law plot needs:
    plan hyperparams, method D_obs/s, model param count, tokens/slice/epochs,
    and the final eval block.
    """
    plan = _sample_plan()
    method = METHODS[plan.method_name]
    # Fake final-eval block mimicking Levanter's eval_metrics.jsonl last row.
    final_eval = {
        "step": plan.train_steps,
        "eval/paloma/macro_bpb": 3.49,
        "eval/uncheatable_eval/macro_bpb": 3.38,
        "eval/bpb": 3.45,
    }
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-east5",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=final_eval,
    )
    # plan block
    assert summary["plan"]["method_name"] == plan.method_name
    assert summary["plan"]["experiment_tag"] == plan.experiment_tag
    assert summary["plan"]["budget_flops"] == plan.budget
    assert summary["plan"]["hidden_dim"] == plan.hidden_dim
    assert summary["plan"]["train_steps"] == plan.train_steps
    assert summary["plan"]["t_exp"] == plan.t_exp
    assert summary["plan"]["t_target"] == plan.t_target
    # method block
    assert summary["method"]["name"] == method.name
    assert summary["method"]["d_obs_tokens"] == method.d_obs_tokens
    assert summary["method"]["d_proj_tokens"] == pytest.approx(method.d_proj)
    assert summary["method"]["s_scale_factor"] == pytest.approx(method.s)
    # model block
    assert summary["model"]["total_trainable_params"] > 0
    assert summary["model"]["vocab_size"] == 128256
    # tokens block
    assert summary["tokens"]["tokens_trained"] == plan.batch_size * plan.seq_len * plan.train_steps
    # ExpA: slice = D_obs, so epochs ~= tokens_trained / D_obs
    assert summary["tokens"]["slice_tokens"] == method.d_obs_tokens
    assert 1.0 <= summary["tokens"]["effective_epochs"] <= 3.0
    # run metadata
    assert summary["run"]["region"] == "us-east5"
    assert summary["run"]["output_path"] == "gs://bucket/out"
    # final eval pass-through
    assert summary["eval"]["eval/paloma/macro_bpb"] == 3.49


def test_build_summary_exp_b_slice_matches_intended_formula():
    """For ExpB plans, the summary's slice_tokens must reflect the fixed
    Levanter formula slice = D_obs × T_exp / (T_target / s) = D_proj × T_exp / T_target.
    """
    plan = _sample_plan_exp_b(t_target=20e12)
    method = METHODS[plan.method_name]
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-east5",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    expected_slice = int(method.d_proj * plan.t_exp / plan.t_target)
    assert summary["tokens"]["slice_tokens"] == pytest.approx(expected_slice, rel=1e-3)
    # Effective epochs in ExpB should equal target-regime epoch count.
    target_epochs = plan.t_target / method.d_proj
    assert summary["tokens"]["effective_epochs"] == pytest.approx(target_epochs, rel=1e-3)


# =============================================================================
# ExpC regime gate: data-rich (LC, Res) skip slicing; sliced (dclm_10k) apply
# =============================================================================


def test_build_summary_expc_data_rich_lc_does_not_slice():
    """For LC at T=33T (target_epochs ≈ 0.22 < 1), the runner trains naturally on
    D_obs. _build_summary must report slice_tokens = D_obs (NOT the slicing-formula
    output, which would be > D_obs and meaningless in this regime).
    """
    plans = curation_plan.enumerate_plans([METHODS["llm_curated_bos_fixed"]], [("C", 33e12)])
    assert plans, "expected ExpC LC plans"
    plan = plans[0]
    method = METHODS["llm_curated_bos_fixed"]
    # Sanity: LC at T=33T must be in the data-rich regime.
    assert plan.t_target < method.d_proj, "test premise broken: LC should be data-rich at T=33T"
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-central1",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    # No slicing — slice_tokens should equal D_obs (full cache):
    assert summary["tokens"]["slice_tokens"] == method.d_obs_tokens
    # Epoch count = T_exp / D_obs (NOT target-regime epochs):
    expected_epochs = (plan.batch_size * plan.seq_len * plan.train_steps) / method.d_obs_tokens
    assert summary["tokens"]["effective_epochs"] == pytest.approx(expected_epochs, rel=1e-6)


def test_build_summary_expc_data_rich_resiliparse_does_not_slice():
    """Same regime gate for resiliparse (target_epochs ≈ 0.09 at T=33T)."""
    plans = curation_plan.enumerate_plans([METHODS["resiliparse"]], [("C", 33e12)])
    assert plans, "expected ExpC resiliparse plans"
    plan = plans[0]
    method = METHODS["resiliparse"]
    assert plan.t_target < method.d_proj, "test premise broken"
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-central1",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    assert summary["tokens"]["slice_tokens"] == method.d_obs_tokens
    expected_epochs = (plan.batch_size * plan.seq_len * plan.train_steps) / method.d_obs_tokens
    assert summary["tokens"]["effective_epochs"] == pytest.approx(expected_epochs, rel=1e-6)


def test_build_summary_expc_sliced_method_applies_slicing():
    """Synthetic dclm_10k (D_proj ≈ 7T at sampled_warcs=10,364): target_epochs at
    T=33T is ≈ 4.7, so the slicing branch fires. The slice formula applies.
    """
    # Synthesize a dclm_10k plan via PlannedRun so we don't depend on the
    # placeholder-d_obs guard. Use realistic 10k D_obs (≈ DCLM_3k × 10364/3000).
    from experiments.scaling_law_sweeps.data_curation_math import CurationMethod

    method = CurationMethod(
        name="dclm_10k",
        tokenized_rel_path="tokenized/dclm_10k_test/",
        d_obs_tokens=int(2_663_454_015 * 10_364 / 3_000),  # ≈ 9.20B
        sampled_warcs=10_364,
    )
    # Sanity: target_epochs >= 1 at T=33T.
    assert 33e12 / method.d_proj >= 1.0, "test premise broken: dclm_10k should be sliced regime"

    # Build a plan via the LC ExpC enumeration (we just need a valid PlannedRun
    # frame; the method+t_target are what drive _build_summary's branching).
    template = curation_plan.enumerate_plans([METHODS["llm_curated_bos_fixed"]], [("C", 33e12)])[0]
    sliced_plan = curation_plan.PlannedRun(
        method_name="dclm_10k",
        experiment_tag=template.experiment_tag,  # "expC_T33T"
        budget=template.budget,
        hidden_dim=template.hidden_dim,
        num_layers=template.num_layers,
        num_heads=template.num_heads,
        intermediate_dim=template.intermediate_dim,
        batch_size=template.batch_size,
        train_steps=template.train_steps,
        learning_rate=template.learning_rate,
        adam_lr=template.adam_lr,
        epsilon=template.epsilon,
        beta1=template.beta1,
        beta2=template.beta2,
        t_exp=min(template.t_exp, method.d_obs_tokens * 0.8),  # safe slice
        t_target=template.t_target,
    )
    summary = standalone._build_summary(
        plan=sliced_plan,
        method=method,
        region="us-east5",
        run_name=sliced_plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    # Slicing applied: slice_tokens = D_proj × T_exp / T_target.
    expected_slice = int(method.d_proj * sliced_plan.t_exp / sliced_plan.t_target)
    assert summary["tokens"]["slice_tokens"] == pytest.approx(expected_slice, rel=1e-3)
    # Effective epochs = target_regime_epochs.
    target_epochs = sliced_plan.t_target / method.d_proj
    assert summary["tokens"]["effective_epochs"] == pytest.approx(target_epochs, rel=1e-3)


def test_build_summary_expa_unaffected_by_regime_gate():
    """ExpA plans (expA_natural tag) MUST NOT enter the slicing branch regardless
    of t_target / d_proj relationship — slice = D_obs always."""
    plan = _sample_plan()  # ExpA dclm
    method = METHODS["dclm"]
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-east5",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    assert plan.experiment_tag == "expA_natural"
    assert summary["tokens"]["slice_tokens"] == method.d_obs_tokens


def test_build_summary_expfm_natural_unaffected_by_regime_gate():
    """fixed_model uses tag 'expFM_natural' which doesn't start with expB/expC.
    Even on a data-rich method like LC, it must NOT slice — preserves the
    existing fixed_model contract."""
    plan = curation_plan.enumerate_plans([METHODS["llm_curated_bos_fixed"]], [None])[0]
    # Force-mock the tag to expFM_natural (simulating fixed_model output).
    fm_plan = curation_plan.PlannedRun(
        method_name=plan.method_name,
        experiment_tag="expFM_natural",
        budget=plan.budget,
        hidden_dim=plan.hidden_dim,
        num_layers=plan.num_layers,
        num_heads=plan.num_heads,
        intermediate_dim=plan.intermediate_dim,
        batch_size=plan.batch_size,
        train_steps=plan.train_steps,
        learning_rate=plan.learning_rate,
        adam_lr=plan.adam_lr,
        epsilon=plan.epsilon,
        beta1=plan.beta1,
        beta2=plan.beta2,
        t_exp=plan.t_exp,
        t_target=plan.t_target,
    )
    method = METHODS[fm_plan.method_name]
    summary = standalone._build_summary(
        plan=fm_plan,
        method=method,
        region="us-central1",
        run_name=fm_plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval=None,
    )
    # No slicing — slice_tokens = D_obs.
    assert summary["tokens"]["slice_tokens"] == method.d_obs_tokens


# =============================================================================
# WandB offline-mode cache: upload / download / sync-pending
# =============================================================================


# -- WandB probe: realistic scenarios ----------------------------------------
#
# The probe POSTs `{ __typename }` to https://api.wandb.ai/graphql. We test
# the EXACT scenarios the probe will see in production:
#
#   Scenario                         Expected probe verdict
#   -------------------------------  ----------------------
#   WandB healthy, unauthed          healthy (401 is a valid alive-signal)
#   WandB healthy, authed            healthy (200 + JSON response)
#   WandB API server down            unhealthy (5xx)
#   WandB unreachable                unhealthy (DNS/timeout/refused)


def _mock_urlopen_raising(exc):
    def raiser(*args, **kwargs):
        raise exc

    return raiser


def test_probe_alive_when_server_returns_401_to_unauthed_graphql(monkeypatch):
    """Realistic: we POST GraphQL without an API key. WandB returns 401 — this
    is the ACTUAL response in production, and it means WandB is up and
    responding. Must count as healthy.
    """
    import urllib.error

    err_401 = urllib.error.HTTPError("https://api.wandb.ai/graphql", 401, "Unauthorized", hdrs=None, fp=None)
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen_raising(err_401))
    assert standalone._probe_wandb_healthy() is True


def test_probe_alive_when_server_returns_200(monkeypatch):
    """If we happened to be authed (e.g., a future version that sends the API
    key), we'd get 200 with a real GraphQL response. Alive.
    """

    class FakeOK:
        pass

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeOK())
    assert standalone._probe_wandb_healthy() is True


def test_probe_unhealthy_when_server_returns_5xx(monkeypatch):
    """WandB's API server is responding with 5xx → API itself is unhealthy.
    The init call is unlikely to succeed; fall back to offline.
    """
    import urllib.error

    err_503 = urllib.error.HTTPError("https://api.wandb.ai/graphql", 503, "Service Unavailable", hdrs=None, fp=None)
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen_raising(err_503))
    assert standalone._probe_wandb_healthy() is False


def test_probe_unhealthy_when_connection_refused(monkeypatch):
    """Can't reach WandB at all — network dead, DNS broken, or they're offline.
    Fall back to offline mode.
    """
    import urllib.error

    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen_raising(urllib.error.URLError("connection refused")))
    assert standalone._probe_wandb_healthy() is False


def test_probe_unhealthy_when_request_times_out(monkeypatch):
    """Request hangs past `timeout_seconds` — WandB is hanging or firewalled.
    Fall back to offline.
    """
    monkeypatch.setattr("urllib.request.urlopen", _mock_urlopen_raising(TimeoutError("timed out")))
    assert standalone._probe_wandb_healthy() is False


def test_decide_wandb_mode_respects_explicit_pref():
    """--wandb-mode online/offline MUST skip the probe and return as-is."""
    assert standalone._decide_wandb_mode("online") == "online"
    assert standalone._decide_wandb_mode("offline") == "offline"


def test_decide_wandb_mode_auto_uses_probe(monkeypatch):
    """'auto' runs the probe and maps success→online, failure→offline."""
    monkeypatch.setattr(standalone, "_probe_wandb_healthy", lambda timeout_seconds=10.0: True)
    assert standalone._decide_wandb_mode("auto") == "online"
    monkeypatch.setattr(standalone, "_probe_wandb_healthy", lambda timeout_seconds=10.0: False)
    assert standalone._decide_wandb_mode("auto") == "offline"


def test_upload_wandb_cache_empty_local_is_noop(tmp_path):
    """If the local wandb dir is missing or empty, upload is a no-op (returns False)."""
    dest = f"file://{tmp_path}/gcs_wandb"
    assert standalone._upload_wandb_cache(str(tmp_path / "does_not_exist"), dest) is False
    empty_dir = tmp_path / "empty_wandb"
    empty_dir.mkdir()
    assert standalone._upload_wandb_cache(str(empty_dir), dest) is False


def test_upload_and_download_wandb_cache_roundtrips(tmp_path):
    """Round-trip: local wandb dir → GCS-like path → local restore."""
    # Stage a fake wandb cache.
    src = tmp_path / "wandb"
    run_dir = src / "run-abc"
    run_dir.mkdir(parents=True)
    (run_dir / "run-abc.wandb").write_text("offline log payload")
    (run_dir / "files").mkdir()
    (run_dir / "files" / "config.yaml").write_text("key: val")

    dest = f"file://{tmp_path}/gcs_wandb/"
    uploaded = standalone._upload_wandb_cache(str(src), dest)
    assert uploaded is True

    # Now download into a fresh local dir.
    restore = tmp_path / "wandb_restored"
    found = standalone._download_wandb_cache_if_exists(dest, str(restore))
    assert found is True
    assert (restore / "run-abc" / "run-abc.wandb").read_text() == "offline log payload"
    assert (restore / "run-abc" / "files" / "config.yaml").read_text() == "key: val"


def test_download_wandb_cache_missing_is_noop(tmp_path):
    """If the GCS cache path doesn't exist, download returns False and creates nothing."""
    found = standalone._download_wandb_cache_if_exists(
        f"file://{tmp_path}/nothing_here/",
        str(tmp_path / "restored"),
    )
    assert found is False


def test_append_sync_pending_row_writes_jsonl(tmp_path):
    """Rows append correctly, multiple calls produce multiple lines, JSON parses."""
    jsonl = f"file://{tmp_path}/sync_pending.jsonl"
    standalone._append_sync_pending_row({"run_name": "run1", "region": "us-east5"}, jsonl)
    standalone._append_sync_pending_row({"run_name": "run2", "region": "us-central1"}, jsonl)
    lines = (tmp_path / "sync_pending.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2
    import json as _json

    assert _json.loads(lines[0])["run_name"] == "run1"
    assert _json.loads(lines[1])["run_name"] == "run2"
    assert _json.loads(lines[1])["region"] == "us-central1"


def test_write_summary_roundtrips_through_fsspec(tmp_path):
    """End-to-end: _write_summary writes a readable JSON that round-trips."""
    plan = _sample_plan()
    method = METHODS[plan.method_name]
    summary = standalone._build_summary(
        plan=plan,
        method=method,
        region="us-east5",
        run_name=plan.run_name_core,
        output_path="gs://bucket/out",
        final_eval={"step": 42, "eval/bpb": 1.23},
    )
    results_prefix = f"file://{tmp_path}/results"
    (tmp_path / "results").mkdir()
    standalone._write_summary(summary, results_prefix, plan.run_name_core)
    # Read back.
    written_path = tmp_path / "results" / f"{plan.run_name_core}.json"
    assert written_path.exists()
    import json

    with open(written_path) as f:
        roundtripped = json.load(f)
    assert roundtripped["plan"]["method_name"] == plan.method_name
    assert roundtripped["eval"]["step"] == 42


def test_prediction_helpers_reproduce_documented_smoke_values():
    """Pin the documented smoke-run numbers so any future heuristic change
    is caught by a concrete assert rather than silent drift.
    """
    plan = _sample_plan()
    method = METHODS["dclm"]
    # Documented in scratch/dclm_smoke_babysit.md:
    #   tokens trained ≈ 4.30 B, slice = 2.66 B (D_obs), epochs ≈ 1.62
    assert _predicted_tokens_trained(plan) == pytest.approx(4.30e9, rel=1e-2)
    assert _predicted_levanter_slice(plan, method) == 2_663_454_015
    assert _predicted_epochs(plan, method) == pytest.approx(1.62, abs=0.05)


def test_early_is_process_0_unset_returns_true(monkeypatch):
    """Single-host TPU: TPU_WORKER_ID unset -> rank 0."""
    monkeypatch.delenv("TPU_WORKER_ID", raising=False)
    assert standalone._early_is_process_0() is True


def test_early_is_process_0_zero_returns_true(monkeypatch):
    """Multi-host TPU: TPU_WORKER_ID=0 -> rank 0 (writer)."""
    monkeypatch.setenv("TPU_WORKER_ID", "0")
    assert standalone._early_is_process_0() is True


def test_early_is_process_0_nonzero_returns_false(monkeypatch):
    """Multi-host TPU: TPU_WORKER_ID>0 -> not rank 0 (skip side effects)."""
    monkeypatch.setenv("TPU_WORKER_ID", "1")
    assert standalone._early_is_process_0() is False
    monkeypatch.setenv("TPU_WORKER_ID", "7")
    assert standalone._early_is_process_0() is False


def test_main_non_rank_zero_skips_summary_and_done_marker(tmp_path, monkeypatch):
    """A non-rank-0 VM (TPU_WORKER_ID=1) must NOT write summary/DONE — those
    side effects are gated to rank 0 to avoid N replicas racing on the same
    GCS object. Training itself still runs (Levanter handles per-process
    gating internally).
    """
    p = _sample_plan()
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()
    results_prefix = f"file://{tmp_path}/results/"
    (tmp_path / "results").mkdir()

    monkeypatch.setenv("MARIN_REGION", "us-central1")
    monkeypatch.delenv("MARIN_PREFIX", raising=False)
    monkeypatch.setenv("TPU_WORKER_ID", "1")  # non-rank-0 VM

    called = []
    fake_train_lm_module = MagicMock()
    fake_train_lm_module.main = lambda cfg: called.append(cfg)
    write_summary_calls = []
    with (
        patch.object(standalone, "_prepare_training_run", lambda c: (c, c.train_config, {}, [])),
        patch.object(standalone, "_decide_wandb_mode", return_value="online"),
        patch.object(
            standalone.importlib,
            "import_module",
            lambda name, *a, **kw: (
                fake_train_lm_module if name == "levanter.main.train_lm" else importlib.import_module(name, *a, **kw)
            ),
        ),
        patch.object(standalone, "_read_last_eval_metrics", return_value=None),
        patch.object(standalone, "_write_summary", side_effect=lambda *a, **kw: write_summary_calls.append((a, kw))),
    ):
        standalone.main(
            p.to_cli_args()
            + [
                "--tracker-prefix",
                tracker_prefix,
                "--tpu-type",
                "v4-16",
                "--results-prefix",
                results_prefix,
            ]
        )

    assert len(called) == 1, "training must still run on non-rank-0 VMs"
    assert len(write_summary_calls) == 0, "non-rank-0 VM must NOT call _write_summary — racing on same GCS path"


def test_main_subsequent_run_in_same_region_succeeds(tmp_path, monkeypatch):
    """Second run with same region should be a no-op claim and proceed normally."""
    p = _sample_plan()
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()

    region_tracker.claim_or_read_region(
        p.run_key,
        "us-central1",
        tracker_prefix=tracker_prefix,
    )

    monkeypatch.setenv("MARIN_REGION", "us-central1")
    monkeypatch.delenv("MARIN_PREFIX", raising=False)

    called = []
    fake_train_lm_module = MagicMock()
    fake_train_lm_module.main = lambda cfg: called.append(cfg)
    with (
        patch.object(standalone, "_prepare_training_run", lambda c: (c, c.train_config, {}, [])),
        patch.object(standalone, "_decide_wandb_mode", return_value="online"),
        patch.object(
            standalone.importlib,
            "import_module",
            lambda name, *a, **kw: (
                fake_train_lm_module if name == "levanter.main.train_lm" else importlib.import_module(name, *a, **kw)
            ),
        ),
        patch.object(standalone, "_read_last_eval_metrics", return_value=None),
        patch.object(standalone, "_write_summary", return_value=None),
    ):
        standalone.main(p.to_cli_args() + ["--tracker-prefix", tracker_prefix, "--tpu-type", "v4-8"])

    assert len(called) == 1  # training was invoked successfully
