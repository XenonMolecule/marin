# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Safety tests: mirror:// usage and region-lock preflight.

These tests pin down the cross-region safety invariants the user explicitly
asked us to enforce:

1. Every training component path uses `mirror://` via `InputName.hardcoded(...)`.
2. Raw `gs://` paths are rejected at `CurationMethod` construction.
3. Before any sweep submission, the preflight region-lock fires for every run
   and raises on mismatch.
4. Checkpoints written in region A cannot be resumed in region B (because the
   preflight catches it).
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps import data_curation_isoflop as dc
from experiments.scaling_law_sweeps import region_tracker as rt
from experiments.scaling_law_sweeps.data_curation_math import CurationMethod
from marin.execution.executor import InputName

MIRROR_PATH = "mirror://tokenized/baseline_dclm-23e9be/"


# =============================================================================
# Mirror path enforcement at CurationMethod construction
# =============================================================================


def test_mirror_path_accepted():
    m = CurationMethod("ok", MIRROR_PATH, d_obs_tokens=1_000_000_000)
    input_name = m.tokenized_input()
    assert isinstance(input_name, InputName)
    assert input_name.name == MIRROR_PATH


def test_raw_gs_path_rejected_in_tokenized_input():
    m = CurationMethod("bad", "gs://marin-us-central2/tokenized/baseline_dclm-23e9be/", d_obs_tokens=1_000_000_000)
    with pytest.raises(ValueError, match="mirror://"):
        m.tokenized_input()


def test_raw_gs_path_rejected_in_as_lm_mixture_config():
    m = CurationMethod("bad", "gs://marin-us-central2/tokenized/baseline_dclm-23e9be/", d_obs_tokens=1_000_000_000)
    with pytest.raises(ValueError, match="mirror://"):
        m.as_lm_mixture_config()


def test_local_filesystem_path_rejected():
    """A local `/tmp/...` path lacks the cross-region safety semantics."""
    m = CurationMethod("local", "/tmp/some/cache/", d_obs_tokens=1_000_000_000)
    with pytest.raises(ValueError, match="mirror://"):
        m.tokenized_input()


def test_http_path_rejected():
    m = CurationMethod("http", "http://example.com/cache/", d_obs_tokens=1_000_000_000)
    with pytest.raises(ValueError, match="mirror://"):
        m.tokenized_input()


# =============================================================================
# Every emitted sweep step's cache_dir is InputName(mirror://)
# =============================================================================


@pytest.fixture
def dclm() -> CurationMethod:
    return CurationMethod("dclm", MIRROR_PATH, d_obs_tokens=2_663_454_015)


def test_every_step_uses_mirror_for_cache_dir(dclm):
    """Across a full Experiment B sweep, every step's cache_dir must be a mirror:// string."""
    steps = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18, 9e18, 3e19))
    assert len(steps) > 0
    for step in steps:
        data = step.config.train_config.data
        component = data.components[dclm.name]
        assert isinstance(
            component.cache_dir, str
        ), f"component.cache_dir must be str for Levanter, got {type(component.cache_dir)}"
        assert component.cache_dir.startswith(
            "mirror://"
        ), f"component.cache_dir must be mirror://, got {component.cache_dir}"
        assert isinstance(component.source.cache_dir, str)
        assert component.source.cache_dir.startswith("mirror://")


def test_no_raw_gs_paths_anywhere_in_training_component(dclm):
    """Training component must have NO raw gs:// strings (cross-region risk)."""
    steps = dc.build_curation_sweep(dclm, t_target=20e12, budgets=(3e18,))
    for step in steps:
        component = step.config.train_config.data.components[dclm.name]
        # cache_dir should be InputName, not a raw string
        assert not isinstance(component.cache_dir, str) or not component.cache_dir.startswith("gs://")
        # source.cache_dir same
        if component.source is not None:
            src_cd = component.source.cache_dir
            assert not isinstance(src_cd, str) or not src_cd.startswith("gs://")


def test_experiment_a_and_b_both_use_mirror(dclm):
    """Both experiments must produce mirror:// configs — not just B."""
    for t_target in [None, 20e12]:
        steps = dc.build_curation_sweep(dclm, t_target=t_target, budgets=(3e18,))
        for step in steps:
            component = step.config.train_config.data.components[dclm.name]
            assert isinstance(component.cache_dir, str)
            assert component.cache_dir.startswith("mirror://")


# =============================================================================
# Preflight region-lock
# =============================================================================


@pytest.fixture
def tracker_dir(tmp_path) -> str:
    d = tmp_path / "locks"
    d.mkdir()
    return f"file://{d}"


def _small_plan(method: CurationMethod) -> list[tuple[CurationMethod, str, str]]:
    """Build a toy plan list: one entry per candidate at budget 3e18."""
    plans = []
    tag = "expB_T20T"
    steps = dc.build_curation_sweep(method, t_target=20e12, budgets=(3e18,))
    for step in steps:
        run_name_core = step.override_output_path.split("/", maxsplit=2)[-1]
        plans.append((method, tag, run_name_core))
    return plans


def test_preflight_first_launch_claims_all(dclm, tracker_dir):
    """First launch in a region should claim every run successfully."""
    plans = _small_plan(dclm)
    pinned = dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)
    assert len(pinned) == len(plans)
    for region in pinned.values():
        assert region == "us-central1"


def test_preflight_second_launch_same_region_succeeds(dclm, tracker_dir):
    """Re-launching in the same region should be idempotent (no-op)."""
    plans = _small_plan(dclm)
    dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)
    # Second launch, same region — should pass silently.
    pinned = dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)
    assert len(pinned) == len(plans)


def test_preflight_different_region_raises(dclm, tracker_dir):
    """Re-launching in a different region must raise on the first mismatch."""
    plans = _small_plan(dclm)
    dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)
    with pytest.raises(rt.RegionMismatch, match="us-central1"):
        dc.preflight_region_lock(plans, local_region="us-east5", tracker_prefix=tracker_dir)


def test_preflight_dry_run_skips_tracker(dclm, tracker_dir):
    """dry_run=True should bypass the tracker entirely."""
    plans = _small_plan(dclm)
    pinned = dc.preflight_region_lock(plans, local_region="us-east5", tracker_prefix=tracker_dir, dry_run=True)
    assert pinned == {}
    # And the tracker files should NOT have been created.
    import os

    local_dir = tracker_dir.removeprefix("file://")
    # Tracker writes files, so an empty dir confirms no writes.
    assert not any(os.listdir(local_dir)), "dry_run should not write tracker files"


def test_preflight_partial_launch_leaves_others_free(dclm, tracker_dir):
    """Locking only some runs shouldn't affect others; subsequent launches can claim the rest."""
    plans = _small_plan(dclm)
    half = len(plans) // 2
    dc.preflight_region_lock(plans[:half], local_region="us-central1", tracker_prefix=tracker_dir)
    # The remaining plans are free to claim in a different region.
    pinned_second = dc.preflight_region_lock(
        plans[half:],
        local_region="us-east5",
        tracker_prefix=tracker_dir,
    )
    for region in pinned_second.values():
        assert region == "us-east5"
    # And re-launching the first half in the ORIGINAL region still succeeds.
    dc.preflight_region_lock(plans[:half], local_region="us-central1", tracker_prefix=tracker_dir)
    # But moving them to a different region must fail.
    with pytest.raises(rt.RegionMismatch):
        dc.preflight_region_lock(plans[:half], local_region="us-east5", tracker_prefix=tracker_dir)


def test_preflight_uses_detect_region_when_none(dclm, tracker_dir, monkeypatch):
    """If local_region isn't passed, preflight pulls from MARIN_PREFIX."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    monkeypatch.delenv("MARIN_REGION", raising=False)
    plans = _small_plan(dclm)[:3]
    pinned = dc.preflight_region_lock(plans, tracker_prefix=tracker_dir)
    for region in pinned.values():
        assert region == "us-central1"


# =============================================================================
# End-to-end: the scenario we most need to prevent
# =============================================================================


def test_scenario_checkpoint_resume_from_wrong_region_blocked(dclm, tracker_dir):
    """Simulate: run claims us-central1, preempted, resumed on us-east5 → must abort.

    This is the exact scenario the user asked to prevent.
    """
    plans = _small_plan(dclm)

    # Step 1: First launch in us-central1 claims all runs.
    dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)

    # Step 2: Worker gets preempted. Iris re-schedules in us-east5.
    # The re-launch's preflight MUST abort before any training starts.
    with pytest.raises(rt.RegionMismatch):
        dc.preflight_region_lock(plans, local_region="us-east5", tracker_prefix=tracker_dir)

    # Step 3: User moves back to us-central1 — should succeed.
    dc.preflight_region_lock(plans, local_region="us-central1", tracker_prefix=tracker_dir)
