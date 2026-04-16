# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the iris coordinator script.

Mocks `IrisClient.submit` to verify:
- Each child gets the comma-separated v4,v5p TPU spec
- Each child has the SOFT REGION-IN-ALL_REGIONS constraint (mode=1) — the
  critical "let children float" lever from launch_adaptive.py
- CLI args carry the full PlannedRun spec
- --dry-run prints the plan and does not submit
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from experiments.scaling_law_sweeps import (
    curation_plan,
    launch_curation_sweep as launcher,
)
from experiments.scaling_law_sweeps.curation_plan import METHODS

# =============================================================================
# submit_one: shape of the iris job submission
# =============================================================================


@pytest.fixture
def sample_plan():
    return curation_plan.enumerate_plans([METHODS["dclm"]], [None])[0]


@pytest.fixture
def mock_client():
    client = MagicMock()
    fake_job = MagicMock()
    fake_job.job_id = "/test_user/curation-test-job"
    client.submit.return_value = fake_job
    return client


def test_submit_one_calls_client_submit_once(mock_client, sample_plan):
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="fake-key",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="data-curation-isoflop",
        tracker_prefix="gs://test/locks/",
    )
    assert mock_client.submit.call_count == 1


def test_submit_one_uses_device_variant_constraint_for_tpu_alternatives(mock_client, sample_plan):
    """Critical: iris scheduler must be able to pick EITHER v4 or v5p.

    The canonical pattern (from iris CLI `--tpu v4-X,v5p-Y`): set a primary TPU
    via `tpu_device()` and add `device_variant_constraint([v4, v5p])` as a
    constraint to allow alternative variants.
    """
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
        tracker_prefix="gs://t/",
    )
    kwargs = mock_client.submit.call_args.kwargs
    # Primary TPU is set on the device.
    assert kwargs["resources"].device is not None
    # Constraint list includes a DEVICE_VARIANT constraint allowing both.
    constraints = kwargs["constraints"]
    variant_constraints = [c for c in constraints if "DEVICE_VARIANT" in repr(c.key)]
    assert len(variant_constraints) == 1
    values_str = repr(variant_constraints[0].values)
    assert sample_plan.v4_tpu in values_str
    assert sample_plan.v5p_tpu in values_str


def test_submit_one_passes_cli_args_through_entrypoint(mock_client, sample_plan):
    """Entrypoint cmd must include the PlannedRun's --method, --budget, etc."""
    with patch.object(launcher, "Entrypoint") as mock_entry:
        mock_entry.from_command.return_value = MagicMock()
        launcher.submit_one(
            mock_client,
            sample_plan,
            child_priority_band=0,
            wandb_api_key="k",
            hf_token=None,
            wandb_project="marin",
            wandb_entity="marin-community",
            wandb_group="g",
            tracker_prefix="gs://t/",
        )
        mock_entry.from_command.assert_called_once()
        cmd_args = mock_entry.from_command.call_args[0]
        cmd_str = " ".join(str(a) for a in cmd_args)
        assert "--method" in cmd_str and sample_plan.method_name in cmd_str
        assert "--budget" in cmd_str
        assert launcher.SCRIPT in cmd_str
        assert "--tracker-prefix" in cmd_str


def test_submit_one_uses_soft_all_regions_constraint(mock_client, sample_plan):
    """CRITICAL safety invariant: SOFT preference for ALL_REGIONS prevents region inheritance.

    Hard-required region pinning would force every child to the parent's region,
    blocking the multi-region floating that's the whole point of this architecture.
    """
    from iris.cluster.constraints import ConstraintOp, WellKnownAttribute

    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
        tracker_prefix="gs://t/",
    )
    constraints = mock_client.submit.call_args.kwargs["constraints"]
    region_constraints = [c for c in constraints if c.key == WellKnownAttribute.REGION]
    assert len(region_constraints) == 1
    rc = region_constraints[0]
    assert rc.op == ConstraintOp.IN
    assert rc.mode == 1, "REGION constraint MUST be soft (mode=1) to allow children to float"
    assert set(rc.values) == set(launcher.ALL_REGIONS)


def test_submit_one_includes_preemptible_constraint(mock_client, sample_plan):
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
        tracker_prefix="gs://t/",
    )
    constraints = mock_client.submit.call_args.kwargs["constraints"]
    # Should have PREEMPTIBLE among the constraints.
    assert any(
        getattr(c, "key", None) is not None and "PREEMPTIBLE" in repr(c.key) for c in constraints
    ), f"Expected PREEMPTIBLE constraint, got: {constraints}"


def test_submit_one_passes_wandb_env(mock_client, sample_plan):
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="my-api-key",
        hf_token="my-hf-token",
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="data-curation-isoflop",
        tracker_prefix="gs://t/",
    )
    env = mock_client.submit.call_args.kwargs["environment"]
    assert env.env_vars["WANDB_API_KEY"] == "my-api-key"
    assert env.env_vars["HF_TOKEN"] == "my-hf-token"


def test_submit_one_no_hf_token_when_none(mock_client, sample_plan):
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
        tracker_prefix="gs://t/",
    )
    env = mock_client.submit.call_args.kwargs["environment"]
    assert "HF_TOKEN" not in env.env_vars


def test_submit_one_retries_set(mock_client, sample_plan):
    launcher.submit_one(
        mock_client,
        sample_plan,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
        tracker_prefix="gs://t/",
    )
    kwargs = mock_client.submit.call_args.kwargs
    assert kwargs["max_retries_preemption"] == 100
    # 10 (not 3): transient TPU VMs with stale libtpu state can burn 3 retries
    # without iris rescheduling to a clean worker. See submit_one comment.
    assert kwargs["max_retries_failure"] == 10


def test_submit_one_drops_mismatched_vm_count_variants(mock_client):
    """Multi-host plans (primary vm_count > 1) MUST not include device-variant
    alternatives whose vm_count differs from the primary. iris's adjust_tpu_replicas
    auto-scales replicas to the PRIMARY's vm_count; a mismatched variant means
    iris asks for N coscheduled tasks from a pool that only has M (M != N), and
    the job pends forever.

    Concrete case: v4-32 (vm_count=4) primary + v5p-16 (vm_count=2) alternative
    -> filter must drop v5p-16; constraint either disappears (only primary left)
    or contains only matching-vm_count variants.
    """
    # Construct a synthetic plan with mismatched vm_count (v4-32 vm=4 + v5p-16 vm=2)
    # to verify submit_one's filter. The real enumerator now aligns vm_counts, but
    # submit_one must still handle mismatches defensively.
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])
    base = [p for p in plans if p.v4_tpu == "v4-32"][0]
    import dataclasses
    plan = dataclasses.replace(base, v5p_tpu="v5p-16")  # force mismatch

    launcher.submit_one(
        mock_client, plan,
        child_priority_band=0, wandb_api_key="k", hf_token=None,
        wandb_project="marin", wandb_entity="marin-community",
        wandb_group="g", tracker_prefix="gs://t/",
    )
    constraints = mock_client.submit.call_args.kwargs["constraints"]
    variant_constraints = [c for c in constraints if "DEVICE_VARIANT" in repr(c.key)]
    if variant_constraints:
        # If a variant constraint was added, every value must have the same
        # vm_count as the primary.
        primary_vm_count = launcher._vm_count(plan.v4_tpu)
        for v in variant_constraints[0].values:
            assert launcher._vm_count(v) == primary_vm_count, (
                f"variant {v} has vm_count={launcher._vm_count(v)} but primary "
                f"{plan.v4_tpu} has vm_count={primary_vm_count} — iris coscheduling will fail"
            )


def test_submit_one_single_host_omits_replicas_and_coscheduling(mock_client, sample_plan):
    """Single-host plans (vm_count=1) must NOT pass replicas/coscheduling.
    Observed: adding coscheduling=tpu-name with replicas=1 on v5p-8 coincided
    with repeated libtpu /dev/vfio busy failures. Yesterday's bare submissions
    (no replicas, no coscheduling) worked cleanly on the same pool.

    sample_plan is the smallest DCLM plan, which has v4-8 primary (vm_count=1).
    """
    # Sanity check: sample_plan IS single-host.
    assert launcher._vm_count(sample_plan.v4_tpu) == 1

    launcher.submit_one(
        mock_client, sample_plan,
        child_priority_band=0, wandb_api_key="k", hf_token=None,
        wandb_project="marin", wandb_entity="marin-community",
        wandb_group="g", tracker_prefix="gs://t/",
    )
    kwargs = mock_client.submit.call_args.kwargs
    assert "replicas" not in kwargs, (
        "single-host should NOT set replicas; iris defaults to 1 anyway"
    )
    assert "coscheduling" not in kwargs, (
        "single-host should NOT set coscheduling; observed to bias iris's scheduler"
    )


def test_submit_one_multi_host_sets_replicas_and_tpu_coscheduling(mock_client):
    """Multi-host plans (vm_count>1) MUST pass replicas=1 + coscheduling so
    iris gang-schedules all VMs onto the same TPU slice.
    """
    from iris.cluster.types import CoschedulingConfig

    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])
    multi_host = [p for p in plans if launcher._vm_count(p.v4_tpu) > 1]
    assert multi_host, "expected at least one multi-host plan"
    plan = multi_host[0]

    launcher.submit_one(
        mock_client, plan,
        child_priority_band=0, wandb_api_key="k", hf_token=None,
        wandb_project="marin", wandb_entity="marin-community",
        wandb_group="g", tracker_prefix="gs://t/",
    )
    kwargs = mock_client.submit.call_args.kwargs
    assert kwargs["replicas"] == 1
    cosched = kwargs["coscheduling"]
    assert isinstance(cosched, CoschedulingConfig)
    assert cosched.group_by == "tpu-name"


# =============================================================================
# submit_all
# =============================================================================


def test_submit_all_submits_for_each_plan(mock_client):
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])[:5]
    # skip_if_done=False avoids a real GCS lookup in the test; behavior of the
    # skip-if-done path is covered separately (see test_is_run_already_complete_*).
    submitted, skipped = launcher.submit_all(
        mock_client,
        plans,
        tracker_prefix="gs://t/",
        skip_if_done=False,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
    )
    assert mock_client.submit.call_count == 5
    assert len(submitted) == 5
    assert skipped == []


def test_submit_all_continues_on_individual_failure(mock_client):
    """One failed submission doesn't kill the loop — log and continue."""
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])[:3]
    mock_client.submit.side_effect = [
        MagicMock(job_id="ok-1"),
        Exception("transient iris failure"),
        MagicMock(job_id="ok-3"),
    ]
    submitted, skipped = launcher.submit_all(
        mock_client,
        plans,
        tracker_prefix="gs://t/",
        skip_if_done=False,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
    )
    assert len(submitted) == 2
    assert skipped == []


# =============================================================================
# Skip-if-done
# =============================================================================


def test_is_run_already_complete_returns_false_when_no_tracker(tmp_path):
    """If no region tracker exists, the run never started → not done."""
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])[:1]
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()
    assert launcher.is_run_already_complete(plans[0], tracker_prefix) is False


def test_is_run_already_complete_returns_false_without_marker(tmp_path):
    """Tracker exists but no DONE marker → not done (run still in progress or failed)."""
    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])[:1]
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()

    # Claim region (simulates a child that started but didn't finish)
    import json

    (tmp_path / "locks" / plans[0].run_key).write_text(
        json.dumps({"region": "us-central1", "run_key": plans[0].run_key})
    )
    # No DONE marker anywhere → should NOT be considered done.
    assert launcher.is_run_already_complete(plans[0], tracker_prefix) is False


def test_submit_all_skips_done_runs(mock_client, tmp_path, monkeypatch):
    """If a plan's DONE marker exists, submit_all skips it and doesn't call iris."""
    import json

    plans = curation_plan.enumerate_plans([METHODS["dclm"]], [None])[:3]
    tracker_prefix = f"file://{tmp_path}/locks/"
    (tmp_path / "locks").mkdir()

    # Mark the MIDDLE plan as done. First and third should still submit.
    done_plan = plans[1]
    (tmp_path / "locks" / done_plan.run_key).write_text(
        json.dumps({"region": "us-central1", "run_key": done_plan.run_key})
    )
    done_bucket = tmp_path / "fake-us-central1" / "checkpoints" / "isoflop-curation" / done_plan.run_name_core
    done_bucket.mkdir(parents=True)
    (done_bucket / ".data_curation_DONE").write_text('{"completed_at": "test"}')

    # Point REGION_TO_BUCKET at our local fake-us-central1 for the test.
    from experiments.scaling_law_sweeps import region_tracker as rt

    monkeypatch.setitem(rt.REGION_TO_BUCKET, "us-central1", f"file://{tmp_path}/fake-us-central1")

    submitted, skipped = launcher.submit_all(
        mock_client,
        plans,
        tracker_prefix=tracker_prefix,
        skip_if_done=True,
        child_priority_band=0,
        wandb_api_key="k",
        hf_token=None,
        wandb_project="marin",
        wandb_entity="marin-community",
        wandb_group="g",
    )
    assert len(skipped) == 1
    assert skipped[0].run_name_core == done_plan.run_name_core
    assert len(submitted) == 2
    # iris.submit called only for the two non-done plans
    assert mock_client.submit.call_count == 2


# =============================================================================
# CLI: --dry-run prints and does not submit
# =============================================================================


def test_dry_run_prints_plan_does_not_submit(capsys, monkeypatch):
    monkeypatch.delenv("IRIS_CONTROLLER_ADDRESS", raising=False)
    launcher.main(["--methods", "dclm", "--experiments", "all", "--dry-run"])
    captured = capsys.readouterr().out
    assert "TOTAL: 127 runs" in captured


def test_main_requires_iris_controller_address_when_not_dry_run(monkeypatch):
    monkeypatch.delenv("IRIS_CONTROLLER_ADDRESS", raising=False)
    with pytest.raises(RuntimeError, match="IRIS_CONTROLLER_ADDRESS"):
        launcher.main(["--methods", "dclm", "--experiments", "A", "--max-count", "1", "--no-keep-alive"])


def test_main_requires_wandb_api_key_when_not_dry_run(monkeypatch):
    monkeypatch.setenv("IRIS_CONTROLLER_ADDRESS", "http://fake")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # IrisClient.remote will be called; mock it so we get past that
    with patch.object(launcher, "IrisClient") as mock_iris:
        mock_iris.remote.return_value = MagicMock()
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            launcher.main(["--methods", "dclm", "--experiments", "A", "--max-count", "1", "--no-keep-alive"])


def test_max_count_truncates_plans(capsys, monkeypatch):
    monkeypatch.delenv("IRIS_CONTROLLER_ADDRESS", raising=False)
    launcher.main(["--methods", "dclm", "--experiments", "all", "--max-count", "3", "--dry-run"])
    captured = capsys.readouterr().out
    assert "TOTAL: 3 runs" in captured
