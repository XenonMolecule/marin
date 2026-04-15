# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for region-agnostic orchestration.

Verifies the region tracker's write-once-read-always semantics, checkpoint
prefix resolution, and region-mismatch detection. All tests use local
filesystem (via `tmp_path`) rather than GCS — the logic is the same.
"""

from __future__ import annotations

import pytest

from experiments.scaling_law_sweeps import region_tracker as rt

# =============================================================================
# Region detection
# =============================================================================


def test_detect_region_reads_marin_region_env():
    assert rt.detect_current_region({"MARIN_REGION": "us-central1"}) == "us-central1"


def test_detect_region_falls_back_to_marin_prefix():
    """Marin's standard launch sets MARIN_PREFIX, not MARIN_REGION."""
    assert rt.detect_current_region({"MARIN_PREFIX": "gs://marin-us-central1"}) == "us-central1"
    assert rt.detect_current_region({"MARIN_PREFIX": "gs://marin-us-east5"}) == "us-east5"


def test_detect_region_prefers_explicit_over_prefix():
    result = rt.detect_current_region(
        {
            "MARIN_REGION": "us-central1",
            "MARIN_PREFIX": "gs://marin-us-east5",  # mismatched on purpose
        }
    )
    assert result == "us-central1"


def test_detect_region_no_signals_raises():
    with pytest.raises(ValueError, match="MARIN_PREFIX|MARIN_REGION"):
        rt.detect_current_region({})


def test_detect_region_non_marin_prefix_raises():
    """If MARIN_PREFIX is a local path (e.g. /tmp/marin), we can't detect region."""
    with pytest.raises(ValueError):
        rt.detect_current_region({"MARIN_PREFIX": "/tmp/marin"})


def test_detect_region_unknown_region_raises():
    with pytest.raises(rt.UnknownRegion, match="atlantis"):
        rt.detect_current_region({"MARIN_REGION": "atlantis"})


def test_all_known_regions_have_bucket_mapping():
    """Every region in REGION_TO_BUCKET must map to a non-empty bucket path."""
    for region, bucket in rt.REGION_TO_BUCKET.items():
        assert bucket.startswith("gs://marin"), f"{region} → {bucket} looks wrong"


# =============================================================================
# Tracker key format
# =============================================================================


def test_run_key_deterministic():
    k1 = rt.run_key_for("dclm", "expB_T20T", "isoflop-3e18-d512-L6-B32")
    k2 = rt.run_key_for("dclm", "expB_T20T", "isoflop-3e18-d512-L6-B32")
    assert k1 == k2


def test_run_key_differs_on_method():
    a = rt.run_key_for("dclm", "expA", "run")
    b = rt.run_key_for("fineweb", "expA", "run")
    assert a != b


def test_run_key_differs_on_experiment():
    a = rt.run_key_for("dclm", "expA_natural", "run")
    b = rt.run_key_for("dclm", "expB_T20T", "run")
    assert a != b


def test_run_key_differs_on_run_name():
    a = rt.run_key_for("dclm", "expA", "run-a")
    b = rt.run_key_for("dclm", "expA", "run-b")
    assert a != b


def test_run_key_has_region_suffix():
    """File suffix makes the stored files easy to identify/grep for."""
    k = rt.run_key_for("dclm", "expB_T20T", "isoflop-3e18")
    assert k.endswith(".region")


# =============================================================================
# claim_or_read_region — first claim / read-through semantics
# =============================================================================


@pytest.fixture
def tracker_dir(tmp_path) -> str:
    """A filesystem-backed tracker prefix. Safe for test parallelism via tmp_path."""
    d = tmp_path / "region_locks"
    d.mkdir()
    return f"file://{d}"


def test_first_claim_writes_and_returns_local_region(tracker_dir):
    claim = rt.claim_or_read_region("run1.region", "us-central1", tracker_prefix=tracker_dir)
    assert claim.region == "us-central1"
    assert claim.was_first_claim is True
    assert claim.run_key == "run1.region"


def test_second_call_reads_existing_claim(tracker_dir):
    rt.claim_or_read_region("run2.region", "us-central1", tracker_prefix=tracker_dir)
    # Second call from a "different region" — should still get us-central1 back.
    claim2 = rt.claim_or_read_region("run2.region", "us-east5", tracker_prefix=tracker_dir)
    assert claim2.region == "us-central1"
    assert claim2.was_first_claim is False


def test_different_run_keys_claim_independently(tracker_dir):
    a = rt.claim_or_read_region("run_a.region", "us-central1", tracker_prefix=tracker_dir)
    b = rt.claim_or_read_region("run_b.region", "us-east5", tracker_prefix=tracker_dir)
    assert a.region == "us-central1"
    assert b.region == "us-east5"
    # Re-reading shouldn't flip anything.
    a2 = rt.claim_or_read_region("run_a.region", "us-east5", tracker_prefix=tracker_dir)
    b2 = rt.claim_or_read_region("run_b.region", "us-central1", tracker_prefix=tracker_dir)
    assert a2.region == "us-central1"
    assert b2.region == "us-east5"


def test_claim_persists_across_reads(tracker_dir):
    """The pinned region should be read-through-idempotent across many calls."""
    rt.claim_or_read_region("run3.region", "us-central1", tracker_prefix=tracker_dir)
    for _ in range(5):
        claim = rt.claim_or_read_region("run3.region", "us-east5", tracker_prefix=tracker_dir)
        assert claim.region == "us-central1"
        assert claim.was_first_claim is False


# =============================================================================
# resolve_checkpoint_prefix — high-level entry point
# =============================================================================


def test_resolve_checkpoint_prefix_happy_path(tracker_dir):
    prefix = rt.resolve_checkpoint_prefix(
        "dclm",
        "expB_T20T",
        "run-1",
        local_region="us-central1",
        tracker_prefix=tracker_dir,
    )
    assert prefix == "gs://marin-us-central1"


def test_resolve_checkpoint_prefix_migration_allowed_by_default(tracker_dir):
    """Default behavior: region mismatch triggers migration (tracker overwritten,
    new region's bucket returned). This prevents iris-retry region-pinning
    deadlocks — data is in every region, so moving is safe, only orphaning
    the old region's checkpoints (Levanter restarts from step 0)."""
    rt.resolve_checkpoint_prefix(
        "dclm",
        "expB_T20T",
        "run-2",
        local_region="us-central1",
        tracker_prefix=tracker_dir,
    )
    # Migration: us-east5 wins, tracker is rewritten.
    new_bucket = rt.resolve_checkpoint_prefix(
        "dclm",
        "expB_T20T",
        "run-2",
        local_region="us-east5",
        tracker_prefix=tracker_dir,
    )
    assert new_bucket == "gs://marin-us-east5"


def test_resolve_checkpoint_prefix_strict_mode_raises_on_mismatch(tracker_dir):
    """When `allow_region_migration=False`, mismatch raises (old strict behavior).
    Use this mode only when cross-region data availability is uncertain."""
    rt.resolve_checkpoint_prefix(
        "dclm",
        "expB_T20T",
        "run-2b",
        local_region="us-central1",
        tracker_prefix=tracker_dir,
    )
    with pytest.raises(rt.RegionMismatch, match="us-central1"):
        rt.resolve_checkpoint_prefix(
            "dclm",
            "expB_T20T",
            "run-2b",
            local_region="us-east5",
            tracker_prefix=tracker_dir,
            allow_region_migration=False,
        )


def test_resolve_checkpoint_prefix_rejects_unknown_region(tracker_dir):
    with pytest.raises(rt.UnknownRegion, match="atlantis"):
        rt.resolve_checkpoint_prefix(
            "dclm",
            "expA",
            "run",
            local_region="atlantis",
            tracker_prefix=tracker_dir,
        )


def test_resolve_checkpoint_prefix_uses_detect_when_local_region_none(tracker_dir, monkeypatch):
    monkeypatch.setenv("MARIN_REGION", "us-central1")
    prefix = rt.resolve_checkpoint_prefix(
        "dclm",
        "expA",
        "run-detect",
        tracker_prefix=tracker_dir,
    )
    assert prefix == "gs://marin-us-central1"


def test_resolve_checkpoint_prefix_raises_without_env(tracker_dir, monkeypatch):
    monkeypatch.delenv("MARIN_REGION", raising=False)
    with pytest.raises(ValueError, match="MARIN_REGION"):
        rt.resolve_checkpoint_prefix(
            "dclm",
            "expA",
            "run-noenv",
            tracker_prefix=tracker_dir,
        )


# =============================================================================
# Storage format (JSON body)
# =============================================================================


def test_claim_file_contents_are_readable_json(tracker_dir, tmp_path):
    """Tracker file should contain JSON with region + run_key for debuggability."""
    import json
    from pathlib import Path

    rt.claim_or_read_region("debug.region", "us-central1", tracker_prefix=tracker_dir)
    # tracker_dir is "file:///..." — strip prefix to get filesystem path.
    local_dir = Path(tracker_dir.removeprefix("file://"))
    content = (local_dir / "debug.region").read_text()
    obj = json.loads(content)
    assert obj["region"] == "us-central1"
    assert obj["run_key"] == "debug.region"


def test_claim_backwards_compatible_with_plain_text():
    """Legacy plaintext claims (no JSON, no heartbeat) are correctly parsed AND,
    because they have no heartbeat, treated as stale and reclaimed by the next
    launch. This is intentional: without a heartbeat we have no proof the
    claimer is still alive, so the safer default is to allow reclaim.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        prefix = f"file://{tmp}"
        # Write a legacy plaintext claim manually.
        Path(tmp, "legacy.region").write_text("us-east5\n")
        claim = rt.claim_or_read_region("legacy.region", "us-central1", tracker_prefix=prefix)
        # Reclaimed: our region wins, and the claim records it.
        assert claim.region == "us-central1"
        assert claim.reclaimed_from_stale is True
        assert claim.was_first_claim is False


# =============================================================================
# Integration: using resolve_checkpoint_prefix in a realistic loop
# =============================================================================


def test_sequential_launches_pin_each_run(tracker_dir):
    """Simulate launching many runs; each gets pinned to its launching region."""
    # 3 DCLM experiment B runs launched in us-central1
    for i in range(3):
        prefix = rt.resolve_checkpoint_prefix(
            "dclm",
            "expB_T20T",
            f"run-{i}",
            local_region="us-central1",
            tracker_prefix=tracker_dir,
        )
        assert prefix == "gs://marin-us-central1"

    # 2 DCLM experiment A runs launched in us-east5 (different region)
    for i in range(2):
        prefix = rt.resolve_checkpoint_prefix(
            "dclm",
            "expA_natural",
            f"run-{i}",
            local_region="us-east5",
            tracker_prefix=tracker_dir,
        )
        assert prefix == "gs://marin-us-east5"

    # Re-launching the same run from a DIFFERENT region MIGRATES by default.
    migrated_bucket = rt.resolve_checkpoint_prefix(
        "dclm",
        "expB_T20T",
        "run-0",
        local_region="us-east5",
        tracker_prefix=tracker_dir,
    )
    assert (
        migrated_bucket == "gs://marin-us-east5"
    ), "Migration should return the new region's bucket (us-east5), not the old one"
    # After migration, re-launching in the ORIGINAL region migrates BACK.
    assert (
        rt.resolve_checkpoint_prefix(
            "dclm",
            "expB_T20T",
            "run-0",
            local_region="us-central1",
            tracker_prefix=tracker_dir,
        )
        == "gs://marin-us-central1"
    )
    # In strict mode (opt-out), mismatches still raise.
    with pytest.raises(rt.RegionMismatch):
        rt.resolve_checkpoint_prefix(
            "dclm",
            "expB_T20T",
            "run-0",
            local_region="us-east5",
            tracker_prefix=tracker_dir,
            allow_region_migration=False,
        )


# =============================================================================
# Heartbeat + stale reclaim
# =============================================================================


import datetime as _dt
import json as _json
import fsspec as _fsspec


def _hours_ago(hours: float) -> _dt.datetime:
    return _dt.datetime.utcnow() - _dt.timedelta(hours=hours)


def _write_tracker_with_ts(tracker_dir: str, run_key: str, region: str, heartbeat: _dt.datetime | None) -> None:
    """Helper: directly write a tracker file at a specific heartbeat (for testing staleness)."""
    payload = {"region": region, "run_key": run_key}
    if heartbeat is not None:
        payload["last_heartbeat_ts"] = heartbeat.isoformat() + "Z"
    path = f"{tracker_dir.rstrip('/')}/{run_key}"
    fs, urlpath = _fsspec.core.url_to_fs(path)
    with fs.open(urlpath, "wb") as f:
        f.write(_json.dumps(payload).encode())


def test_first_claim_writes_heartbeat(tracker_dir):
    """First claim must persist a heartbeat timestamp so future stale-detection works."""
    rt.claim_or_read_region("run_hb1.region", "us-central1", tracker_prefix=tracker_dir)
    # Read back and inspect.
    fs, urlpath = _fsspec.core.url_to_fs(f"{tracker_dir.rstrip('/')}/run_hb1.region")
    with fs.open(urlpath, "rb") as f:
        obj = _json.loads(f.read().decode())
    assert obj["region"] == "us-central1"
    assert "last_heartbeat_ts" in obj
    # Timestamp is parseable and recent (within last minute).
    ts = _dt.datetime.fromisoformat(obj["last_heartbeat_ts"].rstrip("Z"))
    assert (_dt.datetime.utcnow() - ts).total_seconds() < 60


def test_fresh_tracker_is_not_reclaimed(tracker_dir):
    """A tracker with a recent heartbeat must NOT be overwritten by another region."""
    _write_tracker_with_ts(tracker_dir, "run_hb2.region", "us-central1", _hours_ago(0.1))
    claim = rt.claim_or_read_region("run_hb2.region", "us-east5", tracker_prefix=tracker_dir)
    assert claim.region == "us-central1"
    assert claim.reclaimed_from_stale is False


def test_stale_tracker_is_reclaimed_by_new_region(tracker_dir):
    """A tracker whose heartbeat is over 30 min old is considered stale; a new
    launch in any region should reclaim it."""
    _write_tracker_with_ts(tracker_dir, "run_hb3.region", "us-central1", _hours_ago(1.0))
    claim = rt.claim_or_read_region("run_hb3.region", "us-east5", tracker_prefix=tracker_dir)
    assert claim.region == "us-east5"
    assert claim.reclaimed_from_stale is True


def test_missing_heartbeat_treated_as_stale(tracker_dir):
    """Legacy tracker files without `last_heartbeat_ts` must be treated as stale
    (we have no proof the claiming worker is still alive)."""
    _write_tracker_with_ts(tracker_dir, "run_hb4.region", "us-central1", heartbeat=None)
    claim = rt.claim_or_read_region("run_hb4.region", "us-east5", tracker_prefix=tracker_dir)
    assert claim.region == "us-east5"
    assert claim.reclaimed_from_stale is True


def test_refresh_heartbeat_updates_timestamp(tracker_dir):
    """refresh_heartbeat should bump the tracker's timestamp while preserving region."""
    _write_tracker_with_ts(tracker_dir, "run_hb5.region", "us-central1", _hours_ago(0.25))
    rt.refresh_heartbeat("run_hb5.region", "us-central1", tracker_prefix=tracker_dir)
    fs, urlpath = _fsspec.core.url_to_fs(f"{tracker_dir.rstrip('/')}/run_hb5.region")
    with fs.open(urlpath, "rb") as f:
        obj = _json.loads(f.read().decode())
    assert obj["region"] == "us-central1"
    ts = _dt.datetime.fromisoformat(obj["last_heartbeat_ts"].rstrip("Z"))
    assert (_dt.datetime.utcnow() - ts).total_seconds() < 60


def test_refresh_heartbeat_refuses_mismatched_region(tracker_dir, caplog):
    """If the tracker is pinned to region A but region B tries to refresh,
    refresh_heartbeat must NOT overwrite the tracker (that would be a silent
    region migration)."""
    _write_tracker_with_ts(tracker_dir, "run_hb6.region", "us-central1", _hours_ago(0.1))
    with caplog.at_level("WARNING"):
        rt.refresh_heartbeat("run_hb6.region", "us-east5", tracker_prefix=tracker_dir)
    # Tracker still pinned to us-central1.
    fs, urlpath = _fsspec.core.url_to_fs(f"{tracker_dir.rstrip('/')}/run_hb6.region")
    with fs.open(urlpath, "rb") as f:
        obj = _json.loads(f.read().decode())
    assert obj["region"] == "us-central1"


def test_stale_reclaim_updates_to_fresh(tracker_dir):
    """After reclaiming a stale tracker, the heartbeat must be fresh so the
    next re-check sees it as owned, not immediately re-reclaimable."""
    _write_tracker_with_ts(tracker_dir, "run_hb7.region", "us-central1", _hours_ago(2.0))
    rt.claim_or_read_region("run_hb7.region", "us-east5", tracker_prefix=tracker_dir)
    # Immediately re-read: should NOT reclaim again.
    claim = rt.claim_or_read_region("run_hb7.region", "us-central1", tracker_prefix=tracker_dir)
    assert claim.region == "us-east5"
    assert claim.reclaimed_from_stale is False


def test_is_stale_threshold(tracker_dir):
    """Spot-check the threshold: just-expired heartbeat is stale; just-fresh is not."""
    # STALE_HEARTBEAT_SECONDS = 1800
    assert rt._is_stale(_dt.datetime.utcnow() - _dt.timedelta(seconds=1801))
    assert not rt._is_stale(_dt.datetime.utcnow() - _dt.timedelta(seconds=1799))
    assert rt._is_stale(None)  # missing heartbeat always stale
