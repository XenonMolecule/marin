# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-lock tracker for region-agnostic sweep launches.

Problem: we want to launch training runs without specifying a region, so Iris
can schedule them on whichever cluster has TPUs available (us-central1,
us-east5-a, etc.). But once a run starts writing checkpoints in a region, we
must NOT let a resume/re-launch migrate the run to a different region — that
would trigger cross-region egress on every checkpoint read/write.

Solution: a centralized tracker on GCS that records `(run_key → region)` on
first launch and refuses to re-assign on subsequent launches. The atomic
write-once semantics come from GCS's `if_generation_match=0` precondition.

All reads/writes happen at launch time, not training time — so at most one
small GCS operation per run per launch.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import fsspec

logger = logging.getLogger(__name__)


# A region claim counts as "live" (its run still actively writing in that region)
# if the region's checkpoint dir was modified within this window. Must comfortably
# exceed the child's rolling-checkpoint interval (15 min) so a live-but-mid-step run
# is never mistaken for dead — see run_curation_train_standalone's
# CheckpointerConfig(save_interval=timedelta(minutes=15)).
STALE_CLAIM_SECONDS: float = 45 * 60


# Where the tracker files live. A single shared prefix so any cluster can
# read/write. `marin-us-central1` is a neutral home bucket; tracker files are
# ~50 bytes each, read/written at most once per launch.
DEFAULT_TRACKER_PREFIX: str = "gs://marin-us-central1/metadata/region_locks/data_curation_isoflop/"


# Bucket-per-region mapping — sourced from `rigging.filesystem.REGION_TO_DATA_BUCKET`
# so we stay consistent with the canonical Marin region → bucket map. We
# prepend `gs://` because rigging stores bare bucket names.
def _build_region_to_bucket() -> dict[str, str]:
    try:
        from rigging.filesystem import REGION_TO_DATA_BUCKET as _src

        return {region: f"gs://{bucket}" for region, bucket in _src.items()}
    except Exception:
        # Fallback for tests / environments without rigging available.
        return {
            "us-central1": "gs://marin-us-central1",
            "us-central2": "gs://marin-us-central2",
            "us-east1": "gs://marin-us-east1",
            "us-east5": "gs://marin-us-east5",
            "us-west4": "gs://marin-us-west4",
            "europe-west4": "gs://marin-eu-west4",
        }


REGION_TO_BUCKET: dict[str, str] = _build_region_to_bucket()


class RegionMismatch(RuntimeError):
    """Raised when a run's pinned region doesn't match the current worker's region.

    Signals that Iris (or whoever scheduled the job) placed the worker in the
    wrong region. The correct remediation is to re-schedule the job in the
    pinned region, NOT to silently egress checkpoints across regions.
    """


class UnknownRegion(ValueError):
    """Raised when a region name isn't in `REGION_TO_BUCKET`."""


def detect_current_region(env: dict[str, str] | None = None) -> str:
    """Detect the current worker's region.

    Resolution order:
      1. `MARIN_REGION` env var (explicit override, useful for tests)
      2. Parse `MARIN_PREFIX` env var (if set to `gs://marin-{region}`)
      3. GCP metadata server query via `rigging.filesystem.marin_region()`
         (works on any GCP VM — the canonical path on iris TPU workers,
         which don't get MARIN_REGION/MARIN_PREFIX set by default)
      4. Raise — fail loud rather than silently egress cross-region

    Tests can inject via the `env` dict to avoid the metadata-server call.
    """
    source = env if env is not None else os.environ

    # 1. explicit env override
    region = source.get("MARIN_REGION")

    # 2. parse MARIN_PREFIX
    if not region:
        prefix = source.get("MARIN_PREFIX", "")
        if prefix.startswith("gs://"):
            bucket = prefix[len("gs://") :].split("/", 1)[0]
            if bucket.startswith("marin-"):
                region = bucket[len("marin-") :]

    # 3. GCP metadata server (only when env wasn't explicitly provided — tests
    #    use `env={}` to simulate "no env signals" without actually hitting GCP).
    if not region and env is None:
        try:
            from rigging.filesystem import marin_region as _marin_region

            detected = _marin_region()
            if detected:
                region = detected
        except Exception as e:
            logger.warning("marin_region() metadata lookup failed: %s", e)

    if not region:
        raise ValueError(
            "Could not detect current region: neither MARIN_REGION, MARIN_PREFIX, "
            "nor the GCP metadata server yielded a region. On an iris TPU worker "
            "this should auto-resolve; if you're testing locally set MARIN_REGION."
        )
    if region not in REGION_TO_BUCKET:
        raise UnknownRegion(
            f"Region {region!r} is not in REGION_TO_BUCKET. " f"Known regions: {list(REGION_TO_BUCKET)}."
        )
    return region


def run_key_for(method_name: str, experiment_tag: str, run_name: str) -> str:
    """Deterministic tracker key for a (method, experiment, run) triple.

    Used as the GCS filename under `DEFAULT_TRACKER_PREFIX`. Stable across
    launches so the second launch of the same candidate hits the same file.
    """
    # Slashes would create nested dirs which is fine, but keep it simple.
    return f"{method_name}__{experiment_tag}__{run_name}.region"


@dataclass(frozen=True)
class RegionClaim:
    """Result of `claim_or_read_region`. Self-describing for logging."""

    run_key: str
    region: str
    was_first_claim: bool  # True iff we wrote the tracker; False iff we read existing


def _encode_tracker(region: str, run_key: str) -> bytes:
    """Serialize a tracker payload."""
    return json.dumps({"region": region, "run_key": run_key}).encode()


def _decode_tracker(data: str) -> str:
    """Parse tracker bytes into region. Tolerates legacy plain-text payloads."""
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return data.strip()
    return obj.get("region", "").strip()


def _read_region_from(fs, urlpath: str) -> str:
    with fs.open(urlpath, "rb") as f:
        return _decode_tracker(f.read().decode())


def claim_or_read_region(
    run_key: str,
    local_region: str,
    *,
    tracker_prefix: str = DEFAULT_TRACKER_PREFIX,
) -> RegionClaim:
    """Atomically claim `local_region` for `run_key`, or read existing claim.

    Semantics:
      - No tracker file exists → write `local_region`. First-claim path.
      - Tracker exists → return its region. Caller (`resolve_checkpoint_prefix`)
        decides whether to migrate when the stored region differs from
        `local_region` — with `allow_region_migration=True` it overwrites the
        tracker; with `allow_region_migration=False` it raises `RegionMismatch`.

    Uses GCS's `if_generation_match=0` precondition for atomicity — if two
    workers race, exactly one wins the write and the other reads the winner's
    region. Both end up with the same region string.

    Non-GCS filesystems (local `file://` for tests) fall back to a less-atomic
    exists-then-write pattern — fine for tests where there's no contention.
    """
    path = f"{tracker_prefix.rstrip('/')}/{run_key}"
    fs, urlpath = fsspec.core.url_to_fs(path)

    if fs.exists(urlpath):
        existing_region = _read_region_from(fs, urlpath)
        return RegionClaim(run_key=run_key, region=existing_region, was_first_claim=False)

    # No tracker exists. First-claim path.
    try:
        with fs.open(urlpath, "wb") as f:
            f.write(_encode_tracker(local_region, run_key))
    except Exception:
        logger.exception("Failed to write region tracker at %s", urlpath)
        raise

    # Read back. If the region differs, we lost a race — use the winner's value.
    written_region = _read_region_from(fs, urlpath)
    if written_region != local_region:
        logger.info(
            "Lost region-claim race for %s: wrote %s but stored=%s",
            run_key,
            local_region,
            written_region,
        )
        return RegionClaim(run_key=run_key, region=written_region, was_first_claim=False)
    return RegionClaim(run_key=run_key, region=local_region, was_first_claim=True)


def _latest_mtime_seconds_ago(fs, dir_urlpath: str) -> float | None:
    """Seconds since the most recently-modified object under `dir_urlpath`.

    Returns None when the dir is empty/absent (no checkpoint written yet) or when
    no entry carries a usable timestamp. Tolerates gcsfs (`mtime` datetime /
    `updated` ISO string) and local fs (`mtime` epoch float) detail formats.
    """
    try:
        entries = fs.ls(dir_urlpath, detail=True)
    except FileNotFoundError:
        return None

    times: list[datetime] = []
    for e in entries:
        raw = e.get("mtime") or e.get("updated") or e.get("timeCreated")
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            t = datetime.fromtimestamp(raw, tz=timezone.utc)
        elif isinstance(raw, str):
            t = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        elif isinstance(raw, datetime):
            t = raw
        else:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        times.append(t)

    if not times:
        return None
    return (datetime.now(timezone.utc) - max(times)).total_seconds()


def _claim_is_live(checkpoint_dir: str, staleness_seconds: float) -> bool:
    """True if `checkpoint_dir` shows a recent write — i.e. a run still active there.

    Conservative on read errors: returns True (treat as live → don't migrate → never
    risk a concurrent duplicate). Returns False only when we positively confirm no
    recent activity (empty dir, or newest object older than `staleness_seconds`).
    """
    try:
        fs, urlpath = fsspec.core.url_to_fs(checkpoint_dir, skip_instance_cache=True)
    except Exception:
        logger.warning("liveness: could not open fs for %s; treating claim as live", checkpoint_dir)
        return True
    try:
        ago = _latest_mtime_seconds_ago(fs, urlpath)
    except Exception:
        logger.warning("liveness: listing failed for %s; treating claim as live", checkpoint_dir)
        return True
    if ago is None:
        # No checkpoint written yet → not (yet) a live run with durable progress.
        # The atomic first-claim guards simultaneous starts; the residual window is
        # the run's first ~15 min before its first checkpoint lands.
        return False
    return ago < staleness_seconds


def resolve_checkpoint_prefix(
    method_name: str,
    experiment_tag: str,
    run_name: str,
    *,
    local_region: str | None = None,
    tracker_prefix: str = DEFAULT_TRACKER_PREFIX,
    allow_region_migration: bool = True,
    checkpoint_rel_path: str | None = None,
    live_staleness_seconds: float = STALE_CLAIM_SECONDS,
) -> str:
    """Return the correct GCS checkpoint prefix for this run, respecting the region lock.

    On first call: claims `local_region` and returns its bucket prefix.
    On subsequent calls: returns the originally-claimed region's bucket.

    When `allow_region_migration=True` (default since data is pre-copied to all
    regions): a region mismatch triggers a MIGRATION — the tracker is
    overwritten to the new local region, and the caller gets back the new
    bucket. Old checkpoints in the previous region become orphans, and
    training restarts from step 0 in the new region (Levanter sees an empty
    output_path). This is preferred over raising `RegionMismatch` because
    iris doesn't support per-task region affinity on retry; a mismatch
    usually means iris re-scheduled a preempted child in a different region,
    and refusing to proceed leads to terminal task failure.

    LIVENESS GATE: when `checkpoint_rel_path` is given, a mismatch does NOT
    migrate blindly. We first stat the *claimed* region's checkpoint dir
    (`{claimed_bucket}/{checkpoint_rel_path}`); if it was written within
    `live_staleness_seconds` the original run is still alive there, so we raise
    `RegionMismatch` rather than spawning a concurrent duplicate in this region.
    Migration only proceeds when the claimed region's checkpoint is stale/absent
    (the original was preempted or died). Without `checkpoint_rel_path` the old
    always-migrate behavior is preserved (backward compatible for other callers).

    When `allow_region_migration=False` (strict mode): raises `RegionMismatch`.
    Use only when cross-region data availability is uncertain.

    Example:
        >>> resolve_checkpoint_prefix("dclm", "expB_T20T", "isoflop-3e18-d512-...",
        ...                           local_region="us-central1")
        'gs://marin-us-central1'
    """
    if local_region is None:
        local_region = detect_current_region()
    if local_region not in REGION_TO_BUCKET:
        raise UnknownRegion(f"Unknown local region {local_region!r}")

    key = run_key_for(method_name, experiment_tag, run_name)
    claim = claim_or_read_region(key, local_region, tracker_prefix=tracker_prefix)

    effective_region = claim.region
    if claim.region != local_region:
        if not allow_region_migration:
            raise RegionMismatch(
                f"Run {key!r} is region-locked to {claim.region!r} but this worker "
                f"is in {local_region!r}. Iris should have placed this worker in "
                f"{claim.region!r}; re-schedule the job there instead of migrating."
            )
        # Liveness gate: don't migrate over a still-running claim, or we'd spawn a
        # concurrent duplicate in this region. Only migrate when the claimed region's
        # checkpoint is stale/absent (original preempted or dead).
        if checkpoint_rel_path:
            claimed_bucket = REGION_TO_BUCKET.get(claim.region)
            if claimed_bucket is not None:
                checkpoint_dir = f"{claimed_bucket.rstrip('/')}/{checkpoint_rel_path.lstrip('/')}"
                if _claim_is_live(checkpoint_dir, live_staleness_seconds):
                    raise RegionMismatch(
                        f"Run {key!r} is region-locked to {claim.region!r} and that region's "
                        f"checkpoint was written < {live_staleness_seconds:.0f}s ago (still live). "
                        f"Refusing to migrate to {local_region!r} to avoid a concurrent duplicate; "
                        f"iris should re-schedule this worker in {claim.region!r}."
                    )
        # Migration allowed: overwrite tracker with new region. Old checkpoints
        # are orphaned (stay in gs://marin-{old_region}/...), training restarts
        # from step 0 in the new region's bucket.
        logger.warning(
            "Region migration for %s: was %s, now %s. "
            "Old checkpoints in gs://marin-%s/... become orphans; training restarts at step 0.",
            key,
            claim.region,
            local_region,
            claim.region,
        )
        path = f"{tracker_prefix.rstrip('/')}/{key}"
        fs, urlpath = fsspec.core.url_to_fs(path, skip_instance_cache=True)
        try:
            with fs.open(urlpath, "wb") as f:
                f.write(_encode_tracker(local_region, key))
        except Exception:
            logger.exception("Failed to migrate tracker at %s", urlpath)
            raise
        # After migration, the effective region is the NEW local one —
        # the `claim` object is stale and points at the old region.
        effective_region = local_region

    return REGION_TO_BUCKET[effective_region]
