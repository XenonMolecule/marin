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

import datetime
import json
import logging
import os
from dataclasses import dataclass

import fsspec

logger = logging.getLogger(__name__)


# Where the tracker files live. A single shared prefix so any cluster can
# read/write. `marin-us-central1` is a neutral home bucket; tracker files are
# ~50 bytes each, so cross-region reads cost effectively zero.
DEFAULT_TRACKER_PREFIX: str = "gs://marin-us-central1/metadata/region_locks/data_curation_isoflop/"

# A tracker whose last heartbeat was written more than this many seconds ago is
# considered "stale" — the worker that claimed it is assumed dead, and a new
# launch in ANY region may reclaim the lock. The threshold is generous: the
# child refreshes every ~5 min, and a one-off GCS blip shouldn't trigger reclaim
# — but after 30 min of silence the owner has almost certainly died.
STALE_HEARTBEAT_SECONDS: int = 30 * 60


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
    reclaimed_from_stale: bool = False  # True iff we overwrote a stale claim


def _now_utc() -> datetime.datetime:
    return datetime.datetime.utcnow()


def _encode_tracker(region: str, run_key: str, heartbeat: datetime.datetime | None = None) -> bytes:
    """Serialize a tracker payload. `heartbeat` defaults to now."""
    hb = (heartbeat or _now_utc()).isoformat() + "Z"
    return json.dumps(
        {
            "region": region,
            "run_key": run_key,
            "last_heartbeat_ts": hb,
        }
    ).encode()


def _decode_tracker(data: str) -> tuple[str, datetime.datetime | None]:
    """Parse tracker bytes into (region, heartbeat_or_None).

    Legacy plain-text and heartbeat-less JSON payloads are supported — they
    return `(region, None)`, which downstream callers treat as "stale" since
    no heartbeat means no proof of life.
    """
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return data.strip(), None
    region = obj.get("region", "").strip()
    hb_str = obj.get("last_heartbeat_ts")
    if not hb_str:
        return region, None
    try:
        # Tolerate both "...Z" suffix and plain ISO.
        return region, datetime.datetime.fromisoformat(hb_str.rstrip("Z"))
    except ValueError:
        logger.warning("Could not parse heartbeat ts %r; treating as stale", hb_str)
        return region, None


def _is_stale(heartbeat: datetime.datetime | None, now: datetime.datetime | None = None) -> bool:
    """True if heartbeat is missing or older than STALE_HEARTBEAT_SECONDS."""
    if heartbeat is None:
        return True
    now = now or _now_utc()
    return (now - heartbeat).total_seconds() > STALE_HEARTBEAT_SECONDS


def _read_tracker(fs, urlpath: str) -> tuple[str, datetime.datetime | None]:
    with fs.open(urlpath, "rb") as f:
        data = f.read().decode()
    return _decode_tracker(data)


# Back-compat alias — older callers only need the region.
def _read_region_from(fs, urlpath: str) -> str:
    region, _ = _read_tracker(fs, urlpath)
    return region


def claim_or_read_region(
    run_key: str,
    local_region: str,
    *,
    tracker_prefix: str = DEFAULT_TRACKER_PREFIX,
    now: datetime.datetime | None = None,
) -> RegionClaim:
    """Atomically claim `local_region` for `run_key`, or read existing claim.

    Semantics:
      - No tracker file exists → write `local_region` (with heartbeat=now).
        First-claim path.
      - Tracker exists and is FRESH → return its region (caller handles
        potential RegionMismatch).
      - Tracker exists and is STALE (heartbeat missing or older than
        STALE_HEARTBEAT_SECONDS) → overwrite with `local_region` + fresh
        heartbeat. Enables automatic recovery when a worker died mid-run
        without writing a DONE marker.

    Uses GCS's `if_generation_match=0` precondition for atomicity — if two
    workers race, exactly one wins the write and the other reads the winner's
    region. Both end up with the same region string.

    Non-GCS filesystems (local `file://` for tests) fall back to a less-atomic
    exists-then-write pattern — fine for tests where there's no contention.
    """
    path = f"{tracker_prefix.rstrip('/')}/{run_key}"
    fs, urlpath = fsspec.core.url_to_fs(path)

    if fs.exists(urlpath):
        existing_region, heartbeat = _read_tracker(fs, urlpath)
        if not _is_stale(heartbeat, now=now):
            return RegionClaim(run_key=run_key, region=existing_region, was_first_claim=False)
        # Stale: reclaim for local_region.
        logger.warning(
            "Reclaiming stale tracker for %s (prev region=%s, heartbeat=%s) → %s",
            run_key,
            existing_region,
            heartbeat,
            local_region,
        )
        try:
            with fs.open(urlpath, "wb") as f:
                f.write(_encode_tracker(local_region, run_key))
        except Exception:
            logger.exception("Failed to reclaim stale tracker at %s", urlpath)
            raise
        return RegionClaim(
            run_key=run_key,
            region=local_region,
            was_first_claim=False,
            reclaimed_from_stale=True,
        )

    # No tracker exists. First-claim path.
    try:
        with fs.open(urlpath, "wb") as f:
            f.write(_encode_tracker(local_region, run_key))
    except Exception:
        logger.exception("Failed to write region tracker at %s", urlpath)
        raise

    # Read back. If the region differs, we lost a race — use the winner's value.
    written_region, _ = _read_tracker(fs, urlpath)
    if written_region != local_region:
        logger.info(
            "Lost region-claim race for %s: wrote %s but stored=%s",
            run_key,
            local_region,
            written_region,
        )
        return RegionClaim(run_key=run_key, region=written_region, was_first_claim=False)
    return RegionClaim(run_key=run_key, region=local_region, was_first_claim=True)


def refresh_heartbeat(
    run_key: str,
    local_region: str,
    *,
    tracker_prefix: str = DEFAULT_TRACKER_PREFIX,
) -> None:
    """Overwrite the tracker with a fresh heartbeat. Safe to call repeatedly.

    Preserves the region (it must already match local_region, else this is a
    no-op warning — something went wrong if a different region is trying to
    refresh). Non-fatal on write error; heartbeat will just go stale and the
    next launch can reclaim.

    Bypasses fsspec's instance cache via `skip_instance_cache=True`. In the
    standalone child, the heartbeat runs from a background thread while JAX/
    libtpu/Levanter are running heavy I/O on the main thread; the shared
    fsspec GCS session has been observed to stall/deadlock the daemon
    thread's second write. A fresh fs per call is slower but robust.
    """
    path = f"{tracker_prefix.rstrip('/')}/{run_key}"
    fs, urlpath = fsspec.core.url_to_fs(path, skip_instance_cache=True)
    try:
        if fs.exists(urlpath):
            existing_region, _ = _read_tracker(fs, urlpath)
            if existing_region != local_region:
                logger.warning(
                    "refresh_heartbeat: tracker at %s is pinned to %s, but we're in %s; "
                    "refusing to refresh (likely indicates a region migration gone wrong).",
                    urlpath,
                    existing_region,
                    local_region,
                )
                return
        with fs.open(urlpath, "wb") as f:
            f.write(_encode_tracker(local_region, run_key))
    except Exception as e:
        logger.warning("refresh_heartbeat failed for %s: %s", urlpath, e)


def resolve_checkpoint_prefix(
    method_name: str,
    experiment_tag: str,
    run_name: str,
    *,
    local_region: str | None = None,
    tracker_prefix: str = DEFAULT_TRACKER_PREFIX,
    allow_region_migration: bool = True,
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
