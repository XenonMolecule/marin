# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extraction progress dashboard — localhost web UI for monitoring WARC extraction.

Provides at-a-glance progress tracking, cluster status, and job management.
All expensive GCS queries are behind manual refresh buttons.

Usage::

    uv run python experiments/baseline_collection/dashboard.py
    # Open http://localhost:8080
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from experiments.baseline_collection.extraction_specs import LEGACY_SPEC_ID, SPECS

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WARC manifest is per-spec: high_quality is being scaled to the full DCLM
# 400m-1x pool (10,364 WARCs) while every other spec stays on the 3,000-WARC
# baseline. The denominator for "total" / "unclaimed" / ETA is therefore
# spec-dependent — see ``_manifest_path_for_spec`` / ``_load_manifest``.
DEFAULT_MANIFEST_PATH = "experiments/distill/baseline_warcs_3000.txt"
SPEC_MANIFESTS = {
    "high_quality": "experiments/distill/dclm_400m_1x_warcs.txt",
}
OUTPUT_SUBDIR = "documents/baseline_llm_extraction"
IRIS_CONFIG = "lib/iris/examples/marin.yaml"
USER_PREFIX = "michaelryan"

KNOWN_TPU_TYPES = [
    "v5p-8",
    "v5p-16",
    "v5p-32",
    "v5p-64",
    "v5litepod-4",
    "v5litepod-16",
    "v5litepod-32",
    "v6e-4",
    "v6e-8",
    "v6e-16",
]


LEGACY_NUM_BATCHES_CACHE_FILE = Path(__file__).parent / ".dashboard_legacy_num_batches.json"


def _load_legacy_num_batches() -> dict[str, int]:
    """Read every legacy ``_done`` marker and build a hash → num_batches map.

    Three-tier cache:
    1. Module-level dict (fastest, serves repeated calls in one process).
    2. On-disk JSON (survives dashboard restarts; the legacy data is
       immutable, so this never goes stale).
    3. Cold path: scan GCS in parallel (~30-60s for 3000 _done blobs).

    Returns the map of hash → num_batches; only populated for hashes whose
    ``_done`` JSON contained a parseable ``num_batches`` field.
    """
    global _legacy_num_batches
    if _legacy_num_batches is not None:
        return _legacy_num_batches

    # Disk fallback before going to GCS.
    try:
        if LEGACY_NUM_BATCHES_CACHE_FILE.exists():
            on_disk = json.loads(LEGACY_NUM_BATCHES_CACHE_FILE.read_text())
            if isinstance(on_disk, dict):
                _legacy_num_batches = {h: int(v) for h, v in on_disk.items() if isinstance(v, int)}
                logger.info(
                    "Loaded legacy num_batches map from disk cache: %d hashes (%s)",
                    len(_legacy_num_batches),
                    LEGACY_NUM_BATCHES_CACHE_FILE.name,
                )
                return _legacy_num_batches
    except Exception:
        logger.exception("Failed to read legacy cache (will rebuild from GCS)")

    from concurrent.futures import ThreadPoolExecutor

    from google.cloud import storage as gcs_storage
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    client = gcs_storage.Client()
    targets: list[tuple[str, str, str]] = []  # (bucket_name, blob_path, hash)

    # Use match_glob so the GCS list API only returns the few thousand
    # ``_done`` blobs, not every legacy batch file (~285k total). Without
    # this filter the list step alone takes minutes per region.
    for _, bucket_name in REGION_TO_DATA_BUCKET.items():
        bucket = client.bucket(bucket_name)
        try:
            for blob in bucket.list_blobs(
                match_glob="documents/baseline_llm_extraction/data-*/_done",
            ):
                rel = blob.name[len("documents/baseline_llm_extraction/") :]
                parts = rel.split("/")
                if len(parts) != 2 or parts[1] != "_done":
                    continue
                if not parts[0].startswith("data-"):
                    continue
                h = parts[0][len("data-") :]
                if len(h) != 12:
                    continue
                targets.append((bucket_name, blob.name, h))
        except Exception as e:
            logger.warning("Legacy num_batches scan failed for %s: %s", bucket_name, e)

    def _read(t: tuple[str, str, str]) -> tuple[str, int | None]:
        bucket_name, blob_path, h = t
        try:
            blob = client.bucket(bucket_name).blob(blob_path)
            content = blob.download_as_text()
            stats = json.loads(content)
            nb = stats.get("num_batches")
            return h, int(nb) if isinstance(nb, int) else None
        except Exception:
            return h, None

    result: dict[str, int] = {}
    if targets:
        with ThreadPoolExecutor(max_workers=32) as pool:
            for h, nb in pool.map(_read, targets):
                if nb is not None:
                    result.setdefault(h, nb)

    _legacy_num_batches = result
    logger.info("Loaded legacy num_batches map: %d hashes (from %d _done markers)", len(result), len(targets))

    # Persist to disk so future dashboard restarts skip the GCS scan.
    try:
        LEGACY_NUM_BATCHES_CACHE_FILE.write_text(json.dumps(result))
    except Exception:
        logger.exception("Failed to persist legacy cache (non-fatal)")

    return result


def _legacy_payload(manifest: dict[str, str]) -> dict:
    """Build the legacy-ground-truth fields injected into scan responses.

    Single pass over the manifest with O(1) lookups against the cached
    legacy map. Returns the per-hash map (for the per-WARC progress bars),
    the total batch count (for the aggregate progress bar), and the
    coverage count (for the "exact vs estimate" label).
    """
    legacy_map = _load_legacy_num_batches()
    known: dict[str, int] = {}
    for h in manifest:
        nb = legacy_map.get(h)
        if nb is not None:
            known[h] = nb
    return {
        "legacy_num_batches": known,
        "legacy_num_batches_total_known": sum(known.values()),
        "legacy_num_batches_known_count": len(known),
    }


def _bucket_prefix_for_spec(spec_id: str) -> str:
    """Return the bucket-relative prefix to scan for a given spec_id.

    ``LEGACY_SPEC_ID`` maps to the bare ``OUTPUT_SUBDIR`` (where pre-registry
    runs landed). All other specs nest under their id.
    """
    if spec_id == LEGACY_SPEC_ID:
        return f"{OUTPUT_SUBDIR}/"
    return f"{OUTPUT_SUBDIR}/{spec_id}/"


def _manifest_path_for_spec(spec_id: str) -> str:
    """Resolve the WARC manifest a spec is extracting against.

    Specs absent from ``SPEC_MANIFESTS`` use ``DEFAULT_MANIFEST_PATH``.
    """
    return SPEC_MANIFESTS.get(spec_id, DEFAULT_MANIFEST_PATH)


# The completed-WARC registry always lives in us-central1 (see
# run_extract_standalone.DEFAULT_COMPLETED_REGISTRY_PREFIX / _registry_prefix_for).
REGISTRY_BUCKET = "marin-us-central1"


def _load_done_registry(spec_id: str) -> dict[str, float | None]:
    """Map of hash → completion epoch for the central ``_completed`` registry.

    One cheap list (~one blob per finished WARC). The deep scan uses this to
    SKIP per-batch listing of finished WARCs — without it the scan enumerates
    every batch blob ever written (hundreds of thousands), which is why it was
    pathologically slow. On any failure returns an empty dict, degrading to the
    old (correct but slow) behaviour of listing every dir.

    Each marker's ``updated`` timestamp is the WARC's completion time (written
    once by ``_register_completed_warc``). These timestamps are the authoritative
    throughput signal — the in-progress batch scan can't see a finished WARC's
    batches, so completion times are what the headline rate is computed from.
    """
    from google.cloud import storage as gcs_storage

    prefix = _bucket_prefix_for_spec(spec_id) + "_completed/"
    done: dict[str, float | None] = {}
    try:
        client = gcs_storage.Client()
        for blob in client.bucket(REGISTRY_BUCKET).list_blobs(prefix=prefix):
            leaf = blob.name.rsplit("/", 1)[-1]
            if leaf.startswith("data-"):
                done[leaf[len("data-") :]] = blob.updated.timestamp() if blob.updated else None
    except Exception:
        logger.exception("Failed to load completed registry for %s; deep scan will list all dirs", spec_id)
    return done


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_manifest_hashes_by_spec: dict[str, dict[str, str]] = {}  # spec_id -> {hash -> warc_path}
_iris_client = None
_tunnel_cm = None

# Specs the dashboard knows about. Populated from --specs at startup;
# always includes LEGACY_SPEC_ID as a virtual spec for the unprefixed path.
SPECS_TO_SCAN: list[str] = [LEGACY_SPEC_ID]

# Per-WARC ground truth from the legacy (low_quality) run: hash → num_batches.
# Lazy-loaded on first use, cached for the lifetime of the dashboard process.
# Used to replace the per-WARC and aggregate batch estimates with exact
# counts whenever the WARC was previously extracted under low_quality.
_legacy_num_batches: dict[str, int] | None = None

CACHE_FILE = Path(__file__).parent / ".dashboard_cache.json"

# Per-spec progress caches. Cluster + jobs are spec-independent and stay flat.
_cache: dict[str, object] = {
    "progress": {},  # {spec_id: {"data": ..., "updated_at": ...}}
    "progress_deep": {},  # {spec_id: {"data": ..., "updated_at": ...}}
    "cluster": {"data": None, "updated_at": None},
    "jobs": {},  # {spec_id: {"data": ..., "updated_at": ...}}
    "preemption_history": [],  # list of {"ts": epoch, "total": int, "by_type": {tpu: int}}
}


def _jobs_slot(spec_id: str) -> dict:
    """Lazily-allocated per-spec cache slot for the Jobs tab."""
    bucket = _cache["jobs"]
    if spec_id not in bucket:
        bucket[spec_id] = {"data": None, "updated_at": None}
    return bucket[spec_id]


def _progress_slot(mode: str, spec_id: str) -> dict:
    """Lazily-allocated per-spec cache slot for ``progress`` or ``progress_deep``."""
    bucket = _cache[mode]
    if spec_id not in bucket:
        bucket[spec_id] = {"data": None, "updated_at": None}
    return bucket[spec_id]


def _save_cache() -> None:
    """Persist cache to disk so it survives dashboard restarts."""
    try:
        CACHE_FILE.write_text(json.dumps(_cache, default=str))
    except Exception:
        pass


def _load_cache() -> None:
    """Load cache from disk if available.

    The on-disk shape may be either the old flat layout (single ``data`` per
    mode) or the new per-spec layout. Old files are migrated into the legacy
    spec slot so dashboards started after an upgrade don't lose state.
    """
    global _cache
    try:
        if not CACHE_FILE.exists():
            return
        loaded = json.loads(CACHE_FILE.read_text())
        for key in _cache:
            if key not in loaded:
                continue
            value = loaded[key]
            if key == "preemption_history" and isinstance(value, list):
                _cache[key] = value
            elif key in ("progress", "progress_deep", "jobs"):
                if not isinstance(value, dict):
                    continue
                # New layout: {spec_id: {"data": ..., "updated_at": ...}}
                if value.get("data") is None and not any(isinstance(v, dict) and "data" in v for v in value.values()):
                    continue
                if "data" in value:
                    # Old flat layout — migrate into the legacy slot.
                    _cache[key] = {LEGACY_SPEC_ID: value}
                else:
                    _cache[key] = value
            elif key == "cluster":
                if isinstance(value, dict) and value.get("data") is not None:
                    _cache[key] = value
        logger.info("Loaded cached data from %s", CACHE_FILE)
    except Exception:
        logger.exception("Failed to load cache (ignored)")


def _load_manifest(spec_id: str) -> dict[str, str]:
    """Load a spec's manifest and build hash->path mapping. Cached per spec."""
    cached = _manifest_hashes_by_spec.get(spec_id)
    if cached is not None:
        return cached

    from experiments.baseline_collection.download_warcs import _load_manifest as load_m
    from experiments.baseline_collection.download_warcs import _warc_path_hash

    manifest_path = _manifest_path_for_spec(spec_id)
    warcs = load_m(manifest_path)
    hashes = {_warc_path_hash(w): w for w in warcs}
    _manifest_hashes_by_spec[spec_id] = hashes
    logger.info("Loaded manifest for spec %s: %d WARCs (%s)", spec_id, len(hashes), manifest_path)
    return hashes


def _invalidate_iris_client() -> None:
    """Drop the cached iris client + tunnel so the next call rebuilds.

    Used on connection failures (typically: SSH tunnel died after long idle).
    """
    global _iris_client, _tunnel_cm
    if _tunnel_cm is not None:
        try:
            _tunnel_cm.__exit__(None, None, None)
        except Exception:
            logger.exception("error closing stale iris tunnel (ignored)")
    _iris_client = None
    _tunnel_cm = None


def _get_iris_client():
    """Get or create the Iris client with SSH tunnel.

    Probes the cached client with a cheap RPC (`get_job_states([])`) before
    returning it. If the probe raises a connection error, the tunnel and
    client are invalidated and rebuilt. This handles the case where the
    dashboard sits idle for hours and the underlying SSH tunnel times out.
    """
    global _iris_client, _tunnel_cm

    if _iris_client is not None:
        try:
            _iris_client._cluster_client.get_job_states([])
            return _iris_client
        except Exception as e:
            logger.warning("iris tunnel probe failed (%s); rebuilding", e)
            _invalidate_iris_client()

    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig

    iris_config = IrisConfig.load(IRIS_CONFIG)
    bundle = iris_config.provider_bundle()

    controller_address = iris_config.controller_address()
    if not controller_address:
        controller_address = bundle.controller.discover_controller(iris_config.proto.controller)

    logger.info("Establishing tunnel to controller...")
    _tunnel_cm = bundle.controller.tunnel(address=controller_address)
    tunnel_url = _tunnel_cm.__enter__()
    logger.info("Tunnel ready: %s", tunnel_url)

    _iris_client = IrisClient.remote(
        tunnel_url,
        workspace=Path.cwd(),
    )
    return _iris_client


# ---------------------------------------------------------------------------
# Completed WARC registry
# ---------------------------------------------------------------------------


def _completed_registry_prefix(spec_id: str) -> str:
    """gs:// prefix where the per-spec ``_completed`` markers live.

    Mirrors the layout that ``run_extract_standalone.py:_registry_prefix_for``
    writes to: legacy/low_quality at the unprefixed path, others nested.
    """
    return f"gs://marin-us-central1/{_bucket_prefix_for_spec(spec_id).rstrip('/')}/_completed"


def _backfill_completed_registry(done_hashes: set[str], spec_id: str) -> None:
    """Write registry markers for any done WARCs not already registered.

    Called during deep scan to catch legacy completions from workers that
    predate the registry feature. Each marker is an empty file (~0 bytes)
    at ``_completed/data-{hash}``. Idempotent — re-writing an existing
    marker is a no-op.

    Uses google-cloud-storage directly (not fsspec/gcsfs) to avoid SSL
    issues with aiohttp on some local environments.
    """
    from google.cloud import storage as gcs_storage

    registry_uri = _completed_registry_prefix(spec_id)
    client = gcs_storage.Client()
    registry_bucket_name = registry_uri.replace("gs://", "").split("/", 1)[0]
    registry_prefix = registry_uri.replace(f"gs://{registry_bucket_name}/", "")
    bucket = client.bucket(registry_bucket_name)

    # Load existing registry
    existing: set[str] = set()
    try:
        for blob in bucket.list_blobs(prefix=registry_prefix + "/data-"):
            basename = blob.name.rsplit("/", 1)[-1]
            if basename.startswith("data-"):
                existing.add(basename[5:])
    except Exception:
        pass

    missing = done_hashes - existing
    if not missing:
        return

    logger.info("Backfilling completed registry: %d new entries", len(missing))
    errors = 0
    for h in missing:
        blob_path = f"{registry_prefix}/data-{h}"
        try:
            blob = bucket.blob(blob_path)
            blob.upload_from_string("")
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning("Failed to backfill registry for %s: %s", h, e)
            elif errors == 4:
                logger.warning("Suppressing further backfill errors (too many failures)")
    if errors:
        logger.warning("Backfill completed with %d/%d errors", errors, len(missing))


# ---------------------------------------------------------------------------
# GCS scanning
# ---------------------------------------------------------------------------


def _scan_progress_quick(spec_id: str) -> dict:
    """Quick scan: count claimed dirs per region. ~10s, one list op per region.

    Does NOT check _done markers (too expensive per-WARC). Use deep scan for done counts.
    """
    from google.cloud import storage as gcs_storage
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    client = gcs_storage.Client()
    manifest = _load_manifest(spec_id)

    regions = {}
    all_hashes: dict[str, list[str]] = {}  # hash -> list of regions

    for region, bucket_name in sorted(REGION_TO_DATA_BUCKET.items()):
        bucket = client.bucket(bucket_name)
        prefix = _bucket_prefix_for_spec(spec_id)

        claimed = set()
        done = set()
        done_stats = []

        # Two output layouts coexist for the same spec:
        #   (a) ``data-{hash}/`` directories (run_extract_standalone.py per-batch)
        #   (b) ``data-{hash}.jsonl.gz`` flat files (download_and_extract.py per-WARC)
        # Quick scan counts both: directory entries via delimiter list, and flat
        # files via a second list filtered to the immediate level. Flat files
        # are also marked DONE since they only exist on full-WARC completion
        # (Zephyr ``skip_existing`` only writes them on success).
        iterator = bucket.list_blobs(prefix=prefix, delimiter="/")
        list(iterator)  # consume to populate prefixes
        for dir_prefix in iterator.prefixes:
            dirname = dir_prefix.rstrip("/").split("/")[-1]
            if not dirname.startswith("data-"):
                continue
            h = dirname[5:]
            claimed.add(h)
            if h not in all_hashes:
                all_hashes[h] = []
            if region not in all_hashes[h]:
                all_hashes[h].append(region)

        # Flat-file pass: list blobs immediately under the prefix and pick the
        # ``data-{hash}.jsonl.gz`` ones. Tolerate the cost of listing all blobs
        # because the result set in the immediate level is small (one entry
        # per WARC, plus a few _completed markers).
        for blob in bucket.list_blobs(prefix=prefix + "data-"):
            rel = blob.name[len(prefix) :]
            if "/" in rel:
                continue  # nested entry — owned by a directory-layout WARC
            if rel.startswith("data-") and rel.endswith(".jsonl.gz"):
                h = rel[len("data-") : -len(".jsonl.gz")]
                if len(h) != 12:
                    continue
                claimed.add(h)
                done.add(h)
                if h not in all_hashes:
                    all_hashes[h] = []
                if region not in all_hashes[h]:
                    all_hashes[h].append(region)
                done_stats.append(
                    {
                        "hash": h,
                        "region": region,
                        "completed_at": blob.updated.isoformat() if blob.updated else None,
                    }
                )

        # Quick scan: directory-layout WARCs don't get _done checks here
        # (too expensive). Done markers are checked in deep scan mode.

        regions[region] = {
            "bucket": bucket_name,
            "claimed": len(claimed),
            "done": len(done),
            "claimed_hashes": sorted(claimed),
            "done_hashes": sorted(done),
            "done_stats": done_stats,
        }

    # Compute unique totals
    unique_claimed = set()
    unique_done = set()
    for region_data in regions.values():
        unique_claimed.update(region_data["claimed_hashes"])
        unique_done.update(region_data["done_hashes"])

    # Duplicates
    duplicates = sum(1 for h, r_list in all_hashes.items() if len(r_list) > 1)

    # ETA projection from done timestamps
    all_done_stats = []
    for region_data in regions.values():
        all_done_stats.extend(region_data["done_stats"])

    eta_hours = None
    if len(unique_done) >= 2:
        completion_times = []
        for s in all_done_stats:
            if s.get("completed_at"):
                from datetime import datetime

                try:
                    t = datetime.fromisoformat(s["completed_at"])
                    completion_times.append(t.timestamp())
                except Exception:
                    pass
        if len(completion_times) >= 2:
            completion_times.sort()
            span_hours = (completion_times[-1] - completion_times[0]) / 3600
            if span_hours > 0:
                rate = len(completion_times) / span_hours
                remaining = len(manifest) - len(unique_done)
                eta_hours = remaining / rate if rate > 0 else None

    # Strip large hash lists from response (keep counts only)
    for region_data in regions.values():
        del region_data["claimed_hashes"]
        del region_data["done_hashes"]

    return {
        "spec_id": spec_id,
        "total_manifest": len(manifest),
        "unique_claimed": len(unique_claimed),
        "unique_done": len(unique_done),
        "unclaimed": len(manifest) - len(unique_claimed),
        "duplicates": duplicates,
        "eta_hours": round(eta_hours, 1) if eta_hours else None,
        "regions": regions,
        "done_stats": all_done_stats,
        **_legacy_payload(manifest),
    }


def _scan_progress_deep(spec_id: str) -> dict:
    """Deep scan: list ALL blobs to get batch counts, done markers, and activity. ~60s+."""
    from google.cloud import storage as gcs_storage
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    client = gcs_storage.Client()
    manifest = _load_manifest(spec_id)

    all_hashes: dict[str, list[str]] = {}
    batch_counts: dict[str, int] = {}
    batch_seen: set[tuple[str, str]] = set()  # (hash, batch_filename) for dedup across regions
    latest_activity: dict[str, str] = {}
    earliest_batch_ts: float | None = None
    latest_batch_ts: float | None = None
    # All batch file epoch timestamps, for sliding-window rate calcs.
    # Stored as plain list; sorted once at the end.
    batch_timestamps: list[float] = []
    done_hashes: set[str] = set()
    done_stats: list[dict] = []
    region_data: dict[str, dict] = {}

    # Finished WARCs come from the central registry (one cheap list). We then
    # only deep-list the batches of IN-PROGRESS dirs — that's where all the
    # live activity/rate signal is. Finished WARCs (the vast majority of blobs)
    # are never enumerated, so cost is O(active WARCs), not O(all batch files).
    registry_done = _load_done_registry(spec_id)
    legacy_nb = _load_legacy_num_batches()

    prefix = _bucket_prefix_for_spec(spec_id)

    # Phase 1 (parallel across regions): one delimiter list per region surfaces
    # flat-file done WARCs (as blobs) and ``data-{hash}/`` work dirs (as
    # prefixes, populated only after the blob iterator is consumed).
    def _list_region(region: str, bucket_name: str):
        bucket = client.bucket(bucket_name)
        iterator = bucket.list_blobs(prefix=prefix + "data-", delimiter="/")
        flat_done = []  # (hash, completed_at, size_bytes)
        dir_hashes = []  # (hash, dirname)
        for blob in iterator:
            rel = blob.name[len(prefix) :]
            if "/" in rel:
                continue
            if rel.startswith("data-") and rel.endswith(".jsonl.gz"):
                h = rel[len("data-") : -len(".jsonl.gz")]
                if len(h) == 12:
                    flat_done.append((h, blob.updated.isoformat() if blob.updated else None, blob.size))
        for dir_prefix in iterator.prefixes:
            dirname = dir_prefix[len(prefix) :].rstrip("/")
            if dirname.startswith("data-") and len(dirname) - len("data-") == 12:
                dir_hashes.append((dirname[len("data-") :], dirname))
        return region, bucket_name, flat_done, dir_hashes

    region_items = sorted(REGION_TO_DATA_BUCKET.items())
    with ThreadPoolExecutor(max_workers=len(region_items)) as ex:
        phase1 = list(ex.map(lambda kv: _list_region(*kv), region_items))

    # Merge phase 1 and collect the in-progress dirs that still need a batch
    # listing (claimed but not in the finished-registry).
    inprogress: list[tuple[str, str, str, str]] = []  # (region, bucket, hash, dirname)
    for region, bucket_name, flat_done, dir_hashes in phase1:
        r_claimed = set()
        r_done = set()
        for h, completed_at, size_bytes in flat_done:
            r_claimed.add(h)
            r_done.add(h)
            done_hashes.add(h)
            all_hashes.setdefault(h, [])
            if region not in all_hashes[h]:
                all_hashes[h].append(region)
            done_stats.append(
                {
                    "hash": h,
                    "region": region,
                    "completed_at": completed_at,
                    "warc_path": manifest.get(h, ""),
                    "size_bytes": size_bytes,
                }
            )
        for h, dirname in dir_hashes:
            r_claimed.add(h)
            all_hashes.setdefault(h, [])
            if region not in all_hashes[h]:
                all_hashes[h].append(region)
            if h in registry_done:
                r_done.add(h)
                done_hashes.add(h)
                done_stats.append({"hash": h, "region": region, "warc_path": manifest.get(h, "")})
            else:
                inprogress.append((region, bucket_name, h, dirname))
        region_data[region] = {"bucket": bucket_name, "claimed": len(r_claimed), "done": len(r_done)}

    # Phase 2 (parallel): list each in-progress dir's batches + _done marker.
    def _scan_dir(region: str, bucket_name: str, h: str, dirname: str):
        bucket = client.bucket(bucket_name)
        batches = []  # (filename, epoch, iso)
        is_done = False
        done_at = None
        for blob in bucket.list_blobs(prefix=prefix + dirname + "/"):
            filename = blob.name.rsplit("/", 1)[-1]
            if filename == "_done":
                is_done = True
                done_at = blob.updated.isoformat() if blob.updated else None
            elif filename.startswith("batch_") and filename.endswith(".jsonl.gz"):
                batches.append(
                    (
                        filename,
                        blob.updated.timestamp() if blob.updated else None,
                        blob.updated.isoformat() if blob.updated else None,
                    )
                )
        return region, h, batches, is_done, done_at

    phase2 = []
    if inprogress:
        with ThreadPoolExecutor(max_workers=16) as ex:
            phase2 = list(ex.map(lambda a: _scan_dir(*a), inprogress))

    for region, h, batches, is_done, done_at in phase2:
        if is_done:
            done_hashes.add(h)
            done_stats.append({"hash": h, "region": region, "completed_at": done_at, "warc_path": manifest.get(h, "")})
        for filename, epoch, iso in batches:
            # Deduplicate: same batch in multiple regions counts once.
            key = (h, filename)
            if key not in batch_seen:
                batch_seen.add(key)
                batch_counts[h] = batch_counts.get(h, 0) + 1
            if iso and (h not in latest_activity or iso > latest_activity[h]):
                latest_activity[h] = iso
            if epoch is not None:
                batch_timestamps.append(epoch)
                if earliest_batch_ts is None or epoch < earliest_batch_ts:
                    earliest_batch_ts = epoch
                if latest_batch_ts is None or epoch > latest_batch_ts:
                    latest_batch_ts = epoch

    unique_claimed = set(all_hashes.keys())
    duplicates = sum(1 for r_list in all_hashes.values() if len(r_list) > 1)

    # Backfill the completed registry in a background thread so it doesn't
    # block the deep scan response. The scan results are returned immediately;
    # the backfill writes trickle in afterward.
    import threading

    threading.Thread(
        target=_backfill_completed_registry,
        args=(done_hashes, spec_id),
        daemon=True,
        name=f"registry-backfill-{spec_id}",
    ).start()

    # ETA from batch completion rate (much more reliable than WARC completion rate)
    total_batches = sum(batch_counts.values())
    eta_hours = None
    batches_per_hour = None  # lifetime rate
    batches_per_hour_recent = None  # last 3h rate
    recent_window_seconds = 3 * 3600
    now = time.time()

    # Avg batches per WARC, used to estimate remaining work. num_batches depends
    # only on a WARC's page count (not the spec), so the legacy ground-truth map
    # is valid here; fall back to the batch counts we tallied for in-progress
    # dirs. Finished WARCs are no longer enumerated, so their batches don't
    # appear in ``batch_counts`` — that's fine, ``total_batches`` below then
    # means "batches done on still-in-progress WARCs".
    avg_batches_per_warc = 100  # default
    real_counts = []
    for h in done_hashes:
        nb = legacy_nb.get(h) or batch_counts.get(h)
        if nb:
            real_counts.append(nb)
    if real_counts:
        avg_batches_per_warc = round(sum(real_counts) / len(real_counts))

    # Batches done across ALL WARCs, for the batch-level progress bar: finished
    # WARCs (exact legacy num_batches, else avg) + in-progress partials. Finished
    # WARCs aren't enumerated, so their batches are added back from ground truth
    # rather than counted. ``total_batches`` (in-progress only) stays separate
    # for the per-WARC table; ``in_progress_batches`` drives the ETA.
    in_progress_batches = sum(c for h, c in batch_counts.items() if h not in done_hashes)
    done_batches = sum(legacy_nb.get(h) or avg_batches_per_warc for h in done_hashes)
    batches_done = done_batches + in_progress_batches

    # --- True throughput from the completed-WARC registry --------------------
    # The in-progress batch scan only sees batches in not-yet-finished dirs, so
    # it structurally undercounts: a WARC's ~90 batches vanish from the window
    # the instant it completes and its dir is skipped via the registry. The
    # registry's per-WARC completion timestamps are the honest signal — in
    # steady state, batches/hr = WARCs-completed/hr * batches/WARC.
    completion_times = sorted(t for t in registry_done.values() if t)
    cutoff = now - recent_window_seconds
    warcs_per_hour_recent = None
    warcs_per_hour_lifetime = None
    run_age_hours = None
    if completion_times:
        recent_warcs = sum(1 for t in completion_times if t >= cutoff)
        warcs_per_hour_recent = recent_warcs / (recent_window_seconds / 3600)
        if len(completion_times) > 1:
            run_age_hours = (completion_times[-1] - completion_times[0]) / 3600
            if run_age_hours > 0.1:
                warcs_per_hour_lifetime = len(completion_times) / run_age_hours

    # Headline rates: registry-derived true throughput when timestamps are
    # available, else the in-progress batch-write rate (legacy fallback for
    # specs whose registry markers predate timestamp capture).
    if warcs_per_hour_recent is not None:
        batches_per_hour_recent = warcs_per_hour_recent * avg_batches_per_warc
    else:
        # NOW (not latest_batch_ts) so the rate drops to 0 if all workers stop.
        recent_count = sum(1 for ts in batch_timestamps if ts >= cutoff)
        batches_per_hour_recent = recent_count / (recent_window_seconds / 3600)

    if warcs_per_hour_lifetime is not None:
        batches_per_hour = warcs_per_hour_lifetime * avg_batches_per_warc
    elif earliest_batch_ts and latest_batch_ts and total_batches > 10:
        span_hours = (latest_batch_ts - earliest_batch_ts) / 3600
        if span_hours > 0.1:
            batches_per_hour = total_batches / span_hours

    # Use the RECENT rate for ETA (it's more representative of current throughput)
    rate_for_eta = batches_per_hour_recent if batches_per_hour_recent > 0 else batches_per_hour
    if rate_for_eta and rate_for_eta > 0:
        # Remaining = (not-done WARCs x avg) minus the partial batches already
        # done on the in-progress ones. in_progress_batches excludes finished
        # WARCs, so there's no double subtraction.
        remaining_warcs = len(manifest) - len(done_hashes)
        estimated_remaining_batches = remaining_warcs * avg_batches_per_warc - in_progress_batches
        if estimated_remaining_batches > 0:
            eta_hours = estimated_remaining_batches / rate_for_eta

    return {
        "spec_id": spec_id,
        "total_manifest": len(manifest),
        "unique_claimed": len(unique_claimed),
        "unique_done": len(done_hashes),
        "unclaimed": len(manifest) - len(unique_claimed),
        "duplicates": duplicates,
        "eta_hours": round(eta_hours, 1) if eta_hours else None,
        "batches_per_hour": round(batches_per_hour, 1) if batches_per_hour else None,
        "batches_per_hour_recent": round(batches_per_hour_recent, 1) if batches_per_hour_recent else None,
        "warcs_per_hour_recent": round(warcs_per_hour_recent, 1) if warcs_per_hour_recent else None,
        "warcs_per_hour_lifetime": round(warcs_per_hour_lifetime, 1) if warcs_per_hour_lifetime else None,
        "run_age_hours": round(run_age_hours, 1) if run_age_hours else None,
        "recent_window_hours": recent_window_seconds / 3600,
        "avg_batches_per_warc": avg_batches_per_warc,
        "regions": region_data,
        "done_stats": done_stats,
        "batch_counts": batch_counts,
        "latest_activity": latest_activity,
        "total_batches": total_batches,
        "batches_done": batches_done,
        **_legacy_payload(manifest),
    }


# ---------------------------------------------------------------------------
# Iris operations
# ---------------------------------------------------------------------------


def _fetch_cluster_status() -> dict:
    """Get autoscaler status from Iris."""
    client = _get_iris_client()
    response = client._cluster_client.get_autoscaler_status()

    groups = []
    for group in response.status.groups:
        name = group.name
        # Parse scale group name: tpu_v5p-preemptible_8-us-central1-a
        parts = name.split("_")
        tpu_type = None
        region = None
        if len(parts) >= 3 and parts[0] == "tpu":
            # e.g. tpu_v5p-preemptible_8-us-central1-a
            tpu_type = name  # fallback
            region = ""

        counts = dict(group.slice_state_counts) if group.slice_state_counts else {}
        ready = counts.get("ready", 0)
        booting = counts.get("booting", 0)
        initializing = counts.get("initializing", 0)
        failed = counts.get("failed", 0)
        demand = group.current_demand

        # Count idle slices
        idle = sum(1 for s in group.slices if s.idle)

        if ready == 0 and booting == 0 and demand == 0 and initializing == 0:
            continue  # Skip empty groups

        groups.append(
            {
                "name": name,
                "tpu_type": tpu_type,
                "region": region,
                "ready": ready,
                "booting": booting,
                "initializing": initializing,
                "failed": failed,
                "demand": demand,
                "idle": idle,
            }
        )

    groups.sort(key=lambda g: (-g["ready"], -g["demand"], g["name"]))

    return {
        "groups": groups,
        "total_ready": sum(g["ready"] for g in groups),
        "total_demand": sum(g["demand"] for g in groups),
        "total_idle": sum(g["idle"] for g in groups),
    }


def _format_job(j, enrich_resources: bool = False) -> dict:
    """Convert a JobStatus proto to a JSON-serializable dict.

    The Iris ``list_jobs`` RPC returns JobStatus protos with the ``resources``
    field stripped, so children come back without TPU/CPU/RAM info. Set
    ``enrich_resources=True`` to fall back to a per-job ``client.status()``
    call when ``resources`` is missing — adds one RPC per enrichment but
    surfaces the TPU type for adaptive-style children that don't have a
    Zephyr workers-pool.
    """
    from iris.rpc import job_pb2

    state_name = job_pb2.JobState.Name(j.state).replace("JOB_STATE_", "").lower()

    has_device = False
    tpu_type = None
    if j.HasField("resources"):
        r = j.resources
        if r.HasField("device"):
            has_device = True
            if r.device.HasField("tpu"):
                tpu_type = r.device.tpu.variant

    if enrich_resources and (not has_device or tpu_type is None):
        # list_jobs strips ``resources``; reach out to client.status to grab
        # the full proto. Best-effort: failures fall back to the slim job.
        try:
            from iris.cluster.types import JobName

            client = _get_iris_client()
            full = client.status(JobName.from_wire(j.job_id))
            if full.HasField("resources"):
                fr = full.resources
                if fr.HasField("device"):
                    has_device = True
                    if fr.device.HasField("tpu"):
                        tpu_type = fr.device.tpu.variant
                # Override the slim proto's resources with the enriched one
                # so the res_parts builder below picks them up.
                j = full
        except Exception:
            logger.debug("status enrichment failed for %s", j.job_id, exc_info=True)

    res_parts = []
    if j.HasField("resources"):
        r = j.resources
        if r.cpu_millicores:
            res_parts.append(f"{r.cpu_millicores / 1000:g}cpu")
        if r.memory_bytes:
            gb = r.memory_bytes / (1024 * 1024 * 1024)
            res_parts.append(f"{gb:.0f}GB")
        if tpu_type:
            res_parts.append(tpu_type)

    submitted = ""
    if j.submitted_at.epoch_ms:
        from datetime import datetime, timezone

        submitted = datetime.fromtimestamp(j.submitted_at.epoch_ms / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    reason = j.error or j.pending_reason or ""

    return {
        "job_id": j.job_id or "",
        "name": j.name or "",
        "state": state_name,
        "is_parent": not has_device,
        "tpu_type": tpu_type or "",
        "resources": ", ".join(res_parts),
        "submitted": submitted,
        "reason": (reason or "")[:100],
        "preemption_count": j.preemption_count or 0,
        "failure_count": j.failure_count or 0,
    }


def _list_tasks_for_job(job_id: str, tpu_type_hint: str = "") -> list:
    """Return TaskStatus rows for a job, formatted as child-like dicts.

    Each task in the workers-pool job corresponds to one TPU instance under
    the new Iris/Zephyr layout (CPU coordinator + worker pool of TPU tasks).
    Surfacing these at child-level gives the dashboard back the per-TPU
    granularity it had pre-merge.
    """
    from iris.rpc import controller_pb2, job_pb2

    client = _get_iris_client()
    rpc_client = client._cluster_client._client

    state_name_map = {
        job_pb2.TASK_STATE_RUNNING: "running",
        job_pb2.TASK_STATE_PENDING: "pending",
        job_pb2.TASK_STATE_BUILDING: "building",
        job_pb2.TASK_STATE_ASSIGNED: "pending",
        job_pb2.TASK_STATE_SUCCEEDED: "succeeded",
        job_pb2.TASK_STATE_FAILED: "failed",
        job_pb2.TASK_STATE_KILLED: "killed",
        job_pb2.TASK_STATE_PREEMPTED: "preempted",
        job_pb2.TASK_STATE_UNSCHEDULABLE: "pending",
        job_pb2.TASK_STATE_WORKER_FAILED: "worker_failed",
    }

    try:
        req = controller_pb2.Controller.ListTasksRequest(job_id=job_id)
        resp = rpc_client.list_tasks(req)
    except Exception as e:
        logger.warning("list_tasks failed for %s: %s", job_id, e)
        return []

    rows: list[dict] = []
    for t in resp.tasks:
        state = state_name_map.get(t.state, "unknown")
        submitted = ""
        try:
            ts = t.started_at.epoch_ms if t.HasField("started_at") else 0
        except Exception:
            ts = 0
        if ts:
            from datetime import datetime, timezone

            submitted = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        # Best-effort attempt count (UI prefers preemption_count semantically).
        attempt = t.current_attempt_id or 0
        rows.append(
            {
                "job_id": f"{job_id}/task-{t.task_id}",
                "name": f"task-{t.task_id}",
                "state": state,
                "is_parent": False,
                "tpu_type": tpu_type_hint,
                "resources": tpu_type_hint,
                "submitted": submitted,
                "reason": (t.error or t.pending_reason or "")[:100],
                "preemption_count": max(attempt - 0, 0),  # rough proxy
                "failure_count": 0,
                "task_id": t.task_id,
                "worker_id": t.worker_id,
            }
        )
    rows.sort(key=lambda r: int(r["task_id"]) if r["task_id"].isdigit() else r["task_id"])
    return rows


def _tpu_type_from_parent_name(parent_name: str) -> str:
    """Best-effort TPU variant inference from the orchestrator's job name.

    Job names follow the pattern ``extract-{spec_id}-{tpu_variant}_{batch_id}``
    (see ``_submit_iris_job``). Falls back to "" if the pattern doesn't match.
    """
    if not parent_name:
        return ""
    # Strip trailing _<batch_id> (timestamp_counter)
    base = parent_name.rsplit("_", 2)[0] if "_" in parent_name else parent_name
    # Then strip extract-{spec}- prefix to leave just the variant
    parts = base.split("-", 2)
    return parts[-1] if len(parts) >= 3 else ""


def _list_child_jobs(parent_job_id: str) -> list:
    """List child jobs of a parent using the parent_job_id filter in ListJobsRequest.

    Only returns children in active states (running, pending, building) to avoid
    counting thousands of killed/preempted retry attempts.
    """
    from iris.rpc import controller_pb2, job_pb2

    client = _get_iris_client()
    rpc_client = client._cluster_client._client

    # Paginate in case there are many children
    all_children = []
    offset = 0
    active_states = {
        job_pb2.JOB_STATE_RUNNING,
        job_pb2.JOB_STATE_PENDING,
        job_pb2.JOB_STATE_BUILDING,
    }
    while True:
        # Iris proto drift: filter fields moved into a nested ``query`` field
        # on ListJobsRequest. The old top-level kwargs no longer exist.
        request = controller_pb2.Controller.ListJobsRequest(
            query=controller_pb2.Controller.JobQuery(parent_job_id=parent_job_id, limit=500, offset=offset)
        )
        response = rpc_client.list_jobs(request)
        for j in response.jobs:
            if j.state in active_states:
                all_children.append(j)
        if not response.has_more or len(response.jobs) == 0:
            break
        offset += len(response.jobs)
    return all_children


def _fetch_jobs(spec_id: str | None = None) -> dict:
    """List extraction jobs for the current user, optionally filtered by spec.

    Uses name_filter + state_filter on the gRPC ListJobs API to get running and
    pending jobs directly, avoiding the broken pagination over ALL jobs.

    Job names follow ``extract-{spec_id}-...`` (see ``_submit_iris_job`` in
    ``multi_region_extraction.py``); when a spec is supplied we tighten the
    name filter so the Jobs tab only shows that spec's jobs.
    """
    from iris.rpc import controller_pb2 as ctrl_pb2
    from iris.rpc import job_pb2 as _jpb2

    client = _get_iris_client()
    rpc_client = client._cluster_client._client

    # Match ALL of the user's extract jobs by job-id prefix. We cannot filter by
    # spec here: parent (orchestrator) job names are author-chosen and do NOT
    # embed the spec id (e.g. ``extract-hq-solo-v6e4``), while the spec only
    # appears in the *child* leaf names (``extract-{spec_id}-{tpu}-{seed}``),
    # which sit at job-id depth 3 under the parent path. A spec-prefixed filter
    # therefore matched neither parents nor children and returned nothing.
    # Showing all extract parents is the right call given one spec campaign runs
    # at a time; per-spec accounting lives in the progress scans, not here.
    prefix_str = f"/{USER_PREFIX}/extract-"

    # Use name_filter + state_filter to get only our active extract jobs.
    # Filter fields moved to nested ``query`` (see _list_child_jobs above).
    # The three state queries are independent and dominated by controller RPC
    # round-trip latency, so we fan them out across a small thread pool. gRPC
    # python channels are thread-safe (multiple concurrent unary RPCs on one
    # channel are explicitly supported), so reusing the shared rpc_client is OK.
    def _fetch_state(state_name: str):
        req = ctrl_pb2.Controller.ListJobsRequest(
            query=ctrl_pb2.Controller.JobQuery(
                name_filter=prefix_str,
                state_filter=state_name,
                limit=2000,
            )
        )
        return rpc_client.list_jobs(req).jobs

    state_names = ("running", "pending", "building")
    all_active = []
    with ThreadPoolExecutor(max_workers=len(state_names)) as ex:
        for fut in as_completed(ex.submit(_fetch_state, s) for s in state_names):
            all_active.extend(fut.result())

    raw_parents = []
    raw_children = []
    for j in all_active:
        # Parent jobs have no TPU device; children have device.tpu.variant set.
        # HasField("device") is True even for empty device protos, so check content.
        parts = j.job_id.strip("/").split("/")
        is_parent = len(parts) == 2  # user/jobname = parent, user/jobname/child = child
        if is_parent:
            raw_parents.append(j)
        else:
            raw_children.append(j)

    logger.info("Fetched jobs: %d parents, %d active children", len(raw_parents), len(raw_children))

    # Group active children under running parents by job_id prefix.
    # New Iris/Zephyr layout puts the actual TPU instances inside the
    # workers-pool child as TASKS (not separate jobs). Replace each
    # workers-pool child with its task list so the dashboard shows
    # per-TPU rows like the old single-level layout did.
    parents = []
    for p in raw_parents:
        formatted = _format_job(p)
        if p.state == _jpb2.JOB_STATE_RUNNING:
            parent_prefix = p.job_id + "/"
            tpu_hint = _tpu_type_from_parent_name(p.name)
            children: list[dict] = []
            for c in raw_children:
                if not c.job_id.startswith(parent_prefix):
                    continue
                if "/zephyr-" in c.job_id and "-workers-" in c.job_id:
                    # Expand workers-pool tasks (per-TPU granularity).
                    tasks = _list_tasks_for_job(c.job_id, tpu_type_hint=tpu_hint)
                    if tasks:
                        children.extend(tasks)
                        continue
                # Fallback: surface the bare child (e.g. Zephyr coordinator,
                # or unrelated nested CPU jobs / adaptive TPU children).
                # Enrich with client.status() so adaptive children's TPU type
                # surfaces — list_jobs strips ``resources`` from JobStatus.
                children.append(_format_job(c, enrich_resources=True))
            children.sort(key=lambda c: c.get("submitted") or "", reverse=True)
            formatted["children"] = children
        parents.append(formatted)

    parents.sort(key=lambda p: p.get("submitted") or "", reverse=True)

    # Orphans: active children not matching any running parent prefix
    running_prefixes = [p.job_id + "/" for p in raw_parents if p.state == _jpb2.JOB_STATE_RUNNING]
    orphan_children = [
        _format_job(c) for c in raw_children if not any(c.job_id.startswith(pp) for pp in running_prefixes)
    ]

    # Record preemption snapshot for rate tracking
    by_type: dict[str, int] = {}
    total_preemptions = 0
    for p in parents:
        for c in p.get("children", []):
            pc = c.get("preemption_count", 0)
            total_preemptions += pc
            tpu = c.get("tpu_type", "unknown")
            by_type[tpu] = by_type.get(tpu, 0) + pc

    history = _cache.get("preemption_history", [])
    history.append({"ts": time.time(), "total": total_preemptions, "by_type": by_type})
    # Keep last 100 snapshots
    _cache["preemption_history"] = history[-100:]

    return {
        "parents": parents,
        "children": orphan_children,
        "total": len(raw_parents) + len(raw_children),
        "preemption_history": _cache["preemption_history"],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_file(
        Path(__file__).parent / "dashboard.html",
        mimetype="text/html",
    )


def _resolve_spec(raw: str | None) -> str:
    """Pick a spec_id, falling back to the first registered spec."""
    if raw and raw in SPECS_TO_SCAN:
        return raw
    return SPECS_TO_SCAN[0]


@app.route("/api/specs")
def api_specs():
    """List specs the dashboard knows about."""
    return jsonify({"specs": SPECS_TO_SCAN, "default": SPECS_TO_SCAN[0]})


@app.route("/api/progress")
def api_progress():
    spec_id = _resolve_spec(request.args.get("spec"))
    quick = _progress_slot("progress", spec_id)
    deep = _progress_slot("progress_deep", spec_id)
    return jsonify(
        {
            "spec": spec_id,
            "data": quick["data"],
            "updated_at": quick["updated_at"],
            "deep_updated_at": deep["updated_at"],
        }
    )


@app.route("/api/progress/refresh", methods=["POST"])
def api_progress_refresh():
    body = request.json or {}
    mode = body.get("mode", "quick")
    spec_id = _resolve_spec(body.get("spec"))
    quick = _progress_slot("progress", spec_id)
    deep = _progress_slot("progress_deep", spec_id)
    try:
        if mode == "deep":
            data = _scan_progress_deep(spec_id)
            deep["data"] = data
            deep["updated_at"] = time.time()
            quick["data"] = data
            quick["updated_at"] = time.time()
        else:
            data = _scan_progress_quick(spec_id)
            # Merge: preserve deep scan fields (batch_counts, done, etc.) if available
            deep_data = deep.get("data")
            if deep_data:
                data["unique_done"] = deep_data.get("unique_done", 0)
                data["done_stats"] = deep_data.get("done_stats", [])
                data["batch_counts"] = deep_data.get("batch_counts")
                data["latest_activity"] = deep_data.get("latest_activity")
                data["total_batches"] = deep_data.get("total_batches")
                data["batches_done"] = deep_data.get("batches_done")
                data["batches_per_hour"] = deep_data.get("batches_per_hour")
                data["batches_per_hour_recent"] = deep_data.get("batches_per_hour_recent")
                data["warcs_per_hour_recent"] = deep_data.get("warcs_per_hour_recent")
                data["warcs_per_hour_lifetime"] = deep_data.get("warcs_per_hour_lifetime")
                data["run_age_hours"] = deep_data.get("run_age_hours")
                data["recent_window_hours"] = deep_data.get("recent_window_hours")
                data["avg_batches_per_warc"] = deep_data.get("avg_batches_per_warc")
                data["eta_hours"] = deep_data.get("eta_hours")
                # Update done counts per region from deep scan
                for region, rdata in data["regions"].items():
                    deep_region = deep_data.get("regions", {}).get(region, {})
                    if "done" in deep_region:
                        rdata["done"] = deep_region["done"]
            quick["data"] = data
            quick["updated_at"] = time.time()
        _save_cache()
        return jsonify(
            {
                "ok": True,
                "spec": spec_id,
                "data": quick["data"],
                "updated_at": quick["updated_at"],
                "deep_updated_at": deep["updated_at"],
            }
        )
    except Exception as e:
        logger.exception("Progress scan failed for spec=%s", spec_id)
        return jsonify({"ok": False, "spec": spec_id, "error": str(e)}), 500


@app.route("/api/cluster")
def api_cluster():
    return jsonify(
        {
            "data": _cache["cluster"]["data"],
            "updated_at": _cache["cluster"]["updated_at"],
        }
    )


@app.route("/api/cluster/refresh", methods=["POST"])
def api_cluster_refresh():
    try:
        data = _fetch_cluster_status()
        _cache["cluster"]["data"] = data
        _cache["cluster"]["updated_at"] = time.time()
        _save_cache()
        return jsonify({"ok": True, "data": data, "updated_at": time.time()})
    except Exception as e:
        logger.exception("Cluster status fetch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/jobs")
def api_jobs():
    spec_id = _resolve_spec(request.args.get("spec"))
    slot = _jobs_slot(spec_id)
    return jsonify(
        {
            "spec": spec_id,
            "data": slot["data"],
            "updated_at": slot["updated_at"],
        }
    )


@app.route("/api/jobs/refresh", methods=["POST"])
def api_jobs_refresh():
    body = request.json or {}
    spec_id = _resolve_spec(body.get("spec"))
    slot = _jobs_slot(spec_id)
    try:
        data = _fetch_jobs(spec_id=spec_id)
        slot["data"] = data
        slot["updated_at"] = time.time()
        _save_cache()
        return jsonify({"ok": True, "spec": spec_id, "data": data, "updated_at": slot["updated_at"]})
    except Exception as e:
        logger.exception("Job list fetch failed for spec=%s", spec_id)
        return jsonify({"ok": False, "spec": spec_id, "error": str(e)}), 500


@app.route("/api/jobs/kill", methods=["POST"])
def api_jobs_kill():
    from iris.cluster.types import JobName

    job_id = request.json.get("job_id", "") if request.json else ""
    if not job_id.startswith(f"/{USER_PREFIX}/"):
        return jsonify({"ok": False, "error": f"Can only kill {USER_PREFIX}/ jobs"}), 403

    try:
        client = _get_iris_client()
        name = JobName.from_wire(job_id if job_id.startswith("/") else f"/{job_id}")
        client.terminate(name)
        return jsonify({"ok": True, "terminated": [str(name)]})
    except Exception as e:
        logger.exception("Kill failed")
        return jsonify({"ok": False, "error": str(e)}), 500


VALID_PRIORITIES = {"production", "interactive", "batch", "unspecified"}


@app.route("/api/jobs/launch", methods=["POST"])
def api_jobs_launch():
    data = request.json or {}
    tpu_type = data.get("tpu_type", "")
    job_name = data.get("job_name", "")
    max_count = int(data.get("max_count", 64))
    initial_batch = int(data.get("initial_batch", 5))
    chunk_size = int(data.get("chunk_size", 5))
    check_interval = int(data.get("check_interval", 300))
    patience = int(data.get("patience", 3))
    parent_priority = (data.get("parent_priority") or "").strip()  # "" = default
    child_priority = (data.get("child_priority") or "batch").strip()
    # Spec selection: defaults to the currently selected spec in the UI.
    # Sending empty string explicitly = legacy mode (no --spec, hardcoded prompt).
    raw_spec = data.get("spec", None)
    spec = raw_spec.strip() if isinstance(raw_spec, str) and raw_spec.strip() else None
    manifest = (data.get("manifest") or "").strip()
    start = data.get("start")
    end = data.get("end")

    if tpu_type not in KNOWN_TPU_TYPES:
        return jsonify({"ok": False, "error": f"Unknown TPU type: {tpu_type}"}), 400
    if not job_name:
        return jsonify({"ok": False, "error": "Job name required"}), 400
    if parent_priority and parent_priority not in VALID_PRIORITIES:
        return jsonify({"ok": False, "error": f"Invalid parent_priority: {parent_priority}"}), 400
    if child_priority not in VALID_PRIORITIES:
        return jsonify({"ok": False, "error": f"Invalid child_priority: {child_priority}"}), 400

    try:
        from iris.cli.job import run_iris_job

        controller_url = _iris_client._cluster_client._address if _iris_client else None
        if not controller_url:
            return jsonify({"ok": False, "error": "Iris not connected"}), 500

        command = [
            "python",
            "experiments/baseline_collection/launch_adaptive.py",
            "--tpu-type",
            tpu_type,
            "--max-count",
            str(max_count),
            "--initial-batch",
            str(initial_batch),
            "--chunk-size",
            str(chunk_size),
            "--check-interval",
            str(check_interval),
            "--patience",
            str(patience),
            "--child-priority",
            child_priority,
        ]
        if spec:
            command += ["--spec", spec]
        if manifest:
            command += ["--manifest", manifest]
        if start is not None:
            command += ["--start", str(int(start))]
        if end is not None:
            command += ["--end", str(int(end))]

        run_iris_job_kwargs = dict(
            command=command,
            env_vars={},
            controller_url=controller_url,
            cpu=2,
            memory="2GB",
            job_name=job_name,
            wait=False,
        )
        # Only pass --priority for the parent if explicitly chosen.
        # Otherwise Iris assigns its default (typically interactive).
        if parent_priority:
            run_iris_job_kwargs["priority"] = parent_priority

        exit_code = run_iris_job(**run_iris_job_kwargs)
        return jsonify({"ok": exit_code == 0, "job_name": job_name})
    except Exception as e:
        logger.exception("Launch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="WARC extraction dashboard")
    parser.add_argument(
        "--specs",
        nargs="+",
        default=None,
        help=(
            "Spec ids to expose in the dashboard, in display order. The legacy "
            f"spec ({LEGACY_SPEC_ID!r}) is always available and pinned first if "
            "not listed. Defaults to all registered specs plus legacy."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("DASHBOARD_PORT", "8090")),
    )
    args = parser.parse_args()

    # Resolve spec list: ensure legacy is included and pinned first.
    if args.specs:
        specs = list(args.specs)
    else:
        # Default: legacy + all registered specs, deduped.
        specs = [LEGACY_SPEC_ID] + [s for s in sorted(SPECS) if s != LEGACY_SPEC_ID]
    if LEGACY_SPEC_ID not in specs:
        specs.insert(0, LEGACY_SPEC_ID)
    # Validate non-legacy entries against the registry.
    unknown = [s for s in specs if s != LEGACY_SPEC_ID and s not in SPECS]
    if unknown:
        parser.error(f"Unknown spec ids: {unknown}. Registered: {sorted(SPECS)}")
    SPECS_TO_SCAN[:] = specs
    logger.info("Dashboard specs: %s", SPECS_TO_SCAN)

    # Load disk cache from previous run
    _load_cache()

    # Pre-load each spec's manifest (high_quality pulls the 10,364-WARC pool;
    # others the 3,000 baseline).
    for spec_id in SPECS_TO_SCAN:
        _load_manifest(spec_id)

    # Pre-warm the legacy ground-truth cache in a background thread so the
    # first deep/quick scan doesn't pay the cold-path GCS scan. Disk cache
    # is hit synchronously inside _load_legacy_num_batches when available;
    # only falls back to GCS if the disk file is missing or stale.
    import threading as _threading

    _threading.Thread(
        target=_load_legacy_num_batches,
        daemon=True,
        name="legacy-num-batches-prefetch",
    ).start()

    # Establish Iris tunnel
    try:
        _get_iris_client()
        logger.info("Iris client connected")
    except Exception as e:
        logger.warning("Could not connect to Iris: %s (cluster/jobs tabs will be unavailable)", e)

    logger.info("Starting dashboard on http://localhost:%d", args.port)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
