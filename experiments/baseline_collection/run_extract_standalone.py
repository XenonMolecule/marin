# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone WARC download + LLM extraction — runs directly on a TPU node.

Per-batch checkpointing: each batch of records writes its own output file to GCS
immediately after generation. On preemption + restart, completed batches are
skipped. Maximum work lost = one batch (~2 min at batch_size=100).

Output layout per WARC::

    output_dir/data-{warc_hash}/
        batch_0000.jsonl.gz
        batch_0001.jsonl.gz
        ...
        _done              # empty marker written after all batches complete

Usage::

    iris job run --tpu v5p-8 --memory 128GB --extra marin:vllm --extra marin:tpu \
        -- python experiments/baseline_collection/run_extract_standalone.py \
        --manifest gs://bucket/manifest.txt --output-subdir documents/baseline_llm_extraction
"""

import argparse
import gzip
import json
import logging
import os
import re
import time
from typing import Any

import fsspec

from experiments.baseline_collection.download_warcs import (
    _download_one_warc,
    _load_manifest,
    _warc_path_hash,
)

logger = logging.getLogger(__name__)

# Regex patterns from postprocess_extraction.py
_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_FIELD_MARKERS_RE = re.compile(r"\[\[\s*##\s*(text|completed)\s*##\s*\]\]", flags=re.IGNORECASE)
_FILTER_PATTERNS = [re.compile(r"\[NO_USEFUL_CONTENT\]")]
MIN_OUTPUT_CHARS = 50
MAX_OUTPUT_TOKENS = 6144
MAX_DOC_TOKENS = 26624  # 32768 context - MAX_OUTPUT_TOKENS
MAX_CONTEXT_TOKENS = MAX_DOC_TOKENS + MAX_OUTPUT_TOKENS  # 32768
# 500 records per batch ≈ 12 min on v5p-8. vLLM's continuous batching makes
# larger batches more efficient (slow prompts overlap with fast ones).
# Per-batch checkpointing means max 12 min lost on preemption.
DEFAULT_BATCH_SIZE = 500

# Completed WARC registry: a single GCS directory where each completed WARC
# gets a tiny marker file. Workers list this directory on startup (1 API call)
# to skip already-done WARCs without scanning 6 regional buckets per WARC.
#
# The default targets the legacy unprefixed namespace. When this script is
# launched with --spec, the registry prefix becomes
#   gs://marin-us-central1/documents/baseline_llm_extraction/{spec_id}/_completed
# so each spec has its own registry and they don't cross-contaminate.
DEFAULT_COMPLETED_REGISTRY_PREFIX = "gs://marin-us-central1/documents/baseline_llm_extraction/_completed"

# One google-cloud-storage client per process. Claim probes used to build a
# fresh Client() per call, which re-ran credential discovery and opened a new
# HTTPS pool for every probe — pure overhead at fleet scale.
_gcs_storage_client = None


def _gcs_client():
    global _gcs_storage_client
    if _gcs_storage_client is None:
        from google.cloud import storage as gcs_storage

        _gcs_storage_client = gcs_storage.Client()
    return _gcs_storage_client


def _registry_prefix_for(output_subdir: str) -> str:
    """Build the registry prefix for a given output_subdir, in marin-us-central1."""
    return f"gs://marin-us-central1/{output_subdir}/_completed"


def _register_completed_warc(warc_hash: str, registry_prefix: str) -> None:
    """Write a tiny marker to the central completed registry. Idempotent."""
    path = f"{registry_prefix}/data-{warc_hash}"
    try:
        with fsspec.open(path, "w") as f:
            f.write("")  # empty file, ~0 bytes
    except Exception as e:
        logger.warning("Failed to write completion registry for %s: %s", warc_hash, e)


# Registry freshness: gcsfs caches directory listings for the life of the
# process, so a plain ls() after the first call NEVER sees completions written
# by other workers — the remaining-WARC list stops shrinking and every worker
# re-probes long-finished WARCs forever (billions of class-B GETs fleet-wide).
# We force a fresh listing, throttled per prefix so spinning workers don't turn
# the registry itself into a LIST hotspot.
REGISTRY_FRESH_SECONDS = 60.0
_registry_cache: dict[str, tuple[float, set[str]]] = {}


def _load_completed_registry(registry_prefix: str) -> set[str]:
    """List the completed registry directory and return set of done hashes.

    Single list operation against one GCS bucket — much cheaper than scanning
    6 regional buckets. The listing bypasses gcsfs's process-lifetime dircache
    (see REGISTRY_FRESH_SECONDS above) but is served from a local cache for
    REGISTRY_FRESH_SECONDS between real listings. Returns the last good result
    (or empty set) on error — graceful fallback to per-WARC checks.
    """
    now = time.time()
    cached = _registry_cache.get(registry_prefix)
    if cached is not None and now - cached[0] < REGISTRY_FRESH_SECONDS:
        return cached[1]
    fs = fsspec.filesystem("gcs")
    prefix = registry_prefix.replace("gs://", "")
    try:
        paths = fs.ls(prefix, refresh=True)
        hashes = set()
        for p in paths:
            basename = p.rsplit("/", 1)[-1]
            if basename.startswith("data-"):
                hashes.add(basename[5:])
        logger.info("Loaded completed registry from %s: %d WARCs", registry_prefix, len(hashes))
        _registry_cache[registry_prefix] = (now, hashes)
        return hashes
    except Exception as e:
        logger.warning("Failed to load completed registry %s: %s (falling back to per-WARC checks)", registry_prefix, e)
        return cached[1] if cached is not None else set()


# ---------------------------------------------------------------------------
# Steal mode: batch-level claims for parallel WARC processing
# ---------------------------------------------------------------------------

STEAL_CLAIM_STALE_SECONDS = 30 * 60  # 30 minutes


def _steal_claim_path(warc_dir: str, batch_idx: int) -> str:
    return f"{warc_dir}/_stealing/batch_{batch_idx:04d}"


def _claim_batch_for_stealing(warc_dir: str, batch_idx: int) -> bool:
    """Atomically claim a single batch for stealing. Returns True if we won."""
    path = _steal_claim_path(warc_dir, batch_idx)
    if not path.startswith("gs://"):
        # Local filesystem fallback (for testing)
        if os.path.exists(path):
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("")
        return True

    parts = path.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]

    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_path)

    try:
        blob.upload_from_string("", if_generation_match=0)
        return True
    except Exception as e:
        if "conditionNotMet" in str(e) or "412" in str(e):
            # A marker already exists. Reclaim it if it is STALE (a dead stealer's
            # orphan); otherwise a live worker holds it. Without this, an orphaned
            # marker permanently blocks the batch (no stale override existed).
            try:
                blob.reload()
                if blob.updated is not None and time.time() - blob.updated.timestamp() > STEAL_CLAIM_STALE_SECONDS:
                    blob.upload_from_string("")  # overwrite = reclaim the stale marker
                    logger.info("Reclaimed stale steal-marker for batch %d", batch_idx)
                    return True
            except Exception:
                pass
            return False
        raise


def _list_steal_claims(warc_dir: str) -> set[int]:
    """Batch indices with a FRESH steal-claim.

    STALE claims (older than STEAL_CLAIM_STALE_SECONDS) are IGNORED so a dead
    stealer's orphaned ``_stealing`` marker cannot permanently wedge a batch — on
    giant WARCs that failure mode leaves the WARC one batch short forever. A marker
    whose age can't be determined is treated as fresh (conservative — never worse
    than the old always-exclude behavior).
    """
    from datetime import datetime, timezone

    fs = fsspec.filesystem("gcs") if warc_dir.startswith("gs://") else fsspec.filesystem("file")
    steal_dir = f"{warc_dir}/_stealing"
    if warc_dir.startswith("gs://"):
        steal_dir = steal_dir.replace("gs://", "")
    try:
        entries = fs.ls(steal_dir, detail=True)
    except Exception:
        return set()
    now = datetime.now(timezone.utc)
    fresh: set[int] = set()
    for e in entries:
        name = e.get("name") if isinstance(e, dict) else e
        basename = str(name).rsplit("/", 1)[-1]
        if not basename.startswith("batch_"):
            continue
        try:
            idx = int(basename.split("_")[1])
        except (IndexError, ValueError):
            continue
        ts = e.get("updated") or e.get("timeCreated") if isinstance(e, dict) else None
        age = None
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        if ts is not None:
            try:
                age = (now - ts).total_seconds()
            except (TypeError, ValueError):
                age = None
        if age is None or age < STEAL_CLAIM_STALE_SECONDS:
            fresh.add(idx)
    return fresh


def _batch_exists_any_region(warc_hash: str, batch_idx: int, output_subdir: str) -> bool:
    """Check if a specific batch file exists in any regional bucket."""
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    fs = fsspec.filesystem("gcs")
    batch_name = f"batch_{batch_idx:04d}.jsonl.gz"
    for bucket in REGION_TO_DATA_BUCKET.values():
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/{batch_name}"
        try:
            if fs.exists(path):
                return True
        except Exception:
            continue
    return False


def _find_stealable_batches(
    warc_hash: str,
    warc_dir: str,
    output_subdir: str,
    num_batches: int,
) -> list[int]:
    """Find batch indices available for stealing, in reverse order.

    Returns batch indices that are:
    - Not already completed (no batch file in any region)
    - Not FRESHLY steal-claimed (stale/orphaned claims are reclaimable, so they
      are treated as stealable — see ``_list_steal_claims``)
    Ordered from highest to lowest (reverse iteration).
    """
    completed = _find_completed_batches_all_regions(warc_hash, output_subdir)
    steal_claimed = _list_steal_claims(warc_dir)
    stealable = []
    for idx in range(num_batches - 1, -1, -1):
        if idx not in completed and idx not in steal_claimed:
            stealable.append(idx)
    return stealable


def _process_warc_steal(
    warc_path: str,
    output_dir: str,
    output_subdir: str,
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    template: str,
    system_message: str,
    registry_prefix: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pipeline: Any = None,
) -> dict:
    """Process uncompleted batches of a WARC in steal mode (reverse order).

    Does NOT claim the WARC — only claims individual batches via _stealing/.
    The WARC's forward worker (owner) continues unaware. Stealers work from
    the back, owners work from the front; they meet in the middle.
    """
    h = _warc_path_hash(warc_path)
    warc_dir = f"{output_dir}/data-{h}"

    # Quick check: was this WARC completed between our registry scan and now?
    if _is_warc_done_any_region(h, output_subdir):
        return {"warc": warc_path, "status": "steal_already_done"}

    # Download and build the same deterministic record list as the owner
    records = _download_one_warc(warc_path)
    if not records:
        return {"warc": warc_path, "status": "steal_empty"}

    if pipeline is None:
        # Length gate belongs to the single-greedy-call path only: the multi-call
        # pipelines chunk long docs by design, and their certification included
        # the long-form tail this gate would silently exclude (315KB/595KB docs).
        records = _filter_by_length(records, MAX_DOC_TOKENS)
    if not records:
        return {"warc": warc_path, "status": "steal_empty"}

    num_batches = (len(records) + batch_size - 1) // batch_size

    # Find stealable batches (reverse order, excluding done + claimed)
    stealable = _find_stealable_batches(h, warc_dir, output_subdir, num_batches)
    if not stealable:
        # No batches to steal — but check if all batches are done and _done is missing.
        # This handles the case where the original worker finished all batches but got
        # preempted before writing _done.
        all_completed = _find_completed_batches_all_regions(h, output_subdir)
        if len(all_completed) >= num_batches and not _is_warc_done_any_region(h, output_subdir):
            logger.info("Steal: all %d batches complete but _done missing on %s — writing it", num_batches, warc_path)
            final_kept = _count_records_in_all_batches(h, output_subdir)
            final_filtered = len(records) - final_kept
            stats = {
                "warc": warc_path,
                "total_records": len(records),
                "total_kept": final_kept,
                "total_filtered": final_filtered,
                "num_batches": num_batches,
            }
            done_path = f"{warc_dir}/_done"
            import json as _json

            fsspec.filesystem("gcs").pipe_file(done_path.replace("gs://", ""), _json.dumps(stats).encode())
            _register_completed_warc(h, registry_prefix)
            logger.info("Steal: wrote _done for %s (kept=%d, filtered=%d)", warc_path, final_kept, final_filtered)
            return {"warc": warc_path, "status": "steal_wrote_done"}
        logger.info("Steal: no stealable batches on %s, all done or claimed", warc_path)
        return {"warc": warc_path, "status": "steal_nothing_to_do"}

    logger.info(
        "Steal: %s has %d stealable batches (of %d total), starting from batch %d",
        warc_path,
        len(stealable),
        num_batches,
        stealable[0],
    )

    total_kept = 0
    total_filtered = 0
    batches_stolen = 0

    for batch_idx in stealable:
        # Pre-check: did someone else finish this batch since our scan?
        if _batch_exists_any_region(h, batch_idx, output_subdir):
            logger.info("Steal: batch %d already done, skipping", batch_idx)
            continue

        # Atomic steal claim
        if not _claim_batch_for_stealing(warc_dir, batch_idx):
            logger.info("Steal: lost claim on batch %d, skipping", batch_idx)
            continue

        # Process the batch
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(records))
        batch = records[batch_start:batch_end]

        logger.info(
            "Steal: batch %d/%d (%d records)...",
            batch_idx + 1,
            num_batches,
            len(batch),
        )

        if pipeline is not None:
            output_records, kept, filtered, token_stats, profile = _process_batch_pipeline(
                batch, llm, tokenizer, pipeline
            )
        else:
            output_records, kept, filtered, token_stats = _process_batch(
                batch, llm, sampling_params, tokenizer, template, system_message
            )
            profile = None
        total_kept += kept
        total_filtered += filtered

        batch_path = _batch_output_path(warc_dir, batch_idx)
        _write_batch_output(batch_path, output_records)
        _write_token_stats(warc_dir, batch_idx, token_stats)
        if profile is not None:
            _write_batch_profile(warc_dir, batch_idx, profile)
        batches_stolen += 1

        logger.info(
            "Steal: batch %d/%d done (kept=%d, filtered=%d) -> %s",
            batch_idx + 1,
            num_batches,
            kept,
            filtered,
            batch_path,
        )

    # Exit check: are ALL batches now complete? If so, write _done.
    all_completed = _find_completed_batches_all_regions(h, output_subdir)
    if len(all_completed) >= num_batches:
        final_kept = _count_records_in_all_batches(h, output_subdir)
        final_filtered = len(records) - final_kept
        stats = {
            "warc": warc_path,
            "total_records": len(records),
            "total_kept": final_kept,
            "total_filtered": final_filtered,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "completed_by": "stealer",
        }
        _write_done_marker(warc_dir, stats)
        _register_completed_warc(h, registry_prefix)
        logger.info("Steal: WARC complete! %s (kept=%d, filtered=%d)", warc_path, final_kept, final_filtered)

    return {
        "warc": warc_path,
        "status": "steal_done",
        "batches_stolen": batches_stolen,
        "total_kept": total_kept,
        "total_filtered": total_filtered,
    }


def _normalize_record_id(raw_id: str) -> str:
    """Strip <urn:uuid:...> wrapper → bare UUID for DCLM joins."""
    return raw_id.strip("<>").removeprefix("urn:uuid:")


def _extract_snapshot(warc_path: str) -> str:
    """Extract CC-MAIN-YYYY-WW snapshot from a WARC path for FineWeb-Edu joins."""
    m = re.search(r"CC-MAIN-\d{4}-\d{2}", warc_path)
    return m.group(0) if m else ""


def _clean_text(raw_text: str) -> str:
    text = _THINK_RE.sub("", raw_text)
    # Handle unclosed <think> tags (model didn't output </think>):
    # strip everything from <think> onward
    think_pos = text.find("<think>")
    if think_pos != -1:
        text = text[:think_pos]
    text = _FIELD_MARKERS_RE.sub("", text)
    return text.strip()


def _filter_by_length(records: list[dict], max_tokens: int) -> list[dict]:
    """Character-only length filter (no tokenizer calls — instant)."""
    max_chars = max_tokens * 6
    return [r for r in records if len(r.get("html", "")) <= max_chars]


def _batch_output_path(warc_dir: str, batch_idx: int) -> str:
    return f"{warc_dir}/batch_{batch_idx:04d}.jsonl.gz"


def _done_marker_path(warc_dir: str) -> str:
    return f"{warc_dir}/_done"


def _find_completed_batches_in_dir(warc_dir: str) -> set[int]:
    """Scan a single GCS directory for completed batch files.

    Uses a broad glob and explicit suffix check because GCS/fsspec glob can
    match extensionless ``batch_NNNN`` marker files against ``batch_*.jsonl.gz``.
    """
    try:
        files = fsspec.filesystem("gcs").glob(f"{warc_dir.replace('gs://', '')}/batch_*")
    except Exception:
        return set()
    completed = set()
    for f in files:
        basename = os.path.basename(f)
        if not basename.endswith(".jsonl.gz"):
            continue
        try:
            idx = int(basename.split("_")[1].split(".")[0])
            completed.add(idx)
        except (IndexError, ValueError):
            continue
    return completed


def _find_completed_batches_all_regions(warc_hash: str, output_subdir: str) -> set[int]:
    """Scan ALL regional buckets for completed batch files.

    Enables cross-region resume: Job A writes batches 0-20 in us-central1,
    gets preempted. Job B picks up in eu-west4 and skips batches 0-20.
    """
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    completed = set()
    for bucket in REGION_TO_DATA_BUCKET.values():
        warc_dir = f"gs://{bucket}/{output_subdir}/data-{warc_hash}"
        completed |= _find_completed_batches_in_dir(warc_dir)
    return completed


def _is_warc_done(warc_dir: str) -> bool:
    """Check if the _done marker exists for this WARC."""
    try:
        return fsspec.filesystem("gcs").exists(warc_dir.replace("gs://", "") + "/_done")
    except Exception:
        return False


def _is_warc_done_any_region(warc_hash: str, output_subdir: str) -> bool:
    """Check if a WARC is done in ANY regional bucket."""
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    gcs = fsspec.filesystem("gcs")
    for bucket in REGION_TO_DATA_BUCKET.values():
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/_done"
        try:
            if gcs.exists(path):
                return True
        except Exception:
            continue
    return False


def _is_warc_claimed_any_region(warc_hash: str, output_subdir: str, stale_hours: float = 3.0) -> bool:
    """Check if a WARC is claimed (in-progress) by another job in any region.

    A claim is a ``_claimed`` file in the WARC output dir. Claims older than
    ``stale_hours`` are ignored (the claiming job probably died without
    finishing or releasing the claim).
    """
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    gcs = fsspec.filesystem("gcs")
    now = time.time()
    stale_regions = []
    for bucket in REGION_TO_DATA_BUCKET.values():
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/_claimed"
        try:
            info = gcs.info(path)
            # Check if the claim is stale
            mtime = info.get("updated") or info.get("timeCreated")
            if mtime is not None:
                import datetime

                if isinstance(mtime, str):
                    mtime = datetime.datetime.fromisoformat(mtime.replace("Z", "+00:00"))
                age_hours = (now - mtime.timestamp()) / 3600
                if age_hours > stale_hours:
                    stale_regions.append((bucket, age_hours))
                    continue
            # Fresh claim found — log which region has the active worker
            region = bucket.replace("marin-", "")
            logger.info("Active claim on %s in %s (fresh), skipping", warc_hash, region)
            if stale_regions:
                for sr_bucket, sr_age in stale_regions:
                    logger.info("  (also has stale claim in %s, %.1fh old)", sr_bucket.replace("marin-", ""), sr_age)
            return True
        except FileNotFoundError:
            continue
        except Exception:
            continue
    if stale_regions:
        logger.info(
            "Only stale claims on %s (%s) — eligible for reclaim",
            warc_hash,
            ", ".join(f"{b.replace('marin-', '')}={h:.1f}h" for b, h in stale_regions),
        )
    return False


def _claim_warc_atomic(warc_dir: str, stale_hours: float = 3.0) -> bool:
    """Atomically claim a WARC. Returns True if we won the claim, False if someone else did.

    Uses GCS ``if_generation_match=0`` precondition: the write only succeeds if the
    object does NOT already exist. First writer wins, all others get 412 Precondition Failed.

    If the file already exists but the claim is stale (older than ``stale_hours``),
    reclaims it atomically using ``if_generation_match=<current_generation>`` so that
    only one reclaimer wins.
    """
    import datetime

    claim_path = f"{warc_dir}/_claimed"
    # Parse bucket and blob path from gs:// URL
    if not claim_path.startswith("gs://"):
        # Local filesystem fallback (for testing)
        if os.path.exists(claim_path):
            return False
        os.makedirs(os.path.dirname(claim_path), exist_ok=True)
        with open(claim_path, "w") as f:
            f.write(json.dumps({"pid": os.getpid(), "time": time.time()}))
        return True

    parts = claim_path.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]

    bucket = _gcs_client().bucket(bucket_name)
    blob = bucket.blob(blob_path)

    now = time.time()
    claim_data = json.dumps(
        {
            "pid": os.getpid(),
            "time": now,
            "created_at": now,
            "host": os.environ.get("HOSTNAME", "unknown"),
        }
    )
    try:
        blob.upload_from_string(claim_data, if_generation_match=0)
        return True  # We won the claim — no prior file existed
    except Exception as e:
        if "conditionNotMet" not in str(e) and "412" not in str(e):
            raise  # Unexpected error

    # File already exists. Check if the existing claim is stale.
    try:
        blob.reload()
        mtime = blob.updated or blob.time_created
        if mtime is not None:
            if isinstance(mtime, str):
                mtime = datetime.datetime.fromisoformat(mtime.replace("Z", "+00:00"))
            age_hours = (time.time() - mtime.timestamp()) / 3600
            if age_hours > stale_hours:
                # Stale claim — reclaim atomically using current generation.
                # Only one reclaimer wins; others get 412.
                logger.info(
                    "Reclaiming stale local claim in %s (%.1fh old, generation=%d)",
                    warc_dir,
                    age_hours,
                    blob.generation,
                )
                try:
                    blob.upload_from_string(claim_data, if_generation_match=blob.generation)
                    return True  # We reclaimed the stale claim
                except Exception as e2:
                    if "conditionNotMet" in str(e2) or "412" in str(e2):
                        return False  # Another reclaimer beat us
                    raise
    except Exception:
        pass  # Can't check staleness — conservatively decline

    return False  # Non-stale claim exists, someone else owns it


def _refresh_claim(warc_dir: str) -> None:
    """Refresh an existing claim's timestamp (for stale detection).

    Preserves ``created_at`` so cross-region tiebreaking still works after refresh.
    """
    claim_path = f"{warc_dir}/_claimed"
    created_at = time.time()
    try:
        with fsspec.open(claim_path, "r") as f:
            existing = json.loads(f.read())
            created_at = existing.get("created_at", created_at)
    except Exception:
        pass
    with fsspec.open(claim_path, "w") as f:
        f.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "time": time.time(),
                    "created_at": created_at,
                    "host": os.environ.get("HOSTNAME", "unknown"),
                }
            )
        )


def _list_fresh_claims(claim_root: str, stale_hours: float) -> set[str]:
    """WARC hashes with a FRESH ``_claimed`` object under ``claim_root``, from one listing.

    Claim layout is ``{claim_root}/data-{hash}/_claimed`` (see ``_claim_warc_atomic``).
    A fresh claim cannot be won, so callers iterating a manifest can skip the
    per-WARC conditional-write probe (one write RPC + one metadata GET against the
    central bucket, per WARC, per pass, per worker) for every hash in this set.
    Stale or missing claims still go through ``_claim_warc_atomic``, which remains
    the only authority on winning. ``fs.find`` bypasses gcsfs's process-lifetime
    dircache, so the result is always current. Returns empty set on error, which
    degrades to the old probe-everything behavior.
    """
    import datetime

    fs = fsspec.filesystem("gcs")
    now = time.time()
    fresh: set[str] = set()
    try:
        entries = fs.find(claim_root.replace("gs://", ""), detail=True)
    except Exception as e:
        logger.warning("Failed to list claims under %s: %s", claim_root, e)
        return fresh
    for path, info in entries.items():
        parts = path.rstrip("/").rsplit("/", 2)
        if len(parts) < 3 or parts[-1] != "_claimed" or not parts[-2].startswith("data-"):
            continue
        mtime = info.get("updated") or info.get("timeCreated") or info.get("mtime")
        if isinstance(mtime, str):
            try:
                mtime = datetime.datetime.fromisoformat(mtime.replace("Z", "+00:00"))
            except ValueError:
                mtime = None
        age = now - mtime.timestamp() if hasattr(mtime, "timestamp") else None
        # Unknown age counts as fresh — at worst a stale reclaim waits one more pass.
        if age is None or age < stale_hours * 3600:
            fresh.add(parts[-2][5:])
    return fresh


def _should_yield_to_older_claim(
    warc_hash: str, output_subdir: str, my_created_at: float, stale_hours: float = 3.0
) -> tuple[bool, str | None]:
    """Check if another region has a fresh claim we should yield to.

    Returns (should_yield, winning_region). Yields when:
      1. Another region has a fresh claim with an OLDER ``created_at`` (new vs new race)
      2. Another region has a fresh legacy claim (no ``created_at`` field) — we
         conservatively assume the legacy worker started first to avoid duplicates
         during mixed-fleet transition. Worst case: brief delay until 3h staleness.

    Stale claims (>stale_hours old) are always ignored.
    """
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    now = time.time()
    for bucket in REGION_TO_DATA_BUCKET.values():
        # Skip our own region — same-region collisions are handled by atomic claim
        path = f"{bucket}/{output_subdir}/data-{warc_hash}/_claimed"
        try:
            with fsspec.open(f"gs://{path}", "r") as f:
                claim = json.loads(f.read())
        except Exception:
            continue
        last_refresh = claim.get("time", 0)
        if now - last_refresh > stale_hours * 3600:
            continue  # stale, ignore

        # Skip our own claim (same host wrote it)
        my_host = os.environ.get("HOSTNAME", "unknown")
        if claim.get("host") == my_host:
            continue

        other_created = claim.get("created_at")
        if other_created is None:
            # Legacy claim — assume it started first, yield to it
            return True, bucket.replace("marin-", "") + " (legacy)"
        if other_created < my_created_at:
            return True, bucket.replace("marin-", "")
    return False, None


def _write_batch_output(path: str, records: list[dict]) -> None:
    """Write a batch of records to a gzipped JSONL file on GCS, plus a .count sidecar."""
    with fsspec.open(path, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for rec in records:
                gz.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Write tiny sidecar with the record count for cheap aggregation later
    count_path = path.removesuffix(".jsonl.gz") + ".count"
    with fsspec.open(count_path, "w") as f:
        f.write(str(len(records)))


def _write_token_stats(warc_dir: str, batch_idx: int, token_stats: list[dict]) -> None:
    """Write per-record token stats as a compact JSONL sidecar for FLOP accounting.

    Each line: {"input_tokens": N, "output_tokens": M, "status": "kept"|"filtered_short"|"filtered_pattern"|"empty"}
    File: batch_NNNN.tokens.gz (~5-10KB per batch of 500 records)
    """
    path = f"{warc_dir}/batch_{batch_idx:04d}.tokens.gz"
    try:
        with fsspec.open(path, "wb") as f:
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                for stat in token_stats:
                    gz.write(json.dumps(stat) + "\n")
    except Exception as e:
        logger.warning("Failed to write token stats for batch %d: %s", batch_idx, e)


def _count_records_in_all_batches(warc_hash: str, output_subdir: str) -> int:
    """Sum record counts across all batches for a WARC, in ALL regions.

    Reads tiny ``.count`` sidecar files written next to each batch (one int each).
    No fallback to reading batch files — if a sidecar is missing (legacy batches
    written before the fix), it's simply not counted. Avoids expensive cross-region
    decompression. The count will be accurate once all legacy batches drain.
    """
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    fs = fsspec.filesystem("gcs")
    total = 0
    for bucket in REGION_TO_DATA_BUCKET.values():
        try:
            count_paths = fs.glob(f"{bucket}/{output_subdir}/data-{warc_hash}/batch_*.count")
        except Exception:
            continue
        for path in count_paths:
            try:
                with fsspec.open(f"gs://{path}", "r") as f:
                    total += int(f.read().strip())
            except Exception as e:
                logger.warning("Failed to read sidecar %s: %s", path, e)
    return total


def _write_done_marker(warc_dir: str, stats: dict) -> None:
    """Write the _done marker with summary stats."""
    with fsspec.open(_done_marker_path(warc_dir), "w") as f:
        json.dump(stats, f, indent=2)


def _process_batch(
    batch: list[dict],
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    template: str,
    system_message: str,
) -> tuple[list[dict], int, int, list[dict]]:
    """Process a single batch through vLLM.

    Returns:
        (output_records, kept, filtered, token_stats)
        token_stats is a list of per-record dicts with input/output token counts
        and status (kept/filtered/empty), for FLOP accounting.
    """
    # vLLM moved TokensPrompt's canonical location across versions; try a few.
    try:
        from vllm.inputs import TokensPrompt
    except ImportError:
        try:
            from vllm.inputs.data import TokensPrompt
        except ImportError:
            from vllm import TokensPrompt

    # Format prompts and track input token counts
    prompts = []
    input_token_counts: dict[int, int] = {}
    for i, record in enumerate(batch):
        html = record.get("html", "")
        tokens = tokenizer.encode(html)
        if len(tokens) > MAX_DOC_TOKENS:
            tokens = tokens[:MAX_DOC_TOKENS]
            html = tokenizer.decode(tokens)

        text = template.format(example=html)
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": text})
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_text)
        input_token_counts[i] = len(prompt_ids)
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))

    # Filter empty prompts
    valid = [(i, p) for i, p in enumerate(prompts) if p["prompt_token_ids"]]
    if not valid:
        token_stats = [
            {"input_tokens": input_token_counts.get(i, 0), "thinking_tokens": 0, "response_tokens": 0, "status": "empty"}
            for i in range(len(batch))
        ]
        return [], 0, len(batch), token_stats

    valid_indices, valid_prompts = zip(*valid, strict=True)

    # Resolve <think> and </think> token IDs for splitting thinking vs response.
    # For Qwen3 these are typically single tokens each.
    think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
    think_start_ids = tokenizer.encode("<think>", add_special_tokens=False)

    def _split_thinking_response(token_ids: list[int]) -> tuple[int, int, bool]:
        """Split token_ids into (thinking_tokens, response_tokens, is_thinking_overflow).

        Finds the last occurrence of the </think> token(s). Everything up to
        and including it is thinking; everything after is response. If not
        found but <think> IS present, all tokens are thinking overflow (model
        started thinking but hit max_tokens before closing the tag).
        """
        has_think_start = len(token_ids) > 0 and len(think_start_ids) == 1 and token_ids[0] == think_start_ids[0]

        if len(think_end_ids) == 1:
            end_id = think_end_ids[0]
            last_pos = -1
            for j in range(len(token_ids) - 1, -1, -1):
                if token_ids[j] == end_id:
                    last_pos = j
                    break
            if last_pos >= 0:
                return last_pos + 1, len(token_ids) - last_pos - 1, False
            # No </think> found
            if has_think_start:
                # Started thinking but never closed — all tokens are thinking overflow
                return len(token_ids), 0, True
            return 0, len(token_ids), False
        else:
            needle_len = len(think_end_ids)
            for j in range(len(token_ids) - needle_len, -1, -1):
                if token_ids[j : j + needle_len] == think_end_ids:
                    end_pos = j + needle_len
                    return end_pos, len(token_ids) - end_pos, False
            if has_think_start:
                return len(token_ids), 0, True
            return 0, len(token_ids), False

    # Generate
    t0 = time.monotonic()
    outputs = llm.generate(list(valid_prompts), sampling_params)
    elapsed = time.monotonic() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    logger.info(
        "Generated %d prompts in %.1fs (%.1f tok/s)",
        len(valid_prompts),
        elapsed,
        total_tokens / max(elapsed, 0.01),
    )

    # Map outputs back with split token counts
    output_map: dict[int, str] = {}
    output_thinking: dict[int, int] = {}
    output_response: dict[int, int] = {}
    output_overflow: dict[int, bool] = {}
    for idx, out in zip(valid_indices, outputs, strict=True):
        output_map[idx] = " ".join(o.text for o in out.outputs)
        token_ids = list(out.outputs[0].token_ids)
        think, resp, overflow = _split_thinking_response(token_ids)
        output_thinking[idx] = think
        output_response[idx] = resp
        output_overflow[idx] = overflow

    output_records = []
    token_stats = []
    kept = 0
    filtered = 0
    for i, record in enumerate(batch):
        raw = output_map.get(i, "")
        in_toks = input_token_counts.get(i, 0)
        think_toks = output_thinking.get(i, 0)
        resp_toks = output_response.get(i, 0)
        is_overflow = output_overflow.get(i, False)
        cleaned = _clean_text(raw)

        stat = {"input_tokens": in_toks, "thinking_tokens": think_toks, "response_tokens": resp_toks}

        if len(cleaned) < MIN_OUTPUT_CHARS:
            filtered += 1
            if is_overflow:
                # Distinguish: did we hit max_tokens or the full context window?
                if resp_toks == 0 and think_toks + in_toks >= MAX_CONTEXT_TOKENS - 10:
                    token_stats.append({**stat, "status": "thinking_overflow_context"})
                else:
                    token_stats.append({**stat, "status": "thinking_overflow_max_tokens"})
            else:
                token_stats.append({**stat, "status": "filtered_short"})
            continue
        if any(p.search(cleaned) for p in _FILTER_PATTERNS):
            filtered += 1
            token_stats.append({**stat, "status": "filtered_pattern"})
            continue

        kept += 1
        token_stats.append({**stat, "status": "kept"})
        warc_file = record.get("metadata", {}).get("warc_file", "")
        output_records.append(
            {
                "text": cleaned,
                "generated_text": raw,
                "url": record.get("url", ""),
                # Join keys for DCLM / Nemotron / FineWeb-Edu filtering
                "warc_record_id": _normalize_record_id(record.get("id", "")),
                "warc_file": warc_file,
                "snapshot": _extract_snapshot(warc_file),
            }
        )

    return output_records, kept, filtered, token_stats


def _process_batch_pipeline(
    batch: list[dict], llm: Any, tokenizer: Any, pipeline: Any
) -> tuple[list[dict], int, int, list[dict], dict]:
    """Process a GROUP of records through a multi-call extraction pipeline.

    Mirrors _process_batch's return contract (output_records, kept, filtered,
    token_stats) plus a per-group timing/counts profile. Only KEEP docs are
    written, so consolidate/dedup/tokenize see the same kept-doc schema as the
    single-call path; drops/errors are counted, not persisted.
    """
    from experiments.baseline_collection.pipelines.offline_chat import make_vllm_chat_fn
    from experiments.baseline_collection.pipelines.one_call import run_one_call
    from experiments.baseline_collection.pipelines.pipeline_specs import OneCallPipeline, make_formatter
    from experiments.baseline_collection.pipelines.scheduler_base import DECISION_ERROR, DECISION_KEEP
    from experiments.baseline_collection.pipelines.token_budget import QwenTokenCodec
    from experiments.baseline_collection.pipelines.two_stage import run_two_stage

    codec = QwenTokenCodec(tokenizer)
    formatter = make_formatter()
    chat_fn = make_vllm_chat_fn(llm)
    if isinstance(pipeline, OneCallPipeline):
        results, profile = run_one_call(batch, chat_fn, codec, formatter, pipeline)
    else:
        results, profile = run_two_stage(batch, chat_fn, codec, formatter, pipeline)

    output_records: list[dict] = []
    token_stats: list[dict] = []
    kept = 0
    filtered = 0
    for record, res in zip(batch, results, strict=True):
        token_stats.append(
            {"status": res.decision, "completion_tokens": res.completion_tokens, "num_chunks": res.num_chunks}
        )
        if res.decision == DECISION_KEEP:
            kept += 1
            warc_file = record.get("metadata", {}).get("warc_file", "")
            output_records.append(
                {
                    "text": res.text,
                    "url": record.get("url", ""),
                    # Join keys for DCLM / Nemotron / FineWeb-Edu filtering
                    "warc_record_id": _normalize_record_id(record.get("id", "")),
                    "warc_file": warc_file,
                    "snapshot": _extract_snapshot(warc_file),
                    "pipeline_id": pipeline.pipeline_id,
                    "num_chunks": res.num_chunks,
                }
            )
        elif res.decision == DECISION_ERROR:
            # Errors are NOT drops: record identifiers in the timing sidecar so
            # they are auditable and re-runnable, instead of vanishing into the
            # filtered count.
            filtered += 1
            profile.setdefault("error_docs", []).append(
                {
                    "warc_record_id": _normalize_record_id(record.get("id", "")),
                    "url": record.get("url", ""),
                    "error": res.error,
                }
            )
        else:
            filtered += 1
    return output_records, kept, filtered, token_stats, profile


def _write_batch_profile(warc_dir: str, batch_idx: int, profile: dict) -> None:
    """Write the per-group pipeline timing/counts profile as a JSON sidecar
    (batch_NNNN.timing.json), at the same checkpoint boundary as the batch output."""
    path = f"{warc_dir}/batch_{batch_idx:04d}.timing.json"
    try:
        with fsspec.open(path, "w") as f:
            json.dump(profile, f)
    except Exception as e:
        logger.warning("Failed to write timing profile for batch %d: %s", batch_idx, e)


def _process_warc(
    warc_path: str,
    output_dir: str,
    output_subdir: str,
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    template: str,
    system_message: str,
    registry_prefix: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    pipeline: Any = None,
) -> dict:
    """Download one WARC, extract via LLM with per-batch checkpointing.

    When ``pipeline`` is set, each batch (=checkpoint group) runs the multi-call
    pipeline instead of the single greedy call; the checkpoint/claim/resume/steal
    machinery is unchanged, so a preemption still redoes at most one group.
    """
    h = _warc_path_hash(warc_path)
    warc_dir = f"{output_dir}/data-{h}"

    # Skip if fully done (check local region first, then all regions)
    if _is_warc_done(warc_dir) or _is_warc_done_any_region(h, output_subdir):
        return {"warc": warc_path, "status": "skipped"}

    # Skip if another job claimed it (unless the claim is stale)
    if _is_warc_claimed_any_region(h, output_subdir):
        logger.info("Skipping %s (claimed by another job)", warc_path)
        return {"warc": warc_path, "status": "claimed"}

    # Atomic claim: only one worker can win. if_generation_match=0 on GCS
    # ensures first writer wins, all others get 412 Precondition Failed.
    # NOTE: this only protects against same-region collisions. Cross-region
    # races are handled below via _should_yield_to_older_claim between batches.
    if not _claim_warc_atomic(warc_dir):
        logger.info("Skipping %s (lost atomic claim race)", warc_path)
        return {"warc": warc_path, "status": "claimed"}
    # Capture our own created_at for cross-region tiebreaking
    my_created_at = time.time()
    try:
        with fsspec.open(f"{warc_dir}/_claimed", "r") as f:
            my_created_at = json.loads(f.read()).get("created_at", my_created_at)
    except Exception:
        pass
    logger.info("Claimed %s -> %s", warc_path, warc_dir)

    # Download
    records = _download_one_warc(warc_path)
    if not records:
        logger.warning("No HTML records from %s", warc_path)
        _write_done_marker(warc_dir, {"status": "empty", "records": 0})
        _register_completed_warc(h, registry_prefix)
        return {"warc": warc_path, "status": "empty", "records": 0}

    logger.info("Downloaded %d records from %s", len(records), warc_path)

    # Filter by length (instant). Single-greedy-call path ONLY: the multi-call
    # pipelines chunk long docs by design and were certified WITH the long tail.
    if pipeline is None:
        records = _filter_by_length(records, MAX_DOC_TOKENS)
        logger.info("%d records after length filter", len(records))
    if not records:
        _write_done_marker(warc_dir, {"status": "all_filtered", "records": 0})
        _register_completed_warc(h, registry_prefix)
        return {"warc": warc_path, "status": "all_filtered", "records": 0}

    # Check which batches are already done (scan ALL regions for cross-region resume)
    completed_batches = _find_completed_batches_all_regions(h, output_subdir)
    num_batches = (len(records) + batch_size - 1) // batch_size

    if completed_batches:
        logger.info(
            "Resuming: %d/%d batches already complete (possibly across regions), skipping them",
            len(completed_batches),
            num_batches,
        )

    # Process batches with per-batch checkpointing
    total_kept = 0
    total_filtered = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(records))
        batch = records[batch_start:batch_end]

        # Skip if this batch is already done (from initial scan)
        if batch_idx in completed_batches:
            logger.info("Batch %d/%d: already done, skipping", batch_idx + 1, num_batches)
            continue

        # Live check: a stealer may have written this batch since our initial scan
        if _batch_exists_any_region(h, batch_idx, output_subdir):
            logger.info("Batch %d/%d: completed by another worker, skipping", batch_idx + 1, num_batches)
            continue

        logger.info(
            "Batch %d/%d (%d records, offset %d-%d)...", batch_idx + 1, num_batches, len(batch), batch_start, batch_end
        )

        # Process batch (multi-call pipeline when configured, else single greedy call)
        if pipeline is not None:
            output_records, kept, filtered, token_stats, profile = _process_batch_pipeline(
                batch, llm, tokenizer, pipeline
            )
        else:
            output_records, kept, filtered, token_stats = _process_batch(
                batch, llm, sampling_params, tokenizer, template, system_message
            )
            profile = None
        total_kept += kept
        total_filtered += filtered

        # Write batch output IMMEDIATELY to GCS (checkpoint)
        batch_path = _batch_output_path(warc_dir, batch_idx)
        _write_batch_output(batch_path, output_records)
        _write_token_stats(warc_dir, batch_idx, token_stats)
        if profile is not None:
            _write_batch_profile(warc_dir, batch_idx, profile)

        # Refresh claim so other jobs know we're still alive (3h stale timeout)
        _refresh_claim(warc_dir)

        # Check if we still own the claim in our region. If another worker
        # overwrote it, we lost the race and should stop wasting compute.
        try:
            claim_path = f"{warc_dir}/_claimed"
            with fsspec.open(claim_path, "r") as f:
                claim_data = json.loads(f.read())
            my_host = os.environ.get("HOSTNAME", "unknown")
            if claim_data.get("host") != my_host:
                logger.warning(
                    "Lost claim on %s to %s (we are %s). Abandoning WARC.",
                    warc_path,
                    claim_data.get("host"),
                    my_host,
                )
                return {"warc": warc_path, "status": "lost_claim", "batches_completed": batch_idx + 1}
        except Exception:
            pass  # If we can't read the claim, keep going

        # Cross-region race detection: if another region has a fresh claim with
        # an OLDER created_at, that worker started first and wins. We yield to
        # avoid duplicate work. Costs ~6 list+small reads per batch.
        try:
            should_yield, winner_region = _should_yield_to_older_claim(h, output_subdir, my_created_at)
            if should_yield:
                logger.warning(
                    "Yielding %s to older claim in %s (we started at %.0f). Stopping after batch %d.",
                    warc_path,
                    winner_region,
                    my_created_at,
                    batch_idx + 1,
                )
                return {"warc": warc_path, "status": "yielded_to_older", "batches_completed": batch_idx + 1}
        except Exception as e:
            logger.warning("Cross-region claim check failed: %s", e)

        logger.info(
            "Batch %d/%d: kept=%d, filtered=%d -> %s",
            batch_idx + 1,
            num_batches,
            kept,
            filtered,
            batch_path,
        )

    # Exit check: verify ALL batches are complete (including any written by stealers).
    # This catches the case where stealers finished some batches we skipped via
    # the live check above — we want to write _done if everything's done.
    all_completed = _find_completed_batches_all_regions(h, output_subdir)
    if len(all_completed) < num_batches:
        logger.info(
            "WARC %s: %d/%d batches complete after our pass. Not writing _done yet.",
            warc_path,
            len(all_completed),
            num_batches,
        )
        return {"warc": warc_path, "status": "partial", "batches_completed": len(all_completed)}

    # All batches done — count actual records across all batch files (including
    # batches written by stealers or previous workers). Reads sidecar .count files.
    final_kept = _count_records_in_all_batches(h, output_subdir)
    final_filtered = len(records) - final_kept
    stats = {
        "warc": warc_path,
        "total_records": len(records),
        "total_kept": final_kept,
        "total_filtered": final_filtered,
        "this_run_kept": total_kept,
        "this_run_filtered": total_filtered,
        "num_batches": num_batches,
        "batch_size": batch_size,
    }
    _write_done_marker(warc_dir, stats)
    _register_completed_warc(h, registry_prefix)
    logger.info("WARC complete: %s (kept=%d, filtered=%d)", warc_path, final_kept, final_filtered)
    return {"warc": warc_path, "status": "done", **stats}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="GCS path to WARC manifest")
    parser.add_argument(
        "--spec",
        default=None,
        help=(
            "Extraction spec id (key in extraction_specs.SPECS). When set, overrides "
            "--output-subdir to documents/baseline_llm_extraction/{spec_id} and uses "
            "the spec's system_message + template instead of the legacy hardcoded prompt."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Output subdir. Without --spec, this is the legacy namespace "
            "(default: documents/baseline_llm_extraction_test). With --spec, leave "
            "unset to use the spec's canonical namespace; set it to redirect output + "
            "the skip-registry to a separate namespace while keeping the spec's prompt "
            "(e.g. benchmarking a different model on the same spec)."
        ),
    )
    parser.add_argument("--model", default=None, help="Model path (auto-resolves from region if not set)")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size (auto-detect from JAX)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Records per batch (smaller = finer checkpoints)"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle WARC order with this seed. Use different seeds per job to spread work.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index into manifest (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index into manifest (exclusive). Default: all.")
    parser.add_argument(
        "--pipeline",
        default=None,
        help=(
            "Multi-call pipeline id (key in pipelines.PIPELINES), e.g. llm_pipeline_v1 or "
            "llm_simple_v1. Mutually exclusive with --spec. Writes to "
            "documents/baseline_llm_extraction/{pipeline_id} with a per-group timing sidecar."
        ),
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=(
            "Docs per checkpoint group for --pipeline (defaults to --batch-size). Smaller = "
            "less redo on preemption, larger = fuller offline batches."
        ),
    )
    args = parser.parse_args()
    if args.pipeline and args.spec:
        parser.error("--pipeline and --spec are mutually exclusive")

    # Resolve spec-driven config: output subdir, prompts, and registry prefix.
    # Legacy mode (no --spec) keeps the hardcoded prompt below and writes to the
    # bare --output-subdir, registering completions to the unprefixed registry.
    spec_obj = None
    pipeline_obj = None
    if args.pipeline:
        from experiments.baseline_collection.pipelines.pipeline_specs import get_pipeline

        pipeline_obj = get_pipeline(args.pipeline)
        output_subdir = args.output_subdir or f"documents/baseline_llm_extraction/{pipeline_obj.pipeline_id}"
        registry_prefix = _registry_prefix_for(output_subdir)
        if args.group_size:
            args.batch_size = args.group_size
        logger.info(
            "Pipeline: %s (frozen cert %s), group-size=%d",
            pipeline_obj.pipeline_id,
            pipeline_obj.source_cert,
            args.batch_size,
        )
    elif args.spec:
        from experiments.baseline_collection.extraction_specs import LEGACY_SPEC_ID, get_spec

        spec_obj = get_spec(args.spec)
        if args.output_subdir is not None:
            # Explicit override: keep the spec's prompt but redirect output + the
            # skip-registry to a separate namespace. Used to benchmark a different
            # model on the same spec without skipping (the canonical namespace is
            # already complete) or polluting the canonical dataset.
            output_subdir = args.output_subdir
        elif spec_obj.spec_id == LEGACY_SPEC_ID:
            # The legacy spec maps to the unprefixed GCS path so it shares the
            # namespace with pre-registry data. All other specs nest under their id.
            output_subdir = "documents/baseline_llm_extraction"
        else:
            output_subdir = f"documents/baseline_llm_extraction/{spec_obj.spec_id}"
        registry_prefix = _registry_prefix_for(output_subdir)
        logger.info("Spec: %s — %s", spec_obj.spec_id, spec_obj.description or "(no description)")
    else:
        output_subdir = args.output_subdir or "documents/baseline_llm_extraction_test"
        registry_prefix = DEFAULT_COMPLETED_REGISTRY_PREFIX
    logger.info("Registry prefix: %s", registry_prefix)

    # Resolve output path from runtime region
    from rigging.filesystem import marin_prefix

    output_dir = f"{marin_prefix()}/{output_subdir}"
    logger.info("Output dir: %s", output_dir)

    # Resolve model — use local region's bucket to avoid cross-region egress
    if args.model:
        model_name = args.model
    else:
        model_name = f"{marin_prefix()}/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
    logger.info("Model: %s", model_name)

    # Set JAX cache env
    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Detect TPU
    import jax

    devices = jax.devices()
    logger.info("JAX devices (%d): %s", len(devices), devices)

    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    logger.info("Tensor parallel size: %d", tp)

    # Init vLLM engine
    from vllm import LLM, SamplingParams

    t0 = time.monotonic()
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS)
    tokenizer = llm.get_tokenizer()
    logger.info("Engine loaded in %.1fs", time.monotonic() - t0)

    # Extraction prompt: from spec registry when --spec is set, else legacy hardcoded.
    # A --pipeline run builds its own messages from the vendored SFT templates and
    # ignores template/system_message entirely.
    if pipeline_obj is not None:
        system_message = ""
        template = ""
    elif spec_obj is not None:
        system_message = spec_obj.system_message
        template = spec_obj.extraction_template
    else:
        _legacy_spec_rules = (
            "Extract the content from this HTML page as clean text. Follow all rules below.\n\n"
            "1. Extract the full page content in reading order. Keep all explanatory text, "
            "discussion, and comments that add substantive information. Begin your output "
            "directly with the page content.\n"
            "2. Remove boilerplate: navigation bars, footers, sidebars, ads, share buttons, "
            "related links, breadcrumbs, cookie banners, and user interface elements. "
            "Do not output framework markers or metadata tags.\n"
            "3. Preserve all technical content exactly as written: code, math notation, "
            "formulas, tables, and data. Preserve the original line breaks and structure "
            "of code blocks.\n"
            "4. Decode all HTML entities to their plain characters (e.g. &amp; to &, "
            "&lt; to <, &gt; to >, &#8217; to ', &#8211; to \u2013). Remove any raw HTML tags.\n"
            "5. Every sentence in your output must come from the source page. Do not add, "
            "invent, or embellish content.\n"
            "6. For pages with multiple authors or speakers (forums, reviews, comments), "
            "preserve who said what. Include usernames or speaker labels so contributions "
            "remain distinguishable.\n"
            "7. If the text contains content spinner templates like {{word1|word2|word3}}, "
            "pick the first option and output clean text. If the majority of the page is "
            "spinner templates, output [NO_USEFUL_CONTENT] instead.\n"
            "8. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:\n"
            '   - Login, signup, paywall, registration wall, or "you must sign up to view" page\n'
            "   - Error page, empty page, or cookie/captcha wall\n"
            "   - Terms of use, privacy policy, or legal boilerplate page\n"
            "   - User profile page with no substantive content\n"
            "   - Page whose content has been removed, moved, or is no longer available\n"
            "   - Image gallery, photo album listing, or media archive without articles\n"
            "   - The text is incoherent gibberish or garbled encoding throughout\n"
            "   - The page has under ~50 words of substantive content after removing boilerplate\n"
            "   However, for index pages or directory listings, only filter if they contain "
            "nothing but links and titles. If an index page includes real text like discussion "
            "snippets or descriptions, extract it."
        )
        system_message = (
            "Your input fields are:\n1. `html` (str): \n2. `extraction_spec` (str):\n"
            "Your output fields are:\n1. `text` (str):\n"
            "All interactions will be structured in the following way, "
            "with the appropriate values filled in.\n\n"
            "[[ ## html ## ]]\n{html}\n\n[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
            "[[ ## text ## ]]\n{text}\n\n[[ ## completed ## ]]\n"
            "In adhering to this structure, your objective is: \n"
            "        Extract the main content text from a given HTML document."
        )
        template = (
            "[[ ## html ## ]]\n{example}\n\n"
            "[[ ## extraction_spec ## ]]\n" + _legacy_spec_rules + "\n\n"
            "Respond with the corresponding output fields, "
            "starting with the field `[[ ## text ## ]]`, "
            "and then ending with the marker for `[[ ## completed ## ]]`."
        )

    # Load manifest, optionally slice and shuffle
    import random

    warcs = _load_manifest(args.manifest)
    warcs = warcs[args.start : args.end]  # default: all
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(warcs)
        logger.info(
            "Manifest: %d WARCs [%d:%s], shuffled with seed %d", len(warcs), args.start, args.end, args.shuffle_seed
        )
    else:
        logger.info("Manifest: %d WARCs [%d:%s], sequential order", len(warcs), args.start, args.end)

    # On multi-host slices, each VM gets a different IRIS_TASK_ID.
    # Format is like "/user/job-name/child/TaskIdx:Attempt" — extract the task index.
    # Rotate the manifest so each VM starts at a different position, avoiding
    # claim races between VMs on the same slice.
    raw_task_id = os.environ.get("IRIS_TASK_ID", "0")
    try:
        task_id = int(raw_task_id)
    except ValueError:
        # Parse structured ID: take the last path component, then the part before ':'
        last_part = raw_task_id.rsplit("/", 1)[-1]
        task_id = int(last_part.split(":")[0])
        logger.info("Parsed IRIS_TASK_ID=%s -> task_id=%d", raw_task_id, task_id)
    if task_id > 0 and len(warcs) > 0:
        offset = (task_id * len(warcs)) // max(task_id + 1, 8)
        warcs = warcs[offset:] + warcs[:offset]
        logger.info("Rotated manifest by %d for IRIS_TASK_ID=%d", offset, task_id)

    logger.info("batch_size=%d, output_subdir=%s", args.batch_size, output_subdir)

    # -----------------------------------------------------------------------
    # Main processing loop with steal mode
    # -----------------------------------------------------------------------
    REGISTRY_REFRESH_INTERVAL = 10
    # Steal mode thresholds
    # Steal mode patience tiers: more remaining WARCs = more patience before stealing.
    # Tier 1: < 500 remaining  → try 10 times (endgame, most are claimed)
    # Tier 2: < 1000 remaining → try 50 times (getting close)
    # Tier 3: >= 1000 remaining → try 100 times (plenty of unclaimed WARCs, search harder)
    STEAL_TIER1_THRESHOLD = 500
    STEAL_TIER2_THRESHOLD = 1000
    STEAL_PATIENCE_TIER1 = 10
    STEAL_PATIENCE_TIER2 = 50
    STEAL_PATIENCE_TIER3 = 100
    # Negative-result caches. A WARC observed fresh-claimed by another worker
    # costs ~13 cross-region GETs to re-discover, and a WARC with nothing to
    # steal costs a full ~1GB download to re-discover — neither answer changes
    # for a while, so remember it. TTLs sit well under the 3h claim-stale
    # window, so reclaiming a dead worker's WARC is delayed by at most the TTL.
    CLAIMED_RECHECK_SECONDS = 600.0
    STEAL_RECHECK_SECONDS = 900.0
    # A steal cycle samples a few candidates instead of sweeping the whole
    # manifest (each unsuccessful candidate = one full WARC download), and
    # unsuccessful cycles sleep instead of immediately re-downloading.
    STEAL_CANDIDATES_PER_CYCLE = 8
    STEAL_MAX_EMPTY_CYCLES = 5
    STEAL_CYCLE_SLEEP_SECONDS = 60.0

    import random as random_module

    stats: list[dict] = []
    completed_registry = _load_completed_registry(registry_prefix)
    total_warcs = len(warcs)
    steal_mode = False
    recently_claimed: dict[str, float] = {}
    steal_exhausted: dict[str, float] = {}
    no_steal_cycles = 0

    def _get_remaining():
        return [w for w in warcs if _warc_path_hash(w) not in completed_registry]

    remaining = _get_remaining()
    logger.info(
        "Registry filter: %d/%d WARCs remaining (%d already completed)",
        len(remaining),
        total_warcs,
        total_warcs - len(remaining),
    )

    consecutive_failures = 0  # consecutive WARCs we couldn't claim (claimed/skipped)
    warcs_since_refresh = 0
    warc_idx = 0

    while warc_idx < len(remaining) or steal_mode:
        # Refresh registry periodically
        warcs_since_refresh += 1
        if warcs_since_refresh >= REGISTRY_REFRESH_INTERVAL:
            completed_registry = _load_completed_registry(registry_prefix)
            old_remaining = len(remaining)
            remaining = _get_remaining()
            pruned = old_remaining - len(remaining)
            if pruned > 0:
                logger.info("Registry refresh: %d newly completed, %d remaining", pruned, len(remaining))
                # Only reset index if we actually pruned entries (new completions
                # changed the list). Otherwise keep moving forward to avoid
                # re-scanning the same claimed WARCs endlessly.
                warc_idx = 0
            warcs_since_refresh = 0

        # Check steal mode activation
        unclaimed_count = len(remaining)
        if unclaimed_count < STEAL_TIER1_THRESHOLD:
            patience = STEAL_PATIENCE_TIER1
        elif unclaimed_count < STEAL_TIER2_THRESHOLD:
            patience = STEAL_PATIENCE_TIER2
        else:
            patience = STEAL_PATIENCE_TIER3

        if consecutive_failures >= patience:
            if not steal_mode:
                logger.info(
                    "=== ENTERING STEAL MODE === (%d consecutive failures, %d remaining)",
                    consecutive_failures,
                    unclaimed_count,
                )
            steal_mode = True

        if steal_mode:
            # Pick a few random in-progress WARCs to steal from, skipping ones
            # that recently had nothing stealable (each probe is a download).
            now = time.time()
            candidates = [
                w
                for w in warcs
                if _warc_path_hash(w) not in completed_registry
                and now - steal_exhausted.get(_warc_path_hash(w), 0.0) > STEAL_RECHECK_SECONDS
            ]
            if not candidates:
                logger.info("No WARCs left to steal from. Exiting.")
                break
            random_module.shuffle(candidates)
            stole_something = False
            for steal_target in candidates[:STEAL_CANDIDATES_PER_CYCLE]:
                logger.info("Steal: trying %s", steal_target)
                s = _process_warc_steal(
                    steal_target,
                    output_dir,
                    output_subdir,
                    llm,
                    sampling_params,
                    tokenizer,
                    template,
                    system_message,
                    registry_prefix,
                    batch_size=args.batch_size,
                    pipeline=pipeline_obj,
                )
                stats.append(s)
                logger.info("Steal result: %s", s)
                if s.get("batches_stolen", 0) > 0:
                    stole_something = True
                    break  # Go back to main loop — try claiming again first
                # Nothing stealable on this WARC right now — don't re-download
                # it for STEAL_RECHECK_SECONDS.
                steal_exhausted[_warc_path_hash(steal_target)] = time.time()
            if not stole_something:
                no_steal_cycles += 1
                if len(candidates) <= STEAL_CANDIDATES_PER_CYCLE or no_steal_cycles >= STEAL_MAX_EMPTY_CYCLES:
                    logger.info("No stealable batches found (%d empty steal cycles). Exiting.", no_steal_cycles)
                    break
                logger.info(
                    "Steal: nothing stealable in %d sampled candidates (%d eligible); sleeping %.0fs",
                    STEAL_CANDIDATES_PER_CYCLE,
                    len(candidates),
                    STEAL_CYCLE_SLEEP_SECONDS,
                )
                time.sleep(STEAL_CYCLE_SLEEP_SECONDS)
                completed_registry = _load_completed_registry(registry_prefix)
                continue
            # After a successful steal, reset and try forward claiming again
            steal_mode = False
            consecutive_failures = 0
            no_steal_cycles = 0
            completed_registry = _load_completed_registry(registry_prefix)
            remaining = _get_remaining()
            warc_idx = 0
            continue

        # Normal forward processing
        if warc_idx >= len(remaining):
            break
        warc = remaining[warc_idx]
        warc_idx += 1

        # Skip WARCs recently observed claimed by another worker without
        # re-probing GCS — the claim can't have gone stale inside the TTL.
        seen_claimed_at = recently_claimed.get(_warc_path_hash(warc))
        if seen_claimed_at is not None and time.time() - seen_claimed_at < CLAIMED_RECHECK_SECONDS:
            consecutive_failures += 1
            continue

        logger.info("=== WARC %d/%d (of %d remaining) ===", warc_idx, len(remaining), len(remaining))
        s = _process_warc(
            warc,
            output_dir,
            output_subdir,
            llm,
            sampling_params,
            tokenizer,
            template,
            system_message,
            registry_prefix,
            batch_size=args.batch_size,
            pipeline=pipeline_obj,
        )
        stats.append(s)
        if s["status"] != "skipped":
            logger.info("Result: %s", s)
        if s["status"] == "claimed":
            recently_claimed[_warc_path_hash(warc)] = time.time()

        # Track whether we're making progress (finding unclaimed WARCs)
        if s["status"] in ("done", "partial", "empty", "all_filtered"):
            consecutive_failures = 0
        else:
            consecutive_failures += 1

    # Summary
    done = sum(1 for s in stats if s["status"] == "done")
    skipped = sum(1 for s in stats if s["status"] == "skipped")
    claimed = sum(1 for s in stats if s["status"] == "claimed")
    stolen = sum(1 for s in stats if s["status"] == "steal_done")
    logger.info(
        "Complete: %d done, %d stolen, %d skipped, %d claimed-by-other, %d total",
        done,
        stolen,
        skipped,
        claimed,
        len(stats),
    )


if __name__ == "__main__":
    main()
