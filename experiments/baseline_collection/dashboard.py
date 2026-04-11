# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extraction progress dashboard — localhost web UI for monitoring WARC extraction.

Provides at-a-glance progress tracking, cluster status, and job management.
All expensive GCS queries are behind manual refresh buttons.

Usage::

    uv run python experiments/baseline_collection/dashboard.py
    # Open http://localhost:8080
"""

import json
import logging
import os
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_PATH = "experiments/distill/baseline_warcs_3000.txt"
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

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_manifest_hashes: dict[str, str] | None = None  # hash -> warc_path
_iris_client = None
_tunnel_cm = None

CACHE_FILE = Path(__file__).parent / ".dashboard_cache.json"

_cache: dict[str, dict] = {
    "progress": {"data": None, "updated_at": None},
    "progress_deep": {"data": None, "updated_at": None},
    "cluster": {"data": None, "updated_at": None},
    "jobs": {"data": None, "updated_at": None},
    "preemption_history": [],  # list of {"ts": epoch, "total": int, "by_type": {tpu: int}}
}


def _save_cache() -> None:
    """Persist cache to disk so it survives dashboard restarts."""
    try:
        CACHE_FILE.write_text(json.dumps(_cache, default=str))
    except Exception:
        pass


def _load_cache() -> None:
    """Load cache from disk if available."""
    global _cache
    try:
        if CACHE_FILE.exists():
            loaded = json.loads(CACHE_FILE.read_text())
            for key in _cache:
                if key not in loaded:
                    continue
                # preemption_history is a list, not a dict with "data"
                if key == "preemption_history" and isinstance(loaded[key], list):
                    _cache[key] = loaded[key]
                elif isinstance(loaded[key], dict) and loaded[key].get("data") is not None:
                    _cache[key] = loaded[key]
            logger.info("Loaded cached data from %s", CACHE_FILE)
    except Exception:
        pass


def _load_manifest() -> dict[str, str]:
    """Load manifest and build hash->path mapping. Cached after first call."""
    global _manifest_hashes
    if _manifest_hashes is not None:
        return _manifest_hashes

    from experiments.baseline_collection.download_warcs import _load_manifest as load_m
    from experiments.baseline_collection.download_warcs import _warc_path_hash

    warcs = load_m(MANIFEST_PATH)
    _manifest_hashes = {_warc_path_hash(w): w for w in warcs}
    logger.info("Loaded manifest: %d WARCs", len(_manifest_hashes))
    return _manifest_hashes


def _get_iris_client():
    """Get or create the Iris client with SSH tunnel."""
    global _iris_client, _tunnel_cm

    if _iris_client is not None:
        return _iris_client

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
# GCS scanning
# ---------------------------------------------------------------------------


def _scan_progress_quick() -> dict:
    """Quick scan: count claimed dirs per region. ~10s, one list op per region.

    Does NOT check _done markers (too expensive per-WARC). Use deep scan for done counts.
    """
    from google.cloud import storage as gcs_storage
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    client = gcs_storage.Client()
    manifest = _load_manifest()

    regions = {}
    all_hashes: dict[str, list[str]] = {}  # hash -> list of regions

    for region, bucket_name in sorted(REGION_TO_DATA_BUCKET.items()):
        bucket = client.bucket(bucket_name)
        prefix = f"{OUTPUT_SUBDIR}/"

        claimed = set()
        done = set()
        done_stats = []

        # Use delimiter to get directory prefixes (one list op per region)
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

        # Quick scan: skip _done checks (expensive). Only count directories.
        # Done markers are checked in deep scan mode.

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
                from datetime import datetime, timezone

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
                remaining = 3000 - len(unique_done)
                eta_hours = remaining / rate if rate > 0 else None

    # Strip large hash lists from response (keep counts only)
    for region_data in regions.values():
        del region_data["claimed_hashes"]
        del region_data["done_hashes"]

    return {
        "total_manifest": len(manifest),
        "unique_claimed": len(unique_claimed),
        "unique_done": len(unique_done),
        "unclaimed": len(manifest) - len(unique_claimed),
        "duplicates": duplicates,
        "eta_hours": round(eta_hours, 1) if eta_hours else None,
        "regions": regions,
        "done_stats": all_done_stats,
    }


def _scan_progress_deep() -> dict:
    """Deep scan: list ALL blobs to get batch counts, done markers, and activity. ~60s+."""
    from google.cloud import storage as gcs_storage
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    client = gcs_storage.Client()
    manifest = _load_manifest()

    all_hashes: dict[str, list[str]] = {}
    batch_counts: dict[str, int] = {}
    latest_activity: dict[str, str] = {}
    earliest_batch_ts: float | None = None
    latest_batch_ts: float | None = None
    done_hashes: set[str] = set()
    done_stats: list[dict] = []
    region_data: dict[str, dict] = {}

    for region, bucket_name in sorted(REGION_TO_DATA_BUCKET.items()):
        bucket = client.bucket(bucket_name)
        prefix = f"{OUTPUT_SUBDIR}/"
        r_claimed = set()
        r_done = set()

        for blob in bucket.list_blobs(prefix=prefix + "data-"):
            rel = blob.name[len(prefix):]
            parts = rel.split("/")
            if len(parts) < 2 or not parts[0].startswith("data-"):
                continue
            h = parts[0][5:]
            r_claimed.add(h)
            all_hashes.setdefault(h, [])
            if region not in all_hashes[h]:
                all_hashes[h].append(region)

            filename = parts[1]
            if filename == "_done":
                r_done.add(h)
                done_hashes.add(h)
                try:
                    content = blob.download_as_text()
                    stats = json.loads(content)
                    stats["hash"] = h
                    stats["region"] = region
                    stats["completed_at"] = blob.updated.isoformat() if blob.updated else None
                    stats["warc_path"] = manifest.get(h, "")
                    done_stats.append(stats)
                except Exception:
                    done_stats.append({"hash": h, "region": region, "warc_path": manifest.get(h, "")})
            elif filename.startswith("batch_") and filename.endswith(".jsonl.gz"):
                batch_counts[h] = batch_counts.get(h, 0) + 1
                if blob.updated:
                    ts = blob.updated.isoformat()
                    if h not in latest_activity or ts > latest_activity[h]:
                        latest_activity[h] = ts
                    # Track global earliest/latest batch timestamps for rate calc
                    batch_epoch = blob.updated.timestamp()
                    if earliest_batch_ts is None or batch_epoch < earliest_batch_ts:
                        earliest_batch_ts = batch_epoch
                    if latest_batch_ts is None or batch_epoch > latest_batch_ts:
                        latest_batch_ts = batch_epoch

        region_data[region] = {
            "bucket": bucket_name,
            "claimed": len(r_claimed),
            "done": len(r_done),
        }

    unique_claimed = set(all_hashes.keys())
    duplicates = sum(1 for r_list in all_hashes.values() if len(r_list) > 1)

    # ETA from batch completion rate (much more reliable than WARC completion rate)
    total_batches = sum(batch_counts.values())
    eta_hours = None
    batches_per_hour = None

    # Compute avg batches per WARC from completed WARCs
    avg_batches_per_warc = 100  # default
    real_counts = [s["num_batches"] for s in done_stats if s.get("num_batches")]
    if real_counts:
        avg_batches_per_warc = round(sum(real_counts) / len(real_counts))

    if earliest_batch_ts and latest_batch_ts and total_batches > 10:
        span_hours = (latest_batch_ts - earliest_batch_ts) / 3600
        if span_hours > 0.1:
            batches_per_hour = total_batches / span_hours
            # Estimated total batches needed
            done_warc_batches = sum(real_counts) if real_counts else 0
            remaining_warcs = len(manifest) - len(done_hashes)
            estimated_remaining_batches = remaining_warcs * avg_batches_per_warc - (total_batches - done_warc_batches)
            if estimated_remaining_batches > 0 and batches_per_hour > 0:
                eta_hours = estimated_remaining_batches / batches_per_hour

    return {
        "total_manifest": len(manifest),
        "unique_claimed": len(unique_claimed),
        "unique_done": len(done_hashes),
        "unclaimed": len(manifest) - len(unique_claimed),
        "duplicates": duplicates,
        "eta_hours": round(eta_hours, 1) if eta_hours else None,
        "batches_per_hour": round(batches_per_hour, 1) if batches_per_hour else None,
        "avg_batches_per_warc": avg_batches_per_warc,
        "regions": region_data,
        "done_stats": done_stats,
        "batch_counts": batch_counts,
        "latest_activity": latest_activity,
        "total_batches": total_batches,
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
            variant_and_region = "_".join(parts[1:])
            # Try to extract TPU variant and region
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

        groups.append({
            "name": name,
            "tpu_type": tpu_type,
            "region": region,
            "ready": ready,
            "booting": booting,
            "initializing": initializing,
            "failed": failed,
            "demand": demand,
            "idle": idle,
        })

    groups.sort(key=lambda g: (-g["ready"], -g["demand"], g["name"]))

    return {
        "groups": groups,
        "total_ready": sum(g["ready"] for g in groups),
        "total_demand": sum(g["demand"] for g in groups),
        "total_idle": sum(g["idle"] for g in groups),
    }


def _format_job(j) -> dict:
    """Convert a JobStatus proto to a JSON-serializable dict."""
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

        submitted = datetime.fromtimestamp(
            j.submitted_at.epoch_ms / 1000, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

    reason = j.error or j.pending_reason or ""

    return {
        "job_id": j.job_id,
        "name": j.name,
        "state": state_name,
        "is_parent": not has_device,
        "tpu_type": tpu_type,
        "resources": ", ".join(res_parts),
        "submitted": submitted,
        "reason": reason[:100],
        "preemption_count": j.preemption_count,
        "failure_count": j.failure_count,
    }


def _list_child_jobs(parent_job_id: str) -> list:
    """List child jobs of a parent using the parent_job_id filter in ListJobsRequest."""
    from iris.rpc import controller_pb2

    client = _get_iris_client()
    rpc_client = client._cluster_client._client

    request = controller_pb2.Controller.ListJobsRequest(parent_job_id=parent_job_id)
    response = rpc_client.list_jobs(request)
    return list(response.jobs)


def _fetch_jobs() -> dict:
    """List extraction jobs for the current user, including children of each parent."""
    from iris.cluster.types import JobName

    client = _get_iris_client()
    prefix = JobName.from_wire(f"/{USER_PREFIX}/extract")
    top_level_jobs = client.list_jobs(prefix=prefix)
    top_level_jobs.sort(key=lambda j: j.submitted_at.epoch_ms, reverse=True)

    parents = []
    orphan_children = []

    for j in top_level_jobs:
        formatted = _format_job(j)
        if formatted["is_parent"]:
            # Fetch children for this parent
            try:
                child_protos = _list_child_jobs(j.job_id)
                children = [_format_job(c) for c in child_protos]
                children.sort(key=lambda c: c["submitted"], reverse=True)
            except Exception as e:
                logger.warning("Failed to fetch children for %s: %s", j.job_id, e)
                children = []
            formatted["children"] = children
            parents.append(formatted)
        else:
            orphan_children.append(formatted)

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
        "total": len(top_level_jobs),
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


@app.route("/api/progress")
def api_progress():
    return jsonify({
        "data": _cache["progress"]["data"],
        "updated_at": _cache["progress"]["updated_at"],
        "deep_updated_at": _cache["progress_deep"]["updated_at"],
    })


@app.route("/api/progress/refresh", methods=["POST"])
def api_progress_refresh():
    mode = request.json.get("mode", "quick") if request.json else "quick"
    try:
        if mode == "deep":
            data = _scan_progress_deep()
            _cache["progress_deep"]["data"] = data
            _cache["progress_deep"]["updated_at"] = time.time()
            _cache["progress"]["data"] = data
            _cache["progress"]["updated_at"] = time.time()
        else:
            data = _scan_progress_quick()
            # Merge: preserve deep scan fields (batch_counts, done, etc.) if available
            deep = _cache["progress_deep"].get("data")
            if deep:
                data["unique_done"] = deep.get("unique_done", 0)
                data["done_stats"] = deep.get("done_stats", [])
                data["batch_counts"] = deep.get("batch_counts")
                data["latest_activity"] = deep.get("latest_activity")
                data["total_batches"] = deep.get("total_batches")
                data["batches_per_hour"] = deep.get("batches_per_hour")
                data["avg_batches_per_warc"] = deep.get("avg_batches_per_warc")
                data["eta_hours"] = deep.get("eta_hours")
                # Update done counts per region from deep scan
                for region, rdata in data["regions"].items():
                    deep_region = deep.get("regions", {}).get(region, {})
                    if "done" in deep_region:
                        rdata["done"] = deep_region["done"]
            _cache["progress"]["data"] = data
            _cache["progress"]["updated_at"] = time.time()
        _save_cache()
        return jsonify({
            "ok": True,
            "data": _cache["progress"]["data"],
            "updated_at": _cache["progress"]["updated_at"],
            "deep_updated_at": _cache["progress_deep"].get("updated_at"),
        })
    except Exception as e:
        logger.exception("Progress scan failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cluster")
def api_cluster():
    return jsonify({
        "data": _cache["cluster"]["data"],
        "updated_at": _cache["cluster"]["updated_at"],
    })


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
    return jsonify({
        "data": _cache["jobs"]["data"],
        "updated_at": _cache["jobs"]["updated_at"],
    })


@app.route("/api/jobs/refresh", methods=["POST"])
def api_jobs_refresh():
    try:
        data = _fetch_jobs()
        _cache["jobs"]["data"] = data
        _cache["jobs"]["updated_at"] = time.time()
        _save_cache()
        return jsonify({"ok": True, "data": data, "updated_at": time.time()})
    except Exception as e:
        logger.exception("Job list fetch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


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

    if tpu_type not in KNOWN_TPU_TYPES:
        return jsonify({"ok": False, "error": f"Unknown TPU type: {tpu_type}"}), 400
    if not job_name:
        return jsonify({"ok": False, "error": "Job name required"}), 400

    try:
        from iris.cli.job import run_iris_job

        controller_url = _iris_client._cluster_client._address if _iris_client else None
        if not controller_url:
            return jsonify({"ok": False, "error": "Iris not connected"}), 500

        command = [
            "python",
            "experiments/baseline_collection/launch_adaptive.py",
            "--tpu-type", tpu_type,
            "--max-count", str(max_count),
            "--initial-batch", str(initial_batch),
            "--chunk-size", str(chunk_size),
            "--check-interval", str(check_interval),
            "--patience", str(patience),
        ]

        exit_code = run_iris_job(
            command=command,
            env_vars={},
            controller_url=controller_url,
            cpu=2,
            memory="2GB",
            job_name=job_name,
            wait=False,
        )
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

    port = int(os.environ.get("DASHBOARD_PORT", "8090"))

    # Load disk cache from previous run
    _load_cache()

    # Pre-load manifest
    _load_manifest()

    # Establish Iris tunnel
    try:
        _get_iris_client()
        logger.info("Iris client connected")
    except Exception as e:
        logger.warning("Could not connect to Iris: %s (cluster/jobs tabs will be unavailable)", e)

    logger.info("Starting dashboard on http://localhost:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
