# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WARC-scaling sweep dashboard — localhost web UI.

Mirrors `experiments/baseline_collection/dashboard.py` but pivots to the
warc_scaling sweep's data model: per-(method, N) cell progress against the
plan, region distribution of children, and per-coord status with the
running/pending/failed tally.

All expensive queries (GCS list, iris job list) are gated behind manual
refresh buttons. In-memory cache persists to ``.warc_scaling_dashboard_cache.json``
so a restart doesn't re-scan.

Usage::

    uv run python -m experiments.scaling_law_sweeps.warc_scaling_dashboard
    # Open http://localhost:8091
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from experiments.scaling_law_sweeps.warc_scaling_plan import (
    WARC_COUNTS,
    WARC_METHOD_BASE_NAMES,
    enumerate_warc_scaling_plans,
)
from experiments.scaling_law_sweeps.fixed_model_plan import (
    enumerate_fixed_model_plans,
    resolve_methods as fm_resolve_methods,
)

# Methods to include in the FM (N=3000, expFM_natural) progress slice. Today
# this is just llm_curated_dedup — the other FM methods (dclm, *_bos_fixed,
# resiliparse) already get tracked via plot_warc_scaling_sweep's FM merge,
# and adding their N=3000 plans here would clutter the warc-scaling Progress
# tab with rows that aren't part of the per-N WARC sweep.
FM_PROGRESS_METHODS: tuple[str, ...] = ("llm_curated_dedup",)

logger = logging.getLogger(__name__)
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DASHBOARD_PORT = 8091
RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_warc_scaling_results/"
# Fixed-model (N=3000, expFM_natural) results — scanned in addition to the
# warc-scaling prefix so FM-only methods (currently llm_curated_dedup) get
# counted on the Progress tab.
FM_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_fixed_model_results/"
TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/data_curation_warc_scaling/"
IRIS_CONFIG = "lib/iris/examples/marin.yaml"
USER_PREFIX = "michaelryan"
# Bare job-name prefixes (compared against `path.split("/")[2]`, which is the
# `--job-name` value passed to `iris job run` — NOT user-prefixed). Add a new
# entry here when launching a coordinator under a new naming convention.
COORD_NAME_PREFIXES = ("warc-", "dedup-")
CACHE_FILE = Path(__file__).parent / ".warc_scaling_dashboard_cache.json"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {
    "progress": {"data": None, "updated_at": None},
    "jobs": {"data": None, "updated_at": None},
    "deep": {"data": None, "updated_at": None},
    "cluster": {"data": None, "updated_at": None},
    "plots": {"data": None, "updated_at": None},
}
_iris_client = None
_tunnel_cm = None


def _save_cache() -> None:
    try:
        CACHE_FILE.write_text(json.dumps(_cache, default=str))
    except Exception as e:
        logger.warning("Failed to persist cache: %s", e)


def _load_cache() -> None:
    global _cache
    try:
        if CACHE_FILE.exists():
            loaded = json.loads(CACHE_FILE.read_text())
            for key in _cache:
                if key not in loaded:
                    continue
                if isinstance(loaded[key], dict) and loaded[key].get("data") is not None:
                    _cache[key] = loaded[key]
            logger.info("Loaded cached data from %s", CACHE_FILE)
    except Exception as e:
        logger.warning("Failed to load cache: %s", e)


# ---------------------------------------------------------------------------
# Iris client wiring (mirrored from baseline_collection/dashboard.py:110-136)
# ---------------------------------------------------------------------------


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
    """Return a live iris client, rebuilding the SSH tunnel if the cached one is dead.

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
    _iris_client = IrisClient.remote(tunnel_url, workspace=Path.cwd())
    return _iris_client


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _decode_method_n(method_full: str) -> tuple[str, int] | None:
    """Decode a method_full string into (method_base, N).

    Two naming conventions coexist on the dashboard:
      - WARC sweep:  ``llm_curated_500`` → ("llm_curated", 500)
      - FM sweep:    ``llm_curated_dedup`` → ("llm_curated_dedup", 3000)
        (no trailing ``_<int>``; N is implicit 3000 for FM.)

    Returns None if the string can't be decoded under either convention.
    """
    if not method_full:
        return None
    if "_" in method_full:
        base, n_str = method_full.rsplit("_", 1)
        try:
            return base, int(n_str)
        except ValueError:
            pass
    # Fall back to FM (N=3000) for any registered FM-progress method.
    if method_full in FM_PROGRESS_METHODS:
        return method_full, 3000
    return None


def _enumerate_fm_progress_plans() -> list:
    """Return the FM (N=3000, expFM_natural) PlannedRuns to count on Progress.

    Scoped to ``FM_PROGRESS_METHODS`` so we don't pollute the WARC tab with
    other FM methods that the warc-scaling sweep already covers per-N.
    """
    methods = fm_resolve_methods(list(FM_PROGRESS_METHODS))
    return enumerate_fixed_model_plans(methods)


def _planned_counts() -> dict[tuple[str, int], int]:
    """Per-(method_base, N) count of planned runs from the enumerator.

    Includes the 2-budget extension for resiliparse + llm_curated so the
    Progress tab's "X/Y planned" never shows over-100% (e.g., 12/11 was a
    base-grid total being compared against base+extension completions).
    Also includes FM dedup plans at N=3000 (see ``FM_PROGRESS_METHODS``).
    """
    plans = enumerate_warc_scaling_plans(
        WARC_METHOD_BASE_NAMES,
        extension_methods=("llm_curated", "resiliparse"),
    )
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for p in plans:
        decoded = _decode_method_n(p.method_name)
        if decoded is not None:
            counts[decoded] += 1
    for p in _enumerate_fm_progress_plans():
        counts[(p.method_name, 3000)] += 1
    return counts


def _planned_lookup() -> dict[str, dict]:
    """Map run_name_core → plan info (train_steps, hidden_dim, budget, method, N).

    Includes the budget-extension plans for resiliparse + llm_curated so the
    deep scan can join with extension runs (otherwise they'd appear as
    "unknown" and get filtered out). Also includes FM dedup plans at N=3000.
    """
    plans = enumerate_warc_scaling_plans(
        WARC_METHOD_BASE_NAMES,
        extension_methods=("llm_curated", "resiliparse"),
    )
    out: dict[str, dict] = {}
    for p in plans:
        decoded = _decode_method_n(p.method_name)
        if decoded is None:
            continue
        base, n = decoded
        out[p.run_name_core] = {
            "train_steps": p.train_steps,
            "hidden_dim": p.hidden_dim,
            "num_layers": p.num_layers,
            "batch_size": p.batch_size,
            "budget": p.budget,
            "method": base,
            "n_warcs": n,
        }
    for p in _enumerate_fm_progress_plans():
        out[p.run_name_core] = {
            "train_steps": p.train_steps,
            "hidden_dim": p.hidden_dim,
            "num_layers": p.num_layers,
            "batch_size": p.batch_size,
            "budget": p.budget,
            "method": p.method_name,
            "n_warcs": 3000,
        }
    return out


def _gcs_list_basenames(prefix: str) -> list[str]:
    """List leaf basenames under a gs:// prefix using google-cloud-storage."""
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket_name = prefix.replace("gs://", "").split("/", 1)[0]
    obj_prefix = prefix.replace(f"gs://{bucket_name}/", "")
    bucket = client.bucket(bucket_name)
    out = []
    for blob in bucket.list_blobs(prefix=obj_prefix):
        out.append(blob.name.rsplit("/", 1)[-1])
    return out


def _gcs_load_jsonl(prefix: str, suffix: str = ".region") -> list[dict]:
    """Load all small JSON files at a gs:// prefix matching suffix."""
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    bucket_name = prefix.replace("gs://", "").split("/", 1)[0]
    obj_prefix = prefix.replace(f"gs://{bucket_name}/", "")
    bucket = client.bucket(bucket_name)
    out: list[dict] = []
    for blob in bucket.list_blobs(prefix=obj_prefix):
        if not blob.name.endswith(suffix):
            continue
        try:
            text = blob.download_as_text()
            out.append(json.loads(text))
        except Exception as e:
            logger.warning("Failed to read %s: %s", blob.name, e)
    return out


# ---------------------------------------------------------------------------
# Progress scan
# ---------------------------------------------------------------------------


def _scan_progress() -> dict:
    """Return {plan, completed, locks, recent_completions, completion_history}.

    `plan`: {(method, N): expected_count}
    `completed`: {(method, N): done_count}
    `locks`: {region: count} — all-time locks (one per dispatched run)
    `recent_completions`: list of {run_name, method, N, completed_at, region} sorted by time desc
    """
    plan = _planned_counts()

    # Completed = each summary.json filename in results bucket. Scan both the
    # warc-scaling prefix (per-N expWARC_natural) AND the fixed-model prefix
    # (expFM_natural at N=3000) so FM-progress methods like llm_curated_dedup
    # contribute to the (method, N=3000) counter.
    completed: Counter = Counter()
    recent: list[dict] = []
    warc_basenames = _gcs_list_basenames(RESULTS_PREFIX)
    fm_basenames = _gcs_list_basenames(FM_RESULTS_PREFIX)
    summary_basenames = warc_basenames  # canonical scan list for region/history below
    for source_name, basenames, expected_tag in (
        ("warc", warc_basenames, "expWARC"),
        ("fm", fm_basenames, "expFM"),
    ):
        for fname in basenames:
            if not fname.startswith("curation-") or not fname.endswith(".json"):
                continue
            body = fname[len("curation-") : -len(".json")]
            # Skip any -<suffix> variants like "-smoke". Canonical run_name_core
            # always ends with -B<batch_size>; anything beyond that is a suffix.
            if "-smoke" in body:
                continue
            # body = <method_full>-{expWARC|expFM}_natural-<budget>-d<H>-...
            chunks = body.split("-", 2)
            if len(chunks) < 2 or expected_tag not in chunks[1]:
                continue
            decoded = _decode_method_n(chunks[0])
            if decoded is None:
                continue
            # FM scan only counts methods we've explicitly opted into Progress
            # tracking — other FM methods (dclm, *_bos_fixed, resiliparse) are
            # tracked by the warc-scaling sweep's per-N plans.
            if source_name == "fm" and decoded[0] not in FM_PROGRESS_METHODS:
                continue
            completed[decoded] += 1

    # Region locks (one per dispatched run).
    locks_raw = _gcs_load_jsonl(TRACKER_PREFIX, suffix=".region")
    region_counts: Counter = Counter()
    region_per_run: dict[str, str] = {}
    for d in locks_raw:
        region_counts[d.get("region", "?")] += 1
        rk = d.get("run_key", "")
        if "__" in rk:
            run_name = rk.split("__", 2)[-1]
            if run_name.endswith(".region"):
                run_name = run_name[: -len(".region")]
            region_per_run[run_name] = d.get("region", "?")

    # Recent completions: read each summary.json's run.completed_at + region.
    # Limit to the most recent 20 by file basename sort across BOTH warc and
    # FM prefixes so dedup completions show up in the recent feed.
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()

    def _bucket_for(prefix: str):
        bucket_name = prefix.replace("gs://", "").split("/", 1)[0]
        obj_prefix = prefix.replace(f"gs://{bucket_name}/", "")
        return client.bucket(bucket_name), obj_prefix

    warc_bucket, warc_obj_prefix = _bucket_for(RESULTS_PREFIX)
    fm_bucket, fm_obj_prefix = _bucket_for(FM_RESULTS_PREFIX)
    bucket = warc_bucket  # used by the history scan below
    obj_prefix = warc_obj_prefix

    sample_warc = [(warc_bucket, warc_obj_prefix, fname) for fname in sorted(warc_basenames, reverse=True)[:60]]
    sample_fm = [(fm_bucket, fm_obj_prefix, fname) for fname in sorted(fm_basenames, reverse=True)[:60]]
    recent_records: list[dict] = []
    for src_bucket, src_prefix, fname in sample_warc + sample_fm:
        try:
            blob = src_bucket.blob(src_prefix + fname)
            d = json.loads(blob.download_as_text())
        except Exception:
            continue
        plan_d = d.get("plan", {})
        method_full = plan_d.get("method_name", "")
        decoded = _decode_method_n(method_full)
        if decoded is None:
            continue
        base, n = decoded
        # Skip non-progress-tracked FM methods so the recent feed mirrors the
        # (method, N) counter scope.
        if plan_d.get("experiment_tag") == "expFM_natural" and base not in FM_PROGRESS_METHODS:
            continue
        run_d = d.get("run", {})
        recent_records.append(
            {
                "run_name": plan_d.get("run_name", ""),
                "method": base,
                "n_warcs": n,
                "completed_at": run_d.get("completed_at"),
                "region": run_d.get("region", "?"),
                "hidden_dim": plan_d.get("hidden_dim"),
                "budget": plan_d.get("budget_flops"),
                "macro_bpb": (d.get("eval") or {}).get("eval/macro_bpb"),
            }
        )
    recent_records.sort(key=lambda r: r.get("completed_at") or "", reverse=True)
    recent_records = recent_records[:20]

    # Completion history: 10-min buckets over the last 24h, for a sparkline.
    # Re-scan recent completions across BOTH prefixes to compute rate, but cap.
    history: dict[str, int] = {}
    now = time.time()
    cutoff = now - 24 * 3600
    history_iter = (
        [(warc_bucket, warc_obj_prefix, f) for f in warc_basenames]
        + [(fm_bucket, fm_obj_prefix, f) for f in fm_basenames]
    )
    for src_bucket, src_prefix, fname in history_iter:
        # Skip reading every file; instead approximate using GCS modification
        # time via the blob's `updated`.
        try:
            blob = src_bucket.blob(src_prefix + fname)
            blob.reload()
            ts = blob.updated.timestamp() if blob.updated else 0
        except Exception:
            continue
        if ts < cutoff:
            continue
        bucket_key = int((ts - cutoff) // 600) * 600  # 10-min buckets
        key = str(bucket_key)
        history[key] = history.get(key, 0) + 1

    plan_serial = {f"{m}|{n}": c for (m, n), c in plan.items()}
    done_serial = {f"{m}|{n}": c for (m, n), c in completed.items()}
    return {
        "plan": plan_serial,
        "completed": done_serial,
        "region_locks": dict(region_counts),
        "region_per_run": region_per_run,
        "recent": recent_records,
        "completion_history_24h": history,
        "history_origin_ts": cutoff,
        "total_completed": sum(completed.values()),
        "total_planned": sum(plan.values()),
    }


# ---------------------------------------------------------------------------
# Deep scan: per-run checkpoint progress across regions
# ---------------------------------------------------------------------------


REGION_TO_BUCKET = {
    "us-central1": "marin-us-central1",
    "us-central2": "marin-us-central2",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "europe-west4": "marin-eu-west4",
}
CHECKPOINT_PREFIX = "checkpoints/isoflop-curation/"


def _scan_deep_progress() -> dict:
    """Walk all regions, collect per-run latest step + last update time.

    Joins with the plan enumerator to compute % complete and exposes a wandb
    URL per run. ~30s for the full sweep across 5 regions.
    """
    from google.cloud import storage as gcs_storage

    client = gcs_storage.Client()
    plan_lookup = _planned_lookup()

    # Source of truth: read every summary.json. It records the run's CANONICAL
    # region (the standalone runner writes its own region in summary.run.region)
    # — this is more reliable than guessing from checkpoint dir locations,
    # which can be polluted by stale pre-emption attempts in other regions.
    summary_region: dict[str, str] = {}
    summary_runs: set[str] = set()
    # Per-run wandb override: when a recovery sync writes a NEW wandb run id
    # for a corrupted-then-recovered cell, the summary records it under
    # `run.wandb_run_id`. The dashboard's wandb link should then point at the
    # clean recovery run, not the original (contaminated) wandb run.
    summary_wandb_id: dict[str, str] = {}
    try:
        rb_name = RESULTS_PREFIX.replace("gs://", "").split("/", 1)[0]
        rb_prefix = RESULTS_PREFIX.replace(f"gs://{rb_name}/", "")
        rb = client.bucket(rb_name)
        for blob in rb.list_blobs(prefix=rb_prefix):
            fname = blob.name.rsplit("/", 1)[-1]
            if not (fname.startswith("curation-") and fname.endswith(".json")):
                continue
            run_name = fname[: -len(".json")]
            summary_runs.add(run_name)
            try:
                summary = json.loads(blob.download_as_text())
                run_block = summary.get("run") or {}
                region = run_block.get("region")
                if region:
                    summary_region[run_name] = region
                wandb_id = run_block.get("wandb_run_id")
                if wandb_id:
                    summary_wandb_id[run_name] = wandb_id
            except Exception:
                pass
    except Exception as e:
        logger.warning("results-bucket scan failed: %s", e)

    # run_name → {region: ?, current_step: int, last_update: epoch, has_done: bool}
    runs: dict[str, dict] = {}

    for region, bucket_name in REGION_TO_BUCKET.items():
        bucket = client.bucket(bucket_name)
        # Recursive list — get every checkpoint/step blob and DONE markers.
        try:
            blobs = list(bucket.list_blobs(prefix=CHECKPOINT_PREFIX))
        except Exception as e:
            logger.warning("list %s failed: %s", region, e)
            continue
        for blob in blobs:
            # blob.name = checkpoints/isoflop-curation/<run_name>/checkpoints/step-NNNN/<file> | .data_curation_DONE
            rel = blob.name[len(CHECKPOINT_PREFIX) :]
            parts = rel.split("/", 2)
            if len(parts) < 1:
                continue
            run_name = parts[0]
            # Only WARC-scaling runs (run_name_core contains "expWARC_natural").
            if "expWARC_natural" not in run_name:
                continue
            entry = runs.setdefault(
                run_name,
                {
                    "region": region,
                    "current_step": 0,
                    "last_update": 0.0,
                    "has_done": False,
                    # ETA tracking: store earliest + latest step-N seen WITH
                    # corresponding ts so we can compute a steps/sec rate.
                    "first_step": None,
                    "first_step_ts": None,
                    "last_step": 0,
                    "last_step_ts": 0.0,
                },
            )
            # Take the FIRST region we see this run in as the canonical region
            # (region locks force a run to live in exactly one region — the
            # tracker enforces this — so we shouldn't see a run in two regions
            # except for the rare race window, which we just first-write-wins).
            if entry["region"] != region:
                continue  # ignore later sightings in other regions
            ts = blob.updated.timestamp() if blob.updated else 0
            if ts > entry["last_update"]:
                entry["last_update"] = ts
            # DONE marker?
            if rel.endswith(".data_curation_DONE"):
                entry["has_done"] = True
                continue
            # step-N parsing — the step appears in part[2] as "step-NNNN/..."
            if len(parts) >= 3 and parts[1] == "checkpoints":
                step_chunk = parts[2].split("/", 1)[0]
                if step_chunk.startswith("step-"):
                    try:
                        step = int(step_chunk[len("step-") :])
                        if step > entry["current_step"]:
                            entry["current_step"] = step
                        # Track earliest+latest step-N timestamps for ETA.
                        # blob.updated for any blob WITHIN step-N/ is approx
                        # when that step was saved.
                        if step > entry["last_step"] or (step == entry["last_step"] and ts > entry["last_step_ts"]):
                            entry["last_step"] = step
                            entry["last_step_ts"] = ts
                        if entry["first_step"] is None or step < entry["first_step"]:
                            entry["first_step"] = step
                            entry["first_step_ts"] = ts
                    except ValueError:
                        pass

    # Build records: union of (1) all PLANNED runs (so we surface untouched
    # cells too), (2) anything we observed in checkpoint dirs, (3) anything
    # with a summary. Prefer summary's region (canonical); fall back to the
    # checkpoint walk for in-flight runs without a summary yet. DONE = summary
    # exists (marker writes occasionally fail; summary is canonical).
    records: list[dict] = []
    now_ts = time.time()
    all_run_names = set(plan_lookup.keys()) | set(runs.keys()) | set(summary_runs)
    for run_name in all_run_names:
        plan = plan_lookup.get(run_name)
        if plan is None:
            continue
        entry = runs.get(run_name)
        started = entry is not None  # any checkpoint blob seen
        if entry is None:
            entry = {
                "region": "(none)", "current_step": 0, "last_update": 0.0, "has_done": False,
                "first_step": None, "first_step_ts": None, "last_step": 0, "last_step_ts": 0.0,
            }
        canonical_region = summary_region.get(run_name) or entry["region"]
        total = plan["train_steps"]
        cur = entry["current_step"]
        is_done = run_name in summary_runs or entry["has_done"]
        pct = 100.0 if is_done else (100.0 * cur / total if total else 0.0)

        # ETA: rate from (first_step → last_step) in steps/sec. Two-step minimum
        # so the timeline isn't undefined. For DONE runs leave eta=None (already
        # finished). For runs with no two distinct step dirs, eta is None.
        eta_hours = None
        steps_per_hour = None
        is_active = False
        if not is_done and started and entry["first_step"] is not None and entry["last_step"] > entry["first_step"]:
            dt = entry["last_step_ts"] - entry["first_step_ts"]
            d_step = entry["last_step"] - entry["first_step"]
            if dt > 0 and d_step > 0:
                rate_per_sec = d_step / dt
                steps_per_hour = rate_per_sec * 3600
                remaining = total - cur
                if remaining > 0 and rate_per_sec > 0:
                    eta_hours = remaining / rate_per_sec / 3600
            # active = wrote a checkpoint in the last 30 min
            is_active = (now_ts - entry["last_update"]) < 1800

        wandb_id = summary_wandb_id.get(run_name, run_name)
        wandb_url = f"https://wandb.ai/marin-community/marin/runs/{wandb_id}"
        records.append(
            {
                "run_name": run_name,
                "method": plan["method"],
                "n_warcs": plan["n_warcs"],
                "hidden_dim": plan["hidden_dim"],
                "budget": plan["budget"],
                "batch_size": plan["batch_size"],
                "region": canonical_region,
                "current_step": total if is_done else cur,
                "total_steps": total,
                "pct_complete": pct,
                "last_update": entry["last_update"],
                "has_done": is_done,
                "started": started or is_done,
                "is_active": is_active,
                "steps_per_hour": steps_per_hour,
                "eta_hours": eta_hours,
                "wandb_url": wandb_url,
            }
        )
    # Sort: in-progress (no DONE) first by % desc, then DONE.
    records.sort(key=lambda r: (r["has_done"], -r["pct_complete"], -r["last_update"]))
    in_progress_records = [r for r in records if not r["has_done"] and r.get("started")]
    active_records = [r for r in in_progress_records if r.get("is_active")]
    # ETA aggregate: only count ACTIVE runs (wrote in last 30m). A stale run
    # with last_update 12h ago will compute a deceptively large ETA from a
    # near-zero rate; including it would mislead the user.
    eta_values = [r["eta_hours"] for r in active_records if r.get("eta_hours") is not None]
    longest_eta = max(eta_values) if eta_values else None
    return {
        "runs": records,
        "now": time.time(),
        "in_progress": len(in_progress_records),
        "active": len(active_records),  # in-progress AND wrote in last 30 min
        "done": sum(1 for r in records if r["has_done"]),
        "untouched": sum(1 for r in records if not r["has_done"] and not r.get("started")),
        "longest_eta_hours": longest_eta,
        "eta_runs_count": len(eta_values),
    }


# ---------------------------------------------------------------------------
# Cluster scan (iris autoscaler — what TPUs/regions have spare capacity?)
# ---------------------------------------------------------------------------


def _scan_cluster() -> dict:
    """Pull iris autoscaler status: which scale groups (region × TPU) have
    capacity ready/idle vs which are saturated, plus my current usage on each.

    Mirrors `experiments/baseline_collection/dashboard.py:_fetch_cluster_status`
    but adapted to surface the targeting decision: for each (region, TPU) pair,
    show idle/ready/booting/demand counts so the user can pick a region+shape
    to launch a new sweep arm against.
    """
    client = _get_iris_client()
    try:
        response = client._cluster_client.get_autoscaler_status()
    except Exception:
        logger.exception("autoscaler status RPC failed")
        raise

    groups = []
    for sg in response.status.groups:
        name = sg.name
        # Name pattern: `tpu_<variant>-<mode>_<size>-<zone>` where zone is the
        # last 3 dash-separated tokens (e.g. us-central1-a, europe-west4-a).
        # Examples:
        #   tpu_v4-preemptible_8-us-central2-b → variant=v4, size=8, zone=us-central2-b
        #   tpu_v6e-preemptible_4-us-east1-d   → variant=v6e, size=4, zone=us-east1-d
        tpu_type = ""
        region = ""
        try:
            after_tpu = name[len("tpu_") :] if name.startswith("tpu_") else name
            parts = after_tpu.split("-")
            if len(parts) >= 4:
                zone = "-".join(parts[-3:])
                tpu_part = "-".join(parts[:-3])  # e.g. "v4-preemptible_8"
                if "-" in tpu_part:
                    variant, mode_size = tpu_part.split("-", 1)
                else:
                    variant, mode_size = tpu_part, ""
                if "_" in mode_size:
                    _, size = mode_size.rsplit("_", 1)
                    tpu_type = f"{variant}-{size}"
                region = zone
        except Exception:
            pass

        counts = dict(sg.slice_state_counts) if sg.slice_state_counts else {}
        ready = counts.get("ready", 0)
        booting = counts.get("booting", 0)
        initializing = counts.get("initializing", 0)
        failed = counts.get("failed", 0)
        demand = sg.current_demand
        idle = sum(1 for s in sg.slices if getattr(s, "idle", False))

        if ready == 0 and booting == 0 and demand == 0 and initializing == 0 and idle == 0:
            continue  # skip empty groups

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

    # Per-TPU rollup (across regions).
    by_tpu: dict[str, dict[str, int]] = defaultdict(lambda: {"ready": 0, "booting": 0, "demand": 0, "idle": 0, "regions": []})
    for g in groups:
        if not g["tpu_type"]:
            continue
        agg = by_tpu[g["tpu_type"]]
        agg["ready"] += g["ready"]
        agg["booting"] += g["booting"]
        agg["demand"] += g["demand"]
        agg["idle"] += g["idle"]
        agg["regions"].append(g["region"])

    # My current usage by tpu_type (run job list ourselves to avoid double-RPC).
    my_usage: Counter = Counter()
    try:
        out = subprocess.check_output(
            [".venv/bin/iris", "--cluster", "marin", "job", "list", "--state", "running"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
        for line in out.splitlines():
            if "michaelryan-warc" not in line:
                continue
            for token in line.split():
                # token like "v5p-8" or "v4-8" or "v6e-4"
                if token.startswith(("v4-", "v5p-", "v5litepod-", "v6e-")) and token[-1].isdigit():
                    my_usage[token] += 1
                    break
    except Exception:
        pass

    return {
        "groups": groups,
        "by_tpu": {k: dict(v) for k, v in by_tpu.items()},
        "my_usage": dict(my_usage),
        "totals": {
            "ready": sum(g["ready"] for g in groups),
            "demand": sum(g["demand"] for g in groups),
            "idle": sum(g["idle"] for g in groups),
        },
    }


# ---------------------------------------------------------------------------
# Plot regeneration
# ---------------------------------------------------------------------------


PLOT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "scratch" / "plots" / "warc_scaling"
PLOT_SUMMARIES_DIR = Path(__file__).parent.parent.parent / "scratch" / "warc_summaries"
PLOT_FM_DIR = Path(__file__).parent.parent.parent / "scratch" / "fm_summaries"
PLOT_FM_LIMA_DIR = Path(__file__).parent.parent.parent / "scratch" / "fm_lima_sidecar"


def _regenerate_plots() -> dict:
    """Pull latest summaries, regenerate plots, and return the file inventory.

    Steps:
      1. gcloud cp the warc_scaling + fixed_model + LIMA-sidecar summaries
         locally (incremental — `gcloud storage cp -r` is idempotent).
      2. Run plot_warc_scaling_sweep.main() in-process.
      3. Walk PLOT_OUTPUT_DIR and return categorized file paths so the
         frontend can render "Open in new tab" buttons.
    """
    PLOT_SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_FM_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_FM_LIMA_DIR.mkdir(parents=True, exist_ok=True)

    # Pull summaries (cheap, gcloud cp is idempotent for unchanged files).
    for src, dst in [
        (RESULTS_PREFIX, PLOT_SUMMARIES_DIR),
        ("gs://marin-us-central1/metadata/data_curation_fixed_model_results/", PLOT_FM_DIR),
        ("gs://marin-us-central1/metadata/data_curation_fixed_model_lima_results/", PLOT_FM_LIMA_DIR),
    ]:
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", f"{src.rstrip('/')}/*", str(dst) + "/"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
        except Exception as e:
            logger.warning("summary fetch failed for %s: %s", src, e)

    # Regenerate plots in-process so we don't fork.
    from experiments.scaling_law_sweeps import plot_warc_scaling_sweep

    plot_warc_scaling_sweep.main(
        argv=[
            "--results-prefix",
            str(PLOT_SUMMARIES_DIR),
            "--fm-results-prefix",
            str(PLOT_FM_DIR),
            "--fm-lima-sidecar-prefix",
            str(PLOT_FM_LIMA_DIR),
            "--output-dir",
            str(PLOT_OUTPUT_DIR),
        ]
    )

    # Walk output dir → categorize by metric and view type.
    plots: dict[str, dict] = {}
    if PLOT_OUTPUT_DIR.exists():
        for metric_dir in sorted(PLOT_OUTPUT_DIR.iterdir()):
            if not metric_dir.is_dir():
                continue
            metric = metric_dir.name
            entry = {"grids": [], "per_n": {}}
            for f in sorted(metric_dir.iterdir()):
                if f.is_file() and f.name.endswith(".html"):
                    entry["grids"].append({"label": f.stem, "path": str(f.relative_to(PLOT_OUTPUT_DIR))})
                elif f.is_dir() and f.name.startswith("N"):
                    n_files = []
                    for ff in sorted(f.iterdir()):
                        if ff.suffix == ".html":
                            n_files.append({"label": ff.stem, "path": str(ff.relative_to(PLOT_OUTPUT_DIR))})
                    entry["per_n"][f.name] = n_files
            plots[metric] = entry
    return {"plots": plots, "summary_count": len(list(PLOT_SUMMARIES_DIR.glob("*.json")))}


# ---------------------------------------------------------------------------
# Jobs scan (iris)
# ---------------------------------------------------------------------------


def _iris_list_jobs(state: str) -> list[str]:
    """Run the iris CLI to list jobs in a given state, filtering to warc coords."""
    try:
        out = subprocess.check_output(
            [".venv/bin/iris", "--cluster", "marin", "job", "list", "--state", state],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
        )
    except Exception as e:
        logger.warning("iris job list --state %s failed: %s", state, e)
        return []
    out_lines = []
    for line in out.splitlines():
        if not line.strip():
            continue
        # Path is the first whitespace-separated token; but lines start with "/<user>/...".
        if not line.startswith(f"/{USER_PREFIX}/"):
            continue
        parts = line.split()
        path = parts[0]
        # Filter to warc-* parents and their children.
        leaf = path.split("/")[2] if len(path.split("/")) > 2 else ""
        if not any(leaf.startswith(p) for p in COORD_NAME_PREFIXES):
            continue
        out_lines.append(path)
    return out_lines


def _parse_child(path: str) -> dict | None:
    """Decompose `/<user>/<coord>/curation-curation-<method_full>-...` into fields.

    Accepts both ``expWARC_natural`` (warc-scaling) and ``expFM_natural`` (FM
    sweep) tags so dedup-fm2 children appear in the Jobs tab cell counters.
    """
    parts = path.split("/")
    if len(parts) < 4:
        return None
    leaf = parts[-1]
    if leaf.startswith("curation-curation-"):
        leaf = leaf[len("curation-") :]
    if not leaf.startswith("curation-"):
        return None
    body = leaf[len("curation-") :]
    chunks = body.split("-")
    if len(chunks) < 4 or chunks[1] not in ("expWARC_natural", "expFM_natural"):
        return None
    decoded = _decode_method_n(chunks[0])
    if decoded is None:
        return None
    base, n = decoded
    try:
        budget = float(chunks[2])
    except ValueError:
        budget = 0.0
    hidden_dim = 0
    for c in chunks[3:]:
        if c.startswith("d") and c[1:].isdigit():
            hidden_dim = int(c[1:])
            break
    return {
        "path": path,
        "coord": parts[2],
        "run_name": leaf,
        "method": base,
        "n_warcs": n,
        "budget": budget,
        "hidden_dim": hidden_dim,
    }


def _scan_jobs() -> dict:
    """Return {by_coord, running_by_method_n, region_running, totals}."""
    states = ("running", "pending", "failed")
    raw_paths: dict[str, list[str]] = {}
    for s in states:
        raw_paths[s] = _iris_list_jobs(s)

    parents: dict[str, dict] = {}  # coord_path -> info
    children: dict[str, list[dict]] = defaultdict(list)
    running_by_cell: Counter = Counter()
    pending_by_cell: Counter = Counter()
    failed_by_cell: Counter = Counter()
    running_paths: list[str] = []

    for state, paths in raw_paths.items():
        for path in paths:
            parts = path.split("/")
            if len(parts) == 3:
                # Parent coord
                parents[path] = {"path": path, "state": state}
            elif len(parts) == 4:
                child = _parse_child(path)
                if child is None:
                    continue
                child["state"] = state
                children[child["coord"]].append(child)
                cell_key = (child["method"], child["n_warcs"])
                if state == "running":
                    running_by_cell[cell_key] += 1
                    running_paths.append(path)
                elif state == "pending":
                    pending_by_cell[cell_key] += 1
                elif state == "failed":
                    failed_by_cell[cell_key] += 1

    by_coord = []
    for path in sorted(parents):
        coord_name = path.split("/")[-1]
        kids = children.get(coord_name, [])
        coord_running = sum(1 for k in kids if k["state"] == "running")
        coord_pending = sum(1 for k in kids if k["state"] == "pending")
        coord_failed = sum(1 for k in kids if k["state"] == "failed")
        by_coord.append(
            {
                "path": path,
                "state": parents[path]["state"],
                "running": coord_running,
                "pending": coord_pending,
                "failed": coord_failed,
                "children_sample": [
                    {"name": k["run_name"], "state": k["state"], "method": k["method"], "n_warcs": k["n_warcs"]}
                    for k in kids[:50]
                ],
            }
        )

    return {
        "by_coord": by_coord,
        "running_by_cell": {f"{m}|{n}": c for (m, n), c in running_by_cell.items()},
        "pending_by_cell": {f"{m}|{n}": c for (m, n), c in pending_by_cell.items()},
        "failed_by_cell": {f"{m}|{n}": c for (m, n), c in failed_by_cell.items()},
        "totals": {
            "running": sum(running_by_cell.values()),
            "pending": sum(pending_by_cell.values()),
            "failed": sum(failed_by_cell.values()),
        },
        "running_paths": running_paths,
    }


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_file(Path(__file__).parent / "warc_scaling_dashboard.html")


@app.after_request
def add_no_cache_headers(resp):
    """Prevent the browser from serving a stale cached JSON across page reloads."""
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp


@app.route("/api/progress")
def api_progress():
    return jsonify(_cache["progress"])


@app.route("/api/progress/refresh", methods=["POST"])
def api_progress_refresh():
    try:
        data = _scan_progress()
        _cache["progress"] = {"data": data, "updated_at": time.time()}
        _save_cache()
        return jsonify(_cache["progress"])
    except Exception as e:
        logger.exception("progress refresh failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/deep")
def api_deep():
    return jsonify(_cache["deep"])


@app.route("/api/deep/refresh", methods=["POST"])
def api_deep_refresh():
    try:
        data = _scan_deep_progress()
        _cache["deep"] = {"data": data, "updated_at": time.time()}
        _save_cache()
        return jsonify(_cache["deep"])
    except Exception as e:
        logger.exception("deep refresh failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster")
def api_cluster():
    return jsonify(_cache["cluster"])


@app.route("/api/cluster/refresh", methods=["POST"])
def api_cluster_refresh():
    try:
        data = _scan_cluster()
        _cache["cluster"] = {"data": data, "updated_at": time.time()}
        _save_cache()
        return jsonify(_cache["cluster"])
    except Exception as e:
        logger.exception("cluster refresh failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/plots")
def api_plots():
    return jsonify(_cache["plots"])


@app.route("/api/plots/refresh", methods=["POST"])
def api_plots_refresh():
    try:
        data = _regenerate_plots()
        _cache["plots"] = {"data": data, "updated_at": time.time()}
        _save_cache()
        return jsonify(_cache["plots"])
    except Exception as e:
        logger.exception("plot refresh failed")
        return jsonify({"error": str(e)}), 500


@app.route("/plots/<path:rel>")
def serve_plot(rel: str):
    """Serve generated HTML plot files. Path is relative to PLOT_OUTPUT_DIR."""
    target = PLOT_OUTPUT_DIR / rel
    if not target.exists() or not target.is_file():
        return ("Not found", 404)
    return send_file(target)


@app.route("/api/jobs")
def api_jobs():
    return jsonify(_cache["jobs"])


@app.route("/api/jobs/refresh", methods=["POST"])
def api_jobs_refresh():
    try:
        data = _scan_jobs()
        _cache["jobs"] = {"data": data, "updated_at": time.time()}
        _save_cache()
        return jsonify(_cache["jobs"])
    except Exception as e:
        logger.exception("jobs refresh failed")
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _load_cache()
    logger.info("Dashboard starting on http://localhost:%d", DASHBOARD_PORT)
    app.run(host="127.0.0.1", port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()
