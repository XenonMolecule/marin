# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fast-curation production dashboard — localhost web UI for the A/B/C cascade.

Designed to scale to the 7M-WARC production run: progress comes from per-worker **heartbeats**
(``telemetry.Heartbeat``), so every refresh is **O(workers)** — it never lists per-WARC outputs or
the registry (which would be O(WARCs) and unusable at production scale). The two headline views:

* **Predictability** — per-phase + per-run ETAs from the realized throughput (slope of the
  aggregate done-counter over a sliding window), shown as a duration and a wall-clock date.
* **In-flight oversight** — every run x phase (A/B/C, CPU/TPU) x region: active workers, rate,
  docs in→out funnel, compute-time mix.

Usage::

    uv run python experiments/fast_curation/dashboard.py
    # open http://localhost:8092

Cluster + Jobs tabs query Iris (reused from baseline_collection.dashboard); the Runs tab needs only
heartbeat reads, so it auto-refreshes cheaply.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, send_file

from experiments.fast_curation.spec import Extractor, get_spec

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Heartbeats + registry live centrally (one read covers all regions), matching the worker side.
HEARTBEAT_BUCKET = "marin-us-central1"
FAST_CURATION_PREFIX = "documents/fast_curation/"
IRIS_CONFIG = "lib/iris/examples/marin.yaml"
USER_PREFIX = "michaelryan"

# A worker slot is "active" if its heartbeat updated within this window (workers write every WARC
# and at least every 30s; preempted/dead slots fall out after this).
ACTIVE_WINDOW = 300.0
# A run is shown if any slot updated within this window.
RUN_RECENCY = 3600.0
# Sliding window for the realized-throughput slope used in ETAs.
RATE_WINDOW = 900.0
PHASES = ("a", "b", "c")
PHASE_LABEL = {"a": "A · decode+fastText", "b": "B · ModernBERT (TPU)", "c": "C · JustText"}


def _phase_labels(spec_id: str) -> dict[str, str]:
    """Stage labels for a run, derived from its spec rather than hardcoded.

    A run's cascade is spec-defined (the lpv11 line adds a pooled pre-filter and swaps jusText for
    the Rust resiliparse engine), so a fixed label set silently mislabels it — the dashboard would
    claim "JustText" over a corpus jusText never touched.
    """
    labels = dict(PHASE_LABEL)
    try:
        spec = get_spec(spec_id)
    except Exception:
        return labels  # an unknown/older run keeps the legacy names
    if spec.pooled_ckpt:
        labels["b"] = "B · pooled+ModernBERT (TPU)"
    if spec.extraction_engine is Extractor.RESILIPARSE_RS:
        labels["c"] = "C · resiliparse-rs"
    return labels


# In-memory throughput series (bounded, O(snapshots)): the slope over a window gives realized rate.
_series: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=240))  # (run, phase) -> [(ts, done)]
_region_series: dict[tuple[str, str, str], deque] = defaultdict(lambda: deque(maxlen=120))  # +region
# Per-slot high-water marks: a heavily-preempted worker's restart can momentarily reset its heartbeat
# (dying+starting worker race on the slot), which would pull the summed counter backwards. Clamping
# each slot to its max-ever value keeps the aggregate monotonic. Registry stays the true count.
_slot_max: dict[tuple[str, str, str, int], dict[str, int]] = {}  # (run,phase,region,seed) -> maxes
# EMA of each phase's rate so the ETA isn't spiky (the raw window-slope jumps as WARCs land in bursts).
_ema_rate: dict[tuple[str, str], float] = {}  # (run, phase) -> smoothed WARCs/hr
EMA_ALPHA = 0.25
_manifest_total = 0  # denominator (WARCs in the pool); set at startup from --manifest

# Iris client (lazy; shared with the reference dashboard's connection pattern).
_iris_client = None
_tunnel_cm = None


# ---------------------------------------------------------------------------
# Heartbeat aggregation (the scalable core)
# ---------------------------------------------------------------------------


# Threads used to fan out heartbeat reads. The GCS client's HTTP pool is sized to match: urllib3
# defaults to 10 connections, so a 64-thread fan-out spends its time discarding and re-opening
# sockets ("Connection pool is full") instead of reading. At 5 regions x ~300 workers that turned a
# sub-second scan into a multi-minute one and the UI just sat on "Loading...".
GCS_FANOUT = 64
_gcs_client = None


def _storage_client():
    """Process-wide GCS client with a pool big enough for the fan-out.

    Cached: a fresh ``gcs.Client()`` per call re-does auth and starts with an empty pool, which is
    most of the cost when every dashboard refresh scans hundreds of blobs.
    """
    global _gcs_client
    if _gcs_client is None:
        import requests
        from google.cloud import storage as gcs

        client = gcs.Client()
        adapter = requests.adapters.HTTPAdapter(pool_connections=GCS_FANOUT, pool_maxsize=GCS_FANOUT, max_retries=3)
        client._http.mount("https://", adapter)
        _gcs_client = client
    return _gcs_client


def list_active_runs() -> list[str]:
    """Discover run subdirs (``documents/fast_curation/{spec}-{hash}``) that have heartbeats.

    One delimiter list (O(runs)). Recency is decided later from the heartbeats themselves.
    """
    client = _storage_client()
    it = client.list_blobs(HEARTBEAT_BUCKET, prefix=FAST_CURATION_PREFIX, delimiter="/")
    list(it)  # consume the (empty) blob page so ``.prefixes`` (the subdirs) is populated
    return [p[len(FAST_CURATION_PREFIX) :].rstrip("/") for p in it.prefixes]


def read_heartbeats(run: str) -> list[dict]:
    """Read all heartbeat JSONs for a run (O(workers)). Best-effort; skips unreadable slots."""
    client = _storage_client()
    bucket = client.bucket(HEARTBEAT_BUCKET)
    prefix = f"{FAST_CURATION_PREFIX}{run}/_heartbeats/"
    blobs = list(bucket.list_blobs(prefix=prefix))

    def _read(b):
        try:
            return json.loads(b.download_as_text())
        except Exception:
            return None

    if not blobs:
        return []
    with ThreadPoolExecutor(max_workers=min(GCS_FANOUT, len(blobs))) as ex:
        return [h for h in ex.map(_read, blobs) if h]


def _counts(run: str, regions: list[str]) -> dict[str, dict[str, int]]:
    """Authoritative done-counts, immune to heartbeat double-counting.

    Global counts come from the **deduplicated central registry** (`_completed_{a,b,c}`); per-region
    counts come from each region's output files. A slow/reclaimed WARC is scored once in the registry
    but counted by multiple heartbeat slots, so the heartbeat sum overcounts — these listings don't.
    O(WARCs); fine at 10k, will move to sharded counters for the 7M run.
    """
    from rigging.filesystem import data_config

    region_to_bucket = {r: s.name for r, s in data_config().region_buckets.items()}
    client = _storage_client()
    sub = {"a": "a_presurvivors", "b": "b_keeplist", "c": "kept"}

    def _n(bucket_name: str, prefix: str) -> int:
        return sum(
            1
            for b in client.bucket(bucket_name).list_blobs(prefix=prefix)
            if b.name.rsplit("/", 1)[-1].startswith("data-")
        )

    tasks = [("global", p, HEARTBEAT_BUCKET, f"{FAST_CURATION_PREFIX}{run}/_completed_{p}/") for p in PHASES]
    for reg in regions:
        b = region_to_bucket.get(reg)
        if b:
            tasks += [(reg, p, b, f"{FAST_CURATION_PREFIX}{run}/{sub[p]}/") for p in PHASES]
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with ThreadPoolExecutor(max_workers=min(16, len(tasks))) as ex:
        for scope, p, n in ex.map(lambda t: (t[0], t[1], _n(t[2], t[3])), tasks):
            out[scope][p] = n
    return out


def _slope_and_conf(series: deque, now: float) -> tuple[float | None, str]:
    """Realized rate (per hour) over ``RATE_WINDOW`` + a confidence label.

    Confidence compares the recent-half rate to the full-window rate: they agreeing means the ETA
    is trustworthy ("stable"); disagreement means throughput is still changing ("settling").
    """
    cutoff = now - RATE_WINDOW
    w = [(t, d) for t, d in series if t >= cutoff]
    if len(w) < 2 or (w[-1][0] - w[0][0]) < 60:
        return None, "warming"
    (t0, d0), (t1, d1) = w[0], w[-1]
    rate = max(0.0, (d1 - d0) / (t1 - t0) * 3600.0)
    if rate == 0:
        return 0.0, "idle"
    tm, dm = w[len(w) // 2]
    if (t1 - tm) > 30:
        r2 = (d1 - dm) / (t1 - tm) * 3600.0
        if abs(r2 - rate) / max(rate, 1e-9) > 0.4:
            return rate, "settling"
    return rate, "stable"


def _sparkline(series: deque, now: float, n: int = 30) -> list[list]:
    """Recent ``(relative_seconds, done)`` points for a cumulative-progress sparkline."""
    cutoff = now - RATE_WINDOW
    w = [(t, d) for t, d in series if t >= cutoff]
    if len(w) < 2:
        return []
    t0 = w[0][0]
    return [[round(t - t0), d] for t, d in w[-n:]]


def _phase_summary(
    run: str,
    phase: str,
    hbs: list[dict],
    reg_done: int,
    region_done: dict[str, dict[str, int]],
    total: int,
    now: float,
    labels: dict[str, str] | None = None,
) -> dict:
    """Aggregate one phase. Done-count is the AUTHORITATIVE registry/file count (``reg_done`` global,
    ``region_done`` per region); heartbeats supply only live-worker counts + the docs funnel."""
    rows = [h for h in hbs if h.get("phase") == phase]
    docs_in = docs_out = 0
    compute: dict[str, float] = defaultdict(float)
    by_region: dict[str, dict] = {}
    for h in rows:
        cum = h.get("cumulative", {})
        # Clamp docs to max-ever per slot (a restart race can transiently reset a heartbeat).
        key = (run, phase, h.get("region", "?"), int(h.get("seed", -1)))
        prev = _slot_max.get(key, {})
        mono = {k: max(int(cum.get(k, 0)), int(prev.get(k, 0))) for k in ("docs_in", "docs_out")}
        _slot_max[key] = mono
        docs_in += mono["docs_in"]
        docs_out += mono["docs_out"]
        for k, v in cum.get("compute_seconds", {}).items():
            compute[k] += float(v)
        r = h.get("region", "?")
        active = (now - float(h.get("updated_at", 0))) <= ACTIVE_WINDOW
        slot = by_region.setdefault(r, {"workers": 0, "active": 0, "kind": h.get("kind", "")})
        slot["workers"] += 1
        slot["active"] += 1 if active else 0

    done = reg_done  # deduplicated registry count — NOT the heartbeat sum (which double-counts reclaims)
    ps = _series[(run, phase)]
    ps.append((now, done))
    rate, conf = _slope_and_conf(ps, now)
    if rate is not None:  # EMA-smooth so the ETA doesn't jump as WARCs land in bursts
        prev = _ema_rate.get((run, phase))
        rate = rate if prev is None else EMA_ALPHA * rate + (1 - EMA_ALPHA) * prev
        _ema_rate[(run, phase)] = rate
    remaining = max(0, total - done)
    eta_hours = (remaining / rate) if rate and rate > 0 else None

    regions_out: dict[str, dict] = {}
    for r, v in sorted(by_region.items(), key=lambda kv: (-kv[1]["active"], -kv[1]["workers"])):
        rdone = int(region_done.get(r, {}).get(phase, 0))  # authoritative per-region output-file count
        rs = _region_series[(run, phase, r)]
        rs.append((now, rdone))
        rrate, _ = _slope_and_conf(rs, now)
        regions_out[r] = {**v, "done": rdone, "rate_per_hour": round(rrate, 1) if rrate is not None else None}

    return {
        "phase": phase,
        "label": labels[phase] if labels else PHASE_LABEL[phase],
        "kind": rows[0].get("kind", "") if rows else "",
        "done": done,
        "total": total,
        "pct": round(100.0 * done / total, 1) if total else 0.0,
        "docs_in": docs_in,
        "docs_out": docs_out,
        "active_workers": sum(s["active"] for s in by_region.values()),
        "total_workers": len(rows),
        "rate_per_hour": round(rate, 1) if rate is not None else None,
        "rate_confidence": conf,
        "eta_hours": round(eta_hours, 2) if eta_hours is not None else None,
        "sparkline": _sparkline(ps, now),
        "compute_seconds": {k: round(v, 1) for k, v in compute.items()},
        "regions": regions_out,
    }


def aggregate_run(run: str, total: int, now: float) -> dict | None:
    """Full per-run rollup: phase summaries + run % + A→B→C funnel + bottleneck ETA."""
    hbs = read_heartbeats(run)
    if not hbs:
        return None
    last_update = max(float(h.get("updated_at", 0)) for h in hbs)
    regions_present = sorted({h.get("region", "?") for h in hbs})
    counts = _counts(run, regions_present)  # authoritative (registry + per-region files)
    gdone = counts.get("global", {})
    region_done = {r: counts.get(r, {}) for r in regions_present}
    labels = _phase_labels(run.rpartition("-")[0])
    phases = [_phase_summary(run, p, hbs, int(gdone.get(p, 0)), region_done, total, now, labels) for p in PHASES]
    by_phase = {p["phase"]: p for p in phases}
    run_done = int(gdone.get("c", 0))  # final corpus = Phase C output (registry)

    # Per-phase in→out retention (each ratio is over that phase's OWN completed WARCs, so it is
    # progress-independent and honest — unlike comparing one phase's cumulative output to another's,
    # which mixes different WARC counts). fastText + ModernBERT are the filters; JustText extracts.
    def _ret(p: dict) -> float | None:
        return round(100.0 * p["docs_out"] / p["docs_in"], 1) if p["docs_in"] else None

    ext_label = "resiliparse-rs extract" if labels["c"].endswith("resiliparse-rs") else "JustText extract"
    b_label = "pooled+ModernBERT filter" if labels["b"].startswith("B · pooled") else "ModernBERT filter"
    funnel = {
        "stages": [
            {
                "phase": "a",
                "label": "fastText filter",
                "in": by_phase["a"]["docs_in"],
                "out": by_phase["a"]["docs_out"],
                "retention": _ret(by_phase["a"]),
            },
            {
                "phase": "b",
                "label": b_label,
                "in": by_phase["b"]["docs_in"],
                "out": by_phase["b"]["docs_out"],
                "retention": _ret(by_phase["b"]),
            },
            {
                "phase": "c",
                "label": ext_label,
                "in": by_phase["c"]["docs_in"],
                "out": by_phase["c"]["docs_out"],
                "retention": _ret(by_phase["c"]),
            },
        ],
    }

    # Bottleneck = the phase whose ETA gates completion (A→B→C pipeline).
    eta_phases = [(p["phase"], p["eta_hours"]) for p in phases if p["eta_hours"] is not None]
    bottleneck = max(eta_phases, key=lambda x: x[1])[0] if eta_phases else None
    for p in phases:
        p["is_bottleneck"] = p["phase"] == bottleneck
    run_eta = max((e for _, e in eta_phases), default=None)

    # Per-region bottleneck (the constraint can differ by region) + a compute-allocation hint. A
    # region where A is far ahead of B has presurvivors piling up → B is that region's wall; where B
    # is ahead of C, keeplists pile up → C is the wall; otherwise A is the source-limiter.
    region_live = {r: {p["phase"]: p["regions"].get(r, {}).get("active", 0) for p in phases} for r in regions_present}
    region_bottlenecks = {}
    for r in regions_present:
        rd = counts.get(r, {})
        ad, bd, cd = int(rd.get("a", 0)), int(rd.get("b", 0)), int(rd.get("c", 0))
        pileup_ab, pileup_bc = max(0, ad - bd), max(0, bd - cd)
        live = region_live[r]
        if ad > 0 and live.get("b", 0) == 0:
            bn, hint = "b", "no live B — WARCs strand here (needs TPU or rescue)"
        elif pileup_ab >= pileup_bc and pileup_ab > max(30, 0.05 * ad):
            bn, hint = "b", f"{pileup_ab} presurvivors queued for B → add TPU / rescue"
        elif pileup_bc > max(30, 0.05 * max(bd, 1)):
            bn, hint = "c", f"{pileup_bc} keeplists queued for C → add CPU (Phase C)"
        else:
            bn, hint = "a", "flowing → add CPU (Phase A) to push this region faster"
        region_bottlenecks[r] = {
            "a": ad,
            "b": bd,
            "c": cd,
            "pileup_ab": pileup_ab,
            "pileup_bc": pileup_bc,
            "live": live,
            "bottleneck": bn,
            "hint": hint,
        }

    spec_id, _, version = run.rpartition("-")
    return {
        "run": run,
        "spec_id": spec_id,
        "version": version,
        "total": total,
        "run_done": run_done,
        "run_pct": round(100.0 * run_done / total, 1) if total else 0.0,
        "run_eta_hours": round(run_eta, 2) if run_eta is not None else None,
        "bottleneck": bottleneck,
        "funnel": funnel,
        "region_bottlenecks": region_bottlenecks,
        "active_workers": sum(p["active_workers"] for p in phases),
        "last_update": last_update,
        "stale": (now - last_update) > RUN_RECENCY,
        "phases": phases,
    }


def fetch_runs() -> dict:
    """All runs with heartbeats, freshest first; computes ETAs from the live series."""
    now = time.time()
    runs = list_active_runs()
    out = []
    for run in runs:
        agg = aggregate_run(run, _manifest_total, now)
        if agg:
            out.append(agg)
    out.sort(key=lambda r: (r["stale"], -r["last_update"]))
    return {"runs": out, "manifest_total": _manifest_total, "now": now}


# ---------------------------------------------------------------------------
# Iris (cluster autoscaler + flat fastcur job states) — reused connection pattern
# ---------------------------------------------------------------------------


def _get_iris_client():
    global _iris_client, _tunnel_cm
    if _iris_client is not None:
        try:
            _iris_client._cluster_client.get_job_states([])
            return _iris_client
        except Exception as e:
            logger.warning("iris tunnel probe failed (%s); rebuilding", e)
            _iris_client = None
            _tunnel_cm = None

    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig

    iris_config = IrisConfig.load(IRIS_CONFIG)
    bundle = iris_config.provider_bundle()
    controller_address = iris_config.controller_address()
    if not controller_address:
        controller_address = bundle.controller.discover_controller(iris_config.proto.controller)
    _tunnel_cm = bundle.controller.tunnel(address=controller_address)
    tunnel_url = _tunnel_cm.__enter__()
    _iris_client = IrisClient.remote(tunnel_url, workspace=Path.cwd())
    return _iris_client


def fetch_cluster() -> dict:
    """Autoscaler group readiness (TPU capacity) — same shape as the reference dashboard."""
    client = _get_iris_client()
    response = client._cluster_client.get_autoscaler_status()
    groups = []
    for group in response.status.groups:
        counts = dict(group.slice_state_counts) if group.slice_state_counts else {}
        ready, booting = counts.get("ready", 0), counts.get("booting", 0)
        initializing, failed = counts.get("initializing", 0), counts.get("failed", 0)
        demand = group.current_demand
        if ready == 0 and booting == 0 and demand == 0 and initializing == 0:
            continue
        groups.append(
            {
                "name": group.name,
                "ready": ready,
                "booting": booting,
                "initializing": initializing,
                "failed": failed,
                "demand": demand,
                "idle": sum(1 for s in group.slices if s.idle),
            }
        )
    groups.sort(key=lambda g: (-g["ready"], -g["demand"], g["name"]))
    return {
        "groups": groups,
        "total_ready": sum(g["ready"] for g in groups),
        "total_demand": sum(g["demand"] for g in groups),
        "total_idle": sum(g["idle"] for g in groups),
    }


def fetch_jobs() -> dict:
    """Flat ``fastcur-*`` job states (running/pending/building/failed), grouped by phase.

    Heartbeats only exist once a worker is RUNNING; this surfaces the scheduling state (pending /
    unschedulable / failed) that heartbeats can't — the other half of in-flight oversight.
    """
    from iris.rpc import controller_pb2 as ctrl_pb2

    client = _get_iris_client()
    rpc_client = client._cluster_client._client

    def _fetch_state(state_name: str):
        req = ctrl_pb2.Controller.ListJobsRequest(
            query=ctrl_pb2.Controller.JobQuery(
                name_filter=f"/{USER_PREFIX}/fastcur-", state_filter=state_name, limit=4000
            )
        )
        return rpc_client.list_jobs(req).jobs

    states = ("running", "pending", "building", "failed")
    by_phase: dict[str, dict[str, int]] = {p: defaultdict(int) for p in ("a", "tpu", "c")}
    total = 0
    for st in states:
        try:
            jobs = _fetch_state(st)
        except Exception as e:
            logger.warning("job state %s fetch failed: %s", st, e)
            continue
        for j in jobs:
            total += 1
            # name: fastcur-{phase}-{spec}-{seed}
            parts = j.name.split("-")
            phase = parts[1] if len(parts) > 1 and parts[1] in by_phase else "?"
            if phase in by_phase:
                by_phase[phase][st] += 1
                by_phase[phase]["preemptions"] += int(getattr(j, "preemption_count", 0))
    totals = {s: sum(by_phase[p].get(s, 0) for p in by_phase) for s in ("running", "pending", "building", "failed")}
    totals["preemptions"] = sum(by_phase[p].get("preemptions", 0) for p in by_phase)
    return {
        "by_phase": {p: dict(c) for p, c in by_phase.items()},
        "totals": totals,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return send_file(Path(__file__).parent / "dashboard.html", mimetype="text/html")


# A full scan costs O(regions x workers) GCS reads — ~19s at 5 regions x 300 workers. The UI polls
# every few seconds, so without this every poll re-scans and the page sits on "Loading...". Serving a
# few seconds stale is the right trade for a monitor; the numbers move on a minutes timescale.
RUNS_TTL_SECONDS = 20.0
_runs_cache: tuple[float, dict] | None = None
_runs_lock = threading.Lock()


def _runs_cached() -> dict:
    """``fetch_runs()`` behind a short TTL, with one in-flight scan shared by all callers."""
    global _runs_cache
    now = time.time()
    cached = _runs_cache
    if cached and now - cached[0] < RUNS_TTL_SECONDS:
        return cached[1]
    with _runs_lock:  # a second request that arrives mid-scan waits and reuses the result
        cached = _runs_cache
        if cached and time.time() - cached[0] < RUNS_TTL_SECONDS:
            return cached[1]
        data = fetch_runs()
        _runs_cache = (time.time(), data)
        return data


@app.route("/api/runs")
def api_runs():
    try:
        return jsonify({"ok": True, **_runs_cached()})
    except Exception as e:
        logger.exception("runs fetch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cluster")
def api_cluster():
    try:
        return jsonify({"ok": True, "data": fetch_cluster(), "updated_at": time.time()})
    except Exception as e:
        logger.exception("cluster fetch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/jobs")
def api_jobs():
    try:
        return jsonify({"ok": True, "data": fetch_jobs(), "updated_at": time.time()})
    except Exception as e:
        logger.exception("jobs fetch failed")
        return jsonify({"ok": False, "error": str(e)}), 500


def _count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fast-curation cascade dashboard")
    parser.add_argument(
        "--manifest",
        default="experiments/distill/dclm_400m_1x.txt",
        help="WARC pool being extracted; its line count is the ETA denominator.",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("FC_DASHBOARD_PORT", "8092")))
    args = parser.parse_args()

    global _manifest_total
    _manifest_total = _count_lines(args.manifest)
    logger.info("manifest %s -> %d WARCs (ETA denominator)", args.manifest, _manifest_total)

    try:
        _get_iris_client()
        logger.info("Iris connected")
    except Exception as e:
        logger.warning("Iris unavailable (%s); Cluster/Jobs tabs will be empty", e)

    logger.info("Fast-curation dashboard on http://localhost:%d", args.port)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
