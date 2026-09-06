# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Light-tier manager for the in-region worker HTTP endpoints (HA, N replicas).

Runs ``DESIRED_REPLICAS`` :class:`~experiments.spec_explorer.worker.SpecExplorerWorker`
HTTP servers per region for high availability — if one preemptible node is
reclaimed, another already-warm replica keeps serving while a replacement warms.
Calls fail over across the region's warm replicas.

Each worker is reached through the controller's **generic HTTP endpoint proxy**
(``{dashboard:8080}/proxy/<name>/<path>``) — the deployed controller has this
reverse proxy but not the newer actor proxy. Because the controller drops a
registered endpoint after a few minutes, a keepalive thread re-registers every
running worker's endpoint (the worker's HTTP server + warm data stay up
regardless), refreshes per-worker warmth, and tops the region back up to
``DESIRED_REPLICAS``. :func:`reconnect` rebuilds handles from RUNNING jobs.

Hardware: in us-central1 (where the data lives) only ``v5p-preemptible`` fits a
96 GB worker, so replicas get node-level (not TPU-type) redundancy there.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import threading
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from experiments.spec_explorer.catalog import INDEX_DATASET_REGION

logger = logging.getLogger(__name__)

IRIS_CONFIG = "lib/iris/examples/marin.yaml"
ENDPOINT_NAME = "spec_worker"
GEMINI_ENV = "GEMINI_API_KEY"
DASHBOARD_PORT = 8080  # controller dashboard / endpoint proxy (gRPC controller is 10000)
PROXY_CALL_TIMEOUT = 28.0  # the controller proxy caps forwarded requests at 30s
KEEPALIVE_INTERVAL = 40.0  # re-register endpoints; the controller drops them after a few minutes
DESIRED_REPLICAS = 2  # warm workers per region for HA (one serves while a preempted one re-warms)

# region -> list of worker entries: {job_name, name, wire_name, started_at,
#           datasets, address?, endpoint_id?, warm: bool}
_workers: dict[str, list[dict]] = {}
_iris_client = None
_tunnel_cm = None
_controller_url: str | None = None
_proxy_cm = None
_proxy_url: str | None = None
_keepalive_started = False
_lock = threading.Lock()


def datasets_in(region: str) -> list[str]:
    return [ds for ds, r in INDEX_DATASET_REGION.items() if r == region]


def all_regions() -> list[str]:
    return sorted(set(INDEX_DATASET_REGION.values()))


def region_of(dataset: str) -> str:
    region = INDEX_DATASET_REGION.get(dataset)
    if region is None:
        raise ValueError(f"no index region for dataset {dataset!r}")
    return region


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ssl_ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


# --- controller / proxy tunnels ---
#
# SSH tunnels drop (idle, network blip, controller bounce). A dropped tunnel
# leaves its local port refusing connections, and the iris client then retries
# for ~40 min, wedging the keepalive so endpoints never re-register. So before
# reusing a cached tunnel we fast-probe its local port and rebuild if it's dead.


def _local_reachable(url: str | None, timeout: float = 2.0) -> bool:
    """True if the tunnel's local port accepts a TCP connection (cheap liveness)."""
    if not url:
        return False
    import socket
    from urllib.parse import urlparse

    u = urlparse(url)
    if u.hostname is None or u.port is None:
        return False
    try:
        with socket.create_connection((u.hostname, u.port), timeout=timeout):
            return True
    except OSError:
        return False


def _get_controller():
    """Return (IrisClient, controller_url), (re)building the SSH tunnel if dead."""
    global _iris_client, _tunnel_cm, _controller_url
    if _iris_client is not None and _local_reachable(_controller_url):
        return _iris_client, _controller_url
    if _tunnel_cm is not None:  # tear down a dead tunnel before rebuilding
        try:
            _tunnel_cm.__exit__(None, None, None)
        except Exception:
            pass
        _iris_client = _tunnel_cm = _controller_url = None
    from iris.client import IrisClient
    from iris.cluster.config import IrisConfig

    cfg = IrisConfig.load(IRIS_CONFIG)
    bundle = cfg.provider_bundle()
    address = cfg.controller_address() or bundle.controller.discover_controller(cfg.proto.controller)
    logger.info("establishing controller tunnel…")
    _tunnel_cm = bundle.controller.tunnel(address=address)
    _controller_url = _tunnel_cm.__enter__()
    _iris_client = IrisClient.remote(_controller_url, workspace=Path.cwd())
    logger.info("controller tunnel ready: %s", _controller_url)
    return _iris_client, _controller_url


def _get_proxy() -> str:
    """Return the dashboard (endpoint-proxy) tunnel URL, (re)building it if dead."""
    global _proxy_cm, _proxy_url
    if _proxy_url is not None and _local_reachable(_proxy_url):
        return _proxy_url
    if _proxy_cm is not None:
        try:
            _proxy_cm.__exit__(None, None, None)
        except Exception:
            pass
        _proxy_cm = _proxy_url = None
    from iris.cluster.config import IrisConfig

    cfg = IrisConfig.load(IRIS_CONFIG)
    bundle = cfg.provider_bundle()
    address = cfg.controller_address() or bundle.controller.discover_controller(cfg.proto.controller)
    host = address.rsplit(":", 1)[0]
    logger.info("establishing dashboard(proxy) tunnel to %s:%d…", host, DASHBOARD_PORT)
    _proxy_cm = bundle.controller.tunnel(address=f"{host}:{DASHBOARD_PORT}")
    _proxy_url = _proxy_cm.__enter__()
    logger.info("dashboard proxy tunnel ready: %s", _proxy_url)
    return _proxy_url


def controller_ping(timeout: float = 10.0) -> bool:
    """True if the controller answers a cheap call within ``timeout``.

    Runs in a daemon thread so a wedged call (dead tunnel mid-flight) can't hang
    the caller; :func:`_get_controller` rebuilds a dead tunnel before the call.
    """
    result: dict[str, bool] = {}

    def _do() -> None:
        try:
            client, _ = _get_controller()
            client._cluster_client.list_endpoints("/michaelryan/spec-worker", exact=False)
            result["ok"] = True
        except Exception:
            result["ok"] = False

    t = threading.Thread(target=_do, daemon=True)
    t.start()
    t.join(timeout)
    return result.get("ok", False)


def _encoded(wire_name: str) -> str:
    """Encode an Iris endpoint wire name for the /proxy/<name>/ path ( / -> . )."""
    return wire_name.lstrip("/").replace("/", ".")


# --- job state / addressing ---


def _job_state(entry: dict) -> str:
    from iris.rpc import job_pb2

    try:
        client, _ = _get_controller()
        state = client.status(entry["job_name"]).state
    except Exception:
        return "GONE"
    return {
        job_pb2.JOB_STATE_RUNNING: "RUNNING",
        job_pb2.JOB_STATE_PENDING: "PENDING",
        job_pb2.JOB_STATE_SUCCEEDED: "DONE",
        job_pb2.JOB_STATE_FAILED: "FAILED",
    }.get(state, "UNKNOWN")


def _worker_address(region: str, entry: dict, force: bool = False) -> str | None:
    """A worker's HTTP address: cached, else the GCS heartbeat (worker-written
    truth, fresh across preemption moves), else whatever is in the endpoint store.

    The heartbeat is preferred over the endpoint store because the worker rewrites
    it with its real address on every (re)start, whereas the store may hold an
    address this light tier itself registered before the worker moved nodes.
    ``force`` skips the cache to re-read the heartbeat (used by the periodic
    re-register so it never pins a stale address).
    """
    if entry.get("address") and not force:
        return entry["address"]
    # Per-worker GCS heartbeat (worker writes it at registration keyed by the
    # encoded wire name, prefixed with the region by _heartbeat).
    from google.cloud import storage

    blob = f"spec_explorer/worker_heartbeat/{region}_{_encoded(entry['wire_name'])}.json"
    try:
        txt = storage.Client().bucket(f"marin-{region}").blob(blob).download_as_text()
        addr = json.loads(txt).get("address")
        if addr:
            entry["address"] = addr
            return addr
    except Exception:
        pass
    try:
        client, _ = _get_controller()
        eps = client._cluster_client.list_endpoints(entry["wire_name"], exact=True)
        if eps:
            entry["address"] = eps[0].address
            return entry["address"]
    except Exception:
        pass
    return None


# --- keepalive: re-register endpoints, refresh warmth, maintain replica count ---


def _start_keepalive() -> None:
    global _keepalive_started
    if _keepalive_started:
        return
    _keepalive_started = True
    threading.Thread(target=_keepalive_loop, daemon=True).start()


def _reregister(region: str, entry: dict) -> None:
    """Register the worker's endpoint at its current address and prune all others.

    Registers the fresh address, then unregisters every other endpoint_id under
    the same name so exactly one live registration remains — otherwise stale rows
    (from a prior node before preemption, or the worker's own registration) pile
    up and the proxy round-robins onto a dead address.
    """
    from iris.cluster.types import JobName, TaskAttempt

    addr = _worker_address(region, entry, force=True)  # re-read heartbeat; never pin a stale cache
    if not addr:
        return
    client, _ = _get_controller()
    ta = TaskAttempt(task_id=JobName.from_wire(entry["job_name"].to_wire() + "/0"), attempt_id=0)
    new_id = client._cluster_client.register_endpoint(name=entry["wire_name"], address=addr, task_attempt=ta)
    entry["endpoint_id"] = new_id
    try:
        for ep in client._cluster_client.list_endpoints(entry["wire_name"], exact=True):
            if ep.endpoint_id and ep.endpoint_id != new_id:
                try:
                    client._cluster_client.unregister_endpoint(ep.endpoint_id)
                except Exception:
                    pass
    except Exception:
        pass


def _refresh_warm(region: str, entry: dict) -> None:
    """Ping the worker's /health; mark it warm when all datasets' text + BM25 are ready.

    On failure, drop the cached address so the next cycle re-resolves it from the
    heartbeat — a health timeout usually means the worker moved nodes (preemption)
    and the cached address is stale.
    """
    try:
        h = _call_entry(entry, "health", None, timeout=PROXY_CALL_TIMEOUT, method="GET")
        need = set(entry["datasets"])
        # Text-resolution readiness is TEXT ONLY — resolve/view-extractions never
        # touch BM25 (that's search retrieval), and BM25 is the slow ~22 GB mirror.
        # Gating text on text_ready lets a dataset's extractions show as soon as its
        # text.parquet mirror is done, without waiting for BM25.
        entry["ready"] = set(h.get("text_ready", []))
        # "Fully warm" (green status, search-ready) still needs both text + BM25.
        entry["warm"] = need.issubset(set(h.get("bm25_ready", []))) and need.issubset(set(h.get("text_ready", [])))
    except Exception:
        entry["warm"] = False
        entry["ready"] = set()
        entry.pop("address", None)


def _keepalive_loop() -> None:
    while True:
        try:
            for region in list(_workers):
                for entry in list(_workers.get(region, [])):
                    state = _job_state(entry)
                    entry["state"] = state  # cache so the hot path avoids per-call gRPC
                    if state in ("FAILED", "DONE", "GONE"):
                        with _lock:
                            if entry in _workers.get(region, []):
                                _workers[region].remove(entry)
                        continue
                    if state != "RUNNING":
                        continue
                    try:
                        _reregister(region, entry)
                    except Exception as e:
                        logger.warning("keepalive register %s failed: %s", entry["name"], e)
                    _refresh_warm(region, entry)
                # Top the region back up to the desired replica count.
                _ensure_replicas(region)
        except Exception:
            logger.exception("keepalive loop error")
        time.sleep(KEEPALIVE_INTERVAL)


# --- launch / ensure replicas ---


def _launch_one(region: str) -> dict:
    from iris.cluster.constraints import region_constraint
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    client, _ = _get_controller()
    datasets = datasets_in(region)
    env_vars = {"PYTHONUNBUFFERED": "1", "SPEC_EXPLORER_REGION": region, "SPEC_EXPLORER_DATASETS": json.dumps(datasets)}
    if os.environ.get(GEMINI_ENV):
        env_vars[GEMINI_ENV] = os.environ[GEMINI_ENV]
    name = f"spec-worker-{region}-{uuid.uuid4().hex[:6]}"
    job = client.submit(
        entrypoint=Entrypoint.from_command("python", "-m", "experiments.spec_explorer.worker"),
        name=name,
        # Big memory/disk, scaling with the us-central1 dataset count (8 as of
        # 2026-07-28: + high_quality_v2, med_quality, llm_pipeline_v1_1). Memory is
        # dominated by the in-RAM BM25 corpora (bm25s load scales with doc count;
        # dclm/nemotron are large) + the resident url_h maps — 96 GB OOM-killed
        # (exit 137) at 8 datasets, so 200 GB (v5p hosts have it). 100 GB is the
        # node's max allocatable disk (120 GB is unschedulable).
        resources=ResourceSpec(cpu=8.0, memory="200g", disk="100g"),
        environment=EnvironmentSpec(extras=["cpu"], pip_packages=["bm25s", "xxhash"], env_vars=env_vars),
        ports=["actor"],
        # Preemptible v5p is the only big-RAM pool in us-central1; replicas give
        # node-level redundancy and re-warm in-region (free) on preemption.
        constraints=[region_constraint([region])],
        max_retries_preemption=10,
    )
    entry = {
        "job_name": job.job_id,
        "name": name,
        "wire_name": f"{job.job_id.to_wire()}/{ENDPOINT_NAME}",
        "started_at": _now(),
        "datasets": datasets,
        "state": "PENDING",
        "warm": False,
    }
    _workers.setdefault(region, []).append(entry)
    logger.info("launched %s", name)
    return entry


def _live(region: str) -> list[dict]:
    return [e for e in _workers.get(region, []) if e.get("state", "PENDING") in ("RUNNING", "PENDING")]


def _ensure_replicas(region: str, replicas: int = DESIRED_REPLICAS) -> None:
    with _lock:
        deficit = replicas - len(_live(region))
        for _ in range(max(0, deficit)):
            try:
                _launch_one(region)
            except Exception:
                logger.exception("launch replica for %s failed", region)


def launch(region: str) -> dict:
    """Ensure ``DESIRED_REPLICAS`` workers for ``region`` (idempotent)."""
    if region not in all_regions():
        raise ValueError(f"no datasets in region {region!r}; known: {all_regions()}")
    _ensure_replicas(region)
    _start_keepalive()
    return status_one(region)


def reconnect() -> list[str]:
    """Rebuild ``_workers`` from RUNNING ``spec-worker-<region>-*`` jobs (survives restarts)."""
    from iris.cluster.types import JobName
    from iris.rpc import job_pb2

    try:
        client, _ = _get_controller()
        jobs = client.list_jobs(state=job_pb2.JOB_STATE_RUNNING)
    except Exception as e:
        logger.warning("reconnect list_jobs failed: %s", e)
        return []
    found = []
    for js in jobs:
        jn = JobName.from_wire(js.job_id)
        if not jn.name.startswith("spec-worker-"):
            continue
        region = jn.name[len("spec-worker-") :].rsplit("-", 1)[0]  # strip the -<shortid>
        if region not in all_regions():
            continue
        with _lock:
            if any(e["job_name"].to_wire() == jn.to_wire() for e in _workers.get(region, [])):
                continue
            _workers.setdefault(region, []).append(
                {
                    "job_name": jn,
                    "name": jn.name,
                    "wire_name": f"{jn.to_wire()}/{ENDPOINT_NAME}",
                    "started_at": _now(),
                    "datasets": datasets_in(region),
                    "state": "RUNNING",  # list_jobs filtered to RUNNING
                    "warm": False,
                }
            )
            found.append(jn.name)
    if found:
        logger.info("reconnected to workers: %s", found)
    if _workers:
        _start_keepalive()
    return found


# --- calling workers (with failover across replicas) ---


def _call_entry(entry: dict, subpath: str, body: dict | None, timeout: float, method: str = "POST") -> dict:
    url = f"{_get_proxy().rstrip('/')}/proxy/{_encoded(entry['wire_name'])}/{subpath}"
    # No Authorization header: the dashboard proxy is auth-optional and rejects a
    # bearer access token (401); the anonymous path works.
    data = json.dumps(body).encode() if (body is not None and method == "POST") else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as r:
        return json.loads(r.read())


def _pick(region: str) -> list[dict]:
    """Running workers for a region, warm ones first (so calls hit ready replicas).

    Reads the cached state (refreshed by the keepalive); a stale "RUNNING" is
    harmless — the actual HTTP call fails and :func:`call_worker` fails over.
    """
    running = [e for e in list(_workers.get(region, [])) if e.get("state") == "RUNNING"]
    return sorted(running, key=lambda e: (not e.get("warm", False), e["started_at"]))


def call_worker(region: str, subpath: str, body: dict, timeout: float = PROXY_CALL_TIMEOUT) -> dict:
    """POST to a region worker through the proxy, failing over across replicas."""
    workers = _pick(region)
    if not workers:
        raise RuntimeError(f"no RUNNING worker for {region}; launch one first")
    last = None
    for entry in workers:
        try:
            return _call_entry(entry, subpath, body, timeout)
        except Exception as e:
            last = e
            logger.warning("worker %s call %s failed (%s); failing over", entry["name"], subpath, str(e)[:80])
    raise last or RuntimeError(f"all workers for {region} failed")


def resolve_text(dataset: str, url_hs: list, max_chars: int = 20000, timeout: float = PROXY_CALL_TIMEOUT) -> list[dict]:
    """Resolve url_h -> extracted text via ``dataset``'s region worker."""
    out = call_worker(
        region_of(dataset), "resolve", {"dataset": dataset, "url_hs": list(url_hs), "max_chars": max_chars}, timeout
    )
    return out.get("results", [])


def bm25(dataset: str, query: str, k: int = 10) -> list[dict]:
    return call_worker(region_of(dataset), "bm25", {"dataset": dataset, "query": query, "k": k}).get("hits", [])


def bm25_multi(region: str, datasets: list, queries: list, k: int = 20) -> dict:
    """Fan out many (dataset, query) BM25 searches in-region in a single proxy call."""
    return call_worker(region, "bm25_multi", {"datasets": datasets, "queries": queries, "k": k})


def health(region: str) -> dict:
    """/health of the first warm (else any RUNNING) worker in the region."""
    workers = _pick(region)
    if not workers:
        raise RuntimeError(f"worker for {region} not RUNNING")
    return _call_entry(workers[0], "health", None, timeout=PROXY_CALL_TIMEOUT, method="GET")


def has_running_worker(region: str) -> bool:
    """True if any worker for ``region`` is RUNNING (regardless of warmth)."""
    return bool(_pick(region))


def ready_datasets(region: str) -> set:
    """Datasets a RUNNING worker in ``region`` can resolve NOW (text + BM25 mirrored).

    Per-dataset (cached from the keepalive health poll) so a partially-warmed
    worker serves what it has instead of hiding the whole region.
    """
    out: set = set()
    for e in _pick(region):
        out |= e.get("ready", set())
    return out


def partition_datasets(datasets: list) -> tuple[list, list, list]:
    """Split ``datasets`` into (ready-now, warming, down) by their region worker state.

    ``ready``: resolvable now; ``warming``: a worker is up but this dataset hasn't
    finished mirroring (retry shortly, no launch needed); ``down``: no worker in
    the region (offer to launch one).
    """
    available, warming, down = [], [], []
    for ds in datasets:
        region = region_of(ds)
        if ds in ready_datasets(region):
            available.append(ds)
        elif has_running_worker(region):
            warming.append(ds)
        else:
            down.append(ds)
    return available, warming, down


# --- stop / status ---


def stop(region: str) -> dict:
    entries = _workers.pop(region, [])
    if not entries:
        return {"region": region, "status": "NONE"}
    client, _ = _get_controller()
    for e in entries:
        try:
            client.terminate(e["job_name"])
        except Exception as ex:
            logger.warning("terminate %s failed: %s", e["name"], ex)
    return {"region": region, "status": "STOPPED", "count": len(entries)}


def status_one(region: str) -> dict:
    entries = _workers.get(region, [])
    workers = [
        {
            "name": e["name"],
            "state": e.get("state", "PENDING"),
            "warm": e.get("warm", False),
            "started_at": e["started_at"],
        }
        for e in entries
    ]
    running = [w for w in workers if w["state"] == "RUNNING"]
    warm = [w for w in running if w["warm"]]
    status = "RUNNING" if warm else ("STARTING" if running or workers else "NONE")
    return {
        "region": region,
        "status": status,
        "datasets": datasets_in(region),
        "desired": DESIRED_REPLICAS,
        "running": len(running),
        "warm": len(warm),
        "workers": workers,
    }


def status_all() -> list[dict]:
    return [status_one(r) for r in all_regions()]
