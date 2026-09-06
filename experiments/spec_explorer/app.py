# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Light-tier Flask backend for the spec-explorer dashboard.

Serves the single-page frontend and read-only ``/api/*`` endpoints. Expensive
GCS scans (eval aggregation) sit behind an in-memory + on-disk cache that is
only rebuilt on an explicit ``POST .../refresh`` — the same cache+refresh shape
as ``warc_scaling_dashboard.py``. Spec display needs no cache (pure imports).

Run::

    uv run python -m experiments.spec_explorer.app         # -> http://localhost:8096

Endpoints cover eval plots/winners, spec display, and coverage; the search and
in-region worker endpoints arrive in later phases.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from experiments.spec_explorer import (
    coverage_service,
    eval_aggregate,
    search_cache,
    search_service,
    specs_service,
    worker_manager,
)
from experiments.spec_explorer.catalog import (
    INDEX_DATASET_REGION,
    METHOD_STYLE,
    MODEL_PARAMS,
    epoch_flops,
    method_color,
    method_tokens,
)

logger = logging.getLogger(__name__)

DASHBOARD_PORT = 8096
STATIC_DIR = Path(__file__).parent / "static"
CACHE_DIR = Path(__file__).parent / "cache"
EVAL_CACHE_FILE = CACHE_DIR / "eval_matrix.json"

app = Flask(__name__)

_cache: dict[str, dict] = {
    "eval": {"rows": None, "summary": None, "updated_at": None},
    "coverage": {},  # keyed by identity key (url_h/rid_h/text_h/dom_h)
}


_reconnected = False


def _ensure_reconnected() -> None:
    """Rebuild worker handles from RUNNING jobs once, lazily (survives app restarts)."""
    global _reconnected
    if _reconnected:
        return
    _reconnected = True
    try:
        worker_manager.reconnect()
    except Exception:
        logger.exception("worker reconnect failed (ignored)")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _save_eval_cache() -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    try:
        EVAL_CACHE_FILE.write_text(json.dumps(_cache["eval"]))
    except Exception as e:
        logger.warning("failed to persist eval cache: %s", e)


def _load_eval_cache() -> None:
    try:
        if EVAL_CACHE_FILE.exists():
            loaded = json.loads(EVAL_CACHE_FILE.read_text())
            if loaded.get("rows") is not None:
                _cache["eval"] = loaded
                logger.info("loaded %d cached eval rows", len(loaded["rows"]))
    except Exception as e:
        logger.warning("failed to load eval cache: %s", e)


def _refresh_eval() -> dict:
    rows = eval_aggregate.build_rows()
    _cache["eval"] = {"rows": rows, "summary": eval_aggregate.summarize(rows), "updated_at": _now()}
    _save_eval_cache()
    return _cache["eval"]


@app.after_request
def _no_cache(resp):
    if request.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# --- routes ---


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:rel>")
def static_files(rel: str):
    return send_from_directory(STATIC_DIR, rel)


@app.route("/api/meta")
def api_meta():
    """Method color/label vocabulary shared by every tab."""
    return jsonify(
        {
            "method_style": {m: {"color": c, "label": lab} for m, (c, lab) in METHOD_STYLE.items()},
            # dim -> params label, so the frontend labels model sizes without a
            # full eval rescan when a new dim appears (rows carry a stale label).
            "model_labels": {str(dim): lab for dim, (lab, _) in MODEL_PARAMS.items()},
            "eval_updated_at": _cache["eval"]["updated_at"],
        }
    )


@app.route("/api/eval/matrix")
def api_eval_matrix():
    ev = _cache["eval"]
    if ev["rows"] is None:
        return jsonify({"rows": [], "summary": None, "updated_at": None, "stale": True})
    return jsonify(ev)


@app.route("/api/eval/refresh", methods=["POST"])
def api_eval_refresh():
    ev = _refresh_eval()
    return jsonify({"updated_at": ev["updated_at"], "summary": ev["summary"]})


@app.route("/api/epochs")
def api_epochs():
    """{method: {e1, e2}} — 1/2-epoch FLOPs per method at a (dim, N) for epoch marks."""
    try:
        dim = int(request.args["dim"])
        n = int(request.args["n"])
    except (KeyError, ValueError):
        return jsonify({"error": "dim and n required"}), 400
    out = {}
    for method in METHOD_STYLE:
        ef = epoch_flops(method, n, dim)
        if ef:
            out[method] = {"e1": ef[0], "e2": ef[1]}
    return jsonify(out)


@app.route("/api/tokens")
def api_tokens():
    """{method: tokens} — extracted-token count per method at a fixed WARC count N.

    Covers every method present in the eval data (not just the styled ones), so
    newer methods like ``high_quality_v2`` / ``med_quality`` get token labels too.
    """
    try:
        n = int(request.args["n"])
    except (KeyError, ValueError):
        return jsonify({"error": "n required"}), 400
    rows = _cache["eval"]["rows"] or []
    methods = {r["method"] for r in rows} or set(METHOD_STYLE)
    return jsonify({m: t for m in methods if (t := method_tokens(m, n)) is not None})


@app.route("/api/coverage/datasets")
def api_coverage_datasets():
    """Index datasets available for coverage, with region + color."""
    return jsonify(
        [{"dataset": ds, "region": region, "color": method_color(ds)} for ds, region in INDEX_DATASET_REGION.items()]
    )


@app.route("/api/coverage")
def api_coverage():
    """Pairwise Jaccard/containment matrix for a key (cached per key in memory)."""
    key = request.args.get("key", "url_h")
    if key not in coverage_service.KEY_CHOICES:
        return jsonify({"error": f"bad key {key!r}"}), 400
    if key not in _cache["coverage"]:
        _cache["coverage"][key] = coverage_service.matrix(key)
    return jsonify(_cache["coverage"][key])


@app.route("/api/coverage/setdiff", methods=["POST"])
def api_coverage_setdiff():
    """Count (+ optional deterministic sample) of a dataset set expression."""
    body = request.get_json(force=True) or {}
    include = body.get("include") or []
    exclude = body.get("exclude") or []
    key = body.get("key", "url_h")
    sample = int(body.get("sample", 0))
    offset = int(body.get("offset", 0))
    try:
        return jsonify(coverage_service.set_expression(include, exclude, key, sample, offset))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/worker/status")
def api_worker_status():
    _ensure_reconnected()
    return jsonify(worker_manager.status_all())


@app.route("/api/controller/ping")
def api_controller_ping():
    """Liveness of the controller tunnel (forces a rebuild if the SSH tunnel died).

    The monitor hits this: worker status is served from cached state and stays 200
    even when the tunnel is dead, so a check that actually touches the controller
    is needed to detect (and self-heal) a wedged connection.
    """
    try:
        ok = worker_manager.controller_ping()
        return jsonify({"ok": ok}), (200 if ok else 503)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 503


@app.route("/api/worker/launch", methods=["POST"])
def api_worker_launch():
    region = (request.get_json(force=True) or {}).get("region")
    try:
        return jsonify(worker_manager.launch(region))
    except Exception as e:
        logger.exception("worker launch failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/worker/stop", methods=["POST"])
def api_worker_stop():
    region = (request.get_json(force=True) or {}).get("region")
    return jsonify(worker_manager.stop(region))


@app.route("/api/text/resolve", methods=["POST"])
def api_text_resolve():
    """Sample a set expression and resolve each doc's extracted text side-by-side.

    Body: ``{include, exclude, key, sample}``. Returns the sampled ``url_h`` and,
    for each *include* dataset, that dataset's extraction — but only if the
    owning region's worker is RUNNING; otherwise reports which regions to launch.
    """
    body = request.get_json(force=True) or {}
    include = body.get("include") or []
    exclude = body.get("exclude") or []
    key = body.get("key", "url_h")
    sample = int(body.get("sample", 8))
    offset = int(body.get("offset", 0))
    if key != "url_h":
        return jsonify({"error": "text resolution requires key=url_h"}), 400
    _ensure_reconnected()

    expr = coverage_service.set_expression(include, exclude, key, sample=sample, offset=offset)
    url_hs = [int(x) for x in expr["sample"]]
    include = expr["include"]

    # Text is resolved in-region by the worker (reached via the controller's HTTP
    # endpoint proxy) so the large text.parquet never leaves its region. Resolve
    # each dataset that is ready NOW (per-dataset), so a partially-warmed worker
    # still serves what it has; report warming (retry) vs down (launch) separately.
    available, warming, down = worker_manager.partition_datasets(include)
    missing_regions = sorted({worker_manager.region_of(ds) for ds in down})
    warming_regions = sorted({worker_manager.region_of(ds) for ds in warming})
    if not available:
        return jsonify(
            {
                "count": expr["count"],
                "offset": offset,
                "sample": [str(h) for h in url_hs],
                "needs_worker": missing_regions,
                "warming": warming,
                "warming_regions": warming_regions,
                "include": include,
                "exclude": exclude,
            }
        )

    per_dataset: dict[str, dict[str, dict]] = {}
    for ds in available:
        try:
            rows = worker_manager.resolve_text(ds, url_hs)
        except Exception as e:
            logger.warning("resolve %s failed: %s", ds, e)
            rows = []
        per_dataset[ds] = {r["url_h"]: r for r in rows}

    docs = []
    for h in url_hs:
        h = str(h)
        extractions = {ds: per_dataset[ds].get(h) for ds in available if per_dataset[ds].get(h)}
        if extractions:
            docs.append({"url_h": h, "extractions": extractions})
    return jsonify(
        {
            "count": expr["count"],
            "offset": offset,
            "include": available,
            "exclude": exclude,
            "docs": docs,
            "unavailable": warming + down,
            "warming": warming,
            "warming_regions": warming_regions,
            "missing_regions": missing_regions,
        }
    )


# --- search (Gemini expand -> in-region BM25 -> Gemini rerank -> cross-ref) ---

_search_jobs: dict[str, dict] = {}


@app.route("/api/search", methods=["POST"])
def api_search():
    """Start (or return cached) a search. Returns a job_id to poll for progress."""
    body = request.get_json(force=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    # Default is the plain about-topic search; example mode is opt-in.
    example_mode = bool(body.get("example_mode", False))
    # One cache entry per query (last result wins), so the saved list stays a clean
    # set of real terms. Reuse the cache only when its mode matches the request.
    if not body.get("force"):
        cached = search_cache.get(query)
        if cached and bool(cached.get("example_mode", False)) == example_mode:
            return jsonify({"cached": True, "result": cached})
    _ensure_reconnected()
    job_id = uuid.uuid4().hex[:12]
    job = {
        "stage": "queued",
        "info": {},
        "elapsed": 0.0,
        "done": False,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "query": query,
    }
    _search_jobs[job_id] = job

    def run():
        def prog(stage, info):
            job["stage"] = stage
            job["info"] = info
            job["elapsed"] = round(time.time() - job["started_at"], 1)

        try:
            result = search_service.search(query, on_progress=prog, example_mode=example_mode)
            search_cache.save(query, result)
            job["result"] = result
            job["stage"] = "done"
        except Exception as e:
            logger.exception("search failed")
            job["error"] = str(e)
            job["stage"] = "error"
        finally:
            job["done"] = True

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/search/progress/<job_id>")
def api_search_progress(job_id: str):
    job = _search_jobs.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(
        {
            "stage": job["stage"],
            "info": job.get("info"),
            "elapsed": job.get("elapsed"),
            "done": job["done"],
            "result": job["result"],
            "error": job["error"],
            "query": job["query"],
        }
    )


@app.route("/api/search/backport", methods=["POST"])
def api_search_backport():
    """Re-run the cross-ref (+ extraction embedding) on every cached search so
    datasets registered after they were run (e.g. ``llm_pipeline_v1_1``) appear in
    the saved results. No Gemini/BM25 — local coverage keys + a bulk resolve.
    """
    _ensure_reconnected()
    reembed = bool((request.get_json(silent=True) or {}).get("reembed", True))
    pairs = []
    for saved in search_cache.list_saved():
        res = search_cache.get(saved["query"])
        if res and res.get("results"):
            pairs.append((saved["query"], res))
    search_service.backport_results([res for _, res in pairs], reembed=reembed)
    for q, res in pairs:
        search_cache.save(q, res)
    return jsonify({"updated": len(pairs), "reembed": reembed})


@app.route("/api/search/doc", methods=["POST"])
def api_search_doc():
    """Resolve one search result's extracted text across the datasets that kept it."""
    body = request.get_json(force=True) or {}
    url_h = body.get("url_h")
    datasets = body.get("datasets") or []
    if not url_h:
        return jsonify({"extractions": {}})
    _ensure_reconnected()
    available, warming, down = worker_manager.partition_datasets(datasets)
    missing_regions = sorted({worker_manager.region_of(ds) for ds in down})
    warming_regions = sorted({worker_manager.region_of(ds) for ds in warming})
    extractions = {}
    for ds in available:
        try:
            rows = worker_manager.resolve_text(ds, [int(url_h)])
            extractions[ds] = rows[0] if rows else None
        except Exception as e:
            logger.warning("doc resolve %s failed: %s", ds, e)
            extractions[ds] = None
    return jsonify(
        {
            "extractions": extractions,
            "unavailable": warming + down,
            "warming": warming,
            "warming_regions": warming_regions,
            "missing_regions": missing_regions,
        }
    )


@app.route("/api/search/saved")
def api_search_saved():
    return jsonify(search_cache.list_saved(request.args.get("favorites") == "1"))


@app.route("/api/search/get")
def api_search_get():
    r = search_cache.get(request.args.get("query", ""))
    return jsonify(r if r else {"error": "not cached"})


@app.route("/api/search/favorite", methods=["POST"])
def api_search_favorite():
    b = request.get_json(force=True) or {}
    search_cache.set_favorite(b.get("query", ""), bool(b.get("favorite")))
    return jsonify({"ok": True})


@app.route("/api/search/delete", methods=["POST"])
def api_search_delete():
    b = request.get_json(force=True) or {}
    search_cache.delete(b.get("query", ""))
    return jsonify({"ok": True})


@app.route("/api/spec/methods")
def api_spec_methods():
    return jsonify(specs_service.list_methods())


@app.route("/api/spec/<method>")
def api_spec(method: str):
    try:
        return jsonify(specs_service.describe_spec(method))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_eval_cache()
    logger.info("spec-explorer on http://localhost:%d", DASHBOARD_PORT)
    app.run(host="127.0.0.1", port=DASHBOARD_PORT, debug=False)


if __name__ == "__main__":
    main()
