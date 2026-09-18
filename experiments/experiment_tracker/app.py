# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment tracker dashboard — one page for every experiment feeding the paper.

Flask + single-page frontend, same shape as `spec_explorer.app`: probes run ONLY on an
explicit ``POST /api/refresh`` (never a polling loop), the snapshot is cached in memory
and on disk so a fresh server (or another session) starts from the last reading, and
``GET /api/experiments`` serves the cached snapshot instantly.

Run::

    uv run --with flask python -m experiments.experiment_tracker.app   # -> http://localhost:8097

Refresh from anywhere (a babysit loop, another session)::

    curl -s -X POST localhost:8097/api/refresh
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, send_from_directory

from experiments.experiment_tracker.fleet import fleet_snapshot
from experiments.experiment_tracker.probes import timed
from experiments.experiment_tracker.registry import EXPERIMENTS

logger = logging.getLogger(__name__)
app = Flask(__name__)

DASHBOARD_PORT = 8097
STATIC_DIR = Path(__file__).parent / "static"
CACHE_PATH = Path(__file__).parent / "cache" / "snapshot.json"

_lock = threading.Lock()
_snapshot: dict = {"generated_at": None, "refreshing": False, "experiments": []}


def _load_disk_cache() -> None:
    global _snapshot
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as fh:
            _snapshot = json.load(fh)
        _snapshot["refreshing"] = False


def _refresh() -> None:
    rows = []
    for exp in EXPERIMENTS:
        reading, elapsed = timed(exp.probe)
        rows.append(
            {
                "id": exp.id,
                "name": exp.name,
                "group": exp.group,
                "unit": exp.unit,
                "detail": exp.detail,
                "links": exp.links,
                "probe_seconds": round(elapsed, 1),
                **asdict(reading),
            }
        )
        logger.info("probed %s in %.1fs", exp.id, elapsed)
    snap = {"generated_at": time.time(), "refreshing": False, "experiments": rows}
    with _lock:
        _snapshot.clear()
        _snapshot.update(snap)
    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump(snap, fh)


def _refresh_async() -> None:
    try:
        _refresh()
    finally:
        with _lock:
            _snapshot["refreshing"] = False


MIXTURES_DIR = Path(__file__).parent.parent / "data_mixing" / "mixtures"


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


_fleet_cache: dict = {}


@app.get("/fleet")
def fleet_page():
    return send_from_directory(STATIC_DIR, "fleet.html")


@app.get("/api/fleet")
def fleet():
    from flask import request

    global _fleet_cache
    if request.args.get("force") or time.time() - _fleet_cache.get("generated_at", 0) > 300:
        snap = fleet_snapshot()
        snap["generated_at"] = time.time()
        _fleet_cache = snap
    return jsonify(_fleet_cache)


@app.get("/mixtures")
def mixtures_page():
    return send_from_directory(STATIC_DIR, "mixtures.html")


@app.get("/api/mixtures")
def mixtures():
    out = {}
    for path in sorted(MIXTURES_DIR.glob("*.json")):
        with open(path) as fh:
            doc = json.load(fh)
        out[path.stem] = {
            "weights": doc.get("weights", {}),
            "natural": doc.get("natural", {}),
            "meta": {k: v for k, v in doc.items() if not isinstance(v, dict)},
        }
    return jsonify(out)


@app.get("/api/experiments")
def experiments():
    with _lock:
        return jsonify(_snapshot)


@app.post("/api/refresh")
def refresh():
    with _lock:
        if _snapshot.get("refreshing"):
            return jsonify({"started": False, "reason": "refresh already running"})
        _snapshot["refreshing"] = True
    threading.Thread(target=_refresh_async, daemon=True).start()
    return jsonify({"started": True})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--refresh-on-start", action="store_true")
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=900.0,
        help="Seconds between automatic probe refreshes (0 disables; default 15 min — "
        "coarse enough to stay within GCS-listing cost discipline, fresh enough that the "
        "page is never an hour behind reality).",
    )
    args = parser.parse_args()
    _load_disk_cache()
    if args.refresh_on_start:
        threading.Thread(target=_refresh_async, daemon=True).start()
        with _lock:
            _snapshot["refreshing"] = True

    if args.refresh_interval > 0:

        def _auto_refresh():
            while True:
                time.sleep(args.refresh_interval)
                with _lock:
                    if _snapshot.get("refreshing"):
                        continue
                    _snapshot["refreshing"] = True
                _refresh_async()

        threading.Thread(target=_auto_refresh, daemon=True).start()
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
