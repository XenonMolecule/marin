# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live ladder dashboard: WARC progress bars vs the 2M target and the full 8M pool.

Serves one auto-refreshing page (default :8093) summarizing the fused run: shards/WARCs/kept
docs completed, completion rate over a rolling window, ETAs, and per-region fleet size.
Reads only bookkeeping objects (one sentinel LIST + one iris job list per refresh).

    uv run python -m experiments.fast_curation.ladder_dashboard --port 8093
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fsspec
import pyarrow.parquet as pq

from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

RUNG1_WARCS = 99_638  # two-phase v2.1 rung 1 (shards 0-403), counted separately
POOL_8M = 7_925_398
REFRESH_SECONDS = 60.0
KEPT_DOCS_PER_WARC = 7_430  # measured mean across 200k verified WARCs

_STATE: dict = {"ts": 0.0, "history": []}
_LOCK = threading.Lock()


def _collect(spec) -> dict:
    # Fresh client per attempt: a hung socket in the cached singleton would poison
    # every subsequent refresh.
    fs = fsspec.filesystem("gcs", skip_instance_cache=True)
    done = set()
    now = time.time()
    # Sentinel mtimes ARE the completion history: trailing-window rates come straight from
    # GCS on every refresh — no local state, no warm-up, restart-proof.
    listing = fs.ls(sw.sentinel_prefix(spec, "b").removeprefix("gs://"), detail=True, refresh=True)
    mtimes: dict[int, float] = {}
    for entry in listing:
        p = entry["name"] if isinstance(entry, dict) else entry
        digits = "".join(c for c in p.rsplit("/", 1)[-1] if c.isdigit())
        if not digits:
            continue
        s = int(digits)
        done.add(s)
        mt = entry.get("mtime") if isinstance(entry, dict) else None
        if mt is not None:
            mtimes[s] = mt.timestamp() if hasattr(mt, "timestamp") else float(mt)
    with fs.open(sw.index_path(spec).removeprefix("gs://")) as f:
        nw = {e["shard"]: e["n_warcs"] for e in pq.read_table(f).to_pylist()}
    fused_shards = [s for s in done if 404 <= s < 8080]
    fused_warcs = sum(nw[s] for s in fused_shards)
    ladder_target = sum(n for s, n in nw.items() if s < 8080)
    rates = {
        h: sum(nw.get(s, 0) for s in fused_shards if mtimes.get(s, 0) >= now - h * 3600) / h for h in RATE_WINDOWS_HOURS
    }
    try:
        out = subprocess.run(
            [
                "uv",
                "run",
                "iris",
                "--cluster",
                "marin",
                "job",
                "list",
                "--prefix",
                "/michaelryan/fastcur-coord",
                "--limit",
                "900",
                "--state",
                "running",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        ).stdout
        workers = sum(1 for line in out.splitlines() if "/fastcur-fused" in line)
    except Exception:
        workers = -1  # unknown, never zero
    return {
        "ts": time.time(),
        "warcs_done": RUNG1_WARCS + fused_warcs,
        "ladder_target": ladder_target,
        "fused_shards_done": len(fused_shards),
        "fused_shards_target": 8080 - 404,
        "workers": workers,
        "rates": rates,
    }


COLLECT_DEADLINE_SECONDS = 120.0


def _refresher(spec) -> None:
    # Each refresh runs in a throwaway thread with a hard deadline: gcsfs calls have no
    # timeout of their own, and a single hung LIST otherwise freezes this loop forever
    # while the page keeps re-serving stale numbers (observed 2026-09-04: 66 min stale).
    # A timed-out worker thread is abandoned (daemon) and the next cycle starts clean.
    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
    from concurrent.futures import TimeoutError as FutTimeout  # noqa: PLC0415

    while True:
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            snap = ex.submit(_collect, spec).result(timeout=COLLECT_DEADLINE_SECONDS)
            with _LOCK:
                _STATE.update(snap)
        except FutTimeout:
            logger.warning("refresh exceeded %ss (hung GCS call?) — abandoned, retrying", COLLECT_DEADLINE_SECONDS)
        except Exception as e:
            logger.warning("refresh failed (stale data shown): %s", e)
        finally:
            ex.shutdown(wait=False)
        time.sleep(REFRESH_SECONDS)


RATE_WINDOWS_HOURS = (1, 3, 6, 12, 24)  # trailing windows from sentinel mtimes


def _fmt_h(h: float) -> str:
    if h <= 0:
        return "—"
    return f"~{h:.1f} h" if h < 48 else f"~{h / 24:.1f} days"


def _windows_table(s: dict) -> str:
    """Rows: window | rate | ETA 2M | 8M timeframe. Sentinel mtimes make these exact."""
    w1 = s.get("warcs_done", 0)
    rows = []
    for h in RATE_WINDOWS_HOURS:
        rate = s.get("rates", {}).get(h) or s.get("rates", {}).get(str(h), 0)
        if rate and rate > 0:
            eta2m = _fmt_h((s["ladder_target"] - w1) / rate)
            eta8m = _fmt_h((POOL_8M - w1) / rate)
            rows.append(f"<tr><td>{h}h</td><td>{rate:,.0f}/h</td><td>{eta2m}</td><td>{eta8m}</td></tr>")
        else:
            rows.append(f"<tr><td>{h}h</td><td>measuring…</td><td>—</td><td>—</td></tr>")
    return "".join(rows)


PAGE = """<!doctype html><html><head><title>Ladder Progress</title>
<meta http-equiv="refresh" content="60">
<style>
 body {{ font-family: ui-monospace, monospace; background: #0d1117; color: #e6edf3;
        padding: 2rem; max-width: 900px; margin: auto; }}
 h1 {{ color: #58d68d; font-size: 1.3rem; }}
 .bar {{ background: #21262d; border-radius: 8px; height: 34px; margin: .4rem 0 1.4rem; overflow: hidden; }}
 .fill {{ height: 100%; border-radius: 8px 0 0 8px; display: flex; align-items: center;
          padding-left: .8rem; font-weight: bold; color: #0d1117; white-space: nowrap; }}
 .l2m {{ background: linear-gradient(90deg, #58d68d, #2ecc71); }}
 .l8m {{ background: linear-gradient(90deg, #5dade2, #2e86c1); }}
 .meta {{ color: #8b949e; margin-bottom: .2rem; }}
 table {{ border-collapse: collapse; margin-top: 1rem; }}
 td {{ padding: .25rem 1.2rem .25rem 0; }}
</style></head><body>
<h1>WARC Curation Ladder &mdash; live</h1>
<div class="meta">2M ladder target ({ladder_target:,} WARCs)</div>
<div class="bar"><div class="fill l2m" style="width:{pct2m:.2f}%">{warcs:,} &nbsp;({pct2m:.1f}%)</div></div>
<div class="meta">Full 8M pool ({pool8m:,} WARCs)</div>
<div class="bar"><div class="fill l8m" style="width:{pct8m:.2f}%">{warcs:,} &nbsp;({pct8m:.1f}%)</div></div>
<table>
<tr><td>Fused shards done</td><td>{shards:,} / {shards_t:,}</td></tr>
<tr><td>Kept documents (est.)</td><td>{kept:,}</td></tr>
<tr><td>Fleet (fused workers running)</td><td>{workers}</td></tr>
<tr><td>Updated</td><td>{age:.0f}s ago (auto-refresh 60s){stale_note}</td></tr>
</table>
<h1 style="margin-top:1.6rem">Rate windows &amp; timeframes</h1>
<table>
<tr class="meta"><td>window</td><td>rate</td><td>ETA 2M ladder</td><td>8M timeframe</td></tr>
{windows_rows}
</table>
<div class="meta" style="margin-top:.6rem">Short windows show current weather; long windows include
overnight troughs. Truth for planning usually sits between the 6h and 24h rows.</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _LOCK:
            s = dict(_STATE)
        if self.path == "/json":
            body = json.dumps(s).encode()
            ctype = "application/json"
        else:
            body = PAGE.format(
                ladder_target=s.get("ladder_target", 0),
                pool8m=POOL_8M,
                warcs=s.get("warcs_done", 0),
                pct2m=100 * s.get("warcs_done", 0) / max(s.get("ladder_target", 1), 1),
                pct8m=100 * s.get("warcs_done", 0) / POOL_8M,
                shards=s.get("fused_shards_done", 0),
                shards_t=s.get("fused_shards_target", 0),
                kept=s.get("warcs_done", 0) * KEPT_DOCS_PER_WARC,
                workers=s.get("workers", "?"),
                windows_rows=_windows_table(s),
                age=(age := time.time() - s.get("ts", time.time())),
                stale_note=(
                    ' &mdash; <span style="color:#f0883e">STALE (refresh failing; numbers frozen)</span>'
                    if age > 300
                    else ""
                ),
            ).encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet request logging
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="lpv11_fastpipe_v2_1_fused")
    ap.add_argument("--port", type=int, default=8093)
    args = ap.parse_args()
    spec = get_spec(args.spec)
    threading.Thread(target=_refresher, args=(spec,), daemon=True).start()
    logger.info("ladder dashboard on http://localhost:%d", args.port)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
