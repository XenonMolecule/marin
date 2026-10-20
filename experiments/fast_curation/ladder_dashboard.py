# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live curation dashboard: WARC progress toward the full 8M pool, rates, ETAs, fleet by pool.

Serves one auto-refreshing page (default :8093) summarizing the fused run: WARCs/shards/kept
docs completed, progress of the current run segment, completion rate over several trailing
windows with projected completion dates, and running fused workers broken down by TPU pool.
Reads only bookkeeping objects (one sentinel LIST + one iris job list per refresh). A slower
thread tallies stored bytes per region every few hours from full object listings.

    uv run python -m experiments.fast_curation.ladder_dashboard --port 8093
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import pathlib
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import fsspec
import pyarrow.parquet as pq
from google.cloud import storage
from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2

from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)

RUNG1_WARCS = 99_638  # two-phase v2.1 rung 1 (shards 0-403), counted separately
FIRST_FUSED_SHARD = 404
SEGMENT_START_SHARD = 8080  # the 8M run's first shard; below it is the completed 2M ladder
POOL_8M = 7_925_398
REFRESH_SECONDS = 60.0
KEPT_DOCS_PER_WARC = 7_407  # measured over the verified 2M ladder
COLLECT_DEADLINE_SECONDS = 120.0
FLEET_DEADLINE_SECONDS = 120.0
FLEET_LIST_LIMIT = 3000
# Trailing rate windows, measured as the change in warcs_done between recorded progress snapshots.
# Sentinel mtimes cannot be used: a worker whose done-registry is stale re-claims a finished shard,
# finds every WARC marked, and rewrites its sentinel, so on 2026-10-13 185 sentinels were written
# in an hour that closed only 61 new shards (mtime rates read ~3x too high).
RATE_WINDOWS_HOURS = (1, 3, 6, 12, 24, 72)
PROGRESS_HISTORY = pathlib.Path.home() / ".cache" / "ladder_dashboard_progress.json"
# A window's rate is shown once the snapshots span this fraction of it.
RATE_MIN_COVERAGE = 0.9

# Child job ids end in "...-<tpu type>-<region>-<seed>", e.g. ...-v6e-8-us-east5-9302.
_CHILD_POOL_RE = re.compile(r"-(v[0-9a-z]+-\d+)-([a-z]+-[a-z]+\d+)-\d+$")

STORAGE_REFRESH_SECONDS = 3 * 3600
STORAGE_CACHE = pathlib.Path.home() / ".cache" / "ladder_dashboard_storage.json"
# kept/ objects created before this belong to the 2M ladder (including its consolidated us-east1
# copy, written 2026-09-23); the 8M segment's first shard finished 2026-09-24 23:44 UTC.
SEGMENT_CUTOVER = datetime.datetime(2026, 9, 24, tzinfo=datetime.UTC)
DEDUP_ROOT = "documents/baseline_lpv11_fastpipe_v2_1_fused_deduped"

_LIVE_JOB_STATES = (job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_PENDING)

_STATE: dict = {"ts": 0.0}
_LOCK = threading.Lock()


def _chips_per_slice(tpu_type: str) -> int:
    """Chips in one slice. v4/v5p names count TensorCores (2 per chip); v5e/v6e names count chips."""
    gen, _, n = tpu_type.rpartition("-")
    return int(n) // 2 if gen in ("v4", "v5p") else int(n)


def _hard_deadline(seconds: float) -> list[str]:
    """Command prefix that SIGKILLs the child after ``seconds`` from outside this process.

    ``subprocess.run(timeout=...)`` alone did not fire on 2026-10-06: a progress child sat 11.5 min
    past its deadline after a DNS outage and stalled both refresh loops until killed by hand.
    """
    return ["timeout", "-s", "KILL", str(int(seconds))]


def _list_child_pools() -> dict[str, dict[str, int]]:
    """Fused worker jobs by "<tpu>@<region>": {"executing": n, "waiting": n} (runs in a subprocess).

    Counted by TASK state: a preempted worker's job stays RUNNING while its task sits pending for
    a new slice, so job state alone overstates the live fleet (2026-10-03: 332 "running", 150
    executing). One query per live state: an unfiltered listing also returns every finished child
    and times out on the controller.
    """
    pools: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"executing": 0, "waiting": 0})
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        for state in _LIVE_JOB_STATES:
            for job in client.list_jobs(state=state, prefix="/michaelryan/fastcur-coord", limit=FLEET_LIST_LIMIT):
                if "/fastcur-fused" not in job.job_id:
                    continue
                m = _CHILD_POOL_RE.search(job.job_id)
                pool = f"{m.group(1)}@{m.group(2)}" if m else "other"
                pools[pool]["executing" if job.task_state_counts.get("running", 0) else "waiting"] += 1
    return dict(pools)


def _child_pools() -> tuple[collections.Counter[str], collections.Counter[str]] | None:
    """(executing, waiting) per pool; None when the listing fails.

    The listing runs in a child process killed at FLEET_DEADLINE_SECONDS: in-process, an abandoned
    call keeps its controller connection retrying for up to 240 attempts and later calls stall
    behind it (2026-10-04: fleet data 22 min stale while direct queries took 0.2 s).
    """
    try:
        out = subprocess.run(
            [
                *_hard_deadline(FLEET_DEADLINE_SECONDS),
                sys.executable,
                "-m",
                "experiments.fast_curation.ladder_dashboard",
                "--fleet-json",
            ],
            capture_output=True,
            text=True,
            timeout=FLEET_DEADLINE_SECONDS + 30,
            check=True,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning("fleet listing failed: %s", type(e).__name__)
        return None
    pools = json.loads(out.strip().splitlines()[-1])
    executing = collections.Counter({k: v["executing"] for k, v in pools.items()})
    waiting = collections.Counter({k: v["waiting"] for k, v in pools.items()})
    return executing, waiting


def _fleet() -> tuple[int, dict[str, dict[str, int]]]:
    """Worker slices per pool: allocated (executing) and requested (executing + waiting).

    Coordinators only submit a worker when they want one, so executing + waiting is the demand we
    have placed and waiting (pending, or preempted and re-queued) is the part not currently
    granted. (-1, {}) when unknown — never zero.
    """
    counts = _child_pools()
    if counts is None:
        return -1, {}
    running, pending = counts
    pools = {}
    for pool in set(running) | set(pending):
        tpu = pool.split("@", 1)[0]
        chips = _chips_per_slice(tpu) if pool != "other" else 0
        pools[pool] = {
            "allocated": running[pool],
            "requested": running[pool] + pending[pool],
            "chips_allocated": running[pool] * chips,
            "chips_requested": (running[pool] + pending[pool]) * chips,
        }
    return sum(running.values()), pools


def _tree_bytes(url: str) -> tuple[int, int] | None:
    """Bytes under ``url`` created (before, at-or-after) SEGMENT_CUTOVER; None when the listing fails.

    Streams the listing through the storage client (application-default credentials) rather than
    the gcloud CLI, whose user login expires and then fails every listing (2026-10-06).
    """
    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    before = after = 0
    try:
        blobs = storage.Client().list_blobs(
            bucket, prefix=prefix.rstrip("/") + "/", fields="items(name,size,timeCreated),nextPageToken"
        )
        for blob in blobs:
            if blob.time_created < SEGMENT_CUTOVER:
                before += blob.size
            else:
                after += blob.size
    except Exception as e:
        logger.warning("listing %s failed: %s", url, e)
        return None
    return before, after


def _region_storage(spec, region: str) -> dict[str, int | None]:
    bucket = sw.REGION_TO_BUCKET[region]
    ns = spec.namespace(bucket)
    kept = _tree_bytes(f"{ns}/kept")
    kept_text = _tree_bytes(f"{ns}/kept_text")
    dedup = _tree_bytes(f"{bucket}/{DEDUP_ROOT}")
    return {
        "kept_2m": kept and kept[0],
        "kept_8m": kept and kept[1],
        "derived": None if kept_text is None or dedup is None else sum(kept_text) + sum(dedup),
    }


def _storage_refresher(spec) -> None:
    # Show the last completed tally across restarts: a full tally takes ~45 min.
    if STORAGE_CACHE.exists():
        with _LOCK:
            _STATE.update(json.loads(STORAGE_CACHE.read_text()))
    while True:
        t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=len(sw.REGION_TO_BUCKET)) as ex:
                futures = {r: ex.submit(_region_storage, spec, r) for r in sw.REGION_TO_BUCKET}
                storage = {r: f.result() for r, f in futures.items()}
            with _LOCK:
                # A failed listing (None) keeps the last good value rather than blanking the table.
                prev = _STATE.get("storage") or {}
                for region, fields in storage.items():
                    for key, value in fields.items():
                        if value is None:
                            fields[key] = prev.get(region, {}).get(key)
                _STATE["storage"] = storage
                _STATE["storage_ts"] = time.time()
            STORAGE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            STORAGE_CACHE.write_text(json.dumps({"storage": storage, "storage_ts": _STATE["storage_ts"]}))
            logger.info("storage tally refreshed in %.0fs", time.time() - t0)
        except Exception as e:
            logger.warning("storage tally failed (previous tally shown): %s", e)
        time.sleep(STORAGE_REFRESH_SECONDS)


def _collect(spec, max_shard: int) -> dict:
    # Fresh client per attempt: a hung socket in the cached singleton would poison every refresh.
    fs = fsspec.filesystem("gcs", skip_instance_cache=True)
    now = time.time()
    names = [
        p.rsplit("/", 1)[-1].removeprefix("data-")
        for p in fs.ls(sw.sentinel_prefix(spec, "b").removeprefix("gs://"), refresh=True)
    ]
    with fs.open(sw.index_path(spec).removeprefix("gs://")) as f:
        nw = {e["shard"]: e["n_warcs"] for e in pq.read_table(f).to_pylist()}

    fused = [s for s in (int(n) for n in names if n.isdigit()) if FIRST_FUSED_SHARD <= s < max_shard]
    segment = [s for s in fused if s >= SEGMENT_START_SHARD]
    return {
        "ts": now,
        "warcs_done": RUNG1_WARCS + sum(nw[s] for s in fused),
        "target_warcs": RUNG1_WARCS + sum(n for s, n in nw.items() if FIRST_FUSED_SHARD <= s < max_shard),
        "shards_done": len(fused) + FIRST_FUSED_SHARD,
        "shards_target": max_shard,
        "segment_warcs_done": sum(nw[s] for s in segment),
        "segment_warcs_target": sum(n for s, n in nw.items() if SEGMENT_START_SHARD <= s < max_shard),
    }


def _load_history() -> list[tuple[float, int]]:
    """``(ts, warcs_done)`` snapshots recorded by earlier refreshes, oldest first."""
    if not PROGRESS_HISTORY.exists():
        return []
    return [(float(t), int(d)) for t, d in json.loads(PROGRESS_HISTORY.read_text())]


def _windowed_rates(history: list[tuple[float, int]]) -> dict[int, float]:
    """WARCs/h over each trailing window, measured from the oldest snapshot inside it.

    A window is omitted until the snapshots span ``RATE_MIN_COVERAGE`` of it. Only snapshots after
    the last drop in the done count count: reopening shards removes their sentinels.
    """
    last_drop = max((i for i in range(1, len(history)) if history[i][1] < history[i - 1][1]), default=0)
    history = history[last_drop:]
    now, done_now = history[-1]
    rates = {}
    for h in RATE_WINDOWS_HOURS:
        t0, d0 = next((t, d) for t, d in history if t >= now - h * 3600)
        if now - t0 >= RATE_MIN_COVERAGE * h * 3600:
            rates[h] = (done_now - d0) / ((now - t0) / 3600)
    return rates


def _refresher(spec_id: str, max_shard: int) -> None:
    # Each refresh runs in a child process killed at COLLECT_DEADLINE_SECONDS. A thread was not
    # enough: gcsfs calls have no timeout of their own (2026-09-04: 66 min stale), and after a local
    # DNS outage the in-process credentials never refreshed again until restart (2026-10-06: 51 and
    # 77 min stale). A fresh process gets fresh credentials every cycle.
    history = _load_history()
    while True:
        try:
            out = subprocess.run(
                [
                    *_hard_deadline(COLLECT_DEADLINE_SECONDS),
                    sys.executable,
                    "-m",
                    "experiments.fast_curation.ladder_dashboard",
                    "--spec",
                    spec_id,
                    "--max-shard",
                    str(max_shard),
                    "--progress-json",
                ],
                capture_output=True,
                text=True,
                timeout=COLLECT_DEADLINE_SECONDS + 30,
                check=True,
            ).stdout
            snap = json.loads(out.strip().splitlines()[-1])
            history.append((snap["ts"], snap["warcs_done"]))
            history = [(t, d) for t, d in history if t >= snap["ts"] - (max(RATE_WINDOWS_HOURS) + 1) * 3600]
            PROGRESS_HISTORY.write_text(json.dumps(history))
            snap["rates"] = _windowed_rates(history)
            with _LOCK:
                _STATE.update(snap)
        except subprocess.TimeoutExpired:
            logger.warning("refresh exceeded %ss — killed, retrying", COLLECT_DEADLINE_SECONDS)
        except subprocess.CalledProcessError as e:
            logger.warning("refresh failed (stale data shown): %s", e.stderr.strip().splitlines()[-1:])
        time.sleep(REFRESH_SECONDS)


def _fleet_refresher() -> None:
    # Separate from _refresher so a slow controller listing never freezes the progress numbers.
    while True:
        workers, pools = _fleet()
        if workers >= 0:
            with _LOCK:
                _STATE.update(workers=workers, pools=pools, fleet_ts=time.time())
        time.sleep(REFRESH_SECONDS)


def _fmt_h(h: float) -> str:
    return f"~{h:.1f} h" if h < 48 else f"~{h / 24:.1f} days"


def _windows_table(s: dict) -> str:
    """Rows: window | rate | time remaining | projected completion date."""
    remaining = s.get("target_warcs", 0) - s.get("warcs_done", 0)
    rows = []
    for h in RATE_WINDOWS_HOURS:
        rate = s.get("rates", {}).get(h) or s.get("rates", {}).get(str(h), 0)
        if not rate:
            rows.append(f"<tr><td>{h}h</td><td>measuring…</td><td>—</td><td>—</td></tr>")
            continue
        hours = remaining / rate
        done_at = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=hours)
        rows.append(
            f"<tr><td>{h}h</td><td>{rate:,.0f} WARCs/h</td><td>{_fmt_h(hours)}</td>" f"<td>{done_at:%a %b %d}</td></tr>"
        )
    return "".join(rows)


def _pools_table(s: dict) -> str:
    """Rows: pool | slices allocated/requested | chips allocated/requested | fill bar."""
    pools = s.get("pools") or {}
    if not pools:
        return '<tr><td class="meta">no worker jobs reported (listing failed or fleet empty)</td></tr>'
    head = (
        '<tr class="meta"><td>pool</td><td>slices<br>granted / asked</td><td>chips<br>granted / asked</td>'
        "<td>granted</td></tr>"
    )
    rows, tot = [head], collections.Counter()
    for pool, p in sorted(pools.items(), key=lambda kv: (-kv[1]["chips_requested"], kv[0])):
        tot.update(p)
        pct = 100 * p["allocated"] / p["requested"] if p["requested"] else 0
        color = "#2ecc71" if pct >= 99.5 else "#f0b429" if pct >= 50 else "#f0883e"
        rows.append(
            f"<tr><td>{pool}</td><td>{p['allocated']} / {p['requested']}</td>"
            f"<td>{p['chips_allocated']:,} / {p['chips_requested']:,}</td>"
            f'<td><div class="mini"><div style="width:{pct:.0f}%;background:{color}"></div></div> {pct:.0f}%</td></tr>'
        )
    tot_pct = 100 * tot["chips_allocated"] / tot["chips_requested"] if tot["chips_requested"] else 0
    rows.append(
        f'<tr class="total"><td>TOTAL</td><td>{tot["allocated"]} / {tot["requested"]}</td>'
        f'<td>{tot["chips_allocated"]:,} / {tot["chips_requested"]:,}</td><td>{tot_pct:.0f}% of chips</td></tr>'
    )
    return "".join(rows)


def _tb(n: int | None) -> str:
    return "?" if n is None else f"{n / 1e12:,.1f} TB"


def _storage_table(s: dict) -> str:
    """Rows: region | 2M-ladder kept | 8M-segment kept (+ share) | derived | total."""
    storage = s.get("storage")
    if not storage:
        return '<tr><td class="meta">first tally in progress (full listings take ~15-30 min)</td></tr>'
    head = (
        '<tr class="meta"><td>region</td><td>2M ladder<br>kept</td><td>8M segment<br>kept</td>'
        "<td>share of<br>8M segment</td><td>kept_text +<br>dedup trees</td><td>total</td></tr>"
    )
    seg_total = sum(v["kept_8m"] or 0 for v in storage.values())
    rows, tot = [head], collections.Counter()
    for region, v in sorted(storage.items(), key=lambda kv: -(kv[1]["kept_8m"] or 0)):
        parts = [v["kept_2m"], v["kept_8m"], v["derived"]]
        total = None if None in parts else sum(parts)
        tot.update({k: x or 0 for k, x in v.items()})
        share = 100 * (v["kept_8m"] or 0) / seg_total if seg_total else 0
        rows.append(
            f"<tr><td>{region}</td><td>{_tb(v['kept_2m'])}</td><td>{_tb(v['kept_8m'])}</td>"
            f'<td><div class="mini"><div style="width:{share:.0f}%;background:#5dade2"></div></div> {share:.0f}%</td>'
            f"<td>{_tb(v['derived'])}</td><td>{_tb(total)}</td></tr>"
        )
    rows.append(
        f'<tr class="total"><td>TOTAL</td><td>{_tb(tot["kept_2m"])}</td><td>{_tb(tot["kept_8m"])}</td><td></td>'
        f'<td>{_tb(tot["derived"])}</td><td>{_tb(sum(tot.values()))}</td></tr>'
    )
    age = (time.time() - s.get("storage_ts", time.time())) / 3600
    rows.append(f'<tr><td class="meta" colspan="6">tallied {age:.1f} h ago (refreshes every 3 h)</td></tr>')
    return "".join(rows)


PAGE = """<!doctype html><html><head><title>8M Curation Run</title>
<meta http-equiv="refresh" content="60">
<style>
 body {{ font-family: ui-monospace, monospace; background: #0d1117; color: #e6edf3;
        padding: 2rem; max-width: 900px; margin: auto; }}
 h1 {{ color: #58d68d; font-size: 1.3rem; }}
 .bar {{ background: #21262d; border-radius: 8px; height: 34px; margin: .4rem 0 1.4rem; overflow: hidden; }}
 .fill {{ height: 100%; border-radius: 8px 0 0 8px; display: flex; align-items: center;
          padding-left: .8rem; font-weight: bold; color: #0d1117; white-space: nowrap; }}
 .all {{ background: linear-gradient(90deg, #5dade2, #2e86c1); }}
 .seg {{ background: linear-gradient(90deg, #58d68d, #2ecc71); }}
 .meta {{ color: #8b949e; margin-bottom: .2rem; }}
 table {{ border-collapse: collapse; margin-top: 1rem; }}
 td {{ padding: .25rem 1.2rem .25rem 0; vertical-align: middle; }}
 .mini {{ display: inline-block; width: 90px; height: 10px; background: #21262d; border-radius: 4px;
          overflow: hidden; vertical-align: middle; }}
 .mini div {{ height: 100%; }}
 .total td {{ border-top: 1px solid #30363d; font-weight: bold; padding-top: .4rem; }}
</style></head><body>
<h1>WARC Curation &mdash; 8M run, live</h1>
<div class="meta">Full pool ({target:,} WARCs)</div>
<div class="bar"><div class="fill all" style="width:{pct_all:.2f}%">{warcs:,} &nbsp;({pct_all:.1f}%)</div></div>
<div class="meta">This run's segment (shards {seg_start:,}+, {seg_target:,} WARCs)</div>
<div class="bar"><div class="fill seg" style="width:{pct_seg:.2f}%">{seg_done:,} &nbsp;({pct_seg:.1f}%)</div></div>
<table>
<tr><td>Shards done</td><td>{shards:,} / {shards_t:,}</td></tr>
<tr><td>Kept documents (est.)</td><td>{kept:,}</td></tr>
<tr><td>Fleet (workers executing)</td><td>{workers} <span class="meta">(listed {fleet_age:.0f}s ago)</span></td></tr>
<tr><td>Updated</td><td>{age:.0f}s ago (auto-refresh 60s){stale_note}</td></tr>
</table>
<h1 style="margin-top:1.6rem">Rate windows &amp; projected completion</h1>
<table>
<tr class="meta"><td>window</td><td>rate</td><td>time remaining</td><td>done by</td></tr>
{windows_rows}
</table>
<div class="meta" style="margin-top:.6rem">Short windows show current weather; long windows include
overnight troughs and capacity swings. Truth for planning usually sits between the 12h and 72h rows.</div>
<h1 style="margin-top:1.6rem">TPU allocation by pool &mdash; granted vs asked</h1>
<div class="meta">asked = worker slices our coordinators have submitted; granted = workers whose task is executing
(preempted workers waiting for a new slice count as asked, not granted).
v4/v5p names count TensorCores (2 per chip); v5e/v6e names count chips.</div>
<table>{pool_rows}</table>
<h1 style="margin-top:1.6rem">Stored data by region</h1>
<div class="meta">kept = filtered output parquet. The 2M ladder was consolidated into us-east1 on 09-23, so the
"share of 8M segment" column is the fair per-region comparison.</div>
<table>{storage_rows}</table>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        with _LOCK:
            s = dict(_STATE)
        if self.path == "/json":
            body = json.dumps(s).encode()
            ctype = "application/json"
        else:
            warcs = s.get("warcs_done", 0)
            target = s.get("target_warcs", POOL_8M) or POOL_8M
            seg_target = s.get("segment_warcs_target", 0) or 1
            age = time.time() - s.get("ts", time.time())
            body = PAGE.format(
                target=target,
                warcs=warcs,
                pct_all=100 * warcs / target,
                seg_start=SEGMENT_START_SHARD,
                seg_target=seg_target,
                seg_done=s.get("segment_warcs_done", 0),
                pct_seg=100 * s.get("segment_warcs_done", 0) / seg_target,
                shards=s.get("shards_done", 0),
                shards_t=s.get("shards_target", 0),
                kept=warcs * KEPT_DOCS_PER_WARC,
                workers=s.get("workers", "?"),
                fleet_age=time.time() - s.get("fleet_ts", time.time()),
                windows_rows=_windows_table(s),
                pool_rows=_pools_table(s),
                storage_rows=_storage_table(s),
                age=age,
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
    ap.add_argument(
        "--max-shard", type=int, default=32_000, help="Exclusive shard bound of the run (32000 = full pool)."
    )
    ap.add_argument("--fleet-json", action="store_true", help="Print the per-pool fleet listing as JSON and exit.")
    ap.add_argument("--progress-json", action="store_true", help="Print one progress snapshot as JSON and exit.")
    args = ap.parse_args()
    if args.fleet_json:
        print(json.dumps(_list_child_pools()))
        return
    spec = get_spec(args.spec)
    if args.progress_json:
        print(json.dumps(_collect(spec, args.max_shard)))
        return
    threading.Thread(target=_refresher, args=(args.spec, args.max_shard), daemon=True).start()
    threading.Thread(target=_fleet_refresher, daemon=True).start()
    threading.Thread(target=_storage_refresher, args=(spec,), daemon=True).start()
    logger.info("curation dashboard on http://localhost:%d", args.port)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
