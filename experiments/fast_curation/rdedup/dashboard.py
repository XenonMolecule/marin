# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live dashboard for one rdedup run: stage progress, frontier rate and ETA, workers, results.

Serves one auto-refreshing page (default :8094). Each refresh reads the run's bookkeeping only:
plan.json, every sealed stage's ``sealed.json``, and the frontier stage's claims and done marks
(``launcher.stage_progress``). Worker counts come from the Iris job list under the launcher job.
Both reads run in child processes killed at a deadline, so a hung GCS or controller call never
freezes the page. Frontier progress snapshots persist under ~/.cache so rates survive restarts.

    uv run python -m experiments.fast_curation.rdedup.dashboard \\
        --run-root gs://marin-us-central2/tmp/ttl=14d/rdedup/8m-v1 \\
        --launcher-job /michaelryan/rdedup-launch-8m-v1
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import html
import itertools
import json
import logging
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2

from experiments.fast_curation.rdedup.launcher import stage_progress
from experiments.fast_curation.rdedup.plan import load_plan
from experiments.fast_curation.rdedup.store import read_json, store_for

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 120.0
DEADLINE_SECONDS = 300.0
CHARS_PER_TOKEN = 4.302  # llama3, measured on the 2M c075 output
RATE_WINDOWS_HOURS = (0.5, 2.0)
HISTORY_HOURS = 6.0
# Fixed stages of a c075 run; the cc rounds in between are added as they appear.
STAGE_ORDER = ("pass1", "exact", "lshmap", "bucket")
TAIL_STAGES = ("canon_a", "canon_b", "attrs", "c075_text", "c075_solve", "c075_marks", "apply_c075", "report")
LIVE_JOB_STATES = (job_pb2.JOB_STATE_RUNNING, job_pb2.JOB_STATE_PENDING)
# Run ETA: a stage not yet started is estimated from a measured stage of this run that does the
# same work (apply_c075 re-reads every unit like c075_text); others are short merges.
STAGE_ANALOG = {"apply_c075": "c075_text", "apply_cc": "c075_text"}
SHORT_STAGE_HOURS = 0.25

_STATE: dict = {}
_LOCK = threading.Lock()


def _hard_deadline(seconds: float) -> list[str]:
    return ["timeout", "-s", "KILL", str(int(seconds))]


def collect(run_root: str) -> dict:
    """One progress snapshot: plan sizes, per-stage progress, and sealed stage results."""
    store = store_for(run_root)
    plan = load_plan(store, run_root)
    rows = stage_progress(store, run_root)
    results = {}
    for row in rows:
        if row["sealed"]:
            sealed = read_json(store, f"{plan.run_root}/stages/{row['stage']}/sealed.json")
            results[row["stage"]] = {k: v for k, v in sealed.items() if not k.startswith("_")} | {
                "finished": sealed.get("_finished")
            }
    return {
        "ts": time.time(),
        "plan": {"files": plan.n_files, "units": plan.n_units, "c075_partitions": plan.c075_partitions},
        "stages": rows,
        "results": results,
    }


def list_workers(launcher_job: str) -> dict:
    """Worker tasks under every launcher whose job name starts with ``launcher_job``.

    A prefix, so a run fed by several launchers (e.g. ``...-8m-v1`` and ``...-8m-v1-cpu``) counts
    all of their workers.
    """
    running = waiting = jobs = 0
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        for state in LIVE_JOB_STATES:
            for job in client.list_jobs(state=state, prefix=launcher_job, limit=3000):
                if "/rdedup-w-" not in job.job_id:
                    continue
                jobs += 1
                counts = dict(job.task_state_counts)
                running += counts.get("running", 0)
                waiting += sum(v for k, v in counts.items() if k in ("pending", "assigned", "building"))
    return {"jobs": jobs, "running": running, "waiting": waiting}


def _child_json(flag: str, value: str) -> dict | None:
    try:
        out = subprocess.run(
            [*_hard_deadline(DEADLINE_SECONDS), sys.executable, "-m", __spec__.name, flag, value],
            capture_output=True,
            text=True,
            timeout=DEADLINE_SECONDS + 30,
            check=True,
        ).stdout
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.warning("%s failed: %s", flag, type(e).__name__)
        return None
    return json.loads(out.strip().splitlines()[-1])


def _history_path(run_root: str) -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / f"rdedup_dashboard_{hashlib.sha1(run_root.encode()).hexdigest()[:10]}.json"


def frontier_rates(history: list[list], stage: str, done: int, now: float) -> dict[float, float]:
    """Tasks/hour on the frontier stage over each trailing window that the history spans."""
    points = [(t, d) for t, s, d in history if s == stage]
    rates = {}
    for hours in RATE_WINDOWS_HOURS:
        inside = [(t, d) for t, d in points if t >= now - hours * 3600]
        if inside and now - inside[0][0] >= 0.8 * hours * 3600:
            rates[hours] = (done - inside[0][1]) / ((now - inside[0][0]) / 3600)
    return rates


def _refresher(run_root: str) -> None:
    path = _history_path(run_root)
    history = json.loads(path.read_text()) if path.exists() else []
    while True:
        snap = _child_json("--snapshot-json", run_root)
        if snap is not None:
            frontier = next((r for r in snap["stages"] if not r["sealed"]), None)
            if frontier is not None:
                history.append([snap["ts"], frontier["stage"], frontier["done"]])
                history = [h for h in history if h[0] >= snap["ts"] - HISTORY_HOURS * 3600]
                path.write_text(json.dumps(history))
                snap["rates"] = frontier_rates(history, frontier["stage"], frontier["done"], snap["ts"])
            with _LOCK:
                _STATE.update(snap)
        time.sleep(REFRESH_SECONDS)


def _worker_refresher(launcher_job: str) -> None:
    while True:
        workers = _child_json("--workers-json", launcher_job)
        if workers is not None:
            with _LOCK:
                _STATE.update(workers=workers, workers_ts=time.time())
        time.sleep(REFRESH_SECONDS)


def _ago(ts: float | None) -> str:
    if not ts:
        return "never"
    return f"{int(time.time() - ts)}s ago"


def _results_rows(results: dict) -> list[tuple[str, str]]:
    rows = []
    if p := results.get("pass1"):
        rows.append(("Records in", f"{p['rows']:,}"))
    if e := results.get("exact"):
        rows.append(("Exact duplicates", f"{e['exact_dups']:,}"))
        rows.append(("Unique after exact", f"{e['unique']:,}"))
    if a := results.get("attrs"):
        rows.append(("Fuzzy-cluster members", f"{a['members']:,}"))
        rows.append(("Components", f"{a['canonicals']:,}"))
    if c := results.get("canon_b"):
        rows.append(("Largest component", f"{c['largest']:,}"))
    if m := results.get("c075_marks"):
        rows.append(("c075 removals", f"{m['removed']:,}"))
    if out := results.get("apply_c075"):
        rows.append(("c075 docs kept", f"{out['docs']:,}"))
        rows.append(("c075 tokens (est.)", f"{out['chars'] / CHARS_PER_TOKEN / 1e12:.2f}T"))
    return rows


def stage_hours(results: dict) -> dict[str, float]:
    """Measured wall hours of each sealed stage: its seal time minus the previous stage's."""
    seals = sorted((r["finished"], name) for name, r in results.items() if r.get("finished"))
    return {name: (t - prev) / 3600 for (prev, _), (t, name) in itertools.pairwise(seals)}


def run_eta(s: dict, frontier: dict | None) -> list[tuple[str, float]]:
    """(milestone, hours from now): the frontier's seal, each later stage's seal, and the end."""
    rates = s.get("rates", {})
    rate = rates.get(0.5) or rates.get("0.5") or rates.get(2.0) or rates.get("2.0")
    if frontier is None or not rate:
        return []
    measured = stage_hours(s.get("results", {}))
    hours = (frontier["tasks"] - frontier["done"]) / rate
    milestones = [(frontier["stage"], hours)]
    later = TAIL_STAGES[TAIL_STAGES.index(frontier["stage"]) + 1 :] if frontier["stage"] in TAIL_STAGES else ()
    for name in later:
        hours += measured.get(STAGE_ANALOG.get(name, ""), SHORT_STAGE_HOURS)
        milestones.append((name, hours))
    return milestones


def _eta_rows(milestones: list[tuple[str, float]]) -> str:
    if not milestones:
        return "<tr><td colspan=2>measuring… (needs the frontier rate)</td></tr>"
    now = datetime.datetime.now()
    rows = []
    for name, hours in milestones:
        when = (now + datetime.timedelta(hours=hours)).strftime("%a %H:%M")
        note = {"c075_solve": " (docs removed known)", "apply_c075": " (final docs + chars)"}.get(name, "")
        rows.append(f"<tr><td>{name} seals{note}</td><td>{when}</td><td>~{hours:.1f} h</td></tr>")
    return "".join(rows)


def render(s: dict, run_root: str) -> str:
    if "stages" not in s:
        return "<html><body style='font-family:monospace'>Collecting first snapshot…</body></html>"
    stages = s["stages"]
    sealed = sum(1 for r in stages if r["sealed"])
    frontier = next((r for r in stages if not r["sealed"]), None)
    body = []
    for r in stages:
        if r["sealed"]:
            finished = s["results"].get(r["stage"], {}).get("finished")
            when = datetime.datetime.fromtimestamp(finished).strftime("%a %H:%M") if finished else "✓"
            body.append(
                f"<tr class=done><td>{r['stage']}</td><td>{r['tasks']:,}</td><td>{r['tasks']:,}</td>"
                f"<td>—</td><td>—</td><td>sealed {when}</td></tr>"
            )
            continue
        pct = 100 * r["done"] / r["tasks"]
        body.append(
            f"<tr><td><b>{r['stage']}</b></td><td>{r['tasks']:,}</td><td>{r['done']:,} ({pct:.1f}%)</td>"
            f"<td>{r['in_flight']:,}</td><td>{r['failures']:,}</td>"
            f"<td><div class=mini><div style='width:{pct:.1f}%'></div></div></td></tr>"
        )
    eta = []
    if frontier is not None:
        left = frontier["tasks"] - frontier["done"]
        for hours, rate in sorted(s.get("rates", {}).items()):
            when = f"~{left / rate:.1f} h" if rate > 0 else "stalled"
            eta.append(f"<tr><td>last {hours:g}h</td><td>{rate:,.0f} tasks/h</td><td>{when}</td></tr>")
        if not eta:
            eta.append("<tr><td colspan=3>measuring… (needs ~25 min of history)</td></tr>")
    w = s.get("workers", {})
    results = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in _results_rows(s.get("results", {})))
    return f"""<!doctype html><html><head><title>8M Dedup Run</title>
<meta http-equiv="refresh" content="60"><style>
body {{ font-family: ui-monospace, monospace; background: #0d1117; color: #e6edf3; padding: 2rem;
       max-width: 960px; margin: auto; }}
h1 {{ color: #58d68d; font-size: 1.3rem; }} h2 {{ font-size: 1rem; color: #8b949e; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; }} td {{ padding: .25rem 1.2rem .25rem 0; }}
tr.done td {{ color: #8b949e; }}
.mini {{ display: inline-block; width: 160px; height: 10px; background: #21262d; border-radius: 4px; overflow: hidden; }}
.mini div {{ height: 100%; background: #58d68d; }}
.meta {{ color: #8b949e; }}
</style></head><body>
<h1>rdedup &mdash; {html.escape(run_root)}</h1>
<div class=meta>{s['plan']['files']:,} files &middot; {s['plan']['units']:,} units &middot;
{sealed} stages sealed &middot; frontier: <b>{frontier['stage'] if frontier else 'DONE'}</b> &middot;
progress {_ago(s.get('ts'))}</div>
<div class=meta>Workers: {w.get('running', '?')} running, {w.get('waiting', '?')} waiting, in {w.get('jobs', '?')} jobs
(listed {_ago(s.get('workers_ts'))})</div>
<h2>Stages</h2>
<table><tr><td>stage</td><td>tasks</td><td>done</td><td>in flight</td><td>failures</td><td></td></tr>
{"".join(body)}</table>
<h2>Frontier rate &amp; time left in this stage</h2><table>{"".join(eta)}</table>
<h2>Projected milestones (frontier at its current rate; later stages from this run's measured stages)</h2>
<table>{_eta_rows(run_eta(s, frontier))}</table>
<h2>Results so far</h2><table>{results or "<tr><td>none yet</td></tr>"}</table>
<div class=meta style="margin-top:1.5rem">Stage order: {" → ".join(STAGE_ORDER)} → cc rounds → {" → ".join(TAIL_STAGES)}.
CC rounds repeat until no label changes (13 at 2M).</div>
</body></html>"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root")
    ap.add_argument("--launcher-job")
    ap.add_argument("--port", type=int, default=8094)
    ap.add_argument("--snapshot-json", help="Print one progress snapshot for this run root and exit.")
    ap.add_argument("--workers-json", help="Print worker counts under this launcher job and exit.")
    args = ap.parse_args()
    if args.snapshot_json:
        print(json.dumps(collect(args.snapshot_json)))
        return
    if args.workers_json:
        print(json.dumps(list_workers(args.workers_json)))
        return
    if not (args.run_root and args.launcher_job):
        ap.error("--run-root and --launcher-job are required to serve")
    threading.Thread(target=_refresher, args=(args.run_root,), daemon=True).start()
    threading.Thread(target=_worker_refresher, args=(args.launcher_job,), daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with _LOCK:
                snap = dict(_STATE)
            page = render(snap, args.run_root).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *a):
            pass

    logger.info("rdedup dashboard on http://localhost:%d", args.port)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
