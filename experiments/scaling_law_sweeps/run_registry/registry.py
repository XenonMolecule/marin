# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Living registry for natural-epoching curation training runs.

A *run* is one (method, sample) campaign — e.g. ``resiliparse_dedup`` on the
``canon-2000w`` WARC draw. A *sample* is one specific WARC draw (a lineage at a
fixed WARC count); WARC count alone is NOT a unique id, because we run multiple
independent draws at the same N (e.g. the nested ``canon`` ladder vs a fresh
``indepA`` 3000-WARC draw). Every run keys on (method, sample).

The YAML at ``registry.yaml`` is the source of truth and is hand-editable.
``REGISTRY.md`` is a generated human view, refreshed after every mutation.

Supervision: ``claim`` / ``check`` stamp ``last_checked`` (UTC, now) and a
``supervisor`` name so each of us can see at a glance who is watching a run and
how long ago they looked — "claude checked it 20m ago" is a real field.

Common usage::

    # See everything, grouped by sample
    python registry.py list

    # Register a new WARC draw, then a run on it
    python registry.py add-sample --id indepA-3000w --lineage indepA --n 3000 \\
        --manifest experiments/distill/independent/baseline_warcs_3000_indepA.txt \\
        --desc "independent random 3000-WARC draw (NOT the canon/10k-prefix ladder)"
    python registry.py add --method resiliparse_dedup --sample indepA-3000w \\
        --status planned

    # Claim supervision (stamps you + now), later ping that you checked it
    python registry.py claim resiliparse_dedup__indepA-3000w --by claude
    python registry.py check resiliparse_dedup__indepA-3000w --by claude --status running

    # Record where it landed
    python registry.py set resiliparse_dedup__indepA-3000w \\
        --status complete --wandb https://wandb.ai/... \\
        --checkpoint gs://marin-us-east5/checkpoints/...

    # Pull live completed-cell counts from GCS result JSONs
    python registry.py sync --all
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

REGISTRY_DIR = Path(__file__).resolve().parent
YAML_PATH = REGISTRY_DIR / "registry.yaml"
MD_PATH = REGISTRY_DIR / "REGISTRY.md"

# Status vocabulary. A (method, sample) cell is the unit.
STATUSES = ("planned", "launching", "running", "partial", "complete", "failed", "abandoned")
REGIMES = ("natural",)  # only natural epoching is tracked for now

# wandb project for the optional `progress` query (training % from _step/num_train_steps).
WANDB_ENTITY_PROJECT = "marin-community/marin"

# Mutable run fields settable via `set`.
RUN_FIELDS = ("status", "wandb", "results_glob", "checkpoint_dir", "iris_job", "cells_done", "cells_total", "notes")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_supervisor() -> str | None:
    """Auto-derive a stable, human-readable supervisor label for this session.

    Each Claude Code session exports a unique ``CLAUDE_CODE_SESSION_ID``; we tag
    with its short prefix (e.g. ``claude-6329ae0a``) so several parallel observer
    sessions self-identify distinctly without anyone assigning names by hand.
    Returns None outside a Claude session (so a human must pass --by explicitly).
    """
    sid = os.environ.get("CLAUDE_CODE_SESSION_ID")
    return f"claude-{sid[:8]}" if sid else None


def _load() -> dict:
    if not YAML_PATH.exists():
        return {"samples": {}, "runs": []}
    data = yaml.safe_load(YAML_PATH.read_text()) or {}
    data.setdefault("samples", {})
    data.setdefault("runs", [])
    return data


def _save(data: dict) -> None:
    # Deterministic, git-friendly ordering: samples by id, runs by (sample, method).
    data["samples"] = dict(sorted(data["samples"].items()))
    data["runs"] = sorted(data["runs"], key=lambda r: (r.get("sample", ""), r.get("method", "")))
    YAML_PATH.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=100))
    _render(data)


def _run_id(method: str, sample: str) -> str:
    return f"{method}__{sample}"


def _find_run(data: dict, run_id: str) -> dict | None:
    return next((r for r in data["runs"] if r["id"] == run_id), None)


def _age(iso: str | None) -> str:
    if not iso:
        return "never"
    then = datetime.fromisoformat(iso)
    delta = datetime.now(UTC) - then
    s = int(delta.total_seconds())
    if s < 90:
        return f"{s}s ago"
    if s < 90 * 60:
        return f"{s // 60}m ago"
    if s < 48 * 3600:
        return f"{s // 3600}h ago"
    return f"{s // 86400}d ago"


def _stale_seconds(iso: str | None) -> float:
    if not iso:
        return float("inf")
    return (datetime.now(UTC) - datetime.fromisoformat(iso)).total_seconds()


# --- commands ----------------------------------------------------------------


def cmd_add_sample(args) -> int:
    data = _load()
    if args.id in data["samples"] and not args.force:
        print(f"sample {args.id!r} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    data["samples"][args.id] = {
        "lineage": args.lineage,
        "n_warcs": args.n,
        "manifest": args.manifest,
        "description": args.desc,
        "created": _now(),
    }
    _save(data)
    print(f"added sample {args.id}")
    return 0


def cmd_add(args) -> int:
    data = _load()
    if args.sample not in data["samples"]:
        print(f"unknown sample {args.sample!r}. Register it first with add-sample.", file=sys.stderr)
        return 1
    run_id = _run_id(args.method, args.sample)
    if _find_run(data, run_id) and not args.force:
        print(f"run {run_id!r} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    data["runs"] = [r for r in data["runs"] if r["id"] != run_id]
    data["runs"].append(
        {
            "id": run_id,
            "method": args.method,
            "sample": args.sample,
            "regime": args.regime,
            "status": args.status,
            "cells_done": args.cells_done,
            "cells_total": args.cells_total,
            "wandb": args.wandb,
            "results_glob": args.results_glob,
            "checkpoint_dir": args.checkpoint,
            "iris_job": args.iris_job,
            "supervisor": None,
            "last_checked": None,
            "last_synced": None,
            "pct_complete": None,
            "last_progress_synced": None,
            "notes": args.notes,
            "created": _now(),
            "updated": _now(),
        }
    )
    _save(data)
    print(f"added run {run_id}")
    return 0


def _require(data: dict, run_id: str) -> dict:
    r = _find_run(data, run_id)
    if r is None:
        print(f"no such run {run_id!r}", file=sys.stderr)
        raise SystemExit(2)
    return r


def cmd_set(args) -> int:
    data = _load()
    r = _require(data, args.id)
    for field in RUN_FIELDS:
        val = getattr(args, field if field != "checkpoint_dir" else "checkpoint", None)
        if val is not None:
            r[field] = val
    if args.note is not None:
        r["notes"] = args.note
    r["updated"] = _now()
    _save(data)
    print(f"updated {args.id}")
    return 0


def cmd_claim(args) -> int:
    by = args.by or _default_supervisor()
    if not by:
        print("--by is required (no CLAUDE_CODE_SESSION_ID to auto-derive a label)", file=sys.stderr)
        return 2
    data = _load()
    for run_id in args.ids:
        r = _require(data, run_id)
        r["supervisor"] = by
        r["last_checked"] = _now()
        r["updated"] = _now()
        print(f"{by} now supervising {run_id}")
    _save(data)
    return 0


def cmd_check(args) -> int:
    by = args.by or _default_supervisor()
    data = _load()
    for run_id in args.ids:
        r = _require(data, run_id)
        r["last_checked"] = _now()
        if by is not None:
            r["supervisor"] = by
        if args.status is not None:
            r["status"] = args.status
        if args.note is not None:
            r["notes"] = args.note
        r["updated"] = _now()
        who = r.get("supervisor") or "?"
        print(f"{who} checked {run_id} ({r['status']})")
    _save(data)
    return 0


def cmd_release(args) -> int:
    data = _load()
    for run_id in args.ids:
        r = _require(data, run_id)
        r["supervisor"] = None
        r["updated"] = _now()
        print(f"released {run_id}")
    _save(data)
    return 0


def _count_json(glob: str) -> int | None:
    proc = subprocess.run(["gcloud", "storage", "ls", glob], capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    return sum(1 for ln in proc.stdout.splitlines() if ln.strip().endswith(".json"))


def cmd_sync(args) -> int:
    data = _load()
    targets = data["runs"] if args.all else [_require(data, i) for i in args.ids]
    for r in targets:
        glob = r.get("results_glob")
        if not glob:
            continue
        n = _count_json(glob)
        if n is None:
            print(f"  {r['id']}: glob unreadable, skipped")
            continue
        r["cells_done"] = n
        r["last_synced"] = _now()
        r["updated"] = _now()
        print(f"  {r['id']}: {n} result JSONs")
    _save(data)
    return 0


def _run_name_prefix(results_glob: str) -> str | None:
    """Derive the wandb run-name prefix from a results_glob.

    e.g. ".../curation-dclm_10k-expFM_natural-*.json" -> "curation-dclm_10k-expFM_natural-"
    """
    if not results_glob:
        return None
    base = results_glob.rstrip("/").split("/")[-1]
    prefix = base.split("*")[0]
    return prefix or None


def cmd_progress(args) -> int:
    """OPTIONAL, occasional: query wandb for average training % per row.

    Each cell's progress = min(1, summary._step / config.trainer.num_train_steps).
    The row %% is the MEAN per-cell progress over the full planned grid (denominator
    = cells_total), so not-yet-started cells count as 0 and one giant aspirational
    cell (e.g. 9e21) can't dominate. Writes pct_complete + last_progress_synced.
    Heavier than `sync` (hits wandb), so run it occasionally, not on every check.
    """
    try:
        import wandb
    except ImportError:
        print("wandb not installed — cannot query progress.", file=sys.stderr)
        return 2
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — cannot query progress.", file=sys.stderr)
        return 2

    data = _load()
    targets = data["runs"] if args.all else [_require(data, i) for i in args.ids]
    api = wandb.Api(timeout=60)
    for r in targets:
        prefix = _run_name_prefix(r.get("results_glob") or "")
        if not prefix:
            print(f"  {r['id']}: no results_glob to derive run names; skipped")
            continue
        try:
            runs = list(api.runs(WANDB_ENTITY_PROJECT, filters={"display_name": {"$regex": "^" + re.escape(prefix)}}))
        except Exception as e:
            print(f"  {r['id']}: wandb query failed ({e})")
            continue
        fracs: list[float] = []
        unloadable = 0
        states: collections.Counter = collections.Counter()
        for x in runs:
            states[x.state] += 1
            try:
                step = x.summary.get("_step")
                target = (x.config.get("trainer") or {}).get("num_train_steps")
            except Exception:
                unloadable += 1
                continue
            if step is None or not target:
                continue
            fracs.append(min(1.0, step / target))
        # Denominator = full planned grid (cells_total) so unstarted cells count as 0.
        denom = max(r.get("cells_total") or 0, len(fracs)) or 1
        pct = min(100.0, round(100 * sum(fracs) / denom, 1))
        r["pct_complete"] = pct
        r["last_progress_synced"] = _now()
        r["updated"] = _now()
        extra = f", {unloadable} unloadable" if unloadable else ""
        print(f"  {r['id']}: {pct}% ({len(fracs)}/{denom} cells w/ data; states={dict(states)}{extra})")
    _save(data)
    return 0


def _matches(r: dict, samples: dict, args) -> bool:
    if args.method and r["method"] != args.method:
        return False
    if args.sample and r["sample"] != args.sample:
        return False
    if args.lineage and samples.get(r["sample"], {}).get("lineage") != args.lineage:
        return False
    if args.n and samples.get(r["sample"], {}).get("n_warcs") != args.n:
        return False
    if args.status and r["status"] != args.status:
        return False
    if args.supervisor and (r.get("supervisor") or "") != args.supervisor:
        return False
    if args.stale_hours is not None and _stale_seconds(r.get("last_checked")) < args.stale_hours * 3600:
        return False
    return True


def _cells(r: dict) -> str:
    done, total = r.get("cells_done"), r.get("cells_total")
    if done is None and total is None:
        return "-"
    return f"{done if done is not None else '?'}/{total if total is not None else '?'}"


def _pct(r: dict) -> str:
    p = r.get("pct_complete")
    return f"{p:g}%" if p is not None else "-"


def cmd_list(args) -> int:
    data = _load()
    samples = data["samples"]
    rows = [r for r in data["runs"] if _matches(r, samples, args)]
    rows.sort(key=lambda r: (samples.get(r["sample"], {}).get("n_warcs", 0), r["sample"], r["method"]))
    if not rows:
        print("(no matching runs)")
        return 0
    hdr = f"{'id':<42} {'status':<10} {'cells':<8} {'train%':<7} {'super':<9} {'checked':<10} notes"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        sup = r.get("supervisor") or "-"
        checked = _age(r.get("last_checked"))
        # Flag a supervised-but-stale running job.
        if r["status"] == "running" and sup != "-" and _stale_seconds(r.get("last_checked")) > 2 * 3600:
            checked = "⚠ " + checked
        note = (r.get("notes") or "")[:40]
        print(f"{r['id']:<42} {r['status']:<10} {_cells(r):<8} {_pct(r):<7} {sup:<9} {checked:<10} {note}")
    return 0


def cmd_show(args) -> int:
    data = _load()
    r = _require(data, args.id)
    print(yaml.safe_dump(r, sort_keys=False, default_flow_style=False))
    return 0


def cmd_render(args) -> int:
    _render(_load())
    print(f"wrote {MD_PATH}")
    return 0


def cmd_whoami(args) -> int:
    print(_default_supervisor() or "(no CLAUDE_CODE_SESSION_ID; pass --by explicitly)")
    return 0


def cmd_validate(args) -> int:
    data = _load()
    ok = True
    ids = [r["id"] for r in data["runs"]]
    for dup in {i for i in ids if ids.count(i) > 1}:
        print(f"DUPLICATE run id: {dup}", file=sys.stderr)
        ok = False
    for r in data["runs"]:
        if r["sample"] not in data["samples"]:
            print(f"{r['id']}: unknown sample {r['sample']!r}", file=sys.stderr)
            ok = False
        if r["status"] not in STATUSES:
            print(f"{r['id']}: bad status {r['status']!r}", file=sys.stderr)
            ok = False
        if r.get("regime") not in REGIMES:
            print(f"{r['id']}: bad regime {r.get('regime')!r}", file=sys.stderr)
            ok = False
    print("OK" if ok else "INVALID")
    return 0 if ok else 1


# --- markdown rendering ------------------------------------------------------


def _render(data: dict) -> None:
    samples = data["samples"]
    runs = data["runs"]
    lines: list[str] = []
    lines.append("# Natural-Epoching Run Registry")
    lines.append("")
    lines.append("> Generated by `registry.py` — **do not hand-edit this file**; edit `registry.yaml` or use the CLI.")
    lines.append(f"> Last rendered: {_now()}")
    lines.append("")
    lines.append("Identity = **(method, sample)**. `sample` = a specific WARC draw; WARC count alone is")
    lines.append("not unique (e.g. `canon-3000w`, the nested ladder / 10k-prefix, vs `indepA-3000w`, an")
    lines.append("independent draw). Only **natural-epoching** runs are tracked here.")
    lines.append("")

    # Samples table.
    lines.append("## WARC samples (draws)")
    lines.append("")
    lines.append("| sample | lineage | N | manifest | description |")
    lines.append("|---|---|---:|---|---|")
    for sid, s in sorted(samples.items(), key=lambda kv: (kv[1].get("n_warcs", 0), kv[0])):
        lines.append(
            f"| `{sid}` | {s.get('lineage','')} | {s.get('n_warcs','')} | "
            f"`{s.get('manifest') or '-'}` | {s.get('description','')} |"
        )
    lines.append("")

    # Supervision summary: who is watching what, and any stale running jobs.
    watched = [r for r in runs if r.get("supervisor")]
    if watched:
        lines.append("## Under supervision")
        lines.append("")
        lines.append("| run | supervisor | status | last checked |")
        lines.append("|---|---|---|---|")
        for r in sorted(watched, key=lambda r: r.get("last_checked") or ""):
            age = _age(r.get("last_checked"))
            flag = " ⚠️" if r["status"] == "running" and _stale_seconds(r.get("last_checked")) > 2 * 3600 else ""
            lines.append(f"| `{r['id']}` | {r['supervisor']} | {r['status']} | {age}{flag} |")
        lines.append("")

    # Runs grouped by sample (sorted by N then sample id).
    lines.append("## Runs")
    lines.append("")
    by_sample: dict[str, list[dict]] = {}
    for r in runs:
        by_sample.setdefault(r["sample"], []).append(r)
    for sid in sorted(by_sample, key=lambda s: (samples.get(s, {}).get("n_warcs", 0), s)):
        s = samples.get(sid, {})
        lines.append(f"### `{sid}`  (N={s.get('n_warcs','?')}, lineage={s.get('lineage','?')})")
        lines.append("")
        lines.append("| method | status | cells | train% | supervisor | last checked | wandb | checkpoint | notes |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in sorted(by_sample[sid], key=lambda r: r["method"]):
            sup = r.get("supervisor") or "-"
            checked = _age(r.get("last_checked"))
            wb = f"[dashboard]({r['wandb']})" if r.get("wandb") else "-"
            ckpt = f"`{r['checkpoint_dir']}`" if r.get("checkpoint_dir") else "-"
            note = (r.get("notes") or "").replace("\n", " ")
            cells = [r["method"], r["status"], _cells(r), _pct(r), sup, checked, wb, ckpt, note]
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
        lines.append("")

    lines.append("---")
    lines.append("Statuses: " + ", ".join(f"`{s}`" for s in STATUSES) + ".")
    MD_PATH.write_text("\n".join(lines) + "\n")


# --- argparse ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add-sample", help="register a WARC draw")
    sp.add_argument("--id", required=True, help="e.g. canon-2000w / indepA-3000w")
    sp.add_argument("--lineage", required=True, help="draw family, e.g. canon / indepA")
    sp.add_argument("--n", type=int, required=True, help="WARC count")
    sp.add_argument("--manifest", default=None)
    sp.add_argument("--desc", default="")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_add_sample)

    ap = sub.add_parser("add", help="add a (method, sample) run")
    ap.add_argument("--method", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--regime", default="natural", choices=REGIMES)
    ap.add_argument("--status", default="planned", choices=STATUSES)
    ap.add_argument("--cells-done", type=int, default=None, dest="cells_done")
    ap.add_argument("--cells-total", type=int, default=None, dest="cells_total")
    ap.add_argument("--wandb", default=None)
    ap.add_argument("--results-glob", default=None, dest="results_glob")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--iris-job", default=None, dest="iris_job")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--force", action="store_true")
    ap.set_defaults(func=cmd_add)

    st = sub.add_parser("set", help="update fields on a run")
    st.add_argument("id")
    st.add_argument("--status", choices=STATUSES, default=None)
    st.add_argument("--wandb", default=None)
    st.add_argument("--results-glob", dest="results_glob", default=None)
    st.add_argument("--checkpoint", default=None)
    st.add_argument("--iris-job", dest="iris_job", default=None)
    st.add_argument("--cells-done", type=int, dest="cells_done", default=None)
    st.add_argument("--cells-total", type=int, dest="cells_total", default=None)
    st.add_argument("--note", default=None)
    st.add_argument("--notes", default=None)  # alias
    st.set_defaults(func=cmd_set)

    cl = sub.add_parser("claim", help="take supervision of run(s); stamps you + now")
    cl.add_argument("ids", nargs="+")
    cl.add_argument("--by", default=None, help="supervisor name (default: auto from CLAUDE_CODE_SESSION_ID)")
    cl.set_defaults(func=cmd_claim)

    ck = sub.add_parser("check", help="record that you just checked run(s)")
    ck.add_argument("ids", nargs="+")
    ck.add_argument("--by", default=None)
    ck.add_argument("--status", choices=STATUSES, default=None)
    ck.add_argument("--note", default=None)
    ck.set_defaults(func=cmd_check)

    rl = sub.add_parser("release", help="drop supervision of run(s)")
    rl.add_argument("ids", nargs="+")
    rl.set_defaults(func=cmd_release)

    sy = sub.add_parser("sync", help="count result JSONs from results_glob into cells_done")
    sy.add_argument("ids", nargs="*")
    sy.add_argument("--all", action="store_true")
    sy.set_defaults(func=cmd_sync)

    pg = sub.add_parser("progress", help="OPTIONAL: query wandb for compute-weighted training %% (occasional)")
    pg.add_argument("ids", nargs="*")
    pg.add_argument("--all", action="store_true")
    pg.set_defaults(func=cmd_progress)

    ls = sub.add_parser("list", help="list runs")
    ls.add_argument("--method", default=None)
    ls.add_argument("--sample", default=None)
    ls.add_argument("--lineage", default=None)
    ls.add_argument("--n", type=int, default=None)
    ls.add_argument("--status", choices=STATUSES, default=None)
    ls.add_argument("--supervisor", default=None)
    ls.add_argument(
        "--stale-hours",
        type=float,
        default=None,
        dest="stale_hours",
        help="only runs not checked within this many hours",
    )
    ls.set_defaults(func=cmd_list)

    sh = sub.add_parser("show", help="dump one run")
    sh.add_argument("id")
    sh.set_defaults(func=cmd_show)

    sub.add_parser("render", help="regenerate REGISTRY.md").set_defaults(func=cmd_render)
    sub.add_parser("validate", help="check schema integrity").set_defaults(func=cmd_validate)
    sub.add_parser("whoami", help="print this session's auto supervisor label").set_defaults(func=cmd_whoami)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
