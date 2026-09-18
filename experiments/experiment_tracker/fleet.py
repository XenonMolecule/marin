# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fleet landscape probe: every live workload family, grouped and state-counted.

One `iris job list` per family prefix, aggregated into
``{group -> family -> {states, band, kind, note, examples}}`` so the dashboard can paint
training vs evaluating vs paused at a glance. Bands are annotated from how each family
was launched (Iris job listings do not expose priority); paused/killed families are
static entries so the page shows what is deliberately NOT running.
"""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor

# (group, family label, iris prefix, band, note). Band = child priority at launch.
FAMILIES: list[tuple[str, str, str, str, str]] = [
    (
        "Training — domain v2",
        "lpv11 k20 (10k-dom-lpv11b)",
        "/michaelryan/10k-dom-lpv11b/",
        "interactive",
        "k5 children killed 09-16",
    ),
    ("Training — domain v2", "fastpipe_v2 k20 (fpv2-k20b)", "/michaelryan/10k-dom-fpv2-k20b/", "interactive", ""),
    ("Training — domain v2", "dclm k20 (10k-dom-dclm)", "/michaelryan/10k-dom-dclm/", "batch", ""),
    (
        "Training — domain v2",
        "nemotron k20 (10k-dom-nem)",
        "/michaelryan/10k-dom-nem/",
        "batch",
        "+ tag-fix repair arms",
    ),
    ("Training — domain v2", "nemotron repair (nem-fix)", "/michaelryan/10k-dom-nem-fix/", "batch", ""),
    (
        "Training — domain v2",
        "resiliparse k20 re-homed (rslp-k20c)",
        "/michaelryan/10k-dom-rslp-k20c/",
        "interactive",
        "east5+central2 after cache copy 09-17",
    ),
    (
        "Training — domain v2",
        "resiliparse k20 legacy (rslp-k20)",
        "/michaelryan/10k-dom-rslp-k20/",
        "batch",
        "running cells finishing out",
    ),
    (
        "Training — domain v2",
        "resiliparse repair (rslp-fix)",
        "/michaelryan/10k-dom-rslp-fix/",
        "batch",
        "k20 rows only alive",
    ),
    (
        "Training — domain v2",
        "gang-crash repairs (*-wfx*)",
        "/michaelryan/10k-dom-lpv11-wfx",
        "mixed",
        "plus per-corpus wfx twins",
    ),
    (
        "Training — olmixexact (headline)",
        "fpv2 cell retrain (10k-fix-fpv2cell)",
        "/michaelryan/10k-fix-fpv2cell/",
        "interactive",
        "bad-loss 2e19-d512 retrain",
    ),
    (
        "Evals — cultural suite",
        "full resweep (cult-full-resweep2)",
        "/michaelryan/cult-full-resweep2/",
        "interactive",
        "all 117 runs, skip-existing",
    ),
    (
        "Evals — MMLU suite",
        "nem+rslp priority (mmlu-nemrslp-prio)",
        "/michaelryan/mmlu-nemrslp-prio/",
        "interactive",
        "13 runs for table dashes",
    ),
    (
        "Evals — MMLU suite",
        "stale-step redo (mmlu-stalestep-redo)",
        "/michaelryan/mmlu-stalestep-redo/",
        "interactive",
        "7 quarantine replacements",
    ),
    (
        "Evals — MMLU suite",
        "broad wave w5 (mmlumodels-eval-w5)",
        "/michaelryan/mmlumodels-eval-w5/",
        "batch",
        "221-run skip-existing",
    ),
    (
        "Evals — cultural suite",
        "stale-step redo (cult-stalestep-redo)",
        "/michaelryan/cult-stalestep-redo/",
        "interactive",
        "3 quarantine replacements",
    ),
    (
        "Evals — fwedu natural",
        "cultural (fwedu-nat-cult-evals)",
        "/michaelryan/fwedu-nat-cult-evals/",
        "interactive",
        "",
    ),
    ("Evals — fwedu natural", "MMLU (fwedu-nat-mmlu-evals2)", "/michaelryan/fwedu-nat-mmlu-evals2/", "interactive", ""),
    (
        "Evals — headline BPB",
        "stale-step redo (olmix-bpb-stalestep)",
        "/michaelryan/olmix-bpb-stalestep/",
        "interactive",
        "9 quarantine replacements",
    ),
    (
        "Evals — headline BPB",
        "last 3 deep cells (olmix-bpb-last3b)",
        "/michaelryan/olmix-bpb-last3b/",
        "interactive",
        "completes 294/294",
    ),
]

# Deliberately not running — shown so "missing" families read as decisions, not gaps.
PAUSED: list[tuple[str, str]] = [
    ("fineweb_cc domain sweep", "taken OFFLINE 2026-09-15 (user); relaunch is one command w/ skip-if-done"),
    ("resiliparse k5 arms", "KILLED 2026-09-14 (user: k20@100B is the way)"),
    ("lpv11 + fastpipe_v2 k5 arms", "KILLED 2026-09-16 (user: going with k20) — 93 children + fpv2-k5 coordinator"),
    ("high_quality + fastpipe_v1", "deprioritized 2026-09-12 (user focus set)"),
    ("lambda-bug generation (10k-cult-*/10k-mmlu-*)", "killed 2026-09-14 after the mislabeled-lambda incident"),
]


def _one_family(entry):
    group, label, prefix, band, note = entry
    out = subprocess.run(
        ["uv", "run", "iris", "--cluster", "marin", "job", "list", "--prefix", prefix, "--limit", "400"],
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    states: dict[str, int] = {}
    pending_reasons: dict[str, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("/"):
            continue
        # skip the coordinator row itself (exact prefix minus trailing slash)
        if parts[0].rstrip("/") == prefix.rstrip("/"):
            continue
        states[parts[1]] = states.get(parts[1], 0) + 1
        if parts[1] == "pending" and "Scheduler:" in line:
            reason = line.split("Scheduler:", 1)[1].strip()[:40]
            pending_reasons[reason] = pending_reasons.get(reason, 0) + 1
    top_reason = max(pending_reasons.items(), key=lambda kv: kv[1])[0] if pending_reasons else ""
    return {
        "group": group,
        "label": label,
        "band": band,
        "note": note,
        "states": states,
        "top_pending_reason": top_reason,
    }


def fleet_snapshot() -> dict:
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(_one_family, FAMILIES))
    return {
        "families": rows,
        "paused": [{"label": label, "note": note} for label, note in PAUSED],
    }
