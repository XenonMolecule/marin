# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Surgically stop ONLY the provisional 9e21 cells of the 10k-natural sweep.

The 9e21 extension cells (d2432 & d3584, both methods → 4 jobs) are aspirational
top anchors that may prove unreasonable to finish. This kills exactly those,
WITHOUT touching the coordinator or any other cell.

WHY NOT just stop the coordinator: a sweep coordinator orphan-kills ALL its
children when stopped (even with --no-include-children). So we target the four
9e21 *leaf* child jobs by name and stop them with --no-include-children (they
have no children). The coordinator submits once and then only sleeps, so it does
NOT resubmit a stopped child — the kill sticks. To also prevent a future
`launch_10k_natural.py` re-run from resurrecting them, delete the (9e21, 4) entry
from EXTENSION in that file.

Dry-run by default (prints what it WOULD stop). Pass --confirm to actually stop.

    python experiments/scaling_law_sweeps/kill_10k_9e21.py            # preview
    python experiments/scaling_law_sweeps/kill_10k_9e21.py --confirm  # do it
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

CLUSTER = "marin"
DEFAULT_COORD = "/michaelryan/10k-natural-coord"
# run_name embeds f"{budget:.0e}" => 9e21 -> "9e+21"; iris lowercases names.
BUDGET_TOKEN = "9e+21"
LIVE_STATES = ("running", "pending")


def _list_children(coord: str) -> list[dict]:
    r = subprocess.run(
        [".venv/bin/iris", "--cluster", CLUSTER, "job", "list", "--prefix", coord, "--json"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(f"iris job list failed:\n{r.stderr}")
    return [j for j in json.loads(r.stdout) if j["name"].startswith(coord + "/")]


def _is_live(job: dict) -> bool:
    return any(job.get("task_state_counts", {}).get(s, 0) for s in LIVE_STATES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coord", default=DEFAULT_COORD, help=f"Coordinator job (default {DEFAULT_COORD}).")
    ap.add_argument("--confirm", action="store_true", help="Actually stop (default: dry-run preview).")
    args = ap.parse_args()

    targets = [j for j in _list_children(args.coord) if BUDGET_TOKEN in j["name"].lower()]
    live = [j for j in targets if _is_live(j)]

    if not targets:
        print(f"No {BUDGET_TOKEN} children found under {args.coord} (already gone, or coord name differs).")
        return 0
    print(f"Matched {len(targets)} {BUDGET_TOKEN} cells ({len(live)} live):")
    for j in targets:
        print(f"  {'LIVE ' if _is_live(j) else 'done '} {j['name']}")

    if not live:
        print("Nothing live to stop.")
        return 0
    # iris job stop needs the canonical mixed-case job_id; the `name` field is
    # lowercased and would 404.
    ids = [j.get("job_id") or j["name"] for j in live]
    if not args.confirm:
        print(f"\nDRY-RUN. Would stop {len(ids)} job(s) with --no-include-children. Re-run with --confirm.")
        return 0

    r = subprocess.run(
        [".venv/bin/iris", "--cluster", CLUSTER, "job", "stop", *ids, "--no-include-children"],
        capture_output=True,
        text=True,
    )
    print(r.stdout, r.stderr)
    if r.returncode != 0:
        return 1
    print(f"Stopped {len(ids)} 9e21 job(s). Coordinator + all other cells untouched.")
    print("Tip: also delete (9e21, 4) from EXTENSION in launch_10k_natural.py so a re-launch won't resurrect them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
