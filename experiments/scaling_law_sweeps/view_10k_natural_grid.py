# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only view of the frozen 10k-natural isoFLOP grid (6 methods x 38 cells = 228).

A *cell* is one (method, budget, width) training point in the frozen ladder that
`launch_10k_natural.enumerate_10k_natural_plans` produces. A cell is DONE when its
result JSON exists at `<DEFAULT_RESULTS_PREFIX><run_name_core>.json`; otherwise it
is still PENDING (queued, running, or not yet launched).

This script ONLY reads: it enumerates the plans in-process and lists the result
JSONs in GCS. It never submits, mutates the registry, or touches iris. Safe to run
from any session concurrently with the supervisor.

    python experiments/scaling_law_sweeps/view_10k_natural_grid.py            # summary + missing cells
    python experiments/scaling_law_sweeps/view_10k_natural_grid.py --all      # every cell, DONE/PENDING
    python experiments/scaling_law_sweeps/view_10k_natural_grid.py --method dclm_10k --all
"""

from __future__ import annotations

import argparse
import subprocess

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX,
    METHOD_NAMES,
    enumerate_10k_natural_plans,
)


def done_run_keys() -> set[str]:
    """Return the set of run_name_cores whose result JSON exists in GCS."""
    listing = subprocess.run(
        ["gcloud", "storage", "ls", f"{DEFAULT_RESULTS_PREFIX}*.json"],
        capture_output=True,
        text=True,
    )
    keys = set()
    for line in listing.stdout.splitlines():
        name = line.rsplit("/", 1)[-1]
        if name.endswith(".json"):
            keys.add(name[: -len(".json")])
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", help="Restrict to one method (e.g. dclm_10k).")
    parser.add_argument("--all", action="store_true", help="List every cell, not just missing ones.")
    args = parser.parse_args()

    methods = [args.method] if args.method else list(METHOD_NAMES)
    done = done_run_keys()

    grand_done = grand_total = 0
    for method in methods:
        plans = enumerate_10k_natural_plans((method,))
        n_done = sum(1 for p in plans if p.run_name_core in done)
        grand_done += n_done
        grand_total += len(plans)
        print(f"\n{method}: {n_done}/{len(plans)} done")
        for p in plans:
            is_done = p.run_name_core in done
            if args.all or not is_done:
                print(f"  {'DONE   ' if is_done else 'PENDING'}  {p.run_name_core}")

    print(f"\nTOTAL: {grand_done}/{grand_total} cells done")


if __name__ == "__main__":
    main()
