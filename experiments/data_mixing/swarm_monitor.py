# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Periodic swarm status line, built so a FAILED READ can never look like DATA.

Three false alarms in one night, all the same defect in different disguises: a truncated
``job list`` read as "running collapsed to 0", expired credentials read as "every completion
vanished", and one region erroring read as "counts went backwards". Each was patched
individually; none of the patches generalised.

The invariant that does generalise: **completion counts are append-only.** Results are written
once and never removed, so a count that goes DOWN is proof of a bad read, never of lost work.
Combined with per-region error detection (rather than silently summing whichever regions
answered), that covers the whole family.

Emits one line per cycle. On a bad read it says so and holds the previous values rather than
publishing a number it does not trust.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time

CORPORA = ("dclm_10k", "high_quality_10k", "resiliparse_10k", "lpv11_fastpipe_v1_10k")
SHORT = {"dclm_10k": "dclm", "high_quality_10k": "hq", "resiliparse_10k": "resil", "lpv11_fastpipe_v1_10k": "lpv11"}
BUCKETS = ("marin-us-east5", "marin-us-central1", "marin-eu-west4", "marin-us-west4")
SWARM_TOTAL = 363


def _listing(uri: str) -> str | None:
    """Raw listing text, or None if the read itself failed.

    None and empty must stay distinguishable -- conflating them produced every false alarm
    this module exists to prevent.
    """
    try:
        p = subprocess.run(["gcloud", "storage", "ls", uri], capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        # "no objects matched" is a legitimate empty listing, not a failure.
        return "" if "matched no objects" in (p.stderr or "").lower() else None
    return p.stdout


def _indices(listing: str) -> set[int]:
    """Swarm indices named in a listing.

    Counting DISTINCT indices rather than objects matters: a run trained in two regions
    before the dedup fix leaves two result files for one index, so an object count overstates
    progress (hq read 344 files for 318 real runs).
    """
    return {int(m) for m in re.findall(r"-i(\d{4})-", listing)}


def snapshot() -> tuple[dict[str, set[int]], dict[str, set[int]], int, int]:
    """(trained_indices, evaluated_indices, failed_reads, running) per corpus."""
    trained = {c: set() for c in CORPORA}
    evaluated = {c: set() for c in CORPORA}
    failed = 0
    for bucket in BUCKETS:
        for corpus in CORPORA:
            txt = _listing(f"gs://{bucket}/metadata/olmix_swarm_results/{corpus}/")
            if txt is None:
                failed += 1
            else:
                trained[corpus] |= _indices(txt)
        txt = _listing(f"gs://{bucket}/metadata/olmix_swarm_bpb/")
        if txt is None:
            failed += 1
        else:
            for corpus in CORPORA:
                evaluated[corpus] |= _indices("\n".join(l for l in txt.splitlines() if corpus in l))
    running = -1
    try:
        p = subprocess.run(
            [
                "iris",
                "--cluster",
                "marin",
                "query",
                "SELECT COUNT(*) n FROM tasks WHERE state=3 AND job_id LIKE '/michaelryan/olmix-coord%'",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        digits = "".join(ch for ch in p.stdout.strip().splitlines()[-1] if ch.isdigit())
        running = int(digits) if digits else -1
    except Exception:
        running = -1
    return trained, evaluated, failed, running


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--interval", type=float, default=1800.0)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    prev_t = {c: 0 for c in CORPORA}
    prev_e = {c: 0 for c in CORPORA}
    announced = False
    while True:
        trained, evaluated, failed, running = snapshot()
        t = {c: len(trained[c]) for c in CORPORA}
        # Only count evals for runs that actually finished training.
        e = {c: len(evaluated[c] & trained[c]) for c in CORPORA}
        regressed = any(t[c] < prev_t[c] or e[c] < prev_e[c] for c in CORPORA)
        stamp = time.strftime("%H:%MZ", time.gmtime())

        if failed or regressed:
            print(
                f"{stamp} WARN unreliable read ({failed} failed queries"
                f"{', counts regressed' if regressed else ''}) -- counts are append-only, so this is a "
                f"read fault, not lost work. Holding previous.",
                flush=True,
            )
            time.sleep(args.interval if not args.once else 0)
            if args.once:
                return
            continue

        per_corpus = " | ".join(f"{SHORT[c]} {t[c]}/{SWARM_TOTAL} trained, {e[c]} eval" for c in CORPORA)
        line = f"{stamp} {per_corpus} | running={running if running>=0 else '?'}"
        if all(t[c] == SWARM_TOTAL for c in CORPORA) and all(e[c] == SWARM_TOTAL for c in CORPORA):
            if not announced:
                total = SWARM_TOTAL * len(CORPORA)
                print(
                    f"{stamp} *** SWEEP AND EVALS COMPLETE -- {total}/{total} trained, {total}/{total} evaluated. "
                    f"READY FOR THE FINAL FIT. ***",
                    flush=True,
                )
                announced = True
        else:
            remaining_t = sum(SWARM_TOTAL - t[c] for c in CORPORA)
            remaining_e = sum(t[c] - e[c] for c in CORPORA)
            line += f" | {remaining_t} to train, {remaining_e} to eval"
            print(line, flush=True)
        prev_t, prev_e = t, e
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
