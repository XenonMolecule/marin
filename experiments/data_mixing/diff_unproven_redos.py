# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diff quarantined UNPROVEN evals against their provenance-backed redos.

For every run in the unproven-redo manifests, compares the quarantined
``results.unproven.json`` with the freshly re-scored ``results.json`` (same directory)
task by task on ``choice_logprob,none``. This measures how many of the 212 unprovable
evals were actually wrong: IDENTICAL (bitwise-equal scores), CLOSE (max |dclp| < 0.005,
i.e. harness nondeterminism), or DIFFERENT (>= 0.005 on any task — a real stale-eval
casualty). Also reports runs whose redo has not landed (NO_NEW) and redos lacking a
provenance sidecar (UNPROVEN_AGAIN — should be none).

Writes ``gs://marin-us-central1/metadata/eval_integrity/unproven_diff_<ns>.csv``.
Run ON-CLUSTER: MMLU documents are ~228 MB and each run loads two of them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import logging

import fsspec

logger = logging.getLogger(__name__)

BUCKETS = ("marin-us-east5", "marin-us-east1", "marin-us-central1",
           "marin-us-central2", "marin-us-west4", "marin-eu-west4")
OUT_PREFIX = "marin-us-central1/metadata/eval_integrity"
NAMESPACES = {
    "mmlu_target_models": ("metadata/mmlu_target_models/5shot",
                           "experiments/core_eval_manifests/mmlu_unproven_redo.txt"),
    "cultural_target_models": ("metadata/cultural_target_models/5shot",
                               "experiments/core_eval_manifests/cultural_unproven_redo.txt"),
}
CLOSE_TOLERANCE = 0.005


def clp_by_task(fs, path: str) -> dict[str, float]:
    doc = json.load(fs.open(path))
    return {t: float(m["choice_logprob,none"]) for t, m in (doc.get("results") or {}).items()
            if isinstance(m, dict) and "choice_logprob,none" in m}


def diff_run(fs, subdir: str, run: str) -> tuple:
    base = next((f"{b}/{subdir}/{run}" for b in BUCKETS
                 if fs.exists(f"{b}/{subdir}/{run}/results.unproven.json")), None)
    if base is None:
        return (run, "NO_OLD", "", "")
    if not fs.exists(f"{base}/results.json"):
        return (run, "NO_NEW", "", base.split("/")[0])
    old = clp_by_task(fs, f"{base}/results.unproven.json")
    new = clp_by_task(fs, f"{base}/results.json")
    shared = old.keys() & new.keys()
    max_delta = max((abs(old[t] - new[t]) for t in shared), default=float("nan"))
    sidecar = fs.exists(f"{base}/provenance.json")
    if not sidecar:
        verdict = "UNPROVEN_AGAIN"
    elif max_delta == 0.0:
        verdict = "IDENTICAL"
    elif max_delta < CLOSE_TOLERANCE:
        verdict = "CLOSE"
    else:
        verdict = "DIFFERENT"
    return (run, verdict, f"{max_delta:.6f}", base.split("/")[0])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    for ns, (subdir, manifest) in NAMESPACES.items():
        with open(manifest, newline="") as f:
            runs = sorted(r["run_name"] for r in csv.DictReader(f))
        logger.info("%s: diffing %d runs", ns, len(runs))
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, row in enumerate(pool.map(lambda r: diff_run(fs, subdir, r), runs), 1):
                rows.append(row)
                if i % 25 == 0:
                    logger.info("%s: %d/%d", ns, i, len(runs))
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["run", "verdict", "max_abs_dclp", "bucket"])
        w.writerows(sorted(rows))
        with fs.open(f"{OUT_PREFIX}/unproven_diff_{ns}.csv", "w") as fh:
            fh.write(buf.getvalue())
        counts: dict[str, int] = {}
        for r in rows:
            counts[r[1]] = counts.get(r[1], 0) + 1
        logger.info("%s VERDICTS: %s", ns, counts)
        for r in rows:
            if r[1] in ("DIFFERENT", "UNPROVEN_AGAIN", "NO_OLD"):
                logger.warning("  %s %s max|dclp|=%s", r[1], r[0], r[2])


if __name__ == "__main__":
    main()
