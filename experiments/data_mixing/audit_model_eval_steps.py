# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit trained-model eval results for stale-checkpoint provenance.

For every ``results.json`` under ``metadata/<namespace>/5shot``, extract the evaluated
checkpoint step from the lm-eval ``configs`` block (``model_args`` carries the
``.../hf/step-N`` path) and join the run's true final step from its training result.
Writes ``gs://marin-us-central1/metadata/<namespace>/step_audit.csv`` with columns
run,evaled_step,final_step,verdict. Runs on-cluster: each results.json is ~228 MB.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import logging
import re

import fsspec

logger = logging.getLogger(__name__)
BUCKETS = (
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-west4",
    "marin-eu-west4",
)
TRAIN_PREFIX = "marin-us-central1/metadata/data_curation_10k_natural_results"
STEP_RE = re.compile(r"step-(\d+)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--namespace", required=True, choices=["mmlu_target_models", "cultural_target_models"])
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    paths = []
    for b in BUCKETS:
        try:
            paths += [
                p for p in fs.find(f"{b}/metadata/{args.namespace}/5shot", maxdepth=2) if p.endswith("/results.json")
            ]
        except FileNotFoundError:
            continue
    logger.info("auditing %d results", len(paths))

    def one(path: str):
        run = path.rsplit("/", 2)[-2]
        doc = json.load(fs.open(path))
        steps = set()
        for cfg in (doc.get("configs") or {}).values():
            blob = json.dumps(cfg)
            steps.update(int(m) for m in STEP_RE.findall(blob))
        evaled = max(steps) if steps else -1
        try:
            tdoc = json.load(fs.open(f"{TRAIN_PREFIX}/{run}.json"))
            final = int(tdoc["plan"]["train_steps"])
        except FileNotFoundError:
            final = -1
        if evaled < 0 or final < 0:
            verdict = "unknown"
        elif evaled in (final, final - 1):
            verdict = "ok"
        else:
            verdict = "STALE"
        return run, evaled, final, verdict

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, r in enumerate(pool.map(one, paths), 1):
            rows.append(r)
            if i % 25 == 0:
                logger.info("audited %d/%d", i, len(paths))

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "evaled_step", "final_step", "verdict"])
    w.writerows(rows)
    out = f"marin-us-central1/metadata/{args.namespace}/step_audit.csv"
    with fs.open(out, "w") as fh:
        fh.write(buf.getvalue())
    stale = [r for r in rows if r[3] == "STALE"]
    logger.info("wrote %s | STALE: %d / %d", out, len(stale), len(rows))
    for r in stale:
        logger.info("  STALE: %s evaled=%s final=%s", *r[:3])


if __name__ == "__main__":
    main()
