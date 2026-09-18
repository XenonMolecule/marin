# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll up trained-model eval results into one small CSV per namespace.

The per-run ``results.json`` files under ``metadata/{mmlu,cultural}_target_models/5shot``
embed per-sample logs (an MMLU file is ~228 MB), so anything that "just reads the
results" pulls tens of GB. This job runs ON-CLUSTER, extracts only the per-task
``choice_logprob,none`` / ``acc,none`` aggregates, and writes
``gs://<out-bucket>/metadata/<namespace>/rollup.csv`` (run,task,clp,acc — a few hundred
KB). Plot scripts read the rollup, never the raw files. Skips runs already present in
the existing rollup, so re-running is an incremental top-up.

    iris --cluster marin job run ... -- python -m experiments.data_mixing.consolidate_model_evals \\
        --namespace mmlu_target_models
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import logging

import fsspec

from experiments.data_mixing.verify_eval_integrity import NAMESPACES, verify_run

logger = logging.getLogger(__name__)

BUCKETS = (
    "marin-us-east5",
    "marin-us-east1",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-west4",
    "marin-eu-west4",
)
OUT_BUCKET = "marin-us-central1"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--namespace", required=True, choices=["mmlu_target_models", "cultural_target_models"])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs", use_listings_cache=False)
    out_path = f"{OUT_BUCKET}/metadata/{args.namespace}/rollup.csv"
    done_runs: set[str] = set()
    rows: list[tuple[str, str, float, float]] = []
    if fs.exists(out_path):
        with fs.open(out_path, "r") as fh:
            for row in csv.DictReader(fh):
                rows.append((row["run"], row["task"], float(row["clp"]), float(row["acc"])))
                done_runs.add(row["run"])
        logger.info("existing rollup: %d rows, %d runs", len(rows), len(done_runs))

    paths = []
    for bucket in BUCKETS:
        try:
            paths += [
                p
                for p in fs.find(f"{bucket}/metadata/{args.namespace}/5shot", maxdepth=2)
                if p.endswith("/results.json") and p.rsplit("/", 2)[-2] not in done_runs
            ]
        except FileNotFoundError:
            continue
    logger.info("new result files to roll up: %d", len(paths))

    subdir, _, expected_tasks, has_provenance = NAMESPACES[args.namespace]

    def one(path: str) -> list[tuple[str, str, float, float]]:
        run = path.rsplit("/", 2)[-2]
        # The rollup is what plots and tables read, so it only ever ingests datapoints
        # the integrity verifier passes: final-checkpoint provenance + complete task set.
        verdict = verify_run(fs, args.namespace, subdir, expected_tasks, has_provenance, run)
        if verdict[1] != "VERIFIED":
            logger.warning("REFUSED (eval integrity %s: %s): %s", verdict[1], verdict[2], run)
            return []
        doc = json.load(fs.open(path))
        out = []
        for task, metrics in (doc.get("results") or {}).items():
            if isinstance(metrics, dict) and "choice_logprob,none" in metrics:
                out.append(
                    (run, task, float(metrics["choice_logprob,none"]), float(metrics.get("acc,none", float("nan"))))
                )
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, out in enumerate(pool.map(one, paths), 1):
            rows.extend(out)
            if i % 20 == 0:
                logger.info("rolled up %d/%d", i, len(paths))

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run", "task", "clp", "acc"])
    w.writerows(rows)
    with fs.open(out_path, "w") as fh:
        fh.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    main()
