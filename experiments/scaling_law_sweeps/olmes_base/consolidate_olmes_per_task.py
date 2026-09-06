# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate OLMES results into a LONG per-task CSV (one row per run x task).

Sibling of `consolidate_olmes_results.py`, but instead of collapsing each run to a
single mean it emits every task's primary metric separately, so downstream analysis
can ask "which curation method is best on each benchmark" at matched (width, budget).

Runs as a CPU iris job (reads the many tiny results.json in-cloud). Writes one CSV.

Columns: run_stem, method, hidden_dim, num_layers, budget, tokens, task, value
  * value = the task's primary metric (acc_norm,none else acc,none else acc_norm/acc).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re

from rigging.filesystem import filesystem as marin_filesystem

logger = logging.getLogger(__name__)

BUCKETS = ["marin-us-east5", "marin-eu-west4", "marin-us-central1", "marin-us-central2", "marin-us-east1"]
RESULTS_GLOB = "metadata/olmes_base_results/*/results.json"

N_BY_DL = {
    (512, 6): 1.161464e08,
    (1024, 11): 4.086971e08,
    (1536, 16): 1.003494e09,
    (2432, 24): 3.064112e09,
    (3584, 35): 9.169166e09,
}

RUN_RE = re.compile(
    r"curation-(?P<method>.+?)-expFM_natural-(?P<budget>[0-9]+e\+[0-9]+)-d(?P<d>[0-9]+)-L(?P<L>[0-9]+)-B(?P<B>[0-9]+)"
)


def norm_method(raw: str) -> str:
    return raw[:-4] if raw.endswith("_10k") else raw


def task_value(m: dict) -> float | None:
    v = m.get("acc_norm,none", m.get("acc,none", m.get("acc_norm", m.get("acc"))))
    return float(v) if isinstance(v, (int, float)) else None


def tokens_for(budget: str, d: int, L: int) -> float | None:
    n = N_BY_DL.get((d, L))
    return float(budget.replace("e+", "e")) / (6 * n) if n else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gs://marin-us-central1/metadata/olmes_base_per_task.csv")
    args = ap.parse_args()

    fs = marin_filesystem("gcs")
    rows = []
    n_runs = 0
    for b in BUCKETS:
        for path in fs.glob(f"gs://{b}/{RESULTS_GLOB}"):
            run = path.split("/olmes_base_results/")[1].split("/")[0]
            m = RUN_RE.match(run)
            if not m:
                logger.warning("unparsed run: %s", run)
                continue
            with fs.open(path if path.startswith("gs://") else f"gs://{path}", "r") as f:
                doc = json.load(f)
            res = doc.get("results", doc)
            d, L = int(m["d"]), int(m["L"])
            tok = tokens_for(m["budget"], d, L)
            got = False
            for task, metrics in res.items():
                if not isinstance(metrics, dict):
                    continue
                v = task_value(metrics)
                if v is None:
                    continue
                rows.append(
                    dict(
                        run_stem=run,
                        method=norm_method(m["method"]),
                        hidden_dim=d,
                        num_layers=L,
                        budget=m["budget"],
                        tokens=tok,
                        task=task,
                        value=v,
                    )
                )
                got = True
            n_runs += got
    logger.info("emitted %d task-rows from %d runs", len(rows), n_runs)

    buf = io.StringIO()
    w = csv.DictWriter(
        buf, fieldnames=["run_stem", "method", "hidden_dim", "num_layers", "budget", "tokens", "task", "value"]
    )
    w.writeheader()
    w.writerows(rows)
    with fs.open(args.out, "w") as f:
        f.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
