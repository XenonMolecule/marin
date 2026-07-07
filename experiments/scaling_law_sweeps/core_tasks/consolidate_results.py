# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Consolidate all CORE_TASKS results.json (5 buckets) into ONE CSV on GCS.

Runs as a CPU iris job so the many tiny per-run results.json are read IN-CLOUD
(each is a few KB, so reading them cross-region is negligible egress) instead of
being pulled to a laptop one-by-one. Emits a single CSV a human can pull with one
`gcloud storage cp`, then plot locally.

Columns: run_stem, method, hidden_dim, num_layers, budget, tokens, mean_core
  * mean_core = mean of the 12 CORE_TASKS' primary metric (acc_norm else acc).
  * tokens = budget / (6 * N(d,L)) with N(d,L) calibrated from the core_v2 CSV
    (isoflop C=6ND); method/batch-independent, so it covers every cell.
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
RESULTS_GLOB = "metadata/data_curation_10k_core_tasks_results/*/results.json"

# N(d,L) calibrated from scratch/plots/core_v2/core_v2_grid_x_tokens.csv.
N_BY_DL = {
    (512, 6): 1.161464e08,
    (1024, 11): 4.086971e08,
    (1536, 16): 1.003494e09,
    (2432, 24): 3.064112e09,
    (3584, 35): 9.169166e09,
}

RUN_RE = re.compile(r"curation-(?P<method>.+?)-expFM_natural-(?P<budget>[0-9]+e\+[0-9]+)-d(?P<d>[0-9]+)-L(?P<L>[0-9]+)-B(?P<B>[0-9]+)")


def norm_method(raw: str) -> str:
    return raw[:-4] if raw.endswith("_10k") else raw


def mean_core(doc: dict) -> float | None:
    res = doc.get("results", doc)
    vals = []
    for _t, m in res.items():
        if not isinstance(m, dict):
            continue
        v = m.get("acc_norm,none", m.get("acc,none", m.get("acc_norm", m.get("acc"))))
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def tokens_for(budget: str, d: int, L: int) -> float | None:
    n = N_BY_DL.get((d, L))
    return float(budget.replace("e+", "e")) / (6 * n) if n else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gs://marin-us-central1/metadata/core_tasks_summary.csv",
                    help="Single consolidated CSV to write.")
    args = ap.parse_args()

    fs = marin_filesystem("gcs")
    rows = []
    for b in BUCKETS:
        for path in fs.glob(f"gs://{b}/{RESULTS_GLOB}"):
            run = path.split("/data_curation_10k_core_tasks_results/")[1].split("/")[0]
            m = RUN_RE.match(run)
            if not m:
                logger.warning("unparsed run: %s", run)
                continue
            with fs.open(path if path.startswith("gs://") else f"gs://{path}", "r") as f:
                mc = mean_core(json.load(f))
            if mc is None:
                continue
            d, L = int(m["d"]), int(m["L"])
            tok = tokens_for(m["budget"], d, L)
            rows.append(dict(run_stem=run, method=norm_method(m["method"]), hidden_dim=d,
                             num_layers=L, budget=m["budget"], tokens=tok, mean_core=mc))
    logger.info("consolidated %d runs", len(rows))

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["run_stem", "method", "hidden_dim", "num_layers", "budget", "tokens", "mean_core"])
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r["method"], r["hidden_dim"], r["tokens"] or 0)))
    with fs.open(args.out, "w") as f:
        f.write(buf.getvalue())
    logger.info("wrote %s (%d rows)", args.out, len(rows))

    # coverage summary to logs
    from collections import defaultdict
    cnt = defaultdict(int)
    for r in rows:
        cnt[(r["hidden_dim"], r["method"])] += 1
    methods = sorted(set(r["method"] for r in rows))
    logger.info("coverage (method x width):")
    for mth in methods:
        logger.info("  %-16s %s", mth, {d: cnt[(d, mth)] for d in (512, 1024, 1536, 2432, 3584) if cnt[(d, mth)]})


if __name__ == "__main__":
    main()
