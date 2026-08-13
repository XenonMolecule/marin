# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a devset run against the vendored reference metrics.

Reads a run dir (local or a shared gs:// run dir written by parallel shards) and
the devset (local dir or a staged gs:// tarball), then prints acc / macro / topQ
/ gold-lev — the numbers to compare against the frozen certs (two-stage
~93.7/90.8 topQ>=133; one-call ~91.3/87.3). This is the single scoring pass after
a multi-shard run.

Usage::

    python -m experiments.baseline_collection.score_devset \
        gs://marin-us-central1/devset/runs/llm_pipeline_v1 \
        --devset-gcs gs://marin-us-central1/devset/marin_devset_1934.tar.gz
"""

from __future__ import annotations

import argparse
import tempfile

from experiments.baseline_collection.devset.dataset import load_devset
from experiments.baseline_collection.devset.metrics import run_summary
from experiments.baseline_collection.devset.runs import load_run
from experiments.baseline_collection.devset.staging import download_run_dir, fetch_devset_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run dir: local path OR gs:// shared run dir")
    parser.add_argument("--devset-gcs", default=None, help="Staged devset tarball (gs://...tar.gz)")
    parser.add_argument("--devset-dir", default=None, help="Local devset dir (alternative to --devset-gcs)")
    parser.add_argument("--subset", default=None, help="Record-id file to restrict scoring to")
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="score_")
    devset_src = args.devset_dir or args.devset_gcs
    if not devset_src:
        parser.error("provide --devset-gcs or --devset-dir")
    devset_dir = fetch_devset_dir(devset_src, work)
    run_dir = download_run_dir(args.run_dir, work) if args.run_dir.startswith("gs://") else args.run_dir

    docs = load_devset(devset_dir)
    if args.subset:
        wanted = set(open(args.subset, encoding="utf-8").read().split())
        docs = [d for d in docs if d.record_id in wanted]

    summary = run_summary(load_run(run_dir), docs)
    st = summary["keep_drop_strong"]["overall"]
    all_ = summary["keep_drop_all"]["overall"]
    tq = summary["top_quality"]
    g = summary["gold"]
    d = summary["decisions"]
    print(
        f"decisions: kept={d.get('keep', 0)} dropped={d.get('drop', 0)} "
        f"context={d.get('context', 0)} error={d.get('error', 0)}"
    )
    print(f"strong : acc={100 * st['accuracy']:.1f}  macro={100 * st['macro_f1']:.1f}")
    print(f"all    : acc={100 * all_['accuracy']:.1f}  macro={100 * all_['macro_f1']:.1f}")
    print(f"topQ   : {tq['kept']}/{tq['kept'] + tq['dropped']} kept")
    print(
        f"gold   : kept-lev={g['kept_lev_sim']['mean']:.3f}  overall-lev={g['overall_lev_sim']['mean']:.3f}  "
        f"(n_gold={g['n_gold']})"
    )


if __name__ == "__main__":
    main()
