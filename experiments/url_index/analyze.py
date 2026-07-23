# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate + pairwise coverage over already-built url_index artifacts.

Given a collection and the datasets that built, this reads their compact
``keys``/``meta`` parquets, writes a lookup DuckDB and coverage CSVs to a results
prefix, and logs the containment/Jaccard matrices to stdout. Runnable locally
(small, tiny keys) or as an in-region Iris job (full, to keep large ``keys``
reads in-region). Datasets whose ``meta`` is absent (``--keys-only`` tiers like
resiliparse) still contribute to coverage but are skipped from the lookup index.

    python -m experiments.url_index.analyze --collection full \
        --datasets dclm high_quality nemotron_full resiliparse \
        --out-prefix gs://marin-us-central1/url_index/analysis/full
"""

import argparse
import logging
import os
import shutil
import tempfile

from marin.utils import fsspec_exists

from experiments.infinigram.targets import DATASETS, Collection
from experiments.url_index import consolidate, coverage, layout
from experiments.url_index.build import _upload

logger = logging.getLogger(__name__)


def _fmt_matrix(title: str, datasets: list[str], cell) -> str:
    """Render a square matrix (cell(a, b) -> float) as an aligned text table."""
    w = max([14, *(len(d) for d in datasets)])
    head = " " * (w + 2) + "".join(f"{d[:w]:>{w + 2}}" for d in datasets)
    lines = [title, head]
    for a in datasets:
        lines.append(f"{a:<{w + 2}}" + "".join(f"{cell(a, b):>{w + 2}.3f}" for b in datasets))
    return "\n".join(lines)


def analyze(collection: Collection, datasets: list[str], out_prefix: str, local_dir: str) -> None:
    """Consolidate lookup + coverage for one collection; upload results, log matrices."""
    keys_paths, meta_paths = [], []
    for ds in datasets:
        region = DATASETS[ds].region
        kp = layout.keys_path(region, collection, ds)
        mp = layout.meta_path(region, collection, ds)
        if fsspec_exists(kp):
            keys_paths.append(kp)
            if fsspec_exists(mp):
                meta_paths.append(mp)
            else:
                logger.info("%s has no meta (keys-only tier); coverage-only", ds)
        else:
            logger.warning("skip %s: keys.parquet missing at %s", ds, kp)
    if not keys_paths:
        logger.warning("no built datasets for collection %s; nothing to analyze", collection.value)
        return

    col = collection.value
    if meta_paths:
        db_local = os.path.join(local_dir, f"url_lookup_{col}.duckdb")
        consolidate.build_index(meta_paths, collection, db_local)
        _upload(db_local, f"{out_prefix}/url_lookup_{col}.duckdb")

    # url_h is the universal cross-dataset key (every doc has a url); rid_h works
    # only for tiers carrying warc_record_id; text_h compares identical extracted text.
    for key in ("url_h", "rid_h", "text_h"):
        ds_list, sizes, rows = coverage.compute(keys_paths, key)
        cov_dir = os.path.join(local_dir, f"cov_{col}_{key}")
        coverage._write_csvs(ds_list, sizes, rows, cov_dir, key)
        for fn in os.listdir(cov_dir):
            _upload(os.path.join(cov_dir, fn), f"{out_prefix}/{fn}")
        containment = {(r["a"], r["b"]): r["containment_a_in_b"] for r in rows}
        jac = {(r["a"], r["b"]): r["jaccard"] for r in rows}
        logger.info("===== coverage [%s / %s] =====  sizes: %s", col, key, sizes)
        logger.info(
            "\n%s",
            _fmt_matrix(
                f"containment  cell(a,b)=|a and b|/|a|  ({col}/{key})",
                ds_list,
                lambda a, b, c=containment: 1.0 if a == b else c.get((a, b), 0.0),
            ),
        )
        logger.info(
            "\n%s",
            _fmt_matrix(f"jaccard  ({col}/{key})", ds_list, lambda a, b, j=jac: 1.0 if a == b else j.get((a, b), 0.0)),
        )
    logger.info("Uploaded %s results to %s", col, out_prefix)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidate + coverage over built url_index artifacts.")
    p.add_argument("--collection", required=True, choices=[c.value for c in Collection])
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--out-prefix", required=True, help="gs:// (or local) results prefix.")
    p.add_argument("--local-dir", default=None, help="Local scratch dir (default: a tempdir).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    local_dir = args.local_dir or tempfile.mkdtemp(prefix="url_index_analyze_")
    os.makedirs(local_dir, exist_ok=True)
    try:
        analyze(Collection(args.collection), args.datasets, args.out_prefix, local_dir)
    finally:
        if not args.local_dir:
            shutil.rmtree(local_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
