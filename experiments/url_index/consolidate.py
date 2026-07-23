# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate per-dataset ``meta.parquet`` files into one local lookup DuckDB.

The result -- ``url_lookup_keys.duckdb`` -- is the small, text-free index the
interactive :mod:`~experiments.url_index.lookup` client opens locally for
sub-second lookups. It reads only the compact routing columns (never text), so
building it pulls megabytes, not the in-region text stores.

    python -m experiments.url_index.consolidate \
        --meta gs://marin-us-central1/url_index/small/*/meta.parquet \
               gs://marin-us-central2/url_index/small/*/meta.parquet \
        --collection small --out /tmp/url_lookup_small.duckdb
"""

import argparse
import logging

import duckdb

from experiments.infinigram.targets import DATASETS, Collection
from experiments.url_index.duckdb_gcs import maybe_register_gcs

logger = logging.getLogger(__name__)


def _dataset_region_case() -> str:
    """A SQL CASE mapping the ``dataset`` column to its registry region."""
    whens = " ".join(f"WHEN '{ds}' THEN '{spec.region}'" for ds, spec in DATASETS.items())
    return f"CASE dataset {whens} ELSE 'unknown' END"


def build_index(meta_globs: list[str], collection: Collection, out_path: str) -> int:
    """Build the local lookup DuckDB from meta parquets. Returns the row count."""
    con = duckdb.connect(out_path)
    maybe_register_gcs(con, meta_globs)
    globs = ", ".join(f"'{g}'" for g in meta_globs)
    con.execute(
        f"CREATE OR REPLACE TABLE docs AS "
        f"SELECT dataset, {_dataset_region_case()} AS region, url_key, domain, "
        f"       warc_record_id, snapshot, warc_file, text_len "
        f"FROM read_parquet([{globs}])"
    )
    con.execute("CREATE INDEX idx_url_key ON docs (url_key)")
    con.execute("CREATE INDEX idx_domain ON docs (domain)")
    # Record which collection this index covers so lookup can resolve text paths.
    con.execute("CREATE OR REPLACE TABLE index_meta AS SELECT ? AS collection", [collection.value])
    n = con.execute("SELECT count(*) FROM docs").fetchone()[0]
    con.close()
    logger.info("Built %s: %d docs across %d datasets", out_path, n, len(meta_globs))
    return n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Consolidate meta.parquet files into a local lookup DuckDB.")
    p.add_argument("--meta", nargs="+", required=True, help="Glob(s) to meta.parquet files (gs:// or local).")
    p.add_argument("--collection", required=True, choices=[c.value for c in Collection])
    p.add_argument("--out", required=True, help="Output .duckdb path.")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args()
    build_index(args.meta, Collection(args.collection), args.out)


if __name__ == "__main__":
    main()
