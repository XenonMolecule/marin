# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Consolidate the recovered FineWeb-CC kept-URL shards into a single kept_fwcc manifest.

The recovery re-extraction wrote per-segment shards {url, file_path, dump} to
fwcc_membership/_shards/. This dedups to the distinct URL set (the 4th-verifier signal),
reports the total + AO3/BBC sanity counts, and writes the consolidated manifest. In-region.
"""

from __future__ import annotations

import sys

import fsspec

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
SHARDS = f"{WORKSPACE}/fwcc_membership/_shards/*.parquet"
OUT = f"{WORKSPACE}/fwcc_membership/kept_fwcc.parquet"


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    # union_by_name tolerates any near-empty shards; distinct url = the kept_fwcc set.
    con.execute(
        f"CREATE TEMP TABLE f AS SELECT DISTINCT url FROM read_parquet('{SHARDS}', union_by_name=true) WHERE url IS NOT NULL AND url <> ''"
    )
    n = con.execute("SELECT count(*) FROM f").fetchone()[0]
    ao3 = con.execute("SELECT count(*) FROM f WHERE url LIKE '%archiveofourown.org%'").fetchone()[0]
    bbc = con.execute("SELECT count(*) FROM f WHERE url LIKE '%bbc.co.uk%' OR url LIKE '%bbc.com%'").fetchone()[0]
    fanfic = con.execute("SELECT count(*) FROM f WHERE url LIKE '%fanfiction.net%'").fetchone()[0]
    print(f"[fwcc] distinct kept_fwcc urls: {n:,}")
    print(f"[fwcc] AO3 (archiveofourown.org): {ao3:,}  |  BBC (bbc.co.uk/.com): {bbc:,}  |  fanfiction.net: {fanfic:,}")
    con.execute(f"COPY f TO '{OUT}' (FORMAT parquet)")
    print(f"[fwcc] wrote consolidated manifest -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
