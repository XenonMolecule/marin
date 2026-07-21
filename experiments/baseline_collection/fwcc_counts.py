# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the consolidated kept_fwcc manifest: total, AO3/BBC recovery, and in-pool overlap.

Writes a small counts.json (avoids the flaky job-log path). In-region.
"""

from __future__ import annotations

import json
import sys

import fsspec

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
M = f"{WORKSPACE}/fwcc_membership/kept_fwcc.parquet"
MEM = f"{WORKSPACE}/membership/*.parquet"
OUT = f"{WORKSPACE}/fwcc_membership/counts.json"


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(f"CREATE TEMP TABLE f AS SELECT url FROM read_parquet('{M}')")
    con.execute(f"CREATE TEMP TABLE m AS SELECT url FROM read_parquet('{MEM}')")
    n = con.execute("SELECT count(*) FROM f").fetchone()[0]
    ao3 = con.execute("SELECT count(*) FROM f WHERE url LIKE '%archiveofourown.org%'").fetchone()[0]
    bbc = con.execute("SELECT count(*) FROM f WHERE url LIKE '%bbc.co.uk%' OR url LIKE '%bbc.com%'").fetchone()[0]
    ff = con.execute("SELECT count(*) FROM f WHERE url LIKE '%fanfiction.net%'").fetchone()[0]
    in_pool = con.execute("SELECT count(*) FROM m SEMI JOIN f ON m.url = f.url").fetchone()[0]
    pool_total = con.execute("SELECT count(*) FROM m").fetchone()[0]
    r = {
        "kept_fwcc_total": n,
        "ao3": ao3,
        "bbc": bbc,
        "fanfiction": ff,
        "in_pool": in_pool,
        "pool_total": pool_total,
    }
    with fsspec.open(OUT, "w") as fh:
        json.dump(r, fh)
    print(f"[fwcc] {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
