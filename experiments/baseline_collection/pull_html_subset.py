# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull the raw HTML for a specific url list out of the devset_html shards (Iris CPU job, in-region).

Reads a GCS json list of dev urls and writes a small `{url: html}` json so a laptop can grab just
those pages without egressing the whole HTML corpus. Used to feed targeted extraction experiments.
"""

from __future__ import annotations

import argparse
import json
import sys

import fsspec

HTML = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_html/*.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True, help="GCS json list of dev urls")
    ap.add_argument("--out", required=True, help="GCS json output {url: html}")
    args = ap.parse_args()

    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    with fsspec.open(args.urls, "r") as f:
        urls = json.load(f)
    placeholders = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    rows = con.execute(
        f"SELECT dev_url, html FROM read_parquet('{HTML}') WHERE dev_url IN ({placeholders})"
    ).fetchall()
    best: dict[str, str] = {}
    for u, html in rows:
        if html and len(html) > len(best.get(u, "")):
            best[u] = html
    with fsspec.open(args.out, "w") as f:
        json.dump(best, f)
    print(f"pulled {len(best)}/{len(urls)} docs' html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
