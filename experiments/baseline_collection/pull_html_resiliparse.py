# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Pull raw HTML for a url list from devset_html AND compute resiliparse main-content, to a small json."""
from __future__ import annotations

import argparse
import json
import sys

import fsspec

HTML = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_html/*.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import duckdb

    from experiments.baseline_collection.extractors import resiliparse_main

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    with fsspec.open(args.urls, "r") as f:
        urls = json.load(f)
    placeholders = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    rows = con.execute(f"SELECT dev_url, html FROM read_parquet('{HTML}') WHERE dev_url IN ({placeholders})").fetchall()
    best: dict[str, dict] = {}
    for u, html in rows:
        if html and len(html) > len(best.get(u, {}).get("html", "")):
            best[u] = {"html": html}
    for u, d in best.items():
        try:
            d["resiliparse"] = resiliparse_main(d["html"])
        except Exception as e:
            d["resiliparse"] = f"[resiliparse error: {e}]"
    with fsspec.open(args.out, "w") as f:
        json.dump(best, f)
    print(f"pulled {len(best)}/{len(urls)} docs (html + resiliparse)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
