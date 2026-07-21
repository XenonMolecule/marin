# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull a few representative example docs per target domain from the resiliparse hits.

For the dev-set priorities WIP doc: fiction / Q&A / arxiv / science registers whose examples
aren't in the 0-vote residual sample (e.g. fiction is kept by dclm/nemo, so it's not 0-vote).
Reads resiliparse hits in-region, filters to a target domain list, keeps a few docs each with
a clear eval link (>=2 matched examples), writes a tiny examples parquet to pull locally.
"""

from __future__ import annotations

import sys

import fsspec

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k"
RES_HITS = f"{WORKSPACE}/keyword_hits/resiliparse/*.parquet"
OUT = f"{WORKSPACE}/register_examples.parquet"
TARGETS = [
    "fanfiction.net",
    "fictionpress.com",
    "literotica.com",
    "archiveofourown.org",
    "wattpad.com",  # fiction
    "stackexchange.com",
    "reddit.com",
    "answers.com",  # Q&A
    "arxiv.org",  # preprints
    "phys.org",
    "biomedcentral.com",
    "sciencedaily.com",
    "plos.org",  # science
]


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false; SET enable_progress_bar=false;")
    doms = ", ".join(f"'{d}'" for d in TARGETS)
    con.execute(
        f"""COPY (
          SELECT * EXCLUDE (rn) FROM (
            SELECT url, domain, subjects, tasks, n_examples, snippet,
                   row_number() OVER (PARTITION BY domain ORDER BY n_examples DESC) AS rn
            FROM read_parquet('{RES_HITS}')
            WHERE domain IN ({doms}) AND n_examples >= 2
          ) WHERE rn <= 4
        ) TO '{OUT}' (FORMAT parquet)"""
    )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
