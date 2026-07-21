# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Report dev-set HTML coverage and pull the 10 gold-extraction docs' raw HTML (in-region, on Iris).

Runs after `collect_devset_html.py`. Over the `devset_html` shards it (1) computes which of the ~1934
dev urls actually got HTML (coverage / genuine misses), and (2) extracts the raw HTML for the 10
selected gold-extraction docs into a small json that can be pulled to a laptop without egressing the
whole HTML corpus.
"""

from __future__ import annotations

import json
import sys

import fsspec

HTML = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_html/*.parquet"
ALLOWLIST = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_urls.json"
COVERAGE_OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_html_coverage.json"
GOLD10_OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/gold_html_10.json"

GOLD_10 = [
    "http://www.patheos.com/blogs/crossexamined/2013/12/argument-for-god-from-differential-equations/",
    "http://everything2.com/title/polyunsaturated",
    "http://phpkode.com/source/p/h-tracker/forum/db/mssql.php",
    "http://quizlet.com/15825567/developmental-psychology-final-quizlet-flash-cards/",
    "http://www.tolkienfanfiction.com/Story_Read_Chapter.php?CHid=525",
    "http://sacred-texts.com/lcr/abs/abs22.htm",
    "https://www.coursehero.com/file/8194696/lalit-yo-fasolut-ionmadebydisso-lving100gofC/",
    "https://kiss.kstudy.com/thesis/thesis-view.asp?key=3938868",
    "https://blog.wolfram.com/2020/02/20/15-ways-wolframalpha-can-help-with-your-classes/",
    "http://www.interviewgig.com/electronics-engineering/",
]


def main() -> int:
    import duckdb

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))

    found = [r[0] for r in con.execute(f"SELECT DISTINCT dev_url FROM read_parquet('{HTML}')").fetchall()]
    with fsspec.open(ALLOWLIST, "r") as f:
        allow = set(json.load(f))
    found_set = set(found)
    missing = sorted(allow - found_set)
    coverage = {"total": len(allow), "found": len(found_set), "missing_count": len(missing), "missing": missing}
    with fsspec.open(COVERAGE_OUT, "w") as f:
        json.dump(coverage, f)
    print(f"coverage: {len(found_set)}/{len(allow)} dev urls have HTML ({len(missing)} missing)")

    # 10 gold docs — longest html per url in case of dup snapshots
    placeholders = ", ".join(f"'{u}'" for u in GOLD_10)
    rows = con.execute(
        f"SELECT dev_url, html FROM read_parquet('{HTML}') WHERE dev_url IN ({placeholders})"
    ).fetchall()
    best: dict[str, str] = {}
    for url, html in rows:
        if html and len(html) > len(best.get(url, "")):
            best[url] = html
    with fsspec.open(GOLD10_OUT, "w") as f:
        json.dump(best, f)
    print(f"gold-10 html: pulled {len(best)}/{len(GOLD_10)} (missing: {sorted(set(GOLD_10) - set(best))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
