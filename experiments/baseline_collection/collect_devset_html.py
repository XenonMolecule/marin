# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join the hand-label dev set to its raw HTML (Zephyr, on Iris) so specs can be re-run on it.

The dev set (~1934 docs) is only useful for testing new extractor specs if every doc is paired
with the raw HTML the extractor sees. That HTML already exists, decoded, in two us-east5 sources:

  * matched + mined-negative docs  -> `bert_pipeline/decoded_10k`      (the 10k-WARC pool decode)
  * random 2013-2026 docs          -> `internet_timespan_sample/v1`    (the full-timeline HTML sample)

Both carry `{doc_id, warc_hash, url, snapshot, html}`. This scans both, keeps only rows whose url is
in the dev allowlist (exact OR normalized match — url quirks like trailing slash / query / www cost
join hits otherwise), and writes `{dev_url, src_url, snapshot, source, html}`. Coverage (which dev
urls were NOT found) is computed afterwards from the tiny output url column, so re-fetch-from-WARC is
only ever needed for the genuine misses.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import fsspec
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

DECODED = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k"
TIMESPAN = "gs://marin-us-east5/documents/internet_timespan_sample/v1"
ALLOWLIST = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_urls.json"
OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/devset_html"

_SCHEME_WWW = re.compile(r"^https?://(www\.)?")


def _nrm(u: str) -> str:
    """Normalize a url for tolerant matching: drop scheme/www, query, fragment, trailing slash."""
    u = _SCHEME_WWW.sub("", (u or "").lower())
    u = u.split("?", 1)[0].split("#", 1)[0]
    return u.rstrip("/")


def _load_allow() -> tuple[dict[str, str], dict[str, str]]:
    with fsspec.open(ALLOWLIST, "r") as f:
        urls = json.load(f)
    exact = {u: u for u in urls}
    # normalized -> canonical dev url; first writer wins on the rare normalized collision
    norm: dict[str, str] = {}
    for u in urls:
        norm.setdefault(_nrm(u), u)
    return exact, norm


def _make_matcher(source: str):
    exact, norm = _load_allow()

    def match(rec: dict) -> dict | None:
        url = rec.get("url") or ""
        dev = exact.get(url) or norm.get(_nrm(url))
        if not dev:
            return None
        html = rec.get("html") or ""
        if not html:
            return None
        return {
            "dev_url": dev,
            "src_url": url,
            "snapshot": rec.get("snapshot", ""),
            "source": source,
            "html": html,
        }

    return match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["decoded", "timespan"], required=True)
    ap.add_argument("--files", default="*")
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()

    root = DECODED if args.source == "decoded" else TIMESPAN
    src = f"{root}/data-{args.files}.parquet"
    out = f"{OUT}/{args.source}-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(src)
        .load_parquet()
        .map(_make_matcher(args.source))
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(out, skip_existing=True)
    )
    ctx = ZephyrContext(
        name=f"collect-devset-html-{args.source}",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.execute(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
