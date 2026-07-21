# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan the decoded 10k-pool HTML for LOW-QUALITY pages — a source of per-register hard NEGATIVES.

The dev set's matched pool is keyword+verifier selected, so it's almost all keep-worthy: several
registers end up 100% keep, which makes the spec un-tunable ("keep all code/educational/…"). To get
genuine drop examples we mine the raw web (decoded_10k) for low-quality pages — thin content, link
farms, error/parked pages, spam — that still carry enough topic to be assigned to a content register
downstream (an essay-mill stub -> educational-drop, an SEO listicle -> history-drop). Emits candidates
with text; an LLM classifies register + confirms drop-worthiness downstream.
"""

from __future__ import annotations

import argparse
import sys

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

DECODED = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k"
OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/warc_junk"

_ERR = (
    "page not found",
    "404 error",
    "error 404",
    "under construction",
    "domain is for sale",
    "this domain is",
    "parked domain",
    "buy this domain",
    "account suspended",
    "access denied",
)
_SPAM = (
    "viagra",
    "cialis",
    "casino",
    "payday loan",
    "replica watch",
    "escort",
    "porn",
    "cheap flights",
    "make money fast",
    "weight loss pill",
)


def junk_record(rec: dict) -> dict | None:
    html = rec.get("html") or ""
    url = rec.get("url") or ""
    tb = (rec.get("text_body") or "").strip()
    tl = len(tb)
    if tl < 30 or not url:
        return None
    hl = html.lower()
    low = tb.lower()
    n_links = hl.count("<a ")
    reasons = []
    if 40 <= tl <= 900:
        reasons.append("thin")
    if n_links >= 60 and tl < 4000:
        reasons.append("linkfarm")
    if any(s in low for s in _ERR):
        reasons.append("error/parked")
    if any(s in low for s in _SPAM):
        reasons.append("spam")
    if not reasons:
        return None
    return {
        "url": url,
        "domain": (url.split("/")[2] if "//" in url else "")[:60],
        "text_len": tl,
        "n_links": n_links,
        "reason": ",".join(reasons),
        "snippet": tb[:2000],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="[0-4]*", help="filename-hash glob suffix")
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()

    src = f"{DECODED}/data-{args.files}.parquet"
    out = f"{OUT}/j-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(src)
        .load_parquet()
        .map(junk_record)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(out, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="warc-junk-scan",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.execute(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
