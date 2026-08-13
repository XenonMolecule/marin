# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan the decoded 10k-pool HTML for SUBSTANTIAL general-forum threads (Zephyr, on Iris).

The dev set's qa_forum register is well-sized (~112) but skewed to technical forums (physics/math
stackexchange, physicsforums) from keyword-matching. This mines DIVERSE general/discussion forums —
reddit + lifestyle/support/hobby boards — with real thread content (not the thin nav/login stubs the
junk scan already covers). Emits candidates with text; sampled + LLM-verified downstream.
"""

from __future__ import annotations

import argparse
import sys

from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.extractors import resiliparse_main

DECODED = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k"
OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/warc_forum"

_FORUM = (
    "reddit.com",
    "quora.com",
    "ubuntuforums",
    "styleforum",
    "proboards",
    "phpbb",
    "vbulletin",
    "disqus",
    "discourse",
    "serverfault",
    "superuser",
    "askubuntu",
    "healthboards",
    "avforums",
    "forums.",
    "boards.",
    "community.",
    "bogleheads",
    "somethingawful",
    "gaiaonline",
    "neogaf",
)


def _is_forum(dom: str) -> bool:
    return "forum" in dom or any(f in dom for f in _FORUM)


def forum_record(rec: dict) -> dict | None:
    url = rec.get("url") or ""
    dom = (url.split("/")[2] if "//" in url else "").lower()
    if not url or not _is_forum(dom):
        return None
    # The decoded text_body is unreliable (often raw HTML/nav). Run resiliparse main-content on the
    # raw html to strip nav/boilerplate and keep the actual thread posts.
    text = resiliparse_main(rec.get("html") or "").strip()
    if not (1500 <= len(text) <= 40000):
        return None
    return {"url": url, "domain": dom[:60], "text_len": len(text), "snippet": text[:3000]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="[0-5]*")
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()

    src = f"{DECODED}/data-{args.files}.parquet"
    out = f"{OUT}/f-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(src)
        .load_parquet()
        .map(forum_record)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(out, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="warc-forum-scan",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.execute(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
