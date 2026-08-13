# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose WHY resiliparse drops pages the LLM extractors keep (Zephyr, on Iris).

Hypothesis A ("JS-rendered"): the article body is injected by JavaScript, so it is absent from the
static HTML — resiliparse (no JS) AND any HTML-reading extractor should fail.
Hypothesis B ("main-content heuristic too aggressive"): the body IS in the static HTML, but
resiliparse `main_content=True` discards it as boilerplate — so `main_content=False` (take-everything)
still recovers it, and LLM extractors that read the raw HTML succeed.

For each decoded doc we compute resiliparse_main len, resiliparse_full len, and the static
text-body len. If `main==0` docs mostly have `full>0` / `body>0`, the content is present in static
HTML → Hypothesis B (heuristic), not JS. Emits per-doc lens to aggregate downstream.
"""

from __future__ import annotations

import argparse
import sys

from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.extractors import resiliparse_full, resiliparse_main

DECODED = "gs://marin-us-east5/documents/bert_pipeline/decoded_10k"
OUT = "gs://marin-us-east5/scratch/provenance_10k_devset/resiliparse_diag"


def diag_record(rec: dict) -> dict | None:
    html = rec.get("html") or ""
    if not html:
        return None
    rm = resiliparse_main(html)
    rf = resiliparse_full(html)
    tb = rec.get("text_body") or ""
    return {
        "url": rec.get("url", ""),
        "domain": (rec.get("url", "").split("/")[2] if "//" in rec.get("url", "") else "")[:60],
        "main_len": len(rm.strip()),
        "full_len": len(rf.strip()),
        "body_len": len(tb.strip()),
        "html_len": len(html),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", default="0[0-1]*", help="filename-hash glob suffix (small subset for speed)")
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()

    src = f"{DECODED}/data-{args.files}.parquet"
    out = f"{OUT}/d-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(src)
        .load_parquet()
        .map(diag_record)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(out, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="resiliparse-diag",
        max_workers=args.max_workers,
        resources=ResourceConfig(cpu=1, ram="8g", regions=["us-east5"], preemptible=True),
    )
    ctx.execute(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
