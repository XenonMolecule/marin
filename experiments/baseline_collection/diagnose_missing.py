# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diagnose the 62 URLs missing from resiliparse's output.

Fans out over the cluster with ThreadPoolExecutor (head-node I/O) and checks
each missing URL against:

1. ``metadata/baseline_warc_metadata-d671da`` — the canonical URL universe from
   our 3,000 WARCs. If a missing URL is here, the filter is retaining a URL
   that IS in our WARC set.
2. ``raw/commoncrawl/baseline_3000-265ff5`` — the raw download_warcs output,
   which is resiliparse's direct input. If a missing URL is here, then
   resiliparse received the HTML and dropped it (almost certainly via
   ``_is_non_empty`` filtering out an empty main-content extraction).

Reports breakdown and HTML-length distribution for the dropped cases.

Run via::

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -- python experiments/baseline_collection/diagnose_missing.py
"""

from __future__ import annotations

import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("diagnose")

MISSING_URLS_PATH = "gs://marin-us-central2/scratch/baseline_compare/missing_urls.json"
META_ROOT = "gs://marin-us-central2/metadata/baseline_warc_metadata-d671da"
DOWNLOAD_ROOT = "gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5"
REPORT_PATH = "gs://marin-us-central2/scratch/baseline_compare/missing_diagnosis.json"


def scan_shard(path, targets, html_field=False):
    hits: dict[str, dict] = {}
    try:
        with fsspec.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                url = rec.get("url")
                if url and url in targets and url not in hits:
                    hits[url] = {
                        "shard": path.rsplit("/", 1)[-1],
                        "html_len": len(rec.get("html") or "") if html_field else None,
                    }
    except Exception as exc:
        return {"__err__": str(exc)}
    return hits


def parallel_scan(root, targets, html_field=False, max_workers=64):
    fs = fsspec.filesystem("gcs")
    shards = [f"gs://{p}" if not p.startswith("gs://") else p for p in fs.glob(f"{root}/*.jsonl.gz")]
    logger.info("scanning %d shards at %s", len(shards), root)
    all_hits: dict[str, dict] = {}
    scanned = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(scan_shard, p, targets, html_field) for p in shards]
        for fut in as_completed(futs):
            r = fut.result()
            if "__err__" in r:
                continue
            for url, info in r.items():
                all_hits.setdefault(url, info)
            scanned += 1
            if scanned % 500 == 0 or scanned == len(shards):
                logger.info("  scanned %d / %d · unique hits: %d", scanned, len(shards), len(all_hits))
    return all_hits


def main() -> int:
    with fsspec.open(MISSING_URLS_PATH, "r") as f:
        targets_list = json.load(f)
    targets = frozenset(targets_list)
    logger.info("Checking %d missing URLs", len(targets))

    logger.info("=== pass 1: metadata scan ===")
    meta_hits = parallel_scan(META_ROOT, targets, html_field=False)

    logger.info("=== pass 2: download_warcs scan ===")
    dl_hits = parallel_scan(DOWNLOAD_ROOT, targets, html_field=True)

    not_in_meta = []
    in_meta_not_dl = []
    in_meta_in_dl = []
    for u in targets_list:
        if u in meta_hits:
            if u in dl_hits:
                in_meta_in_dl.append({"url": u, "meta": meta_hits[u], "dl": dl_hits[u]})
            else:
                in_meta_not_dl.append({"url": u, "meta": meta_hits[u]})
        else:
            not_in_meta.append(u)

    report = {
        "total_missing": len(targets_list),
        "in_metadata_and_downloaded": len(in_meta_in_dl),
        "in_metadata_but_not_downloaded": len(in_meta_not_dl),
        "not_in_metadata": len(not_in_meta),
        "not_in_metadata_urls": not_in_meta,
        "in_metadata_but_not_downloaded_urls": [x["url"] for x in in_meta_not_dl],
        "in_metadata_and_downloaded_examples": in_meta_in_dl[:20],
        "html_len_stats": {},
    }
    html_lens = sorted(x["dl"]["html_len"] for x in in_meta_in_dl if x["dl"].get("html_len") is not None)
    if html_lens:
        n = len(html_lens)
        report["html_len_stats"] = {
            "n": n,
            "min": html_lens[0],
            "p25": html_lens[n // 4],
            "median": html_lens[n // 2],
            "p75": html_lens[3 * n // 4],
            "max": html_lens[-1],
        }

    logger.info(
        "SUMMARY: %s", {k: v for k, v in report.items() if not k.endswith("_urls") and not k.endswith("examples")}
    )
    with fsspec.open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
