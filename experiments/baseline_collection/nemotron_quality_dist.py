# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tally the Nemotron-CC quality-bucket distribution across filtered samples.

Each ``filter_nemotron_full`` output record carries ``nemotron_quality`` (one of
high / medium-high / medium / medium-low / low) and ``nemotron_kind`` (actual vs
synthetic). This counts records and text chars per quality bucket (and per
quality x kind) for one or more samples, writing a single combined summary JSON.

Purpose: compare the quality distribution of the head-biased 3000-WARC sample,
the uniform-random 3000-WARC sample, and the 10k-WARC sample against each other
(and, offline, against the Nemotron-CC paper's reported distribution). The
head-biased sample concentrates in early crawls, so its mix is expected to skew.

Runs as a single in-region job using a thread pool over the flat
``CC-MAIN-*.jsonl.gz`` shards (mirrors analyze_nemotron_dupes.py) — no
cross-region reads, no large parallel worker pool.

Usage (Iris CPU job, us-central2):
    python experiments/baseline_collection/nemotron_quality_dist.py \\
        --output gs://marin-us-central2/metadata/nemotron_quality_dist/summary.json \\
        --source biased=gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe \\
        --source random=gs://marin-us-central2/filtered/baseline_nemotron_full-2775c6 \\
        --source tenk=gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUALITY_ORDER = ["high", "medium-high", "medium", "medium-low", "low"]
ALLOWED_BUCKET = "marin-us-central2"


def _assert_in_region(root: str) -> None:
    if ALLOWED_BUCKET not in root:
        raise ValueError(f"cross-region read blocked (expected {ALLOWED_BUCKET}): {root}")


def _scan_shard(path: str) -> dict:
    # Track count AND chars per (quality, kind) so we can get organic-only,
    # char-weighted (~token-weighted) distributions comparable to the paper.
    qk_count: dict[str, int] = defaultdict(int)
    qk_chars: dict[str, int] = defaultdict(int)
    try:
        with fsspec.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = rec.get("nemotron_quality") or "unknown"
                k = rec.get("nemotron_kind") or "unknown"
                key = f"{q}|{k}"
                qk_count[key] += 1
                qk_chars[key] += len(rec.get("text") or "")
    except Exception as e:
        return {"__err__": str(e)}
    return {"qk_count": dict(qk_count), "qk_chars": dict(qk_chars)}


def _tally(label: str, root: str, max_workers: int) -> dict:
    _assert_in_region(root)
    fs = fsspec.filesystem("gcs")
    shards = [p if p.startswith("gs://") else f"gs://{p}" for p in fs.glob(f"{root}/*.jsonl.gz")]
    logger.info("[%s] scanning %d shards under %s", label, len(shards), root)

    qk_count: dict[str, int] = defaultdict(int)
    qk_chars: dict[str, int] = defaultdict(int)
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_shard, s) for s in shards]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            if "__err__" in r:
                errors += 1
                continue
            for key, c in r["qk_count"].items():
                qk_count[key] += c
            for key, c in r["qk_chars"].items():
                qk_chars[key] += c
            done += 1
            if done % 2000 == 0:
                logger.info("  [%s] %d/%d shards", label, done, len(shards))

    total = sum(qk_count.values())
    # Organic (actual) char-weighted distribution — the apples-to-apples match
    # for the paper's token-weighted Table 2.
    org_chars = {q: qk_chars.get(f"{q}|actual", 0) for q in QUALITY_ORDER}
    org_total = sum(org_chars.values())
    logger.info("[%s] total records=%d errors=%d organic_chars=%d", label, total, errors, org_total)
    for q in QUALITY_ORDER:
        pct = 100 * org_chars[q] / org_total if org_total else 0.0
        logger.info("  [%s] %-12s organic_chars=%-14d (%5.1f%%)", label, q, org_chars[q], pct)

    return {
        "label": label,
        "root": root,
        "n_shards": len(shards),
        "errors": errors,
        "total_records": total,
        "by_quality_kind_count": dict(qk_count),
        "by_quality_kind_chars": dict(qk_chars),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True, help="label=gs://...root (repeatable)")
    p.add_argument("--output", required=True, help="gs:// path for combined summary JSON")
    p.add_argument("--max-workers", type=int, default=128)
    args = p.parse_args()

    results = {}
    for spec in args.source:
        label, root = spec.split("=", 1)
        results[label] = _tally(label, root.rstrip("/"), args.max_workers)

    with fsspec.open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote combined summary -> %s", args.output)


if __name__ == "__main__":
    main()
