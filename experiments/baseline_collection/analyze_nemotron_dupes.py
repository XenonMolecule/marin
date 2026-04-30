# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantify within-source near-duplication — mainly for nemotron_full.

nemotron_full = organic (kind=actual) + 5 rephraser-synthetic variants per
URL. If the synthetic variants share a lot of text with the originals, the
effective unique-content count is far less than the 5M record count suggests.

This job:
  1. Scans all nemotron_full shards (us-central2, small).
  2. Groups records by URL -> list of texts.
  3. For URLs with >=2 variants, computes pairwise 5-shingle Jaccard.
  4. Reports:
     - variant-count histogram
     - pairwise-Jaccard histogram (per URL: mean of pairwise values)
     - fraction of URL groups where any two variants share Jaccard > 0.5

Also runs the same analysis for nemotron_org (should be ~all 1 variant),
dclm, fineweb_edu as control baselines.

Writes ``duplication_stats.json`` to ``gs://marin-us-central2/scratch/baseline_compare/``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

from experiments.baseline_collection.matched_viewer import _assert_no_cross_region

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("dupes")

SOURCES = {
    "nemotron_full": "gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe",
    "nemotron": "gs://marin-us-central2/filtered/baseline_nemotron-037958",
    "dclm": "gs://marin-us-central2/filtered/baseline_dclm_resharded-1ac313",
    "fineweb_edu": "gs://marin-us-central2/filtered/baseline_fineweb_edu-72c2c7",
}
OUTPUT = "gs://marin-us-central2/scratch/baseline_compare/duplication_stats.json"


def _shingles(text: str, k: int = 5) -> frozenset:
    """K-character shingles (collapses whitespace first).

    5-char shingles strike a balance: small enough for aggressive near-dup
    detection, large enough that trivial formatting differences matter.
    """
    # Lowercase + collapse whitespace for robustness
    import re

    s = re.sub(r"\s+", " ", text.lower())
    if len(s) < k:
        return frozenset([s])
    return frozenset(s[i : i + k] for i in range(len(s) - k + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _scan_source_urls(source: str, root: str, sample_urls: int, seed: int) -> dict:
    """Scan a source's shards, group by URL, compute dup stats.

    Memory: bounded to ``sample_urls * avg_variants * shingle-set-size``.
    We do a FIRST PASS that collects only URL->variant-count, then a SECOND
    PASS that only keeps texts for URLs chosen for the pairwise-Jaccard sample.
    """
    _assert_no_cross_region(root)
    fs = fsspec.filesystem("gcs")
    shards = [f"gs://{p}" if not p.startswith("gs://") else p for p in fs.glob(f"{root}/*.jsonl.gz")]
    logger.info("[%s] %d shards", source, len(shards))

    def scan_shard_counts(path):
        counts = defaultdict(int)
        try:
            with fsspec.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    url = rec.get("url")
                    if url and (rec.get("text") or ""):
                        counts[url] += 1
        except Exception as e:
            return {"__err__": str(e)}
        return dict(counts)

    logger.info("[%s] pass 1: counting variants per URL", source)
    url_variant_count: dict[str, int] = defaultdict(int)
    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = [ex.submit(scan_shard_counts, s) for s in shards]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if "__err__" in r:
                continue
            for u, c in r.items():
                url_variant_count[u] += c
            done += 1
            if done % 500 == 0:
                logger.info("  [%s] scanned %d / %d", source, done, len(shards))

    # Variant-count histogram
    variant_hist: dict[int, int] = defaultdict(int)
    for c in url_variant_count.values():
        variant_hist[c] += 1
    logger.info("[%s] total URLs: %d", source, len(url_variant_count))
    logger.info(
        "[%s] variant-count hist top: %s",
        source,
        dict(sorted(variant_hist.items(), key=lambda kv: -kv[1])[:10]),
    )

    # Pick target URL sample for pairwise-Jaccard: URLs with >=2 variants.
    rng = random.Random(seed + hash(source) % 10**6)
    multi_variant_urls = [u for u, c in url_variant_count.items() if c >= 2]
    rng.shuffle(multi_variant_urls)
    target_urls = set(multi_variant_urls[:sample_urls])
    logger.info("[%s] sampling %d multi-variant URLs for pairwise Jaccard", source, len(target_urls))

    def scan_shard_texts(path):
        buckets = defaultdict(list)
        try:
            with fsspec.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    url = rec.get("url")
                    text = rec.get("text") or ""
                    if url in target_urls and text:
                        # Truncate to keep shingle memory bounded
                        buckets[url].append(text[:8000])
        except Exception as e:
            return {"__err__": str(e)}
        return dict(buckets)

    logger.info("[%s] pass 2: collecting texts for %d sample URLs", source, len(target_urls))
    url_to_texts: dict[str, list[str]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=64) as ex:
        futs = [ex.submit(scan_shard_texts, s) for s in shards]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if "__err__" in r:
                continue
            for u, texts in r.items():
                url_to_texts[u].extend(texts)
            done += 1
            if done % 500 == 0:
                logger.info("  [%s] texts scanned %d / %d", source, done, len(shards))

    # Compute pairwise Jaccard per URL, then aggregate
    jaccard_sum = 0.0
    jaccard_count = 0
    jaccard_hist_edges = [0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.001]
    jaccard_hist = [0] * (len(jaccard_hist_edges) - 1)
    urls_with_any_high_jaccard = 0
    mean_jaccard_per_url = []
    for texts in url_to_texts.values():
        if len(texts) < 2:
            continue
        shingles = [_shingles(t) for t in texts[:6]]  # cap at 6 variants (organic + 5)
        pair_js = []
        for i in range(len(shingles)):
            for j in range(i + 1, len(shingles)):
                val = _jaccard(shingles[i], shingles[j])
                pair_js.append(val)
                jaccard_sum += val
                jaccard_count += 1
                for bi, be in enumerate(jaccard_hist_edges[:-1]):
                    if be <= val < jaccard_hist_edges[bi + 1]:
                        jaccard_hist[bi] += 1
                        break
        if pair_js:
            mean_jaccard_per_url.append(sum(pair_js) / len(pair_js))
            if max(pair_js) > 0.5:
                urls_with_any_high_jaccard += 1

    mean_jaccard = jaccard_sum / jaccard_count if jaccard_count else 0.0
    mean_per_url_jaccard = sum(mean_jaccard_per_url) / len(mean_jaccard_per_url) if mean_jaccard_per_url else 0.0

    return {
        "total_unique_urls": len(url_variant_count),
        "total_records_from_counts": sum(url_variant_count.values()),
        "variant_count_hist": dict(sorted(variant_hist.items())),
        "sample_multi_variant_urls_evaluated": len(mean_jaccard_per_url),
        "mean_pairwise_jaccard": mean_jaccard,
        "mean_per_url_jaccard": mean_per_url_jaccard,
        "jaccard_hist_edges": jaccard_hist_edges,
        "jaccard_hist": jaccard_hist,
        "fraction_urls_with_high_jaccard": (
            urls_with_any_high_jaccard / len(mean_jaccard_per_url) if mean_jaccard_per_url else 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-urls", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    results = {}
    for src, root in SOURCES.items():
        logger.info("=== %s ===", src)
        try:
            results[src] = _scan_source_urls(src, root, args.sample_urls, args.seed)
        except Exception as e:
            logger.exception("%s failed: %s", src, e)
            results[src] = {"error": str(e)}

    _assert_no_cross_region(OUTPUT)
    with fsspec.open(OUTPUT, "w") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
