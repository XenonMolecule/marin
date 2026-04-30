# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""URL-matched baseline source viewer, designed to run in us-central2.

Why this tool exists
--------------------
Resiliparse is the unfiltered "everything" baseline — every URL that survives
nemotron / nemotron_full / dclm / fineweb_edu filtering *must* also appear in
resiliparse (the filters work on a subset of the same 3000 WARCs that
resiliparse extracts from). If any heavily-filtered URL is missing from the
resiliparse output, we have a bug: either resiliparse dropped a doc via
``_is_non_empty``, or a filter pulled in a URL from outside our WARC universe.

Small local samples cannot verify this invariant — resiliparse is ~150 GB.

Design
------
Everything that touches the 150 GB resiliparse corpus runs on the
``us-central2`` Ray cluster to avoid cross-region egress. Local work is limited
to a ~10-40 MB final payload.

Two entrypoints:

``cluster-job`` (launch via ray_run.py on us-central2)
    1. Fan out Ray tasks over every shard of nemotron / nemotron_full / dclm /
       fineweb_edu, extracting URL → record maps per shard.
    2. Reducer intersects URL sets to find URLs retained by ≥ ``--min-filters``
       heavy filters. Downsamples to ``--target-urls``.
    3. Fan out Ray tasks over every resiliparse shard, keeping only records
       whose URL is in the target set.
    4. Verifies superset invariant: every target URL must be in resiliparse.
    5. Writes three outputs to GCS:
         - ``validation.json`` — superset invariant report (tiny).
         - ``matched_full.jsonl.gz`` — every matching record from every source
           (for later analysis).
         - ``matched_viewer.jsonl.gz`` — top-N URLs with text truncated for the
           HTML viewer (small enough to download locally).

``render`` (local, after cluster job)
    Downloads ``validation.json`` + ``matched_viewer.jsonl.gz`` (~10-40 MB),
    emits a self-contained HTML with click-through side-by-side comparison.

Usage
-----
    # Launch cluster job (us-central2 is where the data lives)
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -- python experiments/baseline_collection/matched_viewer.py cluster-job \\
               --target-urls 10000 --min-filters 3 --viewer-urls 500

    # After cluster job finishes, render locally:
    uv run python experiments/baseline_collection/matched_viewer.py render \\
        --output scratch/matched_viewer.html
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("matched_viewer")

SOURCES: dict[str, dict] = {
    "resiliparse": {
        "root": "gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa",
        "color": "#7c3aed",
        "description": "Raw resiliparse main-content extraction (no filtering).",
    },
    "nemotron": {
        "root": "gs://marin-us-central2/filtered/baseline_nemotron-037958",
        "color": "#16a34a",
        "description": "Nemotron-CC organic (kind=actual).",
    },
    "nemotron_full": {
        "root": "gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe",
        "color": "#0d9488",
        "description": "Nemotron-CC organic + 5 rephraser-synthetic variants.",
    },
    "dclm": {
        "root": "gs://marin-us-central2/filtered/baseline_dclm_resharded-1ac313",
        "color": "#2563eb",
        "description": "DCLM-baseline, joined on WARC-Record-ID.",
    },
    "fineweb_edu": {
        "root": "gs://marin-us-central2/filtered/baseline_fineweb_edu-72c2c7",
        "color": "#dc2626",
        "description": "FineWeb-Edu, joined on file_path per snapshot.",
    },
    "llm_curated": {
        # Lives in us-central1, so the llm-curated-job subcommand must run on that cluster.
        "root": "gs://marin-us-central1/documents/baseline_llm_curated-050243",
        "color": "#d97706",
        "description": "LLM-curated extraction (baseline_llm_curated-050243).",
    },
}

HEAVY_FILTERS = ["nemotron", "nemotron_full", "dclm", "fineweb_edu"]

META_FIELDS: dict[str, list[str]] = {
    "nemotron": ["nemotron_quality", "nemotron_kind", "nemotron_id"],
    "nemotron_full": ["nemotron_quality", "nemotron_kind", "nemotron_kind2", "nemotron_id"],
    "dclm": ["warc_record_id", "dclm_fasttext_score", "dclm_language_score"],
    "fineweb_edu": ["fineweb_score", "fineweb_int_score", "dump", "file_path"],
    "resiliparse": [],
    "llm_curated": ["warc_record_id", "warc_file", "snapshot"],
}

WORKSPACE = "gs://marin-us-central2/scratch/baseline_compare"
VALIDATION_PATH = f"{WORKSPACE}/validation.json"
MATCHED_FULL_PATH = f"{WORKSPACE}/matched_full.jsonl.gz"
MATCHED_VIEWER_PATH = f"{WORKSPACE}/matched_viewer.jsonl.gz"
TARGET_URLS_PATH = f"{WORKSPACE}/target_urls.json"

# LLM-curated matches output — small enough (~5 MB for 10K records) that we keep
# it alongside the other artifacts in us-central2 for simple download.
LLM_CURATED_MATCHES_PATH = f"{WORKSPACE}/llm_curated_matches.jsonl.gz"


# ---------------------------------------------------------------------------
# Shared helpers — with cross-region safeguards
# ---------------------------------------------------------------------------


def _detect_gcp_zone() -> str | None:
    """Return the current GCP zone (e.g. ``us-central1-a``) or ``None`` if not on GCP."""
    import os
    import urllib.request

    # Cache negative result so we don't hammer the metadata server.
    cached = getattr(_detect_gcp_zone, "_cache", "unset")
    if cached != "unset":
        return cached
    if os.environ.get("MARIN_DISABLE_ZONE_DETECTION") == "1":
        _detect_gcp_zone._cache = None  # type: ignore[attr-defined]
        return None
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as r:
            zone = r.read().decode().rsplit("/", 1)[-1]
    except Exception:
        zone = None
    _detect_gcp_zone._cache = zone  # type: ignore[attr-defined]
    return zone


def _gcs_path_region(path: str) -> str | None:
    """Infer region from a ``gs://marin-<region>[-<suffix>]/...`` path.

    Returns strings like ``us-central1`` or ``eu-west4``. Returns None for
    non-GCS paths or non-marin buckets.
    """
    if not path.startswith("gs://"):
        return None
    bucket = path[5:].split("/", 1)[0]
    if not bucket.startswith("marin-"):
        return None
    parts = bucket.split("-")
    if len(parts) < 3:
        return None
    return f"{parts[1]}-{parts[2]}"


def _assert_no_cross_region(path: str) -> None:
    """Guard: refuse cross-region GCS I/O when running inside GCP.

    Applies only to jobs running on GCP (has metadata server). On a laptop
    this is a no-op so ``gcloud storage cp`` from a user shell still works.
    Override with ``MARIN_ALLOW_CROSS_REGION=1`` for one-off use.
    """
    import os

    if os.environ.get("MARIN_ALLOW_CROSS_REGION") == "1":
        return
    zone = _detect_gcp_zone()
    if zone is None:
        return
    path_region = _gcs_path_region(path)
    if path_region is None:
        return
    if not zone.startswith(path_region):
        raise RuntimeError(
            f"CROSS-REGION I/O REFUSED: process is in zone {zone!r} "
            f"but path is in region {path_region!r} ({path!r}). "
            f"Set MARIN_ALLOW_CROSS_REGION=1 to override."
        )


def _gcs_ls_glob(glob: str) -> list[str]:
    """List GCS objects matching a glob using fsspec (works on cluster + locally)."""
    _assert_no_cross_region(glob)
    fs = fsspec.filesystem("gcs")
    return [f"gs://{p}" if not p.startswith("gs://") else p for p in fs.glob(glob)]


def _open_read_gzip(path: str):
    """Open a possibly-gzipped JSONL file for streaming record reads."""
    _assert_no_cross_region(path)
    raw = fsspec.open(path, "rb").open()
    return gzip.GzipFile(fileobj=raw, mode="rb")


def _write_jsonl_gz(path: str, records) -> None:
    _assert_no_cross_region(path)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for rec in records:
            gz.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
            gz.write(b"\n")
    with fsspec.open(path, "wb") as out:
        out.write(buf.getvalue())


def _write_json(path: str, obj) -> None:
    _assert_no_cross_region(path)
    with fsspec.open(path, "w") as out:
        out.write(json.dumps(obj, indent=2, ensure_ascii=False))


def _read_jsonl_gz_local(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Cluster job
# ---------------------------------------------------------------------------


def _extract_shard_urls_only(shard_path: str) -> list[str]:
    """Pass-1 worker: return only the non-empty URLs from a filter shard.

    Keeps peak driver memory bounded: ~80 bytes per URL, so 10M URLs fit in
    ~1 GB regardless of how long the actual text is.
    """
    urls: list[str] = []
    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url")
            if url and (rec.get("text") or ""):
                urls.append(url)
    return urls


# Quality score bucket edges and text length histogram edges used by analyze().
LENGTH_BUCKETS = [0, 100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 10**12]
FINEWEB_SCORE_BUCKETS = [-10, 0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10]
DCLM_FT_BUCKETS = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.01]


def _bucketize(value, edges):
    """Return bucket index for a scalar value (lower bound inclusive, upper exclusive)."""
    import bisect

    idx = bisect.bisect_right(edges, value) - 1
    return max(0, min(len(edges) - 2, idx))


def _extract_shard_urls_with_stats(shard_path: str, source: str) -> dict:
    """Pass-1 with instrumentation: URLs + text-length histogram + domain / snapshot /
    quality-score counters.

    Returns a compact dict that the driver merges (rather than holding a
    per-record list) to keep driver memory low even for giant shards.
    """
    from collections import Counter
    from urllib.parse import urlparse

    urls: list[str] = []
    length_hist = [0] * (len(LENGTH_BUCKETS) - 1)
    total_chars = 0
    n_records = 0
    domain_counter: Counter = Counter()
    snapshot_counter: Counter = Counter()
    nemotron_quality: Counter = Counter()
    fineweb_hist = [0] * (len(FINEWEB_SCORE_BUCKETS) - 1)
    dclm_hist = [0] * (len(DCLM_FT_BUCKETS) - 1)

    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url")
            text = rec.get("text") or ""
            if not url or not text:
                continue
            urls.append(url)
            tl = len(text)
            total_chars += tl
            n_records += 1
            length_hist[_bucketize(tl, LENGTH_BUCKETS)] += 1
            try:
                domain_counter[urlparse(url).netloc] += 1
            except Exception:
                pass
            if source in ("nemotron", "nemotron_full") and rec.get("nemotron_quality"):
                nemotron_quality[rec["nemotron_quality"]] += 1
            if source == "fineweb_edu":
                if rec.get("dump"):
                    snapshot_counter[rec["dump"]] += 1
                if isinstance(rec.get("fineweb_score"), (int, float)):
                    fineweb_hist[_bucketize(rec["fineweb_score"], FINEWEB_SCORE_BUCKETS)] += 1
            if source == "dclm" and isinstance(rec.get("dclm_fasttext_score"), (int, float)):
                dclm_hist[_bucketize(rec["dclm_fasttext_score"], DCLM_FT_BUCKETS)] += 1

    return {
        "urls": urls,
        "length_hist": length_hist,
        "total_chars": total_chars,
        "n_records": n_records,
        "domain_counter": dict(domain_counter.most_common(500)),
        "snapshot_counter": dict(snapshot_counter),
        "nemotron_quality": dict(nemotron_quality),
        "fineweb_hist": fineweb_hist,
        "dclm_hist": dclm_hist,
    }


def _scan_resiliparse_with_stats(shard_path: str, target_urls: frozenset[str], max_chars: int) -> dict:
    """Combined pass: resiliparse target matches + global stats in one shard traversal.

    Since scanning resiliparse is the 2-hour bottleneck, we instrument it to
    collect everything at once.

    We intentionally do NOT return a full URL list — resiliparse has ~180M URLs
    which would OOM the head node. Unique URL count is approximated via the
    per-shard n_records sum (slight over-count if a URL appears in multiple
    shards, which should be rare since the 3000 shards are 1:1 with WARCs).
    """
    from collections import Counter
    from urllib.parse import urlparse

    matches: list[dict] = []
    length_hist = [0] * (len(LENGTH_BUCKETS) - 1)
    total_chars = 0
    n_records = 0
    domain_counter: Counter = Counter()

    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url")
            text = rec.get("text") or ""
            if not url or not text:
                continue
            tl = len(text)
            total_chars += tl
            n_records += 1
            length_hist[_bucketize(tl, LENGTH_BUCKETS)] += 1
            try:
                domain_counter[urlparse(url).netloc] += 1
            except Exception:
                pass
            if url in target_urls:
                matches.append(
                    {
                        "source": "resiliparse",
                        "url": url,
                        "text": text[:max_chars],
                        "full_len": tl,
                        "meta": {"shard": shard_path.rsplit("/", 1)[-1]},
                    }
                )

    return {
        "matches": matches,
        "length_hist": length_hist,
        "total_chars": total_chars,
        "n_records": n_records,
        "domain_counter": dict(domain_counter.most_common(500)),
    }


def _extract_shard_matches(shard_path: str, source: str, target_urls: frozenset[str], max_chars: int) -> list[dict]:
    """Pass-2 worker: return full records from a filter shard whose URL is in ``target_urls``."""
    keep_fields = META_FIELDS.get(source, [])
    out: list[dict] = []
    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url")
            if not url or url not in target_urls:
                continue
            text = rec.get("text") or ""
            if not text:
                continue
            out.append(
                {
                    "source": source,
                    "url": url,
                    "text": text[:max_chars],
                    "full_len": len(text),
                    "meta": {k: rec.get(k) for k in keep_fields if k in rec},
                }
            )
    return out


def _scan_resiliparse_shard(shard_path: str, target_urls: frozenset[str], max_chars: int) -> list[dict]:
    """Return resiliparse records for URLs in ``target_urls``."""
    out: list[dict] = []
    with _open_read_gzip(shard_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("url")
            if not url or url not in target_urls:
                continue
            text = rec.get("text") or ""
            out.append(
                {
                    "source": "resiliparse",
                    "url": url,
                    "text": text[:max_chars],
                    "full_len": len(text),
                    "meta": {"shard": shard_path.rsplit("/", 1)[-1]},
                }
            )
    return out


def _parallel_map(fn, items, label: str, max_workers: int):
    """Thread-pool parallel map with periodic progress logging.

    Used instead of Ray because the us-central2 cluster's head has 0 CPUs and
    worker autoscaling is slow; the driver thread pool reads GCS in-region and
    is I/O-bound, so threads give near-linear speedup without needing workers.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list = []
    done = 0
    total = len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fn, item) for item in items]
        last_log = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done - last_log >= max(1, total // 20) or done == total:
                logger.info("[%s] %d / %d done", label, done, total)
                last_log = done
    return results


def cluster_job(args: argparse.Namespace) -> int:
    # --- Pass 1: collect URL sets per heavy filter (memory-bounded, no text) ---
    logger.info("Listing shards for heavy filters...")
    shards_by_source: dict[str, list[str]] = {}
    for s in HEAVY_FILTERS:
        shards = _gcs_ls_glob(f"{SOURCES[s]['root']}/*.jsonl.gz")
        logger.info("  %s: %d shards", s, len(shards))
        shards_by_source[s] = shards

    url_sets: dict[str, set[str]] = {}
    for s in HEAVY_FILTERS:
        shard_results = _parallel_map(
            _extract_shard_urls_only,
            shards_by_source[s],
            f"pass1 {s}",
            max_workers=args.threads,
        )
        urls: set[str] = set()
        for rs in shard_results:
            urls.update(rs)
        url_sets[s] = urls
        logger.info("[%s] %d unique URLs", s, len(urls))
        del shard_results  # drop per-shard lists immediately

    # --- Step 2: tally, pick target URLs ---
    url_to_sources: dict[str, set[str]] = defaultdict(set)
    for s, urls in url_sets.items():
        for u in urls:
            url_to_sources[u].add(s)

    rng = random.Random(args.seed)
    if args.stratify:
        # Balance target URLs across source_count buckets (1..4) so the viewer
        # can show partially-filtered URLs, not just the all-survivors.
        by_bucket: dict[int, list[str]] = defaultdict(list)
        for u, srcs in url_to_sources.items():
            by_bucket[len(srcs)].append(u)
        buckets = sorted(by_bucket.keys())
        per_bucket = max(1, args.target_urls // len(buckets))
        picked: list[str] = []
        for b in buckets:
            urls = by_bucket[b]
            rng.shuffle(urls)
            picked.extend(urls[:per_bucket])
            logger.info("  bucket source_count=%d: %d URLs eligible, took %d", b, len(urls), min(per_bucket, len(urls)))
        target_urls = sorted(set(picked))
    else:
        eligible = [u for u, srcs in url_to_sources.items() if len(srcs) >= args.min_filters]
        logger.info("%d URLs retained by ≥%d heavy filters", len(eligible), args.min_filters)
        rng.shuffle(eligible)
        target_urls = sorted(eligible[: args.target_urls])
    logger.info("Selected %d target URLs", len(target_urls))
    target_frozen = frozenset(target_urls)

    # Keep per-URL source membership for ones we will actually keep.
    target_url_sources = {u: set(url_to_sources[u]) for u in target_urls}
    del url_sets, url_to_sources, eligible  # free Pass-1 memory

    # --- Pass 2: re-read filter shards, keep only target-URL records ---
    index_by_source: dict[str, dict[str, dict]] = {s: {} for s in HEAVY_FILTERS}
    for s in HEAVY_FILTERS:
        shard_results = _parallel_map(
            lambda p, s=s: _extract_shard_matches(p, s, target_frozen, args.max_chars),
            shards_by_source[s],
            f"pass2 {s}",
            max_workers=args.threads,
        )
        for rs in shard_results:
            for rec in rs:
                index_by_source[s].setdefault(rec["url"], rec)
        logger.info("[%s] retained %d target-matched records", s, len(index_by_source[s]))
        del shard_results

    # --- Step 3: scan all resiliparse shards for the target URLs ---
    resiliparse_shards = _gcs_ls_glob(f"{SOURCES['resiliparse']['root']}/data-*.jsonl.gz")
    logger.info("[resiliparse] scanning %d shards for %d target URLs", len(resiliparse_shards), len(target_urls))
    resi_shard_results = _parallel_map(
        lambda p: _scan_resiliparse_shard(p, target_frozen, args.max_chars),
        resiliparse_shards,
        "scan resiliparse",
        max_workers=args.threads,
    )
    resi_url_to_rec: dict[str, dict] = {}
    for rs in resi_shard_results:
        for r in rs:
            resi_url_to_rec.setdefault(r["url"], r)
    del resi_shard_results

    found_in_resiliparse = set(resi_url_to_rec.keys())
    missing = set(target_urls) - found_in_resiliparse
    logger.info(
        "resiliparse superset check: %d / %d target URLs found (%d missing)",
        len(found_in_resiliparse),
        len(target_urls),
        len(missing),
    )

    # --- Step 4: write outputs ---
    validation = {
        "num_target_urls": len(target_urls),
        "num_found_in_resiliparse": len(found_in_resiliparse),
        "num_missing_in_resiliparse": len(missing),
        "superset_invariant_holds": len(missing) == 0,
        "missing_urls_sample": sorted(missing)[:500],
        "min_filters": args.min_filters,
        "counts_per_filter": {s: len(index_by_source[s]) for s in HEAVY_FILTERS},
        "target_urls_from_heavy_intersection": sum(
            1 for u in target_urls if len(target_url_sources[u]) == len(HEAVY_FILTERS)
        ),
    }
    logger.info("Validation: %s", validation)
    _write_json(VALIDATION_PATH, validation)

    # matched_full: every record from every source for target URLs
    full_records: list[dict] = []
    for u in target_urls:
        for s in HEAVY_FILTERS:
            if u in index_by_source[s]:
                full_records.append(index_by_source[s][u])
        if u in resi_url_to_rec:
            full_records.append(resi_url_to_rec[u])
    _write_jsonl_gz(MATCHED_FULL_PATH, full_records)
    logger.info("Wrote %s (%d records)", MATCHED_FULL_PATH, len(full_records))

    # matched_viewer: a bounded sample for the HTML viewer.
    url_score: dict[str, int] = {u: len(target_url_sources[u]) + (1 if u in resi_url_to_rec else 0) for u in target_urls}
    if args.stratify:
        # Mirror the stratified sampling so every source_count bucket shows up
        # in the HTML, not just the full-coverage URLs.
        v_buckets: dict[int, list[str]] = defaultdict(list)
        for u in target_urls:
            v_buckets[len(target_url_sources[u])].append(u)
        v_keys = sorted(v_buckets.keys())
        per_vb = max(1, args.viewer_urls // len(v_keys))
        v_picked: list[str] = []
        for b in v_keys:
            urls = v_buckets[b][:]
            rng.shuffle(urls)
            v_picked.extend(urls[:per_vb])
        viewer_urls = sorted(v_picked)[: args.viewer_urls]
    else:
        viewer_urls = sorted(target_urls, key=lambda u: (-url_score[u], u))[: args.viewer_urls]
    viewer_records: list[dict] = []
    for u in viewer_urls:
        for s in HEAVY_FILTERS:
            if u in index_by_source[s]:
                rec = dict(index_by_source[s][u])
                rec["text"] = rec["text"][: args.viewer_max_chars]
                viewer_records.append(rec)
        if u in resi_url_to_rec:
            rec = dict(resi_url_to_rec[u])
            rec["text"] = rec["text"][: args.viewer_max_chars]
            viewer_records.append(rec)
    _write_jsonl_gz(MATCHED_VIEWER_PATH, viewer_records)
    logger.info("Wrote %s (%d records, %d URLs)", MATCHED_VIEWER_PATH, len(viewer_records), len(viewer_urls))

    # Persist the target URL list as a standalone file so secondary jobs (e.g. the
    # llm-curated scan on us-central1) can pick it up without re-deriving from
    # matched_full.jsonl.gz.
    with fsspec.open(TARGET_URLS_PATH, "w") as out:
        out.write(json.dumps(target_urls))
    logger.info("Wrote %s (%d URLs)", TARGET_URLS_PATH, len(target_urls))

    return 0


# ---------------------------------------------------------------------------
# Comprehensive analysis job
# ---------------------------------------------------------------------------


PER_SOURCE_STATS_PATH = f"{WORKSPACE}/per_source_stats.json"
OVERLAP_MATRIX_PATH = f"{WORKSPACE}/overlap_matrix.json"


def _merge_counters(dicts, top_n=500):
    """Merge a list of Counter-like dicts and keep the top_n entries."""
    from collections import Counter

    c: Counter = Counter()
    for d in dicts:
        c.update(d)
    return dict(c.most_common(top_n))


def _sum_histograms(hists):
    if not hists:
        return []
    n = len(hists[0])
    out = [0] * n
    for h in hists:
        for i, v in enumerate(h):
            out[i] += v
    return out


def analyze(args: argparse.Namespace) -> int:
    """Comprehensive in-region analysis — single pass per source, stratified viewer."""

    # Pass 1 (heavy filters, instrumented) ---------------------------------
    logger.info("Listing shards for heavy filters...")
    shards_by_source: dict[str, list[str]] = {}
    for s in HEAVY_FILTERS:
        shards = _gcs_ls_glob(f"{SOURCES[s]['root']}/*.jsonl.gz")
        logger.info("  %s: %d shards", s, len(shards))
        shards_by_source[s] = shards

    url_sets: dict[str, set[str]] = {}
    per_source_stats: dict[str, dict] = {}
    for s in HEAVY_FILTERS:
        shard_results = _parallel_map(
            lambda p, s=s: _extract_shard_urls_with_stats(p, s),
            shards_by_source[s],
            f"pass1 {s}",
            max_workers=args.threads,
        )
        urls: set[str] = set()
        for rs in shard_results:
            urls.update(rs["urls"])
        url_sets[s] = urls
        per_source_stats[s] = {
            "n_unique_urls": len(urls),
            "n_records": sum(rs["n_records"] for rs in shard_results),
            "total_chars": sum(rs["total_chars"] for rs in shard_results),
            "length_hist_edges": LENGTH_BUCKETS,
            "length_hist": _sum_histograms([rs["length_hist"] for rs in shard_results]),
            "top_domains": _merge_counters([rs["domain_counter"] for rs in shard_results]),
            "snapshot_counts": _merge_counters([rs["snapshot_counter"] for rs in shard_results], top_n=200),
            "nemotron_quality": _merge_counters([rs["nemotron_quality"] for rs in shard_results], top_n=20),
            "fineweb_score_hist_edges": FINEWEB_SCORE_BUCKETS,
            "fineweb_score_hist": _sum_histograms([rs["fineweb_hist"] for rs in shard_results]),
            "dclm_fasttext_hist_edges": DCLM_FT_BUCKETS,
            "dclm_fasttext_hist": _sum_histograms([rs["dclm_hist"] for rs in shard_results]),
        }
        logger.info("[%s] %d unique URLs, %d records", s, len(urls), per_source_stats[s]["n_records"])
        del shard_results

    # Overlap matrix -------------------------------------------------------
    logger.info("Computing overlap matrix across heavy filters...")
    overlap: dict[str, dict[str, int]] = {}
    for a in HEAVY_FILTERS:
        overlap[a] = {}
        for b in HEAVY_FILTERS:
            overlap[a][b] = len(url_sets[a] & url_sets[b])
    # Source-count bucket distribution
    url_to_sources: dict[str, set[str]] = defaultdict(set)
    for s, us in url_sets.items():
        for u in us:
            url_to_sources[u].add(s)
    bucket_counts: dict[int, int] = defaultdict(int)
    for srcs in url_to_sources.values():
        bucket_counts[len(srcs)] += 1
    logger.info("Source-count buckets (heavy filters only): %s", dict(sorted(bucket_counts.items())))

    # Stratified target selection -----------------------------------------
    rng = random.Random(args.seed)
    per_bucket = max(1, args.target_urls // max(1, len(bucket_counts)))
    by_bucket: dict[int, list[str]] = defaultdict(list)
    for u, srcs in url_to_sources.items():
        by_bucket[len(srcs)].append(u)
    picked: list[str] = []
    for b in sorted(by_bucket.keys()):
        urls = by_bucket[b]
        rng.shuffle(urls)
        picked.extend(urls[:per_bucket])
        logger.info("  bucket count=%d: %d eligible, took %d", b, len(urls), min(per_bucket, len(urls)))
    target_urls = sorted(set(picked))
    target_url_sources = {u: set(url_to_sources[u]) for u in target_urls}
    target_frozen = frozenset(target_urls)
    logger.info("Selected %d stratified target URLs", len(target_urls))

    del url_sets, url_to_sources  # free pass-1 memory

    # Pass 2 for heavy filters (target-URL records) -----------------------
    index_by_source: dict[str, dict[str, dict]] = {s: {} for s in HEAVY_FILTERS}
    for s in HEAVY_FILTERS:
        shard_results = _parallel_map(
            lambda p, s=s: _extract_shard_matches(p, s, target_frozen, args.max_chars),
            shards_by_source[s],
            f"pass2 {s}",
            max_workers=args.threads,
        )
        for rs in shard_results:
            for rec in rs:
                index_by_source[s].setdefault(rec["url"], rec)
        logger.info("[%s] retained %d / %d target records", s, len(index_by_source[s]), len(target_urls))
        del shard_results

    # Resiliparse: combined target-match + global stats -------------------
    resiliparse_shards = _gcs_ls_glob(f"{SOURCES['resiliparse']['root']}/data-*.jsonl.gz")
    logger.info("[resiliparse] instrumented scan of %d shards for %d targets", len(resiliparse_shards), len(target_urls))
    resi_results = _parallel_map(
        lambda p: _scan_resiliparse_with_stats(p, target_frozen, args.max_chars),
        resiliparse_shards,
        "scan resiliparse",
        max_workers=args.threads,
    )
    resi_url_to_rec: dict[str, dict] = {}
    resi_length_hist = [0] * (len(LENGTH_BUCKETS) - 1)
    resi_total_chars = 0
    resi_n_records = 0
    for rs in resi_results:
        for r in rs["matches"]:
            resi_url_to_rec.setdefault(r["url"], r)
        for i, v in enumerate(rs["length_hist"]):
            resi_length_hist[i] += v
        resi_total_chars += rs["total_chars"]
        resi_n_records += rs["n_records"]
    resi_domains = _merge_counters([rs["domain_counter"] for rs in resi_results])
    per_source_stats["resiliparse"] = {
        # We do NOT accumulate the full URL set (~180M unique would OOM the head).
        # Since the 3000 resiliparse shards correspond 1:1 to WARC files, a URL
        # appears in ~1 shard, so n_records ≈ n_unique_urls modulo duplicates.
        "n_records": resi_n_records,
        "total_chars": resi_total_chars,
        "length_hist_edges": LENGTH_BUCKETS,
        "length_hist": resi_length_hist,
        "top_domains": resi_domains,
    }
    logger.info("[resiliparse] %d records (unique URL count not tracked — see note)", resi_n_records)

    # Superset check
    found_in_resi = set(resi_url_to_rec.keys())
    missing = set(target_urls) - found_in_resi
    validation = {
        "num_target_urls": len(target_urls),
        "num_found_in_resiliparse": len(found_in_resi),
        "num_missing_in_resiliparse": len(missing),
        "superset_invariant_holds": len(missing) == 0,
        "missing_urls_sample": sorted(missing)[:500],
        "counts_per_filter": {s: len(index_by_source[s]) for s in HEAVY_FILTERS},
        "bucket_counts_heavy": dict(sorted(bucket_counts.items())),
    }
    _write_json(VALIDATION_PATH, validation)
    logger.info("Validation: %s", validation)

    overlap["__note__"] = (
        "resiliparse pairwise overlap not computed — per-filter URL sets were freed to save memory, "
        "and resi URL set is not materialized to avoid OOM (180M URLs). "
        "Use per_source_stats and bucket_counts_heavy for coverage."
    )
    _write_json(OVERLAP_MATRIX_PATH, {"matrix": overlap, "bucket_counts_heavy": dict(sorted(bucket_counts.items()))})
    _write_json(PER_SOURCE_STATS_PATH, per_source_stats)

    # Write full matched records (for stratified 10K) and viewer (stratified 500)
    full_records: list[dict] = []
    for u in target_urls:
        for s in HEAVY_FILTERS:
            if u in index_by_source[s]:
                full_records.append(index_by_source[s][u])
        if u in resi_url_to_rec:
            full_records.append(resi_url_to_rec[u])
    _write_jsonl_gz(MATCHED_FULL_PATH, full_records)
    logger.info("Wrote %s (%d records)", MATCHED_FULL_PATH, len(full_records))

    # Stratified viewer: pick args.viewer_urls, balanced across bucket count.
    v_buckets: dict[int, list[str]] = defaultdict(list)
    for u in target_urls:
        v_buckets[len(target_url_sources[u])].append(u)
    per_vb = max(1, args.viewer_urls // max(1, len(v_buckets)))
    v_picked: list[str] = []
    for b in sorted(v_buckets.keys()):
        urls = v_buckets[b][:]
        rng.shuffle(urls)
        v_picked.extend(urls[:per_vb])
    viewer_urls = sorted(v_picked)[: args.viewer_urls]

    viewer_records: list[dict] = []
    for u in viewer_urls:
        for s in HEAVY_FILTERS:
            if u in index_by_source[s]:
                rec = dict(index_by_source[s][u])
                rec["text"] = rec["text"][: args.viewer_max_chars]
                viewer_records.append(rec)
        if u in resi_url_to_rec:
            rec = dict(resi_url_to_rec[u])
            rec["text"] = rec["text"][: args.viewer_max_chars]
            viewer_records.append(rec)
    _write_jsonl_gz(MATCHED_VIEWER_PATH, viewer_records)
    logger.info("Wrote %s (%d records, %d URLs)", MATCHED_VIEWER_PATH, len(viewer_records), len(viewer_urls))

    with fsspec.open(TARGET_URLS_PATH, "w") as out:
        out.write(json.dumps(target_urls))
    logger.info("Wrote %s", TARGET_URLS_PATH)

    logger.info("=== DONE ===  %s", {k: v for k, v in validation.items() if not isinstance(v, list)})
    return 0


# ---------------------------------------------------------------------------
# LLM-curated job (runs on us-central1 where the curated data lives)
# ---------------------------------------------------------------------------


def llm_curated_job(args: argparse.Namespace) -> int:
    """Scan ``baseline_llm_curated`` on us-central1 for matches to the target URL set.

    To keep reads in-region when run via Iris on us-central1:
      - Pass ``--target-urls-path /path/to/target_urls.json`` pointing at a local
        file bundled into the Iris workspace (avoids reading the central2 copy).
      - Pass ``--output-path gs://marin-us-central1/...`` to keep the output
        write in-region as well.

    Per-shard checkpointing:
      - Each shard writes its matches to ``<output_dir>/per_shard/<shard_name>.jsonl.gz``.
      - On restart, shards whose per-shard file already exists are skipped.
      - After all shards complete, per-shard files are consolidated into
        ``output_path``.
    """
    target_path = args.target_urls_path or TARGET_URLS_PATH
    output_path = args.output_path or LLM_CURATED_MATCHES_PATH
    if target_path.startswith("gs://"):
        _assert_no_cross_region(target_path)
    _assert_no_cross_region(output_path)
    if target_path.startswith("gs://"):
        opener = fsspec.open(target_path, "r")
    else:
        opener = open(target_path)
    with opener as f:
        target_list = json.load(f)
    targets = frozenset(target_list)
    logger.info("Loaded %d target URLs from %s", len(targets), target_path)

    # Per-shard checkpoint directory: siblings of the final output file.
    checkpoint_dir = output_path.rsplit("/", 1)[0] + "/per_shard"
    _assert_no_cross_region(checkpoint_dir)

    shards = _gcs_ls_glob(f"{SOURCES['llm_curated']['root']}/data-*.jsonl.gz")
    logger.info("Scanning %d llm_curated shards (checkpoint dir: %s)", len(shards), checkpoint_dir)

    fs = fsspec.filesystem("gcs")

    def _scan_with_checkpoint(shard_path: str) -> int:
        """Return the count of matching records written for this shard.

        If a prior checkpoint exists, skip the scan and return its record count.
        Otherwise scan + write the per-shard checkpoint.
        """
        shard_name = shard_path.rsplit("/", 1)[-1].removesuffix(".jsonl.gz")
        ckpt_path = f"{checkpoint_dir}/{shard_name}.jsonl.gz"
        # Strip gs:// for fsspec .exists check
        ckpt_exists_path = ckpt_path[len("gs://") :] if ckpt_path.startswith("gs://") else ckpt_path
        if fs.exists(ckpt_exists_path):
            return -1  # sentinel: already checkpointed
        matches = _extract_shard_matches(shard_path, "llm_curated", targets, args.max_chars)
        _write_jsonl_gz(ckpt_path, matches)
        return len(matches)

    _ = _parallel_map(_scan_with_checkpoint, shards, "scan llm_curated", max_workers=args.threads)

    # Consolidate per-shard files → final output.
    logger.info("Consolidating per-shard checkpoints from %s", checkpoint_dir)
    ckpt_files = _gcs_ls_glob(f"{checkpoint_dir}/*.jsonl.gz")
    logger.info("Found %d per-shard files", len(ckpt_files))
    url_to_rec: dict[str, dict] = {}
    for p in ckpt_files:
        with _open_read_gzip(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url_to_rec.setdefault(r["url"], r)

    logger.info(
        "llm_curated retained %d / %d target-matched URLs (%.2f%%)",
        len(url_to_rec),
        len(targets),
        100 * len(url_to_rec) / len(targets) if targets else 0,
    )
    _write_jsonl_gz(output_path, list(url_to_rec.values()))
    logger.info("Wrote %s (%d records)", output_path, len(url_to_rec))
    return 0


# ---------------------------------------------------------------------------
# Local render
# ---------------------------------------------------------------------------


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Matched-URL baseline source viewer</title>
<style>
  :root { --bg:#f5f5f7; --panel:#fff; --border:#d0d0d8; --muted:#7a7a82; --accent:#2563eb; --warn:#b54708; --ok:#065f46; }
  * { box-sizing: border-box; }
  html,body { margin:0; padding:0; height:100%; background:var(--bg); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif; color:#1a1a1f; }
  #app { display:grid; grid-template-columns:340px 1fr; height:100vh; }
  #sidebar { border-right:1px solid var(--border); background:#fafafb; display:flex; flex-direction:column; min-height:0; }
  #sidebar-header { padding:12px 14px; border-bottom:1px solid var(--border); }
  #sidebar-header h1 { margin:0 0 6px 0; font-size:15px; }
  #sidebar-header .sub { font-size:12px; color:var(--muted); }
  #banner { padding:10px 14px; border-bottom:1px solid var(--border); font-size:12px; }
  #banner.ok { background:#ecfdf5; color:var(--ok); }
  #banner.bad { background:#fef2f2; color:#7f1d1d; }
  #filters { padding:8px 14px; border-bottom:1px solid var(--border); font-size:12px; display:flex; gap:6px; flex-wrap:wrap; }
  #filters input[type="search"] { flex:1; min-width:120px; padding:4px 8px; border:1px solid var(--border); border-radius:6px; font-size:12px; }
  #filters select { padding:4px 6px; border:1px solid var(--border); border-radius:6px; font-size:12px; background:white; }
  #list { flex:1; overflow-y:auto; }
  .list-item { padding:8px 14px; border-bottom:1px solid #eef; cursor:pointer; font-size:12px; }
  .list-item:hover { background:#eef2ff; }
  .list-item.active { background:#dbeafe; border-left:3px solid var(--accent); }
  .list-item .url { color:var(--accent); word-break:break-all; font-size:11px; }
  .list-item .meta { color:var(--muted); font-size:10px; margin-top:3px; display:flex; gap:6px; flex-wrap:wrap; }
  .src-chip { display:inline-block; padding:1px 6px; border-radius:9px; font-size:10px; color:white; }
  #main { overflow:auto; padding:18px; min-height:0; }
  #main-header { display:flex; align-items:baseline; gap:16px; margin-bottom:12px; flex-wrap:wrap; }
  #main-header .url { font-size:13px; color:var(--accent); word-break:break-all; flex:1; }
  #main-header .keys { color:var(--muted); font-size:11px; }
  .columns { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); }
  .column { background:var(--panel); border:1px solid var(--border); border-radius:10px; display:flex; flex-direction:column; max-height:calc(100vh - 140px); }
  .column-head { padding:10px 12px; border-bottom:1px solid var(--border); }
  .column-head .src-name { font-weight:600; font-size:13px; }
  .column-head .src-meta { font-size:11px; color:var(--muted); margin-top:4px; display:flex; gap:10px; flex-wrap:wrap; }
  .column-head .src-meta b { color:#333; }
  .column-body { padding:12px; overflow-y:auto; white-space:pre-wrap; font-family:ui-monospace,"SF Mono",Consolas,monospace; font-size:12px; line-height:1.55; flex:1; }
  .missing { color:var(--muted); font-style:italic; padding:16px; text-align:center; }
  details.missing-list { margin:8px 14px; font-size:11px; color:var(--muted); }
  details.missing-list pre { max-height:160px; overflow:auto; background:#fef2f2; padding:8px; border-radius:6px; }
</style>
</head>
<body>
<div id="app">
  <aside id="sidebar">
    <div id="sidebar-header">
      <h1>URL-matched baseline comparison</h1>
      <div class="sub" id="summary-line"></div>
    </div>
    <div id="banner"></div>
    <div id="filters">
      <input type="search" id="q" placeholder="filter urls…">
      <select id="count-mode">
        <option value="ge" selected>at least</option>
        <option value="eq">exactly</option>
      </select>
      <select id="count-n">
        <option value="1">1</option>
        <option value="2" selected>2</option>
        <option value="3">3</option>
        <option value="4">4</option>
        <option value="5">5</option>
        <option value="6">6</option>
      </select>
      <span style="font-size:11px;color:var(--muted);align-self:center">sources</span>
    </div>
    <div id="list"></div>
    <details class="missing-list" id="missing-panel" style="display:none">
      <summary>Missing URLs (resiliparse invariant violations)</summary>
      <pre id="missing-pre"></pre>
    </details>
  </aside>
  <main id="main"></main>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
let selected = 0;
let filtered = [];

function esc(s) { return (s ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function renderBanner() {
  const b = document.getElementById('banner');
  const v = DATA.validation;
  if (!v) { b.textContent = 'No validation report available.'; return; }
  if (v.superset_invariant_holds) {
    b.className = 'ok';
    b.innerHTML = `<b>✓ Superset invariant holds.</b> All ${v.num_target_urls.toLocaleString()} target URLs (retained by ≥${v.min_filters} heavy filters) were found in resiliparse output.`;
  } else {
    b.className = 'bad';
    b.innerHTML = `<b>✗ Superset invariant VIOLATED.</b> ${v.num_missing_in_resiliparse.toLocaleString()} / ${v.num_target_urls.toLocaleString()} target URLs are missing from resiliparse.`;
    if (v.missing_urls_sample && v.missing_urls_sample.length) {
      document.getElementById('missing-panel').style.display = 'block';
      document.getElementById('missing-pre').textContent = v.missing_urls_sample.join('\n');
    }
  }
}

function renderSummary() {
  const src = DATA.source_counts;
  const parts = DATA.source_order.map(s => {
    const color = DATA.source_colors[s];
    const n = src[s] || 0;
    return `<span style="color:${color}">●</span>&nbsp;${esc(s)}: ${n.toLocaleString()}`;
  });
  document.getElementById('summary-line').innerHTML = parts.join(' · ');
}

function currentFiltered() {
  const q = document.getElementById('q').value.toLowerCase();
  const mode = document.getElementById('count-mode').value;
  const n = parseInt(document.getElementById('count-n').value, 10);
  return DATA.matched.filter(m => {
    const ok = mode === 'eq' ? m.source_count === n : m.source_count >= n;
    return ok && (!q || m.url.toLowerCase().includes(q));
  });
}

function renderList() {
  filtered = currentFiltered();
  const list = document.getElementById('list');
  list.innerHTML = filtered.map((m, i) => `
    <div class="list-item ${i === selected ? 'active' : ''}" data-i="${i}">
      <div class="url">${esc(m.url)}</div>
      <div class="meta">
        <span>${m.source_count}/${DATA.source_order.length}</span>
        ${DATA.source_order.filter(s => s in m.sources).map(s =>
          `<span class="src-chip" style="background:${DATA.source_colors[s]}">${esc(s)}</span>`
        ).join('')}
      </div>
    </div>`).join('');
  list.querySelectorAll('.list-item').forEach(el => {
    el.onclick = () => { selected = parseInt(el.dataset.i, 10); renderList(); renderMain(); };
  });
}

function renderMain() {
  const main = document.getElementById('main');
  if (filtered.length === 0) { main.innerHTML = '<div class="missing">No matches.</div>'; return; }
  if (selected >= filtered.length) selected = 0;
  const m = filtered[selected];
  const columns = DATA.source_order.map(s => {
    const color = DATA.source_colors[s];
    if (!(s in m.sources)) {
      return `<div class="column">
        <div class="column-head">
          <div class="src-name" style="color:${color}">${esc(s)}</div>
          <div class="src-meta">not present for this URL</div>
        </div>
        <div class="column-body missing">— not retained —</div>
      </div>`;
    }
    const entry = m.sources[s];
    const metaRow = Object.entries(entry.meta || {}).map(([k, v]) => `<span><b>${esc(k)}:</b> ${esc(typeof v === 'object' ? JSON.stringify(v) : v)}</span>`).join('');
    return `<div class="column">
      <div class="column-head">
        <div class="src-name" style="color:${color}">${esc(s)}</div>
        <div class="src-meta"><span><b>${(entry.full_len || entry.text.length).toLocaleString()}</b> chars total</span>${metaRow}</div>
      </div>
      <div class="column-body">${esc(entry.text)}</div>
    </div>`;
  }).join('');
  main.innerHTML = `
    <div id="main-header">
      <div class="url">${esc(m.url)}</div>
      <div class="keys">${selected + 1}/${filtered.length} · j/k or ↑/↓ to navigate</div>
    </div>
    <div class="columns">${columns}</div>`;
}

document.getElementById('q').addEventListener('input', () => { selected = 0; renderList(); renderMain(); });
document.getElementById('count-mode').addEventListener('change', () => { selected = 0; renderList(); renderMain(); });
document.getElementById('count-n').addEventListener('change', () => { selected = 0; renderList(); renderMain(); });
document.addEventListener('keydown', e => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) return;
  if (e.key === 'j' || e.key === 'ArrowDown') { selected = Math.min(filtered.length - 1, selected + 1); renderList(); renderMain(); }
  if (e.key === 'k' || e.key === 'ArrowUp') { selected = Math.max(0, selected - 1); renderList(); renderMain(); }
});

renderBanner();
renderSummary();
renderList();
renderMain();
</script>
</body>
</html>
"""


def render(args: argparse.Namespace) -> int:
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    viewer_local = local_dir / "matched_viewer.jsonl.gz"
    validation_local = local_dir / "validation.json"
    llm_local = local_dir / "llm_curated_matches.jsonl.gz"
    for gs, local in [(MATCHED_VIEWER_PATH, viewer_local), (VALIDATION_PATH, validation_local)]:
        if not local.exists():
            logger.info("Downloading %s", gs)
            subprocess.run(["gcloud", "storage", "cp", gs, str(local)], check=True, capture_output=True)
    # LLM-curated matches are optional — render gracefully if absent.
    if not llm_local.exists():
        try:
            subprocess.run(
                ["gcloud", "storage", "cp", LLM_CURATED_MATCHES_PATH, str(llm_local)],
                check=True,
                capture_output=True,
            )
            logger.info("Downloaded %s", LLM_CURATED_MATCHES_PATH)
        except subprocess.CalledProcessError:
            logger.info("No llm_curated matches found — rendering without that column")

    with open(validation_local) as f:
        validation = json.load(f)

    by_url: dict[str, dict] = defaultdict(lambda: {"sources": {}})
    for rec in _read_jsonl_gz_local(str(viewer_local)):
        by_url[rec["url"]]["url"] = rec["url"]
        by_url[rec["url"]]["sources"][rec["source"]] = {
            "text": rec["text"],
            "meta": rec.get("meta") or {},
            "full_len": rec.get("full_len"),
        }
    # Merge in llm_curated matches if the file is present. Restrict to URLs
    # already in the viewer payload so the HTML stays compact — viewer URLs
    # are the highest-source-count subset anyway.
    if llm_local.exists():
        viewer_max_chars = 8000
        added = 0
        for rec in _read_jsonl_gz_local(str(llm_local)):
            url = rec["url"]
            if url not in by_url:
                continue
            by_url[url]["sources"]["llm_curated"] = {
                "text": (rec.get("text") or "")[:viewer_max_chars],
                "meta": rec.get("meta") or {},
                "full_len": rec.get("full_len"),
            }
            added += 1
        logger.info("Merged llm_curated for %d / %d viewer URLs", added, len(by_url))

    matched: list[dict] = []
    for entry in by_url.values():
        entry["source_count"] = len(entry["sources"])
        matched.append(entry)
    matched.sort(key=lambda m: (-m["source_count"], m["url"]))

    source_counts: dict[str, int] = defaultdict(int)
    for m in matched:
        for s in m["sources"]:
            source_counts[s] += 1

    payload = {
        "matched": matched,
        "source_order": list(SOURCES.keys()),
        "source_colors": {s: info["color"] for s, info in SOURCES.items()},
        "source_counts": dict(source_counts),
        "validation": validation,
    }
    # Escape "</" so stray "</script>" / "</style>" inside user text cannot close
    # the inline <script> block — a classic HTML-in-JSON footgun.
    serialized = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__PAYLOAD__", serialized)
    Path(args.output).write_text(html, encoding="utf-8")
    logger.info("Wrote %s (%.1f MB)", args.output, Path(args.output).stat().st_size / 1e6)
    logger.info(
        "Invariant holds: %s (%d missing / %d targets)",
        validation.get("superset_invariant_holds"),
        validation.get("num_missing_in_resiliparse", 0),
        validation.get("num_target_urls", 0),
    )
    if args.open:
        subprocess.run(["open", args.output], check=False)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    cj = sub.add_parser("cluster-job", help="run the URL extraction + resiliparse scan on the cluster")
    cj.add_argument("--target-urls", type=int, default=10_000, help="max number of URLs to include in the full output")
    cj.add_argument("--min-filters", type=int, default=3, help="minimum # of heavy filters that must retain a URL")
    cj.add_argument(
        "--viewer-urls",
        type=int,
        default=500,
        help="number of URLs to include in the viewer payload (smaller = smaller HTML)",
    )
    cj.add_argument(
        "--max-chars", type=int, default=20_000, help="truncate each record's text to this many chars in cluster outputs"
    )
    cj.add_argument(
        "--viewer-max-chars", type=int, default=8_000, help="further truncation applied to records in the viewer payload"
    )
    cj.add_argument(
        "--threads",
        type=int,
        default=64,
        help="size of the driver ThreadPoolExecutor used for GCS I/O (head node is CPU=0 so we use threads)",
    )
    cj.add_argument(
        "--stratify",
        action="store_true",
        help="balance target URLs across source_count buckets (1..4 heavy filters) "
        "so partially-filtered URLs are represented in the viewer",
    )
    cj.add_argument("--seed", type=int, default=17)

    an = sub.add_parser(
        "analyze",
        help=(
            "comprehensive one-pass analysis: per-source stats, overlap matrix, "
            "stratified target URLs, superset check (us-central2 only, no cross-region reads)"
        ),
    )
    an.add_argument("--target-urls", type=int, default=10_000, help="total stratified target URL count")
    an.add_argument("--viewer-urls", type=int, default=500, help="URLs included in viewer payload")
    an.add_argument("--max-chars", type=int, default=20_000)
    an.add_argument("--viewer-max-chars", type=int, default=8_000)
    an.add_argument("--threads", type=int, default=64)
    an.add_argument("--seed", type=int, default=17)

    lj = sub.add_parser(
        "llm-curated-job",
        help="scan baseline_llm_curated on us-central1 for matches to the target URL set",
    )
    lj.add_argument(
        "--max-chars", type=int, default=20_000, help="truncate each record's text to this many chars in output"
    )
    lj.add_argument("--threads", type=int, default=64)
    lj.add_argument(
        "--target-urls-path",
        type=str,
        default=None,
        help="path to target URL JSON (local file or gs:// URL). Default reads central2.",
    )
    lj.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="where to write the matches JSONL. Default writes central2.",
    )

    r = sub.add_parser("render", help="download the cluster outputs and render HTML")
    r.add_argument("--output", type=str, default="scratch/matched_viewer.html")
    r.add_argument("--local-dir", type=str, default="scratch/matched_viewer")
    r.add_argument("--open", action="store_true", help="open the generated HTML in the default browser")

    args = parser.parse_args()
    return {
        "cluster-job": cluster_job,
        "analyze": analyze,
        "llm-curated-job": llm_curated_job,
        "render": render,
    }[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
