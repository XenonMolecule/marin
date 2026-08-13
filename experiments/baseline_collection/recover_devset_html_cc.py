# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover raw HTML for dev-set docs from CommonCrawl (Iris CPU job, in-region us-central2).

The local 10k pool was deleted (cleanup 2026-04-28) and `decoded_10k` only covers ~461 of the 1934
dev docs. This recovers the rest straight from CommonCrawl, the permanent public archive the docs came
from. Stages:

  manifest  PARALLEL Zephyr scan of the surviving warc-metadata shards -> each missing url's `snapshot`.
  cdx       Resolve each doc's WARC byte-offset via the COLUMNAR index (CC publishes its index as
            Parquet on the healthy data host). We query one crawl at a time filtering by url_host_name
            (`match_type=host`) — the index is sorted by surt key, so host filtering prunes row groups
            to just those hosts, keeping the read small. This deliberately avoids index.commoncrawl.org's
            CDX HTTP API, which rate-limits at 8+ concurrent and was returning intermittent 502s.
  fetch     HTTP Range-GET just each resolved record from data.commoncrawl.org and decode with the clean
            U+FFFD-safe decoder (`decode_warcs_clean.decode_payload`, NOT naive utf-8). Saves RAW HTML.
  reconcile Aggregate successes (pyarrow, tolerant of empty shards), report coverage, write the residual.

Everything runs on Iris CPU workers; only targeted record bytes + pruned index reads leave the cluster.
"""

from __future__ import annotations

import argparse
import html as htmllib
import io
import json
import logging
import re
import sys
import time
from collections import defaultdict

import fsspec
import pyarrow.parquet as pq
import requests
import warcio
from fray.types import ResourceConfig
from marin.datakit.download.commoncrawl.cdx_query_columnar import (
    _build_domain_filter,
    _get_parquet_urls,
    _query_single_crawl,
)
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.decode_warcs_clean import decode_payload

logger = logging.getLogger(__name__)

WORKSPACE = "gs://marin-us-central2/scratch/provenance_10k_devset"
METADATA_GLOB = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f/*.jsonl.gz"
# Surviving 3000-WARC subsets of the same 10k pool (raw extracted HTML {id, html, url, metadata}),
# preserved through the 10k cleanup. Scanning these recovers covered docs IN-REGION, no CommonCrawl.
LOCAL_POOLS = "gs://marin-us-central2/raw/commoncrawl/baseline_3000*/data-*.jsonl.gz"
MISSING = f"{WORKSPACE}/missing_urls.json"
MANIFEST_DIR = f"{WORKSPACE}/refetch_manifest"  # parquet {url, snapshot}
CDX_DIR = f"{WORKSPACE}/refetch_cdx"  # parquet {dev_url, snapshot, filename, offset, length}
OUT = f"{WORKSPACE}/devset_html_refetch"  # parquet {dev_url, src_url, snapshot, source, html, status}
CC_BASE = "https://data.commoncrawl.org"
MAX_FETCH_RETRIES = 5
RETRY_BASE_DELAY = 4.0


# ── stage: manifest (missing url -> snapshot) ────────────────────────────────
def _build_manifest(max_workers: int) -> None:
    with fsspec.open(MISSING, "r") as f:
        missing = set(json.load(f))
    logger.info(f"building manifest: scanning metadata for {len(missing)} missing urls")

    def keep(r: dict) -> bool:
        return r.get("url") in missing and bool(r.get("snapshot"))

    pipeline = (
        Dataset.from_files(METADATA_GLOB)
        .load_jsonl()
        .filter(keep)
        .map(lambda r: {"url": r["url"], "snapshot": r["snapshot"]})
        .reshard(8)
        .write_parquet(f"{MANIFEST_DIR}/m-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name="recover-manifest",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    )
    ctx.execute(pipeline)


# ── stage: local_pools (recover from surviving 3000-WARC HTML pools, no CommonCrawl) ─────────
def _run_local_pools(max_workers: int) -> None:
    with fsspec.open(MISSING, "r") as f:
        missing = set(json.load(f))
    variant_to_dev = {v: u for u in missing for v in _url_variants(u)}
    logger.info(f"scanning local pools for {len(missing)} missing urls")

    def match(r: dict) -> dict | None:
        u = r.get("url")
        dev = variant_to_dev.get(u)
        html = r.get("html")
        if not dev or not html:
            return None
        meta = r.get("metadata") or {}
        return {
            "dev_url": dev,
            "src_url": u,
            "snapshot": meta.get("snapshot", "") if isinstance(meta, dict) else "",
            "source": "local_pool",
            "html": html,
            "status": "ok",
        }

    pipeline = (
        Dataset.from_files(LOCAL_POOLS)
        .load_jsonl()
        .map(match)
        .filter(lambda x: x is not None)
        .reshard(32)
        .write_parquet(f"{OUT}/local-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name="recover-local-pools",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    )
    ctx.execute(pipeline)


# ── stage: local_targeted (read ONLY the pool files that hold our docs) ──────────────────────
POOLS = [
    "marin-us-central2/raw/commoncrawl/baseline_3000-265ff5",
    "marin-us-central2/raw/commoncrawl/baseline_3000_random-34884d",
]


def _snapshot_from_warc(w: str) -> str:
    # .../crawl-data/CC-MAIN-2015-18/segments/... -> CC-MAIN-2015-18
    part = w.split("/crawl-data/", 1)[-1]
    return part.split("/", 1)[0] if "/" in part else ""


def _run_local_targeted(max_workers: int) -> None:
    """Map missing urls -> their WARC (metadata) -> the pool file `data-<hash>.jsonl.gz` (hash =
    _warc_path_hash(warc_file), verified), then read ONLY those files instead of scanning all 4.5 TB."""
    from zephyr.readers import load_jsonl

    from experiments.baseline_collection.sample_internet_timespan import _warc_path_hash

    fs = fsspec.filesystem("gcs")
    with fsspec.open(MISSING, "r") as f:
        missing = set(json.load(f))

    # 1. parallel metadata scan -> {url: warc_file} for the missing urls
    warc_map_dir = f"{WORKSPACE}/warc_map"
    if not fs.glob(f"{warc_map_dir}/*.parquet".replace("gs://", "")):
        scan = (
            Dataset.from_files(METADATA_GLOB)
            .load_jsonl()
            .filter(lambda r: r.get("url") in missing and r.get("warc_file"))
            .map(lambda r: {"url": r["url"], "warc_file": r["warc_file"]})
            .reshard(8)
            .write_parquet(f"{warc_map_dir}/w-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
        )
        ZephyrContext(
            name="recover-warcmap",
            max_workers=max_workers,
            resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
        ).execute(scan)

    url_warc: dict[str, str] = {}
    for f in fs.glob(f"{warc_map_dir}/*.parquet".replace("gs://", "")):
        d = pq.read_table(f, filesystem=fs).to_pydict()
        for u, w in zip(d["url"], d["warc_file"], strict=True):
            url_warc[u] = w

    # 2. which pool files exist, and which do we actually need
    existing = {p: {x.split("/")[-1] for x in fs.glob(f"{p}/data-*.jsonl.gz")} for p in POOLS}
    needed: set[str] = set()
    for w in url_warc.values():
        fn = f"data-{_warc_path_hash(w)}.jsonl.gz"
        for p in POOLS:
            if fn in existing[p]:
                needed.add(f"gs://{p}/{fn}")
                break
    needed_paths = sorted(needed)
    logger.info(
        f"targeted: {len(url_warc)} missing docs have a warc; {len(needed_paths)} pool files to read "
        f"(of {sum(len(v) for v in existing.values())})"
    )

    # 3. read ONLY those files, extract the missing docs' html
    variant_to_dev = {v: u for u in missing for v in _url_variants(u)}

    def match(r: dict) -> dict | None:
        u = r.get("url")
        dev = variant_to_dev.get(u)
        html = r.get("html")
        if not dev or not html:
            return None
        meta = r.get("metadata") or {}
        warc = meta.get("warc_file", "") if isinstance(meta, dict) else ""
        return {
            "dev_url": dev,
            "src_url": u,
            "snapshot": _snapshot_from_warc(warc),
            "source": "local_pool",
            "html": html,
            "status": "ok",
        }

    read = (
        Dataset.from_list(needed_paths)
        .flat_map(load_jsonl)
        .map(match)
        .filter(lambda x: x is not None)
        .reshard(16)
        .write_parquet(f"{OUT}/targeted-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name="recover-local-targeted",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="6g", regions=["us-central2"], preemptible=True),
    )
    ctx.execute(read)


# ── stage: cdx (columnar offset resolution) ──────────────────────────────────
def _url_variants(u: str) -> list[str]:
    """Equivalent url spellings — the index stores one canonical form (trailing slash / scheme vary)."""
    v = [u]
    v.append(u[:-1] if u.endswith("/") else u + "/")
    if u.startswith("http://"):
        v.append("https://" + u[7:])
    elif u.startswith("https://"):
        v.append("http://" + u[8:])
    return list(dict.fromkeys(v))


def _host_of(u: str) -> str:
    return (u.split("/")[2].lower() if "//" in u else u.split("/")[0].lower()).split(":")[0]


def _resolve_crawl(item: dict) -> list[dict]:
    """Columnar-query one crawl for the given hosts; match rows back to the wanted exact urls."""
    snapshot, urls, hosts = item["snapshot"], item["urls"], item["hosts"]
    parquet_urls = _get_parquet_urls(snapshot)
    if not parquet_urls:
        logger.warning(f"no columnar index for {snapshot}")
        return []
    try:
        recs = _query_single_crawl(parquet_urls, _build_domain_filter(hosts, "host"), ["200"], ["html"])
    except Exception as e:
        logger.warning(f"columnar query failed for {snapshot} ({len(hosts)} hosts): {e}")
        return []
    # map every candidate spelling of a wanted url -> its canonical dev url
    variant_to_dev = {v: u for u in urls for v in _url_variants(u)}
    out, seen = [], set()
    for r in recs:
        dev = variant_to_dev.get(r.get("url"))
        if dev and dev not in seen and r.get("filename"):
            seen.add(dev)
            out.append(
                {
                    "dev_url": dev,
                    "snapshot": snapshot,
                    "filename": r["filename"],
                    "offset": str(r["offset"]),
                    "length": str(r["length"]),
                }
            )
    logger.info(f"{snapshot}: {len(hosts)} hosts, {len(recs)} index rows -> {len(out)}/{len(urls)} resolved")
    return out


def _run_cdx_diag() -> None:
    """Resolve ONE crawl with full visibility (record counts, sample urls, match count, traceback)
    written to GCS — since finelog is down and _resolve_crawl swallows errors into empty shards."""
    import traceback

    fs = fsspec.filesystem("gcs")
    groups: dict[str, dict] = defaultdict(lambda: {"urls": [], "hosts": set()})
    for f in fs.glob(f"{MANIFEST_DIR}/*.parquet".replace("gs://", "")):
        d = pq.read_table(f, columns=["url", "snapshot"], filesystem=fs).to_pydict()
        for u, s in zip(d["url"], d["snapshot"], strict=True):
            groups[s]["urls"].append(u)
            groups[s]["hosts"].add(_host_of(u))
    snap = min(groups, key=lambda s: len(groups[s]["hosts"]))  # fewest hosts = fastest query
    urls = groups[snap]["urls"]
    hosts = sorted(groups[snap]["hosts"])
    diag: dict = {
        "snapshot": snap,
        "n_dev_urls": len(urls),
        "n_hosts": len(hosts),
        "sample_dev_urls": urls[:6],
        "sample_hosts": hosts[:10],
    }
    try:
        purls = _get_parquet_urls(snap)
        diag["n_parquet"] = len(purls) if purls else 0
        filt = _build_domain_filter(hosts, "host")
        diag["filter_sql"] = filt[:400]
        recs = _query_single_crawl(purls, filt, ["200"], ["html"])
        diag["n_recs_returned"] = len(recs)
        diag["sample_rec_urls"] = [r.get("url") for r in recs[:12]]
        vtd = {v: u for u in urls for v in _url_variants(u)}
        matched = [r.get("url") for r in recs if vtd.get(r.get("url"))]
        diag["n_matched"] = len(matched)
        diag["sample_matched"] = matched[:6]
    except Exception:
        diag["error"] = traceback.format_exc()
    with fsspec.open(f"{WORKSPACE}/cdx_diag.json", "w") as f:
        json.dump(diag, f, indent=2)
    print(json.dumps(diag, indent=2))


def _run_cdx(max_workers: int, region: str, shard_idx: int, num_shards: int, tag: str) -> None:
    fs = fsspec.filesystem("gcs")
    groups: dict[str, dict] = defaultdict(lambda: {"urls": [], "hosts": set()})
    for f in fs.glob(f"{MANIFEST_DIR}/*.parquet".replace("gs://", "")):
        d = pq.read_table(f, columns=["url", "snapshot"], filesystem=fs).to_pydict()
        for u, s in zip(d["url"], d["snapshot"], strict=True):
            groups[s]["urls"].append(u)
            groups[s]["hosts"].add(_host_of(u))
    # skip crawls already resolved by any prior cdx run (don't redo the expensive big ones).
    resolved: set[str] = set()
    for f in fs.glob(f"{CDX_DIR}/*.parquet".replace("gs://", "")):
        try:
            resolved.update(pq.read_table(f, columns=["snapshot"], filesystem=fs).to_pydict()["snapshot"])
        except Exception:
            continue
    # sort by snapshot so the crawl->shard partition is STABLE across region jobs (disjoint slices).
    items = [
        {"snapshot": s, "urls": g["urls"], "hosts": sorted(g["hosts"])}
        for s, g in sorted(groups.items())
        if s not in resolved
    ]
    if num_shards > 1:
        items = [it for i, it in enumerate(items) if i % num_shards == shard_idx]
    logger.info(
        f"cdx[{tag} {shard_idx}/{num_shards} @ {region}]: {sum(len(it['urls']) for it in items)} urls across {len(items)} crawls"
    )
    # NO reshard barrier: write each crawl's output as it finishes -> CDX_DIR growth is a live signal.
    # Region-tagged filenames so concurrent region jobs never collide on the shared CDX_DIR.
    pipeline = (
        Dataset.from_list(items)
        .flat_map(_resolve_crawl)
        .write_parquet(f"{CDX_DIR}/{tag}-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name=f"recover-cdx-{tag}",
        max_workers=min(max_workers, max(1, len(items))),
        resources=ResourceConfig(cpu=2, ram="12g", regions=[region], preemptible=True),
    )
    ctx.execute(pipeline)


# ── stage: fetch (range-GET the resolved records + clean decode) ─────────────
def _recover_one(rec: dict, session: requests.Session) -> dict:
    url = rec["dev_url"]
    base = {"dev_url": url, "snapshot": rec["snapshot"], "source": "cc_refetch", "html": ""}
    offset, length = int(rec["offset"]), int(rec["length"])
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    for attempt in range(MAX_FETCH_RETRIES):
        try:
            resp = session.get(f"{CC_BASE}/{rec['filename']}", headers=headers, timeout=120)
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                if attempt < MAX_FETCH_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2**attempt))
                    continue
                return {**base, "status": f"http_{resp.status_code}"}
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < MAX_FETCH_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue
            logger.warning(f"range-fetch failed {url}: {e}")
            return {**base, "status": "fetch_fail"}
    for record in warcio.ArchiveIterator(io.BytesIO(resp.content)):
        if record.rec_type != "response":
            continue
        ct = record.http_headers.get_header("Content-Type") if record.http_headers else None
        html = decode_payload(record.content_stream().read(), ct)  # clean decode -> RAW html (full markup)
        return {
            "dev_url": url,
            "src_url": record.rec_headers.get_header("WARC-Target-URI") or url,
            "snapshot": rec["snapshot"],
            "source": "cc_refetch",
            "html": html,
            "status": "ok",
        }
    return {**base, "status": "no_response"}


def _fetch_shard(entries, _shard_info=None):
    session = requests.Session()
    session.headers.update({"User-Agent": "marin-research-crawler/1.0 (academic research)"})
    for e in entries:
        yield _recover_one(e, session)


def _run_fetch(tag: str, max_workers: int, shards: int) -> None:
    pipeline = (
        Dataset.from_files(f"{CDX_DIR}/*.parquet")
        .load_parquet()
        .reshard(shards)
        .map_shard(_fetch_shard)
        .filter(lambda r: r is not None)
        .write_parquet(f"{OUT}/{tag}-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    # range-GET on data.commoncrawl.org tolerates high concurrency (unlike the CDX API).
    ctx = ZephyrContext(
        name=f"recover-fetch-{tag}",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=0.5, ram="2g", regions=["us-central2"], preemptible=True),
    )
    ctx.execute(pipeline)


# ── stage: reconcile ─────────────────────────────────────────────────────────
def _run_reconcile() -> None:
    """Aggregate successes (pyarrow, tolerating empty shards); report coverage vs the full manifest."""
    fs = fsspec.filesystem("gcs")
    recovered: set[str] = set()
    reason_counts: dict[str, int] = {}
    for f in fs.glob(f"{OUT}/*.parquet".replace("gs://", "")):
        try:
            d = pq.read_table(f, columns=["dev_url", "status", "html"], filesystem=fs).to_pydict()
        except Exception:
            continue
        for u, st, h in zip(d["dev_url"], d["status"], d["html"], strict=True):
            if st == "ok" and h:
                recovered.add(u)
            elif st != "ok":
                reason_counts[st] = reason_counts.get(st, 0) + 1
    targets: dict[str, str] = {}
    for f in fs.glob(f"{MANIFEST_DIR}/*.parquet".replace("gs://", "")):
        d = pq.read_table(f, columns=["url", "snapshot"], filesystem=fs).to_pydict()
        for u, s in zip(d["url"], d["snapshot"], strict=True):
            targets[u] = s
    resolved = 0
    for f in fs.glob(f"{CDX_DIR}/*.parquet".replace("gs://", "")):
        try:
            resolved += pq.read_table(f, columns=["dev_url"], filesystem=fs).num_rows
        except Exception:
            continue
    report = {
        "targets": len(targets),
        "cdx_resolved": resolved,
        "recovered": len(recovered),
        "residual": len(targets) - len(recovered),
        "fetch_miss_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
    }
    with fsspec.open(f"{WORKSPACE}/recovery_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"reconcile: {report}")


GATE_OUT = f"{WORKSPACE}/html_gate_report"  # parquet {dev_url, verdict, contain, nsig, source}
SIG_PATH = f"{WORKSPACE}/devset_signatures.json"  # {dev_url: {"sig": [rare tokens], "head": str}}

_GATE_SCRIPT = re.compile(r"(?is)<(script|style).*?</\1>")
_GATE_TAG = re.compile(r"(?s)<[^>]+>")


def _linear_lower(html: str) -> str:
    s = _GATE_SCRIPT.sub(" ", html or "")
    return htmllib.unescape(_GATE_TAG.sub(" ", s)).lower()


def _run_gate(max_workers: int) -> None:
    """Identity gate: a recovered html must actually BE the page its dev_url/label denotes.

    Wrong-page substitutions (query-string collisions, stale snapshots) are the failure we saw. We
    catch them by requiring the label's DISTINCTIVE (rarest, doc-frequency<=6) tokens to appear in the
    html — shared legal/forum boilerplate can't mask a wrong page because those tokens aren't rare.
    Writes only verdicts (no html leaves the cluster)."""
    with fsspec.open(SIG_PATH, "r") as f:
        sigs = json.load(f)

    def gate(r: dict) -> dict | None:
        u = r.get("dev_url")
        if r.get("status") != "ok" or not r.get("html"):
            return None
        s = sigs.get(u)
        if not s or not s.get("sig"):
            return {"dev_url": u, "verdict": "no_signature", "contain": -1.0, "nsig": 0, "source": r.get("source", "")}
        sig = s["sig"]
        lin = _linear_lower(r["html"])
        contain = sum(1 for w in sig if w in lin) / len(sig)
        return {
            "dev_url": u,
            "verdict": "match" if contain >= 0.40 else "mismatch",
            "contain": round(contain, 3),
            "nsig": len(sig),
            "source": r.get("source", ""),
        }

    pipeline = (
        Dataset.from_files(f"{OUT}/*.parquet")
        .load_parquet(columns=["dev_url", "html", "source", "status"])
        .map(gate)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(f"{GATE_OUT}/g-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=False)
    )
    ZephyrContext(
        name="devset-html-gate",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    ).execute(pipeline)
    logger.info(f"gate report -> {GATE_OUT}")


# ── stage: full_warc (download the exact CC WARC per doc, extract by warc_record_id) ─────────
# The URL-keyed joins collide on query-string URLs (same path, different ?id=/?p= → wrong page). The
# metadata gives every doc's EXACT `warc_file` + `warc_record_id`, so streaming that WARC and matching
# the record id recovers the true page with zero collision. No CDX/offset needed (immune to the CDX
# outage) — we stream the whole WARC and stop once every wanted record in it is found.
TARGET = f"{WORKSPACE}/full_warc_targets.json"  # union of missing + at-risk dev urls to (re)fetch
EXCLUDE = f"{WORKSPACE}/full_warc_exclude.json"  # dev urls already recovered elsewhere — skip their WARCs
WARC_PTR = f"{WORKSPACE}/warc_ptr"  # parquet {dev_url, warc_file, warc_record_id, snapshot}


def _norm_rid(h: str | None) -> str:
    """Metadata stores bare UUIDs; WARC headers wrap them as `<urn:uuid:...>`. Normalize both."""
    return (h or "").strip().lstrip("<").rstrip(">").removeprefix("urn:uuid:")


def _cc_url(warc_file: str) -> str:
    path = warc_file.split("commoncrawl/", 1)[-1]
    return f"{CC_BASE}/{path}"


def _build_warc_ptr(max_workers: int) -> None:
    """Scan the surviving warc-metadata for the target urls -> exact {warc_file, warc_record_id}.

    Match metadata urls to targets by url VARIANT (scheme + trailing slash) rather than exact string —
    this reclaims the http/https/slash spelling misses that exact matching drops. Variants never collapse
    the query string, so there is no wrong-page collision risk (unlike the old url-keyed joins)."""
    with fsspec.open(TARGET, "r") as f:
        target = json.load(f)
    variant_to_dev = {v: u for u in target for v in _url_variants(u)}
    logger.info(f"building warc pointers for {len(target)} target urls ({len(variant_to_dev)} url variants)")
    scan = (
        Dataset.from_files(METADATA_GLOB)
        .load_jsonl()
        .filter(lambda r: r.get("url") in variant_to_dev and r.get("warc_file") and r.get("warc_record_id"))
        .map(
            lambda r: {
                "dev_url": variant_to_dev[r["url"]],
                "warc_file": r["warc_file"],
                "warc_record_id": r["warc_record_id"],
                "snapshot": r.get("snapshot", ""),
            }
        )
        .reshard(16)
        .write_parquet(f"{WARC_PTR}/p-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ZephyrContext(
        name="recover-warcptr",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    ).execute(scan)


def _run_full_warc(max_workers: int, limit_warcs: int = 0, warc_offset: int = 0, out_tag: str = "") -> None:
    fs = fsspec.filesystem("gcs")
    exclude: set[str] = set()
    if fs.exists(EXCLUDE.replace("gs://", "")):
        with fsspec.open(EXCLUDE, "r") as f:
            exclude = set(json.load(f))

    by_warc: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for f in fs.glob(f"{WARC_PTR}/*.parquet".replace("gs://", "")):
        d = pq.read_table(f, filesystem=fs).to_pydict()
        for du, wf, rid, snap in zip(d["dev_url"], d["warc_file"], d["warc_record_id"], d["snapshot"], strict=True):
            if du in exclude or du in seen:
                continue
            seen.add(du)
            by_warc[wf].append((_norm_rid(rid), du, snap))
    all_warcs = sorted(by_warc)
    end = warc_offset + limit_warcs if limit_warcs else None
    warcs = all_warcs[warc_offset:end]
    if limit_warcs or warc_offset:
        logger.info(f"full_warc: SMOKE/slice — warcs[{warc_offset}:{end}] = {len(warcs)} warcs")
    logger.info(
        f"full_warc: {len(seen)} urls across {len(all_warcs)} distinct warcs "
        f"(fetching {len(warcs)}, excluded {len(exclude)})"
    )

    def fetch_warc(wf: str) -> list[dict]:
        wanted = {rid: (du, snap) for rid, du, snap in by_warc[wf]}
        url = _cc_url(wf)
        for attempt in range(MAX_FETCH_RETRIES):
            try:
                resp = requests.get(url, stream=True, timeout=600)
                resp.raise_for_status()
                resp.raw.decode_content = False  # keep the .warc.gz bytes; warcio does gzip itself
                out: list[dict] = []
                for rec in warcio.ArchiveIterator(resp.raw):
                    if rec.rec_type != "response":
                        continue
                    rid = _norm_rid(rec.rec_headers.get_header("WARC-Record-ID"))
                    if rid not in wanted:
                        continue
                    du, snap = wanted[rid]
                    ct = rec.http_headers.get_header("Content-Type") if rec.http_headers else None
                    out.append(
                        {
                            "dev_url": du,
                            "src_url": rec.rec_headers.get_header("WARC-Target-URI") or du,
                            "snapshot": snap,
                            "source": "full_warc",
                            "html": decode_payload(rec.content_stream().read(), ct),
                            "status": "ok",
                        }
                    )
                    if len(out) == len(wanted):
                        break
                return out
            except Exception as e:
                if attempt < MAX_FETCH_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (attempt + 1))
                    continue
                return [
                    {
                        "dev_url": du,
                        "src_url": du,
                        "snapshot": snap,
                        "source": "full_warc",
                        "html": "",
                        "status": f"fetch_fail:{type(e).__name__}",
                    }
                    for _, du, snap in by_warc[wf]
                ]
        return []

    # distinct output prefix per run — two full runs both reshard output to 32, so without a unique
    # tag their shard names collide and skip_existing silently drops the second run's data.
    if limit_warcs or warc_offset:
        prefix = f"fullwarc-smoke{warc_offset}"
    elif out_tag:
        prefix = f"fullwarc-{out_tag}"
    else:
        prefix = "fullwarc"
    # reshard the input to one-warc-per-shard BEFORE flat_map: the shard is Zephyr's unit of
    # parallelism, and from_list is a single shard, so without this every warc would stream on one
    # worker. This fans the ~1.5k downloads across the whole worker pool.
    pipeline = (
        Dataset.from_list(warcs)
        .reshard(len(warcs))
        .flat_map(fetch_warc)
        .filter(lambda r: r is not None)
        .reshard(min(32, max(1, len(warcs))))
        .write_parquet(f"{OUT}/{prefix}-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ZephyrContext(
        name="recover-full-warc",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    ).execute(pipeline)
    logger.info(f"full_warc recovery written to {OUT}")


# ── stage: fallback (docs with no 10k-pool WARC pointer — e.g. the random-timespan category) ──
# These never came from the 10k pool, so full_warc can't reach them. Their raw html survives in two
# other us-east5 corpora keyed by url; join by url variant (in-region) and copy only the small result.
FALLBACK_SOURCES = [
    ("timespan", "gs://marin-us-east5/documents/internet_timespan_sample/v1/*.parquet"),
    ("decoded10k", "gs://marin-us-east5/documents/bert_pipeline/decoded_10k/*.parquet"),
]


EXTRACT_TODO = f"{WORKSPACE}/extraction_todo.json"  # dev urls to gold-extract
EXTRACT_STAGE = f"{WORKSPACE}/extract_stage"  # parquet {dev_url, html, source} — only authoritative html


def _run_stage_extract(max_workers: int) -> None:
    """Filter the recovered set to the extraction to-do urls, keeping ONLY authoritative html
    (full_warc / fallback — never the superseded, collision-prone local_pool/cc_refetch joins)."""
    with fsspec.open(EXTRACT_TODO, "r") as f:
        todo = set(json.load(f))

    def keep(r: dict) -> dict | None:
        if (
            r.get("dev_url") in todo
            and r.get("source") in ("full_warc", "fallback")
            and r.get("status") == "ok"
            and r.get("html")
        ):
            return {"dev_url": r["dev_url"], "html": r["html"], "source": r["source"]}
        return None

    pipeline = (
        Dataset.from_files(f"{OUT}/*.parquet")
        .load_parquet(columns=["dev_url", "html", "source", "status"])
        .map(keep)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(f"{EXTRACT_STAGE}/s-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=False)
    )
    ZephyrContext(
        name="stage-extract-html",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    ).execute(pipeline)
    logger.info(f"staged extraction html -> {EXTRACT_STAGE}")


def _run_fallback(max_workers: int) -> None:
    fs = fsspec.filesystem("gcs")
    with fsspec.open(TARGET, "r") as f:
        target = json.load(f)
    rec: set[str] = set()
    for f2 in fs.glob(f"{OUT}/*.parquet".replace("gs://", "")):
        try:
            t = pq.read_table(f2, columns=["dev_url", "status"], filesystem=fs).to_pydict()
            rec |= {du for du, s in zip(t["dev_url"], t["status"], strict=True) if s == "ok"}
        except Exception:
            continue
    remaining = set(target) - rec
    variant_to_dev = {v: u for u in remaining for v in _url_variants(u)}
    logger.info(f"fallback: {len(remaining)} remaining urls, scanning us-east5 corpora")

    def match(r: dict) -> dict | None:
        dev = variant_to_dev.get(r.get("url"))
        html = r.get("html")
        if not dev or not html:
            return None
        return {
            "dev_url": dev,
            "src_url": r.get("url"),
            "snapshot": r.get("snapshot", "") or "",
            "source": "fallback",
            "html": html,
            "status": "ok",
        }

    for tag, src in FALLBACK_SOURCES:
        pipeline = (
            Dataset.from_files(src)
            .load_parquet(columns=["url", "snapshot", "html"])
            .map(match)
            .filter(lambda x: x is not None)
            .reshard(8)
            .write_parquet(f"{OUT}/fallback-{tag}-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
        )
        ZephyrContext(
            name=f"recover-fallback-{tag}",
            max_workers=max_workers,
            resources=ResourceConfig(cpu=1, ram="4g", regions=["us-east5"], preemptible=True),
        ).execute(pipeline)
    logger.info("fallback recovery done")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=[
            "manifest",
            "local_pools",
            "local_targeted",
            "cdx",
            "cdx_diag",
            "fetch",
            "reconcile",
            "gate",
            "warc_ptr",
            "full_warc",
            "fallback",
            "stage_extract",
        ],
        default="cdx",
    )
    ap.add_argument("--tag", default="r0")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--region", default="us-central2")
    ap.add_argument("--shard-idx", type=int, default=0)  # cdx: this region's slice of crawls
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit-warcs", type=int, default=0)  # full_warc: smoke-test on N warcs (0 = all)
    ap.add_argument("--warc-offset", type=int, default=0)  # full_warc: skip first N warcs (disjoint smokes)
    ap.add_argument("--out-tag", default="")  # full_warc: distinct output prefix per run (avoid shard collisions)
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs")
    if args.mode == "manifest":
        _build_manifest(args.max_workers)
    elif args.mode == "local_pools":
        _run_local_pools(args.max_workers)
    elif args.mode == "local_targeted":
        _run_local_targeted(args.max_workers)
    elif args.mode == "cdx_diag":
        _run_cdx_diag()
    elif args.mode == "cdx":
        if not fs.glob(f"{MANIFEST_DIR}/*.parquet".replace("gs://", "")):
            _build_manifest(48)
        _run_cdx(args.max_workers, args.region, args.shard_idx, args.num_shards, args.tag)
    elif args.mode == "fetch":
        _run_fetch(args.tag, args.max_workers, args.shards)
    elif args.mode == "gate":
        _run_gate(args.max_workers)
    elif args.mode == "warc_ptr":
        _build_warc_ptr(args.max_workers)
    elif args.mode == "full_warc":
        if not fs.glob(f"{WARC_PTR}/*.parquet".replace("gs://", "")):
            _build_warc_ptr(args.max_workers)
        _run_full_warc(args.max_workers, args.limit_warcs, args.warc_offset, args.out_tag)
    elif args.mode == "fallback":
        _run_fallback(args.max_workers)
    elif args.mode == "stage_extract":
        _run_stage_extract(args.max_workers)
    else:
        _run_reconcile()
    return 0


if __name__ == "__main__":
    sys.exit(main())
