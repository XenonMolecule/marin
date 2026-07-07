# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the *whole* Common Crawl timeline, not just the DCLM window.

The DCLM 400m-1x pool (``experiments/distill/dclm_400m_1x.txt``) stops at
``CC-MAIN-2022-49`` — DCLM-baseline-1.0 was built from Common Crawl only through
2022, so *anything* drawn from that manifest is blind to 2023-2026 (~27% of the
CC timeline). This builds a sample that spans every WARC-era crawl,
``CC-MAIN-2013-20 → CC-MAIN-2026-25`` (122 crawls at time of writing), by reading
each crawl's ``warc.paths.gz`` straight from ``data.commoncrawl.org`` — the same
free-ingress source ``decode_warcs_clean``/``download_warcs`` already stream from.

Two stages:

**Stage 1 — manifest** (``build-manifest``, runs anywhere, ~24 MB of downloads):
one crawl = one time stratum. For every WARC-era crawl in ``collinfo.json`` we
draw ONE WARC (seeded) from its ~90k-line ``warc.paths.gz``. Result: a ~122-line
manifest, one WARC per period, maximally even temporal coverage. Written as
plain ``s3://commoncrawl/...`` lines so every existing WARC tool consumes it.

**Stage 2 — sample** (``sample``, Zephyr CPU fleet inside an Iris job): stream
each sampled WARC and keep ~1/50 of its ``text/html`` response records (a stable
hash of the record id, so the 2% is reproducible AND decorrelated from a WARC's
internal domain clustering). Each kept payload is decoded with
:func:`decode_warcs_clean.decode_payload` — WHATWG charset precedence, **never**
``errors="replace"``, so the sample carries ZERO ``U+FFFD``. One parquet per WARC
(stable id, ``skip_existing`` for resumability). 1/50 x ~122 WARCs ≈ "2 WARCs of
records" — a few-GB bundle small enough to pull down locally. Each row is
``{doc_id, warc_hash, url, snapshot, html}`` (full decoded page; no derived text).

Unlike ``decode_warcs_clean._decode_one_warc`` this does NOT drop empty-body docs:
a representative internet sample must not filter on classifier-input non-emptiness.

Run Stage 1::

    python -m experiments.baseline_collection.sample_internet_timespan build-manifest \
        --output experiments/distill/subsets/internet_timespan_warcs.txt

Run Stage 2 (inside an Iris job, us-east5)::

    python -m experiments.baseline_collection.sample_internet_timespan sample \
        --manifest experiments/distill/subsets/internet_timespan_warcs.txt \
        --output-path gs://marin-us-east5/documents/internet_timespan_sample/v1 \
        --sample-rate 50

Regression check (correct decode ⇒ zero U+FFFD)::

    python -m experiments.baseline_collection.sample_internet_timespan scan-fffd \
        --output-path gs://marin-us-east5/documents/internet_timespan_sample/v1
"""

import argparse
import gzip
import hashlib
import io
import logging
import random
import re
import time

import fsspec
import requests
import warcio
from fray.types import ResourceConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.decode_warcs_clean import (
    HTTP_TIMEOUT,
    MAX_RETRIES,
    REPLACEMENT,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
    _normalize_record_id,
    _s3_to_https,
    _snapshot_of,
    _warc_path_hash,
    decode_payload,
)

logger = logging.getLogger(__name__)

COLLINFO_URL = "https://index.commoncrawl.org/collinfo.json"
CC_BASE = "https://data.commoncrawl.org"
# WARC-era crawls are named CC-MAIN-YYYY-WW. The three older collections
# (CC-MAIN-2008-2009, 2009-2010, 2012) are ARC format with a different layout and
# are deliberately excluded — this sampler speaks WARC only.
_WARC_ERA_RE = re.compile(r"^CC-MAIN-\d{4}-\d{2}$")
DEFAULT_SEED = 0
DEFAULT_SAMPLE_RATE = 50  # keep 1 in 50 text/html records ≈ 2%


# ── Stage 1: stratified crawl → WARC manifest ────────────────────────────────


def list_warc_era_crawls() -> list[str]:
    """All CC-MAIN-YYYY-WW crawl ids, oldest → newest (excludes ARC-era collections)."""
    crawls = requests.get(COLLINFO_URL, timeout=60).json()
    ids = [c["id"] for c in crawls if _WARC_ERA_RE.match(c["id"])]
    return sorted(ids)  # lexical sort == chronological for YYYY-WW


def _fetch_warc_paths(crawl_id: str) -> list[str]:
    """The full ``warc.paths.gz`` line list for a crawl (~90k relative paths)."""
    url = f"{CC_BASE}/crawl-data/{crawl_id}/warc.paths.gz"
    resp = requests.get(url, timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    text = gzip.decompress(resp.content).decode("utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def build_manifest(output_path: str, seed: int) -> None:
    """Draw ONE WARC per crawl across the whole WARC-era timeline (seeded).

    Every crawl is an independent time stratum; drawing one WARC from each gives
    perfectly even temporal coverage. The per-crawl pick uses a crawl-specific
    seeded RNG so the choice is reproducible and independent across crawls.
    """
    crawls = list_warc_era_crawls()
    logger.info(f"{len(crawls)} WARC-era crawls: {crawls[0]} → {crawls[-1]}")

    picks: list[str] = []
    for crawl_id in crawls:
        paths = _fetch_warc_paths(crawl_id)
        # Per-crawl seed keeps picks reproducible and independent across crawls.
        crawl_seed = int(hashlib.sha256(f"{seed}:{crawl_id}".encode()).hexdigest()[:16], 16)
        chosen = random.Random(crawl_seed).choice(paths)
        picks.append(f"s3://commoncrawl/{chosen}")
        logger.info(f"{crawl_id}: 1 of {len(paths)} WARCs → {chosen}")

    header = (
        f"# One WARC per crawl across the WARC-era CC timeline ({crawls[0]} → {crawls[-1]})\n"
        f"# crawls: {len(crawls)}  |  seed: {seed}\n"
        f"# method: seeded random.choice over each crawl's warc.paths.gz\n"
    )
    with fsspec.open(output_path, "w") as f:
        f.write(header)
        for p in picks:
            f.write(p + "\n")
    logger.info(f"Wrote {len(picks)} WARCs (one per crawl) → {output_path}")


# ── Stage 2: 1/50 clean-decoded record sample ────────────────────────────────


def _keep_record(doc_id: str, sample_rate: int) -> bool:
    """Deterministic ~1/sample_rate keep, decorrelated from a WARC's record order.

    Hashing the (already ``urn:uuid:``-stripped) record id gives a stable, uniform
    2% draw that does not track the domain clustering of sequential WARC records.
    """
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:16], 16)
    return h % sample_rate == 0


def _sample_one_warc(warc_path: str, sample_rate: int) -> list[dict]:
    """Stream one WARC, clean-decode ~1/sample_rate of its text/html records.

    Emits ``{doc_id, warc_hash, url, snapshot, html}`` — the FULL cleanly-decoded
    page, no derived text column (recompute body text later via
    ``decode_warcs_clean.body_strip``/``fasttext_text`` if needed). Retries on
    transient HTTP errors; raises on permanent failure (never silently drops a
    WARC). No content filtering — every decoded text/html record is representative.
    """
    url = _s3_to_https(warc_path)
    warc_hash = _warc_path_hash(warc_path)
    snapshot = _snapshot_of(warc_path)

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Downloading WARC (attempt {attempt + 1}): {warc_path}")
            response = requests.get(url, stream=True, timeout=HTTP_TIMEOUT)
            if response.status_code in RETRYABLE_STATUS_CODES:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(f"Rate limited ({response.status_code}) on {warc_path}, retry in {delay:.0f}s")

                time.sleep(delay)
                continue
            response.raise_for_status()

            records: list[dict] = []
            raw_stream = io.BytesIO(response.content)
            seen_html = 0
            fffd_skips = 0
            parse_errors = 0

            for record in warcio.ArchiveIterator(raw_stream):
                try:
                    if record.rec_type != "response":
                        continue
                    http_headers = record.http_headers
                    if http_headers is None:
                        continue
                    content_type = http_headers.get_header("Content-Type") or ""
                    if "text/html" not in content_type.lower():
                        continue

                    seen_html += 1
                    doc_id = _normalize_record_id(record.rec_headers.get_header("WARC-Record-ID") or "")
                    if not _keep_record(doc_id, sample_rate):
                        continue

                    payload = record.content_stream().read()
                    html = decode_payload(payload, content_type)
                    if REPLACEMENT in html:
                        # Unreachable given the latin-1 tail, but never ship a corrupt row.
                        fffd_skips += 1
                        continue

                    records.append(
                        {
                            "doc_id": doc_id,
                            "warc_hash": warc_hash,
                            "url": record.rec_headers.get_header("WARC-Target-URI") or "",
                            "snapshot": snapshot,
                            "html": html,
                        }
                    )
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:
                        logger.warning(f"Skipping corrupt record in {warc_path}: {e}")

            if fffd_skips:
                logger.warning(f"{warc_path}: skipped {fffd_skips} records still holding U+FFFD after decode")
            logger.info(
                f"{warc_path} [{snapshot}]: kept {len(records)} of {seen_html} text/html records "
                f"(~1/{sample_rate}); {parse_errors} parse errors"
            )
            return records

        except requests.exceptions.RequestException as e:
            delay = RETRY_BASE_DELAY * (2**attempt)
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Download error on {warc_path} (attempt {attempt + 1}): {e}. Retrying in {delay:.0f}s")

                time.sleep(delay)
            else:
                raise RuntimeError(f"Failed to download {warc_path} after {MAX_RETRIES} attempts: {e}") from e

    raise RuntimeError(f"Failed to download {warc_path} after {MAX_RETRIES} attempts")


def _load_manifest(manifest_path: str) -> list[str]:
    with fsspec.open(manifest_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def sample_warcs(manifest_path: str, output_path: str, sample_rate: int, max_workers: int) -> None:
    """Zephyr CPU fleet: one WARC = one shard = one parquet of ~1/sample_rate records."""
    warc_paths = _load_manifest(manifest_path)
    logger.info(f"Sampling ~1/{sample_rate} records from {len(warc_paths)} WARCs → {output_path}")

    def _output_path_fn(shard_idx: int, total_shards: int) -> str:
        return f"{output_path}/data-{_warc_path_hash(warc_paths[shard_idx])}.parquet"

    pipeline = (
        Dataset.from_list(warc_paths)
        .reshard(len(warc_paths))  # one WARC per shard → one output file per WARC
        .flat_map(lambda w: _sample_one_warc(w, sample_rate))
        .write_parquet(_output_path_fn, skip_existing=True)
    )
    # Each worker holds a full WARC body (~1-1.3 GB compressed) plus decompressed
    # HTML; 24 GiB matches decode_warcs_clean's headroom for adversarial WARCs.
    ctx = ZephyrContext(
        name="sample-internet-timespan",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="24g"),
    )
    ctx.execute(pipeline)
    logger.info(f"Sample complete. {len(warc_paths)} WARCs → {output_path}")


def scan_fffd(output_path: str, sample: int | None) -> None:
    """Regression check: a correct decode yields ZERO U+FFFD across the sample."""
    import pyarrow.parquet as pq

    fs, root = fsspec.core.url_to_fs(output_path)
    files = sorted(fs.glob(f"{root}/data-*.parquet"))
    if sample is not None:
        files = files[:sample]
    total_docs = total_bad = bad_files = 0
    for f in files:
        n_bad = 0
        with fs.open(f, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(columns=["html"], batch_size=2048):
                for h in batch.column("html").to_pylist():
                    total_docs += 1
                    if h and REPLACEMENT in h:
                        n_bad += 1
        total_bad += n_bad
        if n_bad:
            bad_files += 1
            logger.warning(f"{f}: {n_bad} docs still contain U+FFFD")
    logger.info(f"scan-fffd: {len(files)} files, {total_docs} docs, {total_bad} with U+FFFD ({bad_files} bad files)")
    if total_bad:
        raise SystemExit(f"FAIL: {total_bad} docs contain U+FFFD — decode is broken")
    logger.info("scan-fffd PASS: zero U+FFFD")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    m = sub.add_parser("build-manifest", help="Stage 1: one WARC per crawl across the CC timeline.")
    m.add_argument("--output", required=True, help="Manifest output path (local or gs://).")
    m.add_argument("--seed", type=int, default=DEFAULT_SEED)

    s = sub.add_parser("sample", help="Stage 2: clean-decode ~1/N records per WARC into parquet.")
    s.add_argument("--manifest", required=True)
    s.add_argument("--output-path", required=True)
    s.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Keep 1 in N text/html records.")
    s.add_argument("--max-workers", type=int, default=128)

    f = sub.add_parser("scan-fffd", help="Regression: assert zero U+FFFD in the decoded sample.")
    f.add_argument("--output-path", required=True)
    f.add_argument("--scan-sample", type=int, default=None)

    args = ap.parse_args()
    if args.command == "build-manifest":
        build_manifest(args.output, args.seed)
    elif args.command == "sample":
        sample_warcs(args.manifest, args.output_path, args.sample_rate, args.max_workers)
    elif args.command == "scan-fffd":
        scan_fffd(args.output_path, args.scan_sample)


if __name__ == "__main__":
    main()
