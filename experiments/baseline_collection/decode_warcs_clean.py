# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 of the BERT WARC pipeline: re-decode raw WARC bytes CORRECTLY.

The existing 10k download (``download_warcs.py``) decoded payloads with
``content.decode("utf-8", errors="replace")`` — a meta-only / fixed-codec path that
turns every non-ASCII byte in a charset-less page into ``U+FFFD`` (the replacement
character). ~8.8% of docs are corrupted this way, and ``U+FFFD`` is written *after*
the source byte is discarded, so it is **unrecoverable** downstream. The only fix is
to re-decode from the raw WARC bytes resolving the charset in WHATWG order. See
``…/jusText/docs/WARC_DECODING.md``.

This stage:
  * downloads each WARC (warcio; free CommonCrawl S3 ingress via data.commoncrawl.org),
  * for every ``response`` + ``text/html`` record, decodes the *payload bytes* with
    :func:`decode_payload` (BOM → HTTP ``Content-Type charset`` → ``<meta charset>`` →
    ``charset-normalizer`` → cp1252/latin-1; **never** ``errors="replace"``), and
  * writes ONE parquet per WARC: ``{doc_id, warc_hash, url, snapshot, html, text_body}``.

``html`` is the FULL decoded page (Rule 5 — no over-eager pre-strip; jusText/extractors
later want the whole document). ``text_body`` is the BERT-classifier input
(``body_strip`` + whitespace-collapse + lowercase) — the exact preprocessing the
ModernBERT classifier was trained on.

One WARC = one shard = one output file (stable id ``sha256(warc_path)[:12]``),
``skip_existing`` for resumability — identical unit-of-work to ``download_warcs.py``.

Run (Zephyr CPU fleet, inside an Iris job in us-east5)::

    python -m experiments.baseline_collection.decode_warcs_clean \
        --manifest experiments/distill/dclm_400m_1x.txt \
        --output-path gs://marin-us-east5/documents/bert_pipeline/decoded_10k \
        --limit 1000

Regression check (a correct pipeline produces ZERO U+FFFD)::

    python -m experiments.baseline_collection.decode_warcs_clean \
        --scan-fffd --output-path gs://marin-us-east5/documents/bert_pipeline/decoded_10k
"""

import argparse
import codecs
import io
import logging
import re
import time
from dataclasses import dataclass

import charset_normalizer
import fsspec
import requests
import warcio
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5.0  # seconds, doubles each retry
RETRYABLE_STATUS_CODES = {429, 503}
HTTP_TIMEOUT = 600  # 10 minutes — 1GB at 2MB/s = 500s

REPLACEMENT = "�"  # U+FFFD; must NEVER appear in correctly-decoded output

_HTTP_CHARSET_RE = re.compile(r"charset\s*=\s*[\"']?([a-zA-Z0-9_\-:.]+)", re.IGNORECASE)
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset[\s"'=]*([a-zA-Z0-9_\-:.]+)""", re.IGNORECASE)
_SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")

# WHATWG maps the latin-1/ascii family to windows-1252 for HTML: a page that declares
# iso-8859-1 but uses the cp1252 punctuation range (0x80-0x9F: curly quotes, em-dash…)
# must be decoded as cp1252, else those bytes become control chars. This IS the
# curly-apostrophe failure mode (Ghana's -> Ghana<ctrl>s). Browsers do this; so do we.
_LATIN1_FAMILY = {
    "iso-8859-1",
    "iso8859-1",
    "iso_8859-1",
    "iso-8859-1:1987",
    "latin-1",
    "latin1",
    "l1",
    "ascii",
    "us-ascii",
    "cp819",
    "ibm819",
    "8859-1",
    "windows-1252",
    "cp1252",
}
# charset-normalizer's single-byte guess is only trusted when it found real language
# structure (coherence) — otherwise an ambiguous mostly-ASCII Western page defaults to
# cp1252 (the documented right call) instead of a cp1250/cyrillic coin-flip.
_COHERENCE_MIN = 0.50


def _normalize_label(enc: str | None) -> str | None:
    if not enc:
        return None
    if enc.strip().lower() in _LATIN1_FAMILY:
        return "cp1252"
    return enc


# body_strip / fasttext_text are duplicated from cascade_chat_filter.py ON PURPOSE:
# that module imports torch_xla at top level, which is absent (and would crash) on a
# CPU decode worker. These three regexes are the entire shared surface.
_SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
_BODY_TAG_RE = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


def body_strip(html: str) -> str:
    cleaned = _SCRIPT_TAG_RE.sub("", html)
    bodies = [m.group(1) for m in _BODY_TAG_RE.finditer(cleaned)]
    return "".join(bodies) if bodies else cleaned


def fasttext_text(bs: str) -> str:
    """body_strip text -> classifier input (collapse whitespace + lowercase)."""
    return _WS_RE.sub(" ", bs).strip().lower()


def _s3_to_https(s3_path: str) -> str:
    """Convert s3://commoncrawl/... to https://data.commoncrawl.org/... (free ingress)."""
    if s3_path.startswith("s3://commoncrawl/"):
        return "https://data.commoncrawl.org/" + s3_path[len("s3://commoncrawl/") :]
    if s3_path.startswith("https://"):
        return s3_path
    return "https://data.commoncrawl.org/" + s3_path


def _warc_path_hash(warc_path: str) -> str:
    """Deterministic short hash of a WARC path for stable output filenames."""
    import hashlib

    return hashlib.sha256(warc_path.encode()).hexdigest()[:12]


def _snapshot_of(warc_path: str) -> str:
    m = _SNAPSHOT_RE.search(warc_path)
    return m.group(0) if m else ""


def _normalize_record_id(record_id: str) -> str:
    """Strip the ``urn:uuid:`` wrapper so ids join cleanly with DCLM (same as extraction)."""
    rid = record_id.strip()
    if rid.startswith("<") and rid.endswith(">"):
        rid = rid[1:-1]
    if rid.startswith("urn:uuid:"):
        rid = rid[len("urn:uuid:") :]
    return rid


def _charset_from_header(content_type: str | None) -> str | None:
    if not content_type:
        return None
    m = _HTTP_CHARSET_RE.search(content_type)
    return m.group(1) if m else None


def _bom_charset(raw: bytes) -> str | None:
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16"
    return None


def _try_strict(raw: bytes, enc: str | None) -> str | None:
    """Strict decode (raises on bad bytes) -> str, or None if codec unknown / bytes invalid."""
    if not enc:
        return None
    try:
        codecs.lookup(enc)
    except LookupError:
        return None
    try:
        return raw.decode(enc)
    except (UnicodeDecodeError, ValueError):
        return None


def decode_payload(raw: bytes, content_type_header: str | None) -> str:
    """Decode HTML payload bytes to unicode in WHATWG charset precedence.

    BOM → HTTP ``Content-Type charset`` → ``<meta charset>`` (first 1024 B) →
    statistical (``charset-normalizer``) → cp1252 → latin-1. Strict at every step so a
    wrong guess is rejected (no silent ``U+FFFD``); the cp1252/latin-1 tail maps every
    byte and so always succeeds without producing the replacement character.
    """
    # 1. BOM — authoritative.
    bom = _bom_charset(raw)
    if bom:
        s = _try_strict(raw, bom)
        if s is not None and REPLACEMENT not in s:
            return s

    # 2. HTTP Content-Type charset (the header the meta tag is usually missing).
    s = _try_strict(raw, _normalize_label(_charset_from_header(content_type_header)))
    if s is not None and REPLACEMENT not in s:
        return s

    # 3. <meta charset> in the first ~1024 bytes.
    m = _META_CHARSET_RE.search(raw[:1024])
    if m:
        s = _try_strict(raw, _normalize_label(m.group(1).decode("ascii", "ignore")))
        if s is not None and REPLACEMENT not in s:
            return s

    # 4. utf-8 strict. utf-8 is self-validating (cp1252 high bytes almost never form
    #    valid utf-8 sequences), so this reliably catches undeclared-utf8 AND correctly
    #    REJECTS cp1252 content, handing it to the steps below.
    s = _try_strict(raw, "utf-8")
    if s is not None and REPLACEMENT not in s:
        return s

    # 5. Statistical detection — trusted only when confident: a multibyte encoding, or a
    #    single-byte codec with real language coherence. Low-signal Western pages fall
    #    through to the cp1252 default rather than a cp1250/cyrillic coin-flip.
    best = charset_normalizer.from_bytes(raw).best()
    if best is not None and (best.multi_byte_usage > 0 or best.coherence >= _COHERENCE_MIN):
        s = str(best)
        if REPLACEMENT not in s:
            return s

    # 6. cp1252 default (correct for the legacy Western pages that dominate the failure
    #    set), then latin-1 which maps all 256 byte values. Neither produces U+FFFD.
    try:
        return raw.decode("cp1252")
    except (UnicodeDecodeError, ValueError):
        return raw.decode("latin-1")


def _decode_one_warc(warc_path: str) -> list[dict]:
    """Download one WARC and emit cleanly-decoded HTML response records.

    Returns ``[{doc_id, warc_hash, url, snapshot, html, text_body}]``. Retries on
    transient HTTP errors; raises on permanent failure (never silently drops a WARC).
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
            parse_errors = 0
            fffd_skips = 0

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

                    payload = record.content_stream().read()
                    html = decode_payload(payload, content_type)
                    if REPLACEMENT in html:
                        # Rule 4: never ship a corrupted row. Should be unreachable given
                        # the latin-1 tail, but assert-by-skip rather than trust it.
                        fffd_skips += 1
                        continue

                    bs = body_strip(html)
                    text_body = fasttext_text(bs)
                    if not text_body:
                        continue

                    records.append(
                        {
                            "doc_id": _normalize_record_id(record.rec_headers.get_header("WARC-Record-ID") or ""),
                            "warc_hash": warc_hash,
                            "url": record.rec_headers.get_header("WARC-Target-URI") or "",
                            "snapshot": snapshot,
                            "html": html,
                            "text_body": text_body,
                        }
                    )
                except Exception as e:
                    parse_errors += 1
                    if parse_errors <= 3:
                        logger.warning(f"Skipping corrupt record in {warc_path}: {e}")

            if fffd_skips:
                logger.warning(f"{warc_path}: skipped {fffd_skips} records that still held U+FFFD after decode")
            if parse_errors:
                logger.warning(f"Skipped {parse_errors} corrupt records in {warc_path}")
            if not records:
                logger.warning(f"WARC yielded 0 clean HTML records: {warc_path}")
            else:
                logger.info(f"Decoded {len(records)} clean HTML pages from {warc_path}")
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


@dataclass
class DecodeConfig:
    manifest_path: str
    output_path: str
    limit: int | None = None
    start: int = 0
    max_workers: int = 256


def decode_warcs(config: DecodeConfig) -> None:
    """Re-decode WARCs from the manifest into per-WARC clean parquet (resumable)."""
    warc_paths = _load_manifest(config.manifest_path)
    warc_paths = warc_paths[config.start :]
    if config.limit is not None:
        warc_paths = warc_paths[: config.limit]
    logger.info(f"Decoding {len(warc_paths)} WARCs -> {config.output_path}")

    def _output_path_fn(shard_idx: int, total_shards: int) -> str:
        return f"{config.output_path}/data-{_warc_path_hash(warc_paths[shard_idx])}.parquet"

    pipeline = (
        Dataset.from_list(warc_paths)
        .reshard(len(warc_paths))  # one WARC per shard -> one output file per WARC
        .flat_map(_decode_one_warc)
        .write_parquet(_output_path_fn, skip_existing=True)
    )
    # Each worker holds a full WARC body (~1-1.3 GB compressed) plus 5-10x decompressed
    # HTML; 24 GiB matches download_warcs.py's headroom for adversarial WARCs.
    ctx = ZephyrContext(
        name="decode-warcs-clean",
        max_workers=config.max_workers,
        resources=ResourceConfig(cpu=1, ram="24g"),
    )
    ctx.execute(pipeline)
    logger.info(f"Decode complete. {len(warc_paths)} WARCs -> {config.output_path}")


def scan_fffd(output_path: str, sample: int | None) -> None:
    """Regression check: count U+FFFD across decoded parquet. A correct run yields 0."""
    import pyarrow.parquet as pq

    fs, root = fsspec.core.url_to_fs(output_path)
    files = sorted(fs.glob(f"{root}/data-*.parquet"))
    if sample is not None:
        files = files[:sample]
    total_docs = 0
    total_bad = 0
    bad_files = 0
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Decode only the first N WARCs after --start.")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N WARCs of the manifest.")
    ap.add_argument("--max-workers", type=int, default=256)
    ap.add_argument("--scan-fffd", action="store_true", help="Regression mode: scan outputs for U+FFFD and exit.")
    ap.add_argument("--scan-sample", type=int, default=None, help="Limit --scan-fffd to the first N parquet files.")
    args = ap.parse_args()

    if args.scan_fffd:
        scan_fffd(args.output_path, args.scan_sample)
        return

    decode_warcs(
        DecodeConfig(
            manifest_path=args.manifest,
            output_path=args.output_path,
            limit=args.limit,
            start=args.start,
            max_workers=args.max_workers,
        )
    )


if __name__ == "__main__":
    main()
