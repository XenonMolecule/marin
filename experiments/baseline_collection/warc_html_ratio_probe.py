# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick HTML-vs-text length probe: is the long tail of hq-dropped docs context-limited?

The extracted-text probe showed the 3/3-dropped docs are mostly short (median ~1.5k Qwen3
tokens, 0.3% >32k) → hq dropped them by spec, not the 32k context. Open question: the ~10%
tail above 8k EXTRACTED tokens — does the raw HTML the extractor ingests cross 32k there?

Rather than re-fetch all 7.99 TB of WARCs, sample ~24 WARCs (stride over the manifest),
decode each with the clean decoder, and tokenize BOTH the raw HTML and the extracted text
with the Qwen3 tokenizer. That gives the HTML:text token ratio and, directly, the fraction
of long-text docs whose HTML exceeds 32k — enough to settle the tail.

WARCs fetched from CommonCrawl S3 (free ingress); tokenizer staged in-region.
"""

from __future__ import annotations

import io
import os
import sys
import time

import fsspec
import requests
import warcio
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.decode_warcs_clean import (
    HTTP_TIMEOUT,
    MAX_RETRIES,
    REPLACEMENT,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
    _load_manifest,
    _s3_to_https,
    body_strip,
    decode_payload,
    fasttext_text,
)
from experiments.baseline_collection.provenance_audit_10k import registered_domain

MANIFEST = "experiments/distill/dclm_400m_1x.txt"
TOK_GCS = "gs://marin-us-central2/scratch/provenance_10k/qwen3_tok"
OUT = "gs://marin-us-central2/scratch/provenance_10k/html_ratio/r-{shard:05d}-of-{total:05d}.parquet"
STRIDE = 420  # ~24 WARCs spread across snapshots
DOC_CAP = 2500  # HTML docs to sample per WARC (raw WARCs hold ~30-50k records; a sample is plenty for the ratio)
TOK_CAP = 65536  # truncate tokenization here — we only need to know if HTML crosses 32k
MAX_TOK_CHARS = 1_000_000  # char guard for pathological pages (>>32k tokens either way)

_TOK = None


def _tok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        local = "/tmp/qwen3_tok"
        if not os.path.exists(os.path.join(local, "tokenizer.json")):
            os.makedirs(local, exist_ok=True)
            fs = fsspec.filesystem("gcs")
            for p in fs.ls(TOK_GCS.removeprefix("gs://")):
                nm = p.rsplit("/", 1)[-1]
                if nm:
                    fs.get(p, os.path.join(local, nm))
        _TOK = AutoTokenizer.from_pretrained(local)
    return _TOK


def _ntok(s: str) -> int:
    return len(_tok()(s[:MAX_TOK_CHARS], add_special_tokens=False, truncation=True, max_length=TOK_CAP)["input_ids"])


def probe_warc(warc_path: str) -> list[dict]:
    https = _s3_to_https(warc_path)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(https, stream=True, timeout=HTTP_TIMEOUT)
            if resp.status_code in RETRYABLE_STATUS_CODES:
                time.sleep(RETRY_BASE_DELAY * (2**attempt))
                continue
            resp.raise_for_status()
            out: list[dict] = []
            for record in warcio.ArchiveIterator(io.BytesIO(resp.content)):
                if len(out) >= DOC_CAP:
                    break
                if record.rec_type != "response":
                    continue
                hh = record.http_headers
                if hh is None:
                    continue
                ct = hh.get_header("Content-Type") or ""
                if "text/html" not in ct.lower():
                    continue
                try:
                    html = decode_payload(record.content_stream().read(), ct)
                    if REPLACEMENT in html:
                        continue
                    text = fasttext_text(body_strip(html))
                    if not text:
                        continue
                    url = record.rec_headers.get_header("WARC-Target-URI") or ""
                    out.append(
                        {
                            "url": url,
                            "domain": registered_domain(url),
                            "html_tokens": _ntok(html),
                            "text_tokens": _ntok(text),
                        }
                    )
                except Exception:
                    continue
            return out
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    return []


def main() -> int:
    sample = _load_manifest(MANIFEST)[::STRIDE]
    print(f"[html-ratio] {len(sample)} WARCs sampled (stride {STRIDE})")
    pipeline = Dataset.from_list(sample).reshard(len(sample)).flat_map(probe_warc).write_parquet(OUT, skip_existing=True)
    ctx = ZephyrContext(
        name="html-ratio",
        max_workers=len(sample),
        resources=ResourceConfig(cpu=1, ram="24g", regions=["us-central2"], preemptible=True),
    )
    ctx.execute(pipeline)
    print("[html-ratio] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
