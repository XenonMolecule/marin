# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a small RANDOM seed dataset for CurationArena.

Samples a few WARCs, decodes their raw HTML, takes a random URL sample (unbiased — NOT
verifier-filtered), and joins each pipeline's extraction-or-dropped status for every URL.
Emits arena_seed.json: [{url, domain, html, extractions:{pipeline:{status,text}}}]. A
pipeline is "dropped" (== FILTERED / NO_USEFUL_CONTENT / CONTEXT_LENGTH_EXCEEDED, all
equivalent) when the URL is absent from its kept set. In-region except a tiny hq read.

Stage 1 (zephyr): fetch+decode K WARCs -> per-URL {url, domain, html} parquet.
Stage 2 (duckdb): left-join the 5 pipelines' text by URL -> assemble JSON.
"""

from __future__ import annotations

import io
import json
import sys
import time

import fsspec
import requests
import warcio
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.decode_warcs_clean import (
    HTTP_TIMEOUT,
    MAX_RETRIES,
    REPLACEMENT,
    RETRY_BASE_DELAY,
    RETRYABLE_STATUS_CODES,
    _load_manifest,
    _s3_to_https,
    decode_payload,
)
from experiments.baseline_collection.provenance_audit_10k import SOURCES, registered_domain

MANIFEST = "experiments/distill/dclm_400m_1x.txt"
WORKSPACE = "gs://marin-us-central2/scratch/curation_arena"
HTML_PARQUET = f"{WORKSPACE}/warc_html/h-{{shard:05d}}-of-{{total:05d}}.parquet"
OUT_JSON = f"{WORKSPACE}/arena_seed.json"
N_WARCS = 10  # random WARCs to fetch
DOCS_PER_WARC = 40  # random HTML pages to keep per WARC
MAX_HTML_CHARS = 300_000  # cap stored HTML so the seed stays small
PIPELINES = ("hq", "dclm", "nemo", "fwedu", "resiliparse")


def fetch_warc(warc_path: str) -> list[dict]:
    """Decode one WARC -> up to DOCS_PER_WARC {url, domain, html} records."""
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
                if len(out) >= DOCS_PER_WARC:
                    break
                if record.rec_type != "response":
                    continue
                hh = record.http_headers
                if hh is None or "text/html" not in (hh.get_header("Content-Type") or "").lower():
                    continue
                try:
                    html = decode_payload(record.content_stream().read(), hh.get_header("Content-Type") or "")
                    if REPLACEMENT in html or len(html) < 500:
                        continue
                    url = record.rec_headers.get_header("WARC-Target-URI") or ""
                    if not url:
                        continue
                    out.append({"url": url, "domain": registered_domain(url), "html": html[:MAX_HTML_CHARS]})
                except Exception:
                    continue
            return out
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_DELAY * (2**attempt))
    return []


def _read_pipeline(con, method: str) -> str:
    """A SQL sub-select of (url, text) for one pipeline's kept docs."""
    src = SOURCES[method]
    glob = f"{src['root']}/{src['glob']}"
    if src["format"] == "parquet":
        return f"SELECT url, text FROM read_parquet('{glob}')"
    return f"SELECT url, text FROM read_json_auto('{glob}', format='newline_delimited', ignore_errors=true)"


def main() -> int:
    import duckdb

    warcs = _load_manifest(MANIFEST)
    # deterministic spread across snapshots (no RNG): evenly-spaced picks
    step = max(1, len(warcs) // N_WARCS)
    sample = warcs[::step][:N_WARCS]
    print(f"[arena] fetching {len(sample)} WARCs for HTML")

    ctx = ZephyrContext(
        name="arena-warc-html",
        max_workers=len(sample),
        resources=ResourceConfig(cpu=1, ram="24g", regions=["us-central2"], preemptible=True),
    )
    pipeline = (
        Dataset.from_list(sample)
        .reshard(len(sample))
        .flat_map(fetch_warc)
        .write_parquet(HTML_PARQUET, skip_existing=True)
    )
    ctx.execute(pipeline)

    con = duckdb.connect()
    con.register_filesystem(fsspec.filesystem("gcs"))
    con.execute("SET preserve_insertion_order=false;")
    con.execute(
        f"CREATE TEMP TABLE pages AS SELECT DISTINCT ON (url) url, domain, html FROM read_parquet('{WORKSPACE}/warc_html/*.parquet')"
    )
    n_pages = con.execute("SELECT count(*) FROM pages").fetchone()[0]
    print(f"[arena] {n_pages} unique HTML pages sampled")
    # join each pipeline's text (only for our sampled urls — pushed via the semi-join)
    for m in PIPELINES:
        con.execute(
            f"CREATE TEMP TABLE t_{m} AS SELECT p.url, p.text FROM ({_read_pipeline(con, m)}) p SEMI JOIN pages g ON p.url = g.url"
        )
        kept = con.execute(f"SELECT count(*) FROM t_{m}").fetchone()[0]
        print(f"[arena]   {m}: {kept}/{n_pages} kept")

    rows = con.execute("SELECT url, domain, html FROM pages").fetchall()
    texts = {m: dict(con.execute(f"SELECT url, text FROM t_{m}").fetchall()) for m in PIPELINES}
    items = []
    for url, domain, html in rows:
        ext = {}
        for m in PIPELINES:
            txt = texts[m].get(url)
            ext[m] = {"status": "ok", "text": txt} if txt else {"status": "dropped", "text": None}
        # keep only pages where >=2 pipelines produced text (a real contest)
        if sum(1 for m in PIPELINES if ext[m]["status"] == "ok") >= 2:
            items.append({"url": url, "domain": domain, "html": html, "extractions": ext})
    print(f"[arena] {len(items)} seed items (>=2 pipelines produced text)")
    with fsspec.open(OUT_JSON, "w") as f:
        json.dump(items, f)
    print(f"[arena] wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
