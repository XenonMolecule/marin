# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-WARC Phase A bottleneck benchmark — run as an Iris CPU job, prints a timing table.

The canary measured extraction at ~21 docs/s/proc against the artifact's 292 docs/s/core golden-page
benchmark (14x gap) and decode at ~159s/WARC. This isolates where that time actually goes on a real
worker with a real WARC:

* decode split: download | warcio+charset decode | body_strip/fasttext_text prep
* extraction: serial in-process rate on the REAL pages vs `ResiliparseRsPool` at several task sizes
* the parent->child pickle feed cost in isolation

Launch (matches a Phase A worker's shape)::

    iris job run --region us-east5 --cpu 8 --memory 64GB --disk 24GB --enable-extra-resources \\
        --extra cpu --extra dclm --priority interactive --no-wait --job-name bench-phase-a -- \\
        python -m experiments.fast_curation.bench_phase_a \\
        --manifest experiments/distill/random_subsets/random_warcs_300.txt --warc-index 3
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import time

import requests
import warcio

from experiments.baseline_collection.decode_warcs_clean import (
    _load_manifest,
    _s3_to_https,
    body_strip,
    decode_payload,
    fasttext_text,
)
from experiments.fast_curation.cpu_phase_c import _install_rust_extractor, _size_aware_chunks
from experiments.fast_curation.spec import get_spec

logger = logging.getLogger(__name__)


def timed(label: str, fn):
    t0 = time.monotonic()
    out = fn()
    dt = time.monotonic() - t0
    logger.info("BENCH %-52s %8.2fs", label, dt)
    return out, dt


def decode_split(warc_path: str) -> list[dict]:
    """The `_decode_one_warc` work, instrumented per stage."""
    url = _s3_to_https(warc_path)
    resp, _ = timed("download WARC", lambda: requests.get(url, timeout=180.0))
    resp.raise_for_status()
    logger.info("BENCH warc bytes: %.0f MB", len(resp.content) / 1e6)

    t_iter = t_decode = t_prep = 0.0
    records: list[dict] = []
    stream = io.BytesIO(resp.content)
    t0 = time.monotonic()
    it = warcio.ArchiveIterator(stream)
    while True:
        s = time.monotonic()
        try:
            record = next(it)
        except StopIteration:
            t_iter += time.monotonic() - s
            break
        t_iter += time.monotonic() - s
        if record.rec_type != "response" or record.http_headers is None:
            continue
        content_type = record.http_headers.get_header("Content-Type") or ""
        if "text/html" not in content_type.lower():
            continue
        s = time.monotonic()
        payload = record.content_stream().read()
        html = decode_payload(payload, content_type)
        t_decode += time.monotonic() - s
        s = time.monotonic()
        text_body = fasttext_text(body_strip(html))
        t_prep += time.monotonic() - s
        if not text_body:
            continue
        records.append({"html": html, "doc_id": record.rec_headers.get_header("WARC-Record-ID") or ""})
    logger.info(
        "BENCH decode split: iter=%.1fs charset_decode=%.1fs body_strip_prep=%.1fs total=%.1fs (%d docs, %.0f MB html)",
        t_iter,
        t_decode,
        t_prep,
        time.monotonic() - t0,
        len(records),
        sum(len(r["html"]) for r in records) / 1e6,
    )
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments/distill/random_subsets/random_warcs_300.txt")
    ap.add_argument("--warc-index", type=int, default=3, help="Manifest row to benchmark (avoid canary rows 0-2).")
    ap.add_argument("--spec", default="lpv11_fastpipe_v2")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--serial-docs", type=int, default=4000, help="Docs for the serial in-process rate.")
    ap.add_argument(
        "--resiliparse-artifact",
        default="gs://marin-us-east5/artifacts/resiliparse_rs/latest",
    )
    args = ap.parse_args()

    spec = get_spec(args.spec)
    warc_path = _load_manifest(args.manifest)[args.warc_index]
    records = decode_split(warc_path)
    htmls = [r["html"] for r in records]
    doc_ids = [r["doc_id"] for r in records]
    cap = spec.justext_max_html_chars

    # Parent-side pickle feed cost in isolation (what the pool pays before any child works).
    timed("pickle all html (parent-side feed cost)", lambda: sum(len(pickle.dumps(h)) for h in htmls))

    # Serial in-process extraction on REAL pages (page-hardness vs the 69KB golden-page benchmark).
    pool = _install_rust_extractor(args.resiliparse_artifact, spec, 1)
    sample = htmls[: args.serial_docs]

    def serial():
        # In the POOL child the fork import is set up by the initializer; here run one solo task.
        return pool.run(sample, doc_ids[: len(sample)], cap)

    (texts, crashed), t_serial = timed(f"pool procs=1 (serial-ish) on {len(sample)} docs", serial)
    logger.info(
        "BENCH serial rate: %.1f docs/s/proc (%d empty, %d crashed)",
        len(sample) / t_serial,
        sum(1 for t in texts if not t),
        len(crashed),
    )
    pool.close()

    for task_chunk in (25, 100, 400):
        p = _install_rust_extractor(args.resiliparse_artifact, spec, args.procs)
        p._task_chunk = task_chunk
        (texts, crashed), dt = timed(
            f"pool procs={args.procs} task_chunk={task_chunk} ALL {len(htmls)} docs",
            lambda p=p: p.run(htmls, doc_ids, cap),
        )
        n_chunks = len(_size_aware_chunks(htmls, task_chunk))
        logger.info(
            "BENCH pooled rate: %.1f docs/s total, %.1f docs/s/proc (%d tasks, %d crashed)",
            len(htmls) / dt,
            len(htmls) / dt / args.procs,
            n_chunks,
            len(crashed),
        )
        p.close()


if __name__ == "__main__":
    main()
