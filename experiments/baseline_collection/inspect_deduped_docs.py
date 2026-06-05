# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect post-dedup docs that fed the tokenizer.

For a given ``(spec, n)`` whose dedup pipeline finished, reservoir-sample docs
from ``documents/baseline_{spec}_deduped/{n}warcs/deduped/data-*-of-*.jsonl.gz``
and render them as an HTML page. Each shard is a gzipped jsonl of
``{"text": "..."}`` records — exactly what gets handed to the Llama tokenizer
downstream.

The sampler does a reservoir pass over the whole population so every doc has
equal probability of being shown. Streaming is in-region (read on a us-central1
cluster job), so no cross-region egress.

Usage (run on cluster to avoid cross-region read)::

    iris --config lib/iris/examples/marin.yaml job run --priority interactive \\
        --memory 4GB --cpu 4 --enable-extra-resources \\
        --job-name inspect-deduped-high_quality \\
        -- python experiments/baseline_collection/inspect_deduped_docs.py \\
            --spec high_quality --n 100 --num-docs 60 \\
            --output gs://marin-us-central1/scratch/viewers/deduped_high_quality_100.html
"""

from __future__ import annotations

import argparse
import gzip
import html
import io
import json
import logging
import random
from pathlib import Path

from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)


def _list_shards(spec: str, n: int) -> list[str]:
    client = gcs_storage.Client()
    bucket = "marin-us-central1"
    prefix = f"documents/baseline_{spec}_deduped/{n}warcs/deduped/"
    blobs = client.bucket(bucket).list_blobs(prefix=prefix)
    return sorted(f"gs://{bucket}/{b.name}" for b in blobs if b.name.endswith(".jsonl.gz"))


def _stream_records(uri: str):
    """Yield each JSON record from a gs:// gzipped jsonl shard."""
    bucket_name, _, blob_name = uri[len("gs://") :].partition("/")
    blob = gcs_storage.Client().bucket(bucket_name).blob(blob_name)
    raw = blob.download_as_bytes()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _reservoir_sample(spec: str, n: int, sample_size: int, seed: int) -> tuple[list[dict], int]:
    """Reservoir-sample `sample_size` records uniformly over the full deduped set.

    Returns (samples, total_seen).
    """
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0
    for shard in _list_shards(spec, n):
        logger.info("scanning %s", shard)
        for rec in _stream_records(shard):
            total += 1
            if len(reservoir) < sample_size:
                reservoir.append(rec)
            else:
                j = rng.randrange(total)
                if j < sample_size:
                    reservoir[j] = rec
    return reservoir, total


def render_html(samples: list[dict], title: str, total_docs: int) -> str:
    """Render samples as a dark-themed HTML page."""
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
        "<style>",
        "body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background:#1a1a2e; color:#e0e0e0; }",
        "header { padding: 12px 16px; border-bottom: 1px solid #333; background: #16213e; position: sticky; top: 0; }",
        "h1 { font-size: 16px; color: #4ecca3; margin: 0; font-weight: normal; }",
        ".stats { font-size: 12px; color: #888; margin-top: 4px; }",
        ".doc { margin: 16px; padding: 12px; border: 1px solid #2a2a3e; background: #0f1424; border-radius: 4px; }",
        ".doc h2 { font-size: 13px; color: #4ecca3; margin: 0 0 8px; font-weight: normal; }",
        ".doc-text { font-family: 'SF Mono', Consolas, monospace; font-size: 12px; white-space: pre-wrap; max-height: 24em; overflow-y: auto; line-height: 1.4; padding: 8px; background: #16213e; border-radius: 3px; }",
        ".meta { font-size: 11px; color: #888; margin-bottom: 6px; }",
        "</style></head><body>",
        f"<header><h1>{html.escape(title)}</h1>",
        f"<div class='stats'>{total_docs:,} docs in deduped pool · showing {len(samples)} random sample"
        f"{'s' if len(samples) != 1 else ''}</div></header>",
    ]
    for i, rec in enumerate(samples):
        text = rec.get("text", "")
        chars = len(text)
        parts.append(
            f"<div class='doc'><h2>sample #{i + 1}</h2>"
            f"<div class='meta'>{chars:,} chars</div>"
            f"<div class='doc-text'>{html.escape(text[:8000])}"
            f"{'… [truncated, ' + str(chars - 8000) + ' chars more]' if chars > 8000 else ''}</div>"
            f"</div>"
        )
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True, help="Quality spec name, e.g. high_quality")
    p.add_argument("--n", type=int, required=True, help="WARC count, e.g. 100")
    p.add_argument("--num-docs", type=int, default=60, help="Number of docs to sample (default 60)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        default=None,
        help="HTML output path. gs:// for upload, else local. Default scratch/deduped_<spec>_<n>.html",
    )
    args = p.parse_args()

    logger.info("Reservoir-sampling %d docs from %s n=%d", args.num_docs, args.spec, args.n)
    samples, total = _reservoir_sample(args.spec, args.n, args.num_docs, args.seed)
    logger.info("Sampled %d/%d docs", len(samples), total)

    title = f"Post-dedup docs · {args.spec} · n={args.n}"
    html_doc = render_html(samples, title, total)

    out_path = args.output or f"scratch/deduped_{args.spec}_{args.n}.html"
    if out_path.startswith("gs://"):
        bn, _, bp = out_path[len("gs://") :].partition("/")
        gcs_storage.Client().bucket(bn).blob(bp).upload_from_string(html_doc, content_type="text/html")
        logger.info("Rendered → %s (uploaded to GCS)", out_path)
    else:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(html_doc)
        logger.info("Rendered → %s", out_path)


if __name__ == "__main__":
    main()
