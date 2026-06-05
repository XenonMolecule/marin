# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect fuzzy-dup clusters produced by dedup_extracted.py.

For a given ``(spec, n)`` whose dedup pipeline finished, group the per-doc
``dup_cluster_id`` markers from fuzzy/outputs/source_000/*.parquet and join
back to the source text in normalize/outputs/main/*.parquet. Render a few
sampled clusters as an HTML page so we can SEE what a Jaccard-0.75 cluster
actually looks like in our data.

Sampling: a few largest clusters (most aggressive dedup), a few small-but-
multi-member clusters (typical case), and a few clusters near the canonical-
flag boundary (one canonical + 1 dup, to show the just-barely matches).

Usage::

    uv run python experiments/baseline_collection/inspect_fuzzy_clusters.py \\
        --spec low_quality --n 100 \\
        --num-largest 5 --num-medium 5 --num-small 10 \\
        --output scratch/clusters_low_quality_100.html
"""

from __future__ import annotations

import argparse
import html
import logging
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from google.cloud import storage as gcs_storage

logger = logging.getLogger(__name__)


def _list_blobs(bucket: str, prefix: str) -> list[str]:
    client = gcs_storage.Client()
    return [
        f"gs://{bucket}/{b.name}" for b in client.bucket(bucket).list_blobs(prefix=prefix) if b.name.endswith(".parquet")
    ]


def _read_parquet(uri: str, columns: list[str] | None = None) -> list[dict]:
    """Read a parquet file from gs://... into a list of dicts. Streams via pyarrow filesystem."""
    table = pq.read_table(uri, columns=columns)
    return table.to_pylist()


def _gather_clusters(spec: str, n: int) -> dict[str, list[dict]]:
    """Read fuzzy attr parquet shards; return {cluster_id: [{id, is_canonical}, ...]}."""
    base = f"baseline_{spec}_deduped/{n}warcs/fuzzy/outputs"
    bucket = "marin-us-central1"
    prefix = f"documents/{base}"
    blobs = _list_blobs(bucket, prefix)
    if not blobs:
        raise FileNotFoundError(f"no parquet under gs://{bucket}/{prefix}/")

    clusters: dict[str, list[dict]] = defaultdict(list)
    for path in blobs:
        for row in _read_parquet(path, columns=["id", "attributes"]):
            attrs = row["attributes"]
            clusters[attrs["dup_cluster_id"]].append({"id": row["id"], "is_canonical": attrs["is_cluster_canonical"]})
    return clusters


def _gather_text_lookup(spec: str, n: int, needed_ids: set[str]) -> dict[str, str]:
    """For a set of ids, return {id: text} by scanning normalize parquet shards."""
    base = f"baseline_{spec}_deduped/{n}warcs/normalize/outputs/main"
    bucket = "marin-us-central1"
    prefix = f"documents/{base}"
    blobs = _list_blobs(bucket, prefix)
    out: dict[str, str] = {}
    remaining = set(needed_ids)
    for path in blobs:
        if not remaining:
            break
        for row in _read_parquet(path, columns=["id", "text"]):
            if row["id"] in remaining:
                out[row["id"]] = row["text"]
                remaining.discard(row["id"])
                if not remaining:
                    break
    return out


def _pick_samples(
    clusters: dict[str, list[dict]],
    num_largest: int,
    num_medium: int,
    num_small: int,
) -> list[tuple[str, list[dict]]]:
    """Return ordered list of (cluster_id, members) to render."""
    sized = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)
    largest = sized[:num_largest]
    medium_pool = [c for c in sized if 5 <= len(c[1]) <= 20]
    medium = medium_pool[:num_medium] if medium_pool else []
    small = (
        [c for c in sized if len(c[1]) == 2][-num_small:]
        if any(len(c[1]) == 2 for _, c in zip([0], clusters.values()))
        else []
    )
    # Dedup by cluster_id
    seen = set()
    out: list[tuple[str, list[dict]]] = []
    for src, label in [(largest, "largest"), (medium, "medium"), (small, "size-2")]:
        for cid, members in src:
            if cid in seen:
                continue
            seen.add(cid)
            out.append((cid, members))
    return out


def render_html(
    samples: list[tuple[str, list[dict]]],
    text_lookup: dict[str, str],
    title: str,
    total_clusters: int,
    total_members: int,
) -> str:
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>",
        "<style>",
        "body { margin: 0; font-family: -apple-system, system-ui, sans-serif; background:#1a1a2e; color:#e0e0e0; }",
        "header { padding: 12px 16px; border-bottom: 1px solid #333; background: #16213e; }",
        "h1 { font-size: 16px; color: #4ecca3; margin: 0; font-weight: normal; }",
        ".stats { font-size: 12px; color: #888; margin-top: 4px; }",
        ".cluster { margin: 16px; padding: 12px; border: 1px solid #2a2a3e; background: #0f1424; border-radius: 4px; }",
        ".cluster h2 { font-size: 13px; color: #4ecca3; margin: 0 0 8px; font-weight: normal; }",
        ".doc { padding: 8px; margin: 8px 0; border-left: 3px solid #2a2a3e; background: #16213e; }",
        ".doc.canonical { border-left-color: #4ecca3; }",
        ".doc-meta { font-size: 11px; color: #888; margin-bottom: 4px; }",
        ".doc-text { font-family: 'SF Mono', Consolas, monospace; font-size: 12px; white-space: pre-wrap; max-height: 12em; overflow-y: auto; line-height: 1.4; }",
        ".badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10px; margin-right:6px; }",
        ".badge.canonical { background:#4ecca3; color:#0f1424; }",
        ".badge.dup { background:#555; color:#ccc; }",
        "</style></head><body>",
        f"<header><h1>{html.escape(title)}</h1>",
        f"<div class='stats'>{total_clusters:,} multi-member clusters, {total_members:,} total members. "
        f"Showing {len(samples)} sampled clusters below.</div></header>",
    ]
    for cid, members in samples:
        canonical_id = next((m["id"] for m in members if m["is_canonical"]), None)
        parts.append(
            f"<div class='cluster'><h2>cluster {html.escape(cid[:16])}… · {len(members)} docs · canonical: {html.escape(str(canonical_id)[:16])}…</h2>"
        )
        # Order: canonical first, then duplicates
        members_sorted = sorted(members, key=lambda m: (not m["is_canonical"], m["id"]))
        for m in members_sorted:
            text = text_lookup.get(m["id"], "<no text — id missing from normalize parquet>")
            badge_cls = "canonical" if m["is_canonical"] else "dup"
            badge_label = "CANONICAL (kept)" if m["is_canonical"] else "DUP (dropped)"
            parts.append(
                f"<div class='doc {'canonical' if m['is_canonical'] else ''}'>"
                f"<div class='doc-meta'><span class='badge {badge_cls}'>{badge_label}</span>id: {html.escape(m['id'])}</div>"
                f"<div class='doc-text'>{html.escape(text[:4000])}{'… [truncated]' if len(text) > 4000 else ''}</div>"
                f"</div>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--num-largest", type=int, default=5)
    p.add_argument("--num-medium", type=int, default=5)
    p.add_argument("--num-small", type=int, default=10)
    p.add_argument("--output", default=None, help="HTML output path (default scratch/clusters_<spec>_<n>.html)")
    args = p.parse_args()

    logger.info("Gathering clusters for spec=%s n=%d", args.spec, args.n)
    clusters = _gather_clusters(args.spec, args.n)
    total_clusters = len(clusters)
    total_members = sum(len(v) for v in clusters.values())
    logger.info("Found %d clusters with %d members total.", total_clusters, total_members)

    samples = _pick_samples(clusters, args.num_largest, args.num_medium, args.num_small)
    needed_ids = {m["id"] for _, members in samples for m in members}
    logger.info("Sampled %d clusters, fetching text for %d ids", len(samples), len(needed_ids))

    text_lookup = _gather_text_lookup(args.spec, args.n, needed_ids)
    logger.info("Got %d/%d texts", len(text_lookup), len(needed_ids))

    title = f"Fuzzy clusters · {args.spec} · n={args.n}"
    html_doc = render_html(samples, text_lookup, title, total_clusters, total_members)

    out_path = args.output or f"scratch/clusters_{args.spec}_{args.n}.html"
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
