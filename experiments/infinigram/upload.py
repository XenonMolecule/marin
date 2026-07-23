# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest for a finished index at its canonical GCS home.

Destination is ``IndexTarget.index_dir``
(``gs://marin-{region}/infinigram_indices/{collection}/{dataset}/``), always
in-region. ``manifest.json`` records what was indexed (shard dirs, url-index side
table, provenance/efficiency stats) so a build is idempotent and the query side
can discover shard dirs without re-listing. The actual index bytes are uploaded
per-chunk by the pipeline (bounded disk); this module writes the sidecar.
"""

import json
import logging
import subprocess
from dataclasses import dataclass

import fsspec

from experiments.infinigram.resolve import ResolvedTarget

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class IndexStats:
    """Aggregated build stats across all chunks of one index."""

    doc_count: int = 0
    provenance_matched: int = 0
    index_bytes: int = 0


def write_manifest(
    index_dir: str,
    resolved: ResolvedTarget,
    *,
    shard_dir_urls: list[str],
    url_index_url: str,
    stats: IndexStats,
    num_chunks: int,
) -> None:
    """Write ``manifest.json`` for a fully-uploaded index."""
    target = resolved.target
    manifest = {
        "dataset": target.dataset,
        "collection": target.collection.value,
        "region": target.region,
        "shard_dirs": shard_dir_urls,
        "num_chunks": num_chunks,
        "url_index": url_index_url,
        "num_input_shards": resolved.shard_count,
        "input_bytes": resolved.total_bytes,
        "index_bytes": stats.index_bytes,
        "index_ratio": round(stats.index_bytes / resolved.total_bytes, 4) if resolved.total_bytes else None,
        "doc_count": stats.doc_count,
        "provenance_joined": bool(target.provenance_globs),
        "provenance_matched": stats.provenance_matched,
        "source_shard_urls_sample": list(resolved.shard_urls[:5]),
        "git_sha": _git_sha(),
    }
    with fsspec.open(manifest_url(index_dir), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "Wrote manifest %s (%d shard dirs, index_ratio=%s)", index_dir, len(shard_dir_urls), manifest["index_ratio"]
    )
