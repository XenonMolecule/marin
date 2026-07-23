# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload a finished local index to its canonical GCS home + a manifest sidecar.

Destination is ``IndexTarget.index_dir``
(``gs://marin-{region}/infinigram_indices/{collection}/{dataset}/``), always
in-region. A ``manifest.json`` records what was indexed so a build is idempotent
(skipped if the manifest already exists unless ``overwrite=True``) and so the
query side can discover shard dirs without re-listing.
"""

import json
import logging
import os
import subprocess

import fsspec
from marin.utils import fsspec_exists

from experiments.infinigram.build import BuildResult
from experiments.infinigram.gcs_io import upload_dir
from experiments.infinigram.resolve import ResolvedTarget
from experiments.infinigram.stage import StagedCorpus

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


def _manifest_url(index_dir: str) -> str:
    return f"{index_dir.rstrip('/')}/{MANIFEST_NAME}"


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def upload_index(build: BuildResult, resolved: ResolvedTarget, staged: StagedCorpus, *, overwrite: bool = False) -> str:
    """Copy ``build``'s index dir to the target's canonical GCS location.

    Returns the destination index dir. Raises if it already exists and
    ``overwrite`` is False.
    """
    target = resolved.target
    index_dir = target.index_dir
    manifest_url = _manifest_url(index_dir)

    if fsspec_exists(manifest_url) and not overwrite:
        raise FileExistsError(f"{manifest_url} exists; pass overwrite=True to rebuild {target.name}")

    logger.info("Uploading %s -> %s", build.save_dir, index_dir)
    upload_dir(build.save_dir, index_dir)

    def _shard_url(local_shard_dir: str) -> str:
        rel = os.path.relpath(local_shard_dir, build.save_dir)
        return index_dir.rstrip("/") if rel == "." else f"{index_dir.rstrip('/')}/{rel}"

    manifest = {
        "dataset": target.dataset,
        "collection": target.collection.value,
        "region": target.region,
        "shard_dirs": [_shard_url(d) for d in build.shard_dirs],
        "url_index": f"{index_dir.rstrip('/')}/{os.path.basename(staged.url_index_path)}",
        "num_input_shards": resolved.shard_count,
        "input_bytes": resolved.total_bytes,
        "index_bytes": build.index_bytes,
        "doc_count": staged.doc_count,
        "provenance_joined": bool(target.provenance_globs),
        "provenance_matched": staged.matched_provenance,
        "source_shard_urls_sample": list(resolved.shard_urls[:5]),
        "git_sha": _git_sha(),
    }
    with fsspec.open(manifest_url, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest %s", manifest_url)
    return index_dir
