# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS staging helpers shared by the devset harness and scorer.

The devset is staged once as a tarball (stage_devset.py); parallel TPU workers
fetch+extract it locally and write per-doc records into ONE shared GCS run dir.
Scoring downloads that full run dir and reads it with the local loader.
"""

from __future__ import annotations

import logging
import os
import tarfile

import fsspec

logger = logging.getLogger(__name__)


def fetch_devset_dir(src: str, work_dir: str) -> str:
    """Return a local devset dir. If `src` is a gs:// tarball, download+extract it;
    if it is already a local dir, return it unchanged."""
    if not src.startswith("gs://"):
        return src
    logger.info("Fetching staged devset %s", src)
    local_tar = os.path.join(work_dir, "devset.tar.gz")
    with fsspec.open(src, "rb") as s, open(local_tar, "wb") as d:
        d.write(s.read())
    with tarfile.open(local_tar, "r:gz") as t:
        t.extractall(work_dir)
    for root, _dirs, files in os.walk(work_dir):
        if any(f.endswith(".html.meta.json") for f in files):
            return root
    raise FileNotFoundError(f"No devset (*.html.meta.json) found under {work_dir}")


def gcs_done_record_ids(out_gcs: str) -> set[str]:
    """Record ids already present in a shared GCS run dir (for skip-existing)."""
    fs = fsspec.filesystem("gcs")
    base = out_gcs.rstrip("/").replace("gs://", "")
    try:
        paths = fs.glob(f"{base}/records/*.json")
    except Exception:
        return set()
    return {os.path.basename(p)[: -len(".json")] for p in paths}


def upload_record(out_gcs: str, record_id: str, local_path: str) -> None:
    """Upload one record json into the shared GCS run dir's records/."""
    fs = fsspec.filesystem("gcs")
    dst = f"{out_gcs.rstrip('/')}/records/{record_id}.json".replace("gs://", "")
    with open(local_path, "rb") as src, fs.open(dst, "wb") as out:
        out.write(src.read())


def download_run_dir(run_gcs: str, dest: str) -> str:
    """Download a shared GCS run dir (manifest.json + records/*.json) to `dest`."""
    fs = fsspec.filesystem("gcs")
    base = run_gcs.rstrip("/").replace("gs://", "")
    os.makedirs(os.path.join(dest, "records"), exist_ok=True)
    try:
        with fs.open(f"{base}/manifest.json", "rb") as src, open(os.path.join(dest, "manifest.json"), "wb") as d:
            d.write(src.read())
    except Exception:
        pass
    for p in fs.glob(f"{base}/records/*.json"):
        with fs.open(p, "rb") as src, open(os.path.join(dest, "records", os.path.basename(p)), "wb") as d:
            d.write(src.read())
    return dest
