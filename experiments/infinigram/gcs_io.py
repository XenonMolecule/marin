# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parallel GCS<->local copies via gcsfs.

The index builder runs in the marin container, which has gcsfs (fsspec) but not
the ``gsutil`` CLI, so all staging/upload goes through fsspec. Copies are always
in-region (the job is region-pinned), so throughput is bounded by the node, not
egress.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import fsspec

_MAX_WORKERS = 16


def _gcs():
    return fsspec.filesystem("gcs")


def _run(fn, items: list) -> None:
    if not items:
        return
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for _ in ex.map(fn, items):
            pass


def download_files(urls: list[str], dest_dir: str) -> None:
    """Download gs:// files into ``dest_dir`` (flat, by basename)."""
    os.makedirs(dest_dir, exist_ok=True)
    fs = _gcs()
    _run(lambda u: fs.get_file(u, os.path.join(dest_dir, os.path.basename(u))), urls)


def download_dir(gs_dir: str, dest_dir: str) -> None:
    """Mirror a gs:// directory tree into ``dest_dir``, preserving structure."""
    fs = _gcs()
    base = fs._strip_protocol(gs_dir.rstrip("/"))
    remotes = fs.find(gs_dir)

    def one(remote: str) -> None:
        rel = fs._strip_protocol(remote)[len(base) :].lstrip("/")
        local = os.path.join(dest_dir, rel)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        fs.get_file(remote, local)

    _run(one, remotes)


def upload_dir(local_dir: str, gs_dir: str) -> None:
    """Upload a local directory tree to ``gs_dir``, preserving structure."""
    fs = _gcs()
    local_dir = local_dir.rstrip("/")
    gs_dir = gs_dir.rstrip("/")
    tasks: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(local_dir):
        for f in files:
            local = os.path.join(root, f)
            rel = os.path.relpath(local, local_dir)
            tasks.append((local, f"{gs_dir}/{rel}"))
    _run(lambda t: fs.put_file(t[0], t[1]), tasks)
