#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Upload the high_quality 10364-WARC HF export to the Hugging Face Hub.

Streams the joined parquet shards straight from the us-central1 bucket to HF
(via marin's `_actually_upload_to_hf`), so this MUST run as an in-region
us-central1 Iris job — otherwise the 59 GB egresses twice (GCS -> here -> HF).

Repo layout:
  *.parquet         the dataset shards (HF auto-loads them as the `train` split,
                    made explicit by the README's `configs:` block)
  README.md         the dataset card (ODC-BY)
  source_warcs.txt  the 10,364 source WARC paths
  REPORT.json       the build/validation report

Launch (in-region, after `--dry-run` looks right)::

    uv run iris --cluster marin job run \\
        --region us-central1 \\
        --cpu 4 --memory 16GB --disk 80GB \\
        --priority interactive --no-wait \\
        --extra cpu --enable-extra-resources \\
        --job-name hq-hf-upload \\
        -e HF_TOKEN <write-scoped-token> \\
        -- python experiments/baseline_collection/upload_high_quality_to_hf.py \\
           --repo-id <org>/<name> [--private]

Dry-run locally (lists files + sizes + egress estimate, no token, no push)::

    .venv/bin/python experiments/baseline_collection/upload_high_quality_to_hf.py \\
        --repo-id placeholder/placeholder --dry-run
"""

from __future__ import annotations

import argparse
import io
import logging
import os

from huggingface_hub import CommitOperationAdd
from marin.export.hf_upload import _wrap_in_buffered_base, retrying_create_commit
from rigging.filesystem import url_to_fs

COMMIT_BATCH_BYTES = 1 << 30  # 1 GiB of shards per HF commit

logger = logging.getLogger(__name__)

BUCKET = "gs://marin-us-central1"
EXPORT_PREFIX = f"{BUCKET}/documents/baseline_high_quality_hf_export/10364warcs"
JOINED_PREFIX = f"{EXPORT_PREFIX}/joined"
LOCAL_README = os.path.join(os.path.dirname(__file__), "hq_hf_export_README.md")
INTERNET_EGRESS_USD_PER_GB = 0.12

# Small sibling files (in GCS) copied to the repo root alongside the card.
AUX_FILES = {
    "source_warcs.txt": f"{EXPORT_PREFIX}/source_warcs.txt",
    "REPORT.json": f"{EXPORT_PREFIX}/REPORT.json",
}


def _gcs_bytes(url: str) -> bytes:
    fs, path = url_to_fs(url)
    with fs.open(path, "rb") as f:
        return f.read()


def _joined_size_bytes() -> tuple[int, int]:
    fs, path = url_to_fs(JOINED_PREFIX)
    shards = fs.glob(f"{path}/*.parquet")
    return len(shards), sum(fs.size(s) for s in shards)


def dry_run() -> None:
    n_shards, size_bytes = _joined_size_bytes()
    size_gb = size_bytes / 1e9
    logger.info("parquet shards: %d", n_shards)
    logger.info("data size: %.2f GB", size_gb)
    logger.info(
        "est. internet egress to HF (@ $%.2f/GB): ~$%.2f",
        INTERNET_EGRESS_USD_PER_GB,
        size_gb * INTERNET_EGRESS_USD_PER_GB,
    )
    logger.info("would also upload: README.md (%s) + %s", LOCAL_README, ", ".join(AUX_FILES))
    assert os.path.exists(LOCAL_README), f"missing dataset card: {LOCAL_README}"
    logger.info("DRY RUN ok — nothing uploaded.")


def upload(repo_id: str, token: str, private: bool) -> None:
    from huggingface_hub.hf_api import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()
    existing: set[str] = set()
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
        existing = {s.rfilename for s in info.siblings}
        logger.info("repo %s exists with %d files already", repo_id, len(existing))
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type="dataset", token=token, private=private)
        logger.info("created repo %s (private=%s)", repo_id, private)

    # Card + small sibling files at the repo root (idempotent re-put, cheap), so the
    # dataset page renders even while the large parquet commits stream in.
    with open(LOCAL_README, "rb") as f:
        api.upload_file(path_or_fileobj=f, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset", token=token)
    for name, url in AUX_FILES.items():
        api.upload_file(
            path_or_fileobj=io.BytesIO(_gcs_bytes(url)),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    logger.info("uploaded README.md + %d aux files", len(AUX_FILES))

    # Stream parquet shards GCS -> HF in ~1 GiB commits, SKIPPING shards already in
    # the repo. This makes the job resumable: a preemption restart re-reads only the
    # remaining shards instead of all 59 GB.
    fs, jpath = url_to_fs(JOINED_PREFIX)
    shards = sorted(fs.glob(f"{jpath}/*.parquet"))
    todo = [s for s in shards if os.path.basename(s) not in existing]
    logger.info("parquet: %d total, %d already uploaded, %d to upload", len(shards), len(shards) - len(todo), len(todo))

    batch: list = []
    batch_bytes = 0

    def flush() -> None:
        nonlocal batch, batch_bytes
        if not batch:
            return
        retrying_create_commit(
            repo_id,
            operations=batch,
            commit_message=f"Add {len(batch)} parquet shards",
            token=token,
            repo_type="dataset",
        )
        logger.info("committed %d shards (%.2f GB)", len(batch), batch_bytes / 1e9)
        batch = []
        batch_bytes = 0

    for s in todo:
        fileobj = fs.open(s, "rb")
        if not isinstance(fileobj, io.BufferedIOBase):
            fileobj = _wrap_in_buffered_base(fileobj)
        batch.append(CommitOperationAdd(path_in_repo=os.path.basename(s), path_or_fileobj=fileobj))
        batch_bytes += fs.size(s)
        if batch_bytes > COMMIT_BATCH_BYTES:
            flush()
    flush()
    logger.info("DONE -> https://huggingface.co/datasets/%s", repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Target HF dataset repo, e.g. org/name")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF write token (default: $HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="Create the repo private.")
    parser.add_argument("--dry-run", action="store_true", help="List files + egress estimate; do not upload.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dry_run:
        dry_run()
        return
    if not args.token:
        raise SystemExit("no HF token: pass --token or set HF_TOKEN (must be write-scoped for the target org)")
    upload(args.repo_id, args.token, args.private)


if __name__ == "__main__":
    main()
