# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the 1934-doc gold devset to GCS as a tarball (one-time, ~283MB).

run_devset_pipeline.py downloads + extracts this on a TPU node, so the parity
benchmark needs no sibling-repo filesystem. Run once from a machine that has the
devset locally (the small-rephraser static tree).

Usage::

    python -m experiments.baseline_collection.stage_devset \
        --devset-dir /path/to/small-rephraser/static/warcs/marin_devset_1934_html \
        --out-gcs gs://marin-us-central1/devset/marin_devset_1934.tar.gz
"""

from __future__ import annotations

import argparse
import logging
import os
import tarfile
import tempfile

import fsspec

logger = logging.getLogger(__name__)

TARBALL_TOP_DIR = "marin_devset_1934_html"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset-dir", required=True, help="Local devset dir (flat record_*.html tree + gold/)")
    parser.add_argument("--out-gcs", required=True, help="Target GCS path for the tarball (gs://.../*.tar.gz)")
    args = parser.parse_args()

    n_meta = sum(1 for f in os.listdir(args.devset_dir) if f.endswith(".html.meta.json"))
    if n_meta == 0:
        raise FileNotFoundError(f"No *.html.meta.json under {args.devset_dir} — not a devset dir")
    logger.info("Staging %d devset docs from %s", n_meta, args.devset_dir)

    with tempfile.TemporaryDirectory() as tmp:
        local_tar = os.path.join(tmp, "devset.tar.gz")
        with tarfile.open(local_tar, "w:gz") as tar:
            tar.add(args.devset_dir, arcname=TARBALL_TOP_DIR)
        size_mb = os.path.getsize(local_tar) / 1e6
        logger.info("Built tarball (%.0f MB); uploading to %s", size_mb, args.out_gcs)
        with open(local_tar, "rb") as src, fsspec.open(args.out_gcs, "wb") as dst:
            dst.write(src.read())
    logger.info("Staged devset -> %s", args.out_gcs)


if __name__ == "__main__":
    main()
