# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a re-downloaded 10k WARC pool against the saved manifest.

Use after re-running ``pipeline_10k.py`` (or any pipeline that drops
``data-<hash>.jsonl.gz`` files into a new pool) to confirm 1:1 byte equivalence
with the original.

Compares each manifest entry against the re-downloaded object on size + crc32c.
crc32c is GCS's native checksum so this is genuine byte-level verification.

Usage::

    uv run python experiments/baseline_collection/verify_10k_manifest.py \\
        --new-pool gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-XXXXXX/

The manifest is read from
``experiments/baseline_collection/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz``
by default (with a GCS fallback to ``gs://marin-us-central2/manifests/...``).
"""

import argparse
import csv
import gzip
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_MANIFEST_REPO = Path(__file__).resolve().parent / "dclm_400m_1x_10k-ee2365_manifest.jsonl.gz"
DEFAULT_MANIFEST_GCS = "gs://marin-us-central2/manifests/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _load_manifest(path: str) -> list[dict]:
    if path.startswith("gs://"):
        with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run(["gcloud", "storage", "cp", path, tmp_path], check=True)
        body = Path(tmp_path).read_bytes()
        Path(tmp_path).unlink(missing_ok=True)
    else:
        body = Path(path).read_bytes()
    return [json.loads(line) for line in gzip.decompress(body).splitlines() if line.strip()]


def _list_new_pool(new_pool: str) -> dict[str, dict]:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "w") as fh:
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "objects",
                    "list",
                    new_pool.rstrip("/") + "/*",
                    "--format=csv[no-heading](name,size,crc32c_hash)",
                ],
                check=True,
                stdout=fh,
            )
        out: dict[str, dict] = {}
        with open(tmp_path) as fh:
            for name, size, crc32c in csv.reader(fh):
                basename = name.rsplit("/", 1)[-1]
                if basename.startswith("data-") and basename.endswith(".jsonl.gz"):
                    out[basename] = {"size_bytes": int(size), "crc32c": crc32c}
        return out
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-pool", required=True, help="gs:// path to the re-downloaded pool")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_REPO) if DEFAULT_MANIFEST_REPO.exists() else DEFAULT_MANIFEST_GCS,
        help=f"Manifest path (default: repo copy if present, else {DEFAULT_MANIFEST_GCS})",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    logger.info("Loaded %d manifest entries from %s", len(manifest), args.manifest)

    new_pool = _list_new_pool(args.new_pool)
    logger.info("Listed %d data-*.jsonl.gz objects in %s", len(new_pool), args.new_pool)

    missing: list[str] = []
    size_mismatch: list[tuple[str, int, int]] = []
    crc_mismatch: list[tuple[str, str, str]] = []

    for entry in manifest:
        bn = entry["data_filename"]
        new = new_pool.get(bn)
        if new is None:
            missing.append(bn)
            continue
        if new["size_bytes"] != entry["size_bytes"]:
            size_mismatch.append((bn, entry["size_bytes"], new["size_bytes"]))
        if new["crc32c"] != entry["crc32c"]:
            crc_mismatch.append((bn, entry["crc32c"], new["crc32c"]))

    extras = sorted(set(new_pool) - {e["data_filename"] for e in manifest})

    logger.info("Verification result:")
    logger.info("  matched (size+crc32c):     %d", len(manifest) - len(missing) - len(size_mismatch) - len(crc_mismatch))
    logger.info("  missing in new pool:       %d", len(missing))
    logger.info("  size mismatches:           %d", len(size_mismatch))
    logger.info("  crc32c mismatches:         %d", len(crc_mismatch))
    logger.info("  extras (in new, not orig): %d", len(extras))

    for bn in missing[:10]:
        logger.warning("  missing: %s", bn)
    for bn, exp, got in size_mismatch[:10]:
        logger.warning("  size mismatch: %s expected=%d got=%d", bn, exp, got)
    for bn, exp, got in crc_mismatch[:10]:
        logger.warning("  crc32c mismatch: %s expected=%s got=%s", bn, exp, got)
    for bn in extras[:10]:
        logger.warning("  extra: %s", bn)

    if missing or size_mismatch or crc_mismatch or extras:
        sys.exit(1)


if __name__ == "__main__":
    main()
