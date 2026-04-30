# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a verifiable manifest of the 10k WARC download pool before deletion.

The pool at ``gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/``
holds 10,364 ``data-<hash>.jsonl.gz`` files — one per source WARC, where
``<hash> = sha256(warc_path)[:12]`` per ``download_warcs._warc_path_hash``.

This script captures, per source WARC:
- ``warc_path`` (canonical CC path from ``experiments/distill/dclm_400m_1x.txt``)
- ``data_filename`` (derived from the hash; the on-disk output name)
- ``size_bytes``, ``crc32c``, ``md5`` (from GCS object metadata)

So a future re-download can be verified 1:1: re-run the pipeline against the
same WARC list, then compare each new ``data-<hash>.jsonl.gz`` against the
manifest's size + crc32c. Match means byte-identical extraction.

Outputs (both written; keep both — repo for git history, GCS for durability):
- ``experiments/baseline_collection/dclm_400m_1x_10k-ee2365_manifest.jsonl``
- ``gs://marin-us-central2/manifests/dclm_400m_1x_10k-ee2365_manifest.jsonl``

Run locally (no cluster needed; ~30s for the GCS listing):

    uv run python experiments/baseline_collection/build_10k_manifest.py
"""

import csv
import gzip
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path

POOL_PATH = "gs://marin-us-central2/raw/commoncrawl/dclm_400m_1x_10k-ee2365/"
WARC_MANIFEST = Path(__file__).resolve().parent.parent / "distill" / "dclm_400m_1x.txt"
REPO_OUT = Path(__file__).resolve().parent / "dclm_400m_1x_10k-ee2365_manifest.jsonl.gz"
GCS_OUT = "gs://marin-us-central2/manifests/dclm_400m_1x_10k-ee2365_manifest.jsonl.gz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _warc_path_hash(warc_path: str) -> str:
    """Same hash as download_warcs._warc_path_hash. Sha256 of utf-8 bytes, first 12 hex chars."""
    return hashlib.sha256(warc_path.encode()).hexdigest()[:12]


def _load_warc_paths() -> list[str]:
    """Read source WARC list, stripping blank lines."""
    paths = [line.strip() for line in WARC_MANIFEST.read_text().splitlines() if line.strip()]
    logger.info("Loaded %d WARC paths from %s", len(paths), WARC_MANIFEST)
    return paths


def _list_pool_objects() -> dict[str, dict]:
    """Return {basename: {size, crc32c, md5}} for every data-*.jsonl.gz in the pool.

    Uses ``gcloud storage objects list`` (not gcsfs) because gcsfs hits a macOS
    SSL cert issue on this machine. Streams the CSV result to a tmp file to avoid
    holding 10k * ~150B in subprocess pipe buffers.
    """
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        logger.info("Listing pool via gcloud (this takes ~30-60s for 10k objects)…")
        with open(tmp_path, "w") as fh:
            subprocess.run(
                [
                    "gcloud",
                    "storage",
                    "objects",
                    "list",
                    POOL_PATH.rstrip("/") + "/*",
                    "--format=csv[no-heading](name,size,crc32c_hash,md5_hash)",
                ],
                check=True,
                stdout=fh,
            )

        out: dict[str, dict] = {}
        with open(tmp_path) as fh:
            for name, size, crc32c, md5 in csv.reader(fh):
                basename = name.rsplit("/", 1)[-1]
                if not basename.startswith("data-") or not basename.endswith(".jsonl.gz"):
                    continue
                out[basename] = {
                    "size_bytes": int(size),
                    "crc32c": crc32c,
                    "md5": md5,
                }
        logger.info("Listed %d data-*.jsonl.gz objects in %s", len(out), POOL_PATH)
        return out
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    warc_paths = _load_warc_paths()
    pool = _list_pool_objects()

    records: list[dict] = []
    missing: list[str] = []
    for warc_path in warc_paths:
        h = _warc_path_hash(warc_path)
        basename = f"data-{h}.jsonl.gz"
        meta = pool.get(basename)
        if meta is None:
            missing.append(warc_path)
            continue
        records.append(
            {
                "warc_path": warc_path,
                "data_filename": basename,
                "size_bytes": meta["size_bytes"],
                "crc32c": meta["crc32c"],
                "md5": meta["md5"],
            }
        )

    extras = sorted(set(pool) - {f"data-{_warc_path_hash(p)}.jsonl.gz" for p in warc_paths})

    logger.info("Manifest entries: %d", len(records))
    if missing:
        logger.warning("WARCs with no output (will not be in manifest): %d", len(missing))
        for p in missing[:10]:
            logger.warning("  missing: %s", p)
    if extras:
        logger.warning("Pool objects without a source WARC: %d", len(extras))
        for b in extras[:10]:
            logger.warning("  extra: %s", b)

    body = "\n".join(json.dumps(r, sort_keys=True) for r in records).encode() + b"\n"
    payload = gzip.compress(body)

    REPO_OUT.write_bytes(payload)
    logger.info("Wrote %s (%.2f MB)", REPO_OUT, REPO_OUT.stat().st_size / 1e6)

    subprocess.run(["gcloud", "storage", "cp", str(REPO_OUT), GCS_OUT], check=True)
    logger.info("Wrote %s", GCS_OUT)

    total_bytes = sum(r["size_bytes"] for r in records)
    logger.info(
        "Manifest summary: %d files, %.2f TB total (sum of pool object sizes)",
        len(records),
        total_bytes / 1e12,
    )

    if missing or extras:
        raise SystemExit(
            f"Manifest is incomplete: {len(missing)} missing source WARCs, {len(extras)} extra objects. "
            "Investigate before deletion."
        )


if __name__ == "__main__":
    main()
