# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify that resiliparse is a true URL-level superset of the filtered pipelines.

Invariant being tested
----------------------
The download/extract/filter pipeline is sharded 1:1 per WARC: for a given shard
index, ``download_warcs``, ``extract_metadata``, and ``extracted/baseline_resiliparse``
all cover the same source WARC file. Therefore every URL in metadata shard ``i``
that survived resiliparse's ``_is_non_empty`` filter must appear in resiliparse
shard ``i``.

The filter pipelines (nemotron, dclm, fineweb) join on the full metadata set
across all 3000 shards. For a given nemotron URL, we cannot tell a priori which
WARC shard it belongs to. But we can:

1. Fix a pair of shards ``i`` (e.g. 0, 100, 1000).
2. Compute ``M_i = URLs in metadata shard i``.
3. Compute ``R_i = URLs in resiliparse shard i``.
4. Download the entire nemotron output (~400MB, ~3M URLs) and intersect with M_i.
   Call that set ``N_i``: "nemotron URLs whose WARC is shard ``i``".
5. Assert ``N_i ⊆ R_i`` (modulo the ``_is_non_empty`` filter, which should be
   near-universal).

We report:
- |R_i ∩ M_i| / |M_i| — fraction of metadata URLs kept by resiliparse.
- |N_i ∩ R_i| / |N_i| — fraction of nemotron-retained URLs that appear in resiliparse.
- Example URLs where N_i \\ R_i (candidate bugs).
- And the same for nemotron_full, dclm, fineweb.

If |N_i ∩ R_i| / |N_i| is ~100%, resiliparse is a superset. If not, we have
direct evidence that resiliparse is dropping things the filters kept — or that
the filters are pulling in URLs outside our 3000-WARC set.

Usage:
    uv run python experiments/baseline_collection/verify_resiliparse_superset.py \\
        --shards 0 100 1000
"""

from __future__ import annotations

import argparse
import gzip
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import orjson

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("verify")

METADATA_ROOT = "gs://marin-us-central2/metadata/baseline_warc_metadata-d671da"
RESILIPARSE_ROOT = "gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa"
NEMOTRON_ROOT = "gs://marin-us-central2/filtered/baseline_nemotron-037958"
NEMOTRON_FULL_ROOT = "gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe"
DCLM_ROOT = "gs://marin-us-central2/filtered/baseline_dclm_resharded-1ac313"
FINEWEB_ROOT = "gs://marin-us-central2/filtered/baseline_fineweb_edu-72c2c7"


def download(gs: str, local: Path) -> Path:
    if local.exists() and local.stat().st_size > 0:
        return local
    subprocess.run(["gcloud", "storage", "cp", gs, str(local)], check=True, capture_output=True)
    return local


def read_urls(path: Path) -> set[str]:
    urls: set[str] = set()
    with gzip.open(path, "rb") as f:
        for line in f:
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            u = rec.get("url")
            if u:
                urls.add(u)
    return urls


def list_dir(gs: str) -> list[str]:
    out = subprocess.run(
        ["gcloud", "storage", "ls", f"{gs}/*.jsonl.gz"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [l.strip() for l in out.splitlines() if l.strip()]


def download_many(paths: list[str], local_dir: Path, max_workers: int = 16) -> list[Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download, p, local_dir / p.rsplit("/", 1)[-1]): p for p in paths}
        for fut in as_completed(futs):
            results.append(fut.result())
    return results


def collect_all_urls(root: str, local_dir: Path, label: str) -> set[str]:
    logger.info("[%s] listing + downloading full output from %s", label, root)
    shards = list_dir(root)
    paths = download_many(shards, local_dir)
    urls: set[str] = set()
    for p in paths:
        urls |= read_urls(p)
    logger.info("[%s] %d shards → %d unique URLs", label, len(shards), len(urls))
    return urls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--shards", type=int, nargs="+", default=[0, 100, 1000], help="WARC shard indices to probe (0..2999)."
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("scratch/verify_superset_cache"))
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # These are small enough to fully materialize: nemotron ~400MB, fineweb ~2.5GB.
    # DCLM resharded is ~5GB. We fetch them all once since we need the full URL sets.
    nemotron_urls = collect_all_urls(NEMOTRON_ROOT, args.cache_dir / "nemotron", "nemotron")
    nemotron_full_urls = collect_all_urls(NEMOTRON_FULL_ROOT, args.cache_dir / "nemotron_full", "nemotron_full")
    fineweb_urls = collect_all_urls(FINEWEB_ROOT, args.cache_dir / "fineweb", "fineweb_edu")
    dclm_urls = collect_all_urls(DCLM_ROOT, args.cache_dir / "dclm", "dclm")

    # Now for each probed shard, fetch metadata + resiliparse for that index.
    for idx in args.shards:
        meta_url = f"{METADATA_ROOT}/data-{idx:05d}-of-03000.jsonl.gz"
        resi_url = f"{RESILIPARSE_ROOT}/data-{idx:05d}-of-03000.jsonl.gz"
        logger.info("=" * 72)
        logger.info("Probing shard %d", idx)
        meta_local = download(meta_url, args.cache_dir / f"meta-{idx:05d}.jsonl.gz")
        resi_local = download(resi_url, args.cache_dir / f"resi-{idx:05d}.jsonl.gz")

        M = read_urls(meta_local)
        R = read_urls(resi_local)
        logger.info("  metadata URLs:    %d", len(M))
        logger.info("  resiliparse URLs: %d", len(R))
        logger.info("  R ⊆ M:            %s (|R \\ M| = %d)", R.issubset(M), len(R - M))
        if M:
            logger.info("  |R ∩ M| / |M| = %.4f (resiliparse retention of metadata URLs)", len(R & M) / len(M))

        for name, F in [
            ("nemotron", nemotron_urls),
            ("nemotron_full", nemotron_full_urls),
            ("dclm", dclm_urls),
            ("fineweb_edu", fineweb_urls),
        ]:
            N_i = F & M  # filter URLs that belong to THIS WARC
            missing = N_i - R  # filter URLs in this WARC but NOT in resiliparse
            if not N_i:
                logger.info("  %-14s no URLs intersect metadata shard %d", name, idx)
                continue
            rate = len(N_i & R) / len(N_i)
            logger.info(
                "  %-14s |N ∩ M|=%6d  |N ∩ R|/|N ∩ M|=%.4f  missing=%d",
                name,
                len(N_i),
                rate,
                len(missing),
            )
            if missing:
                sample_missing = list(missing)[:5]
                for u in sample_missing:
                    logger.info("      MISSING: %s", u)

    return 0


if __name__ == "__main__":
    sys.exit(main())
