# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find WARC files present in both DCLM and FineWeb-Edu.

Reads the DCLM 400m-1x WARC file list (bundled in repo) and scans FineWeb-Edu
parquet files on GCS to find WARC files that appear in both datasets.

Parallelizes across snapshots using Zephyr (CPU-only, I/O-bound).

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \
        -- python experiments/distill/find_overlapping_warcs.py
"""

import json
import logging
import re
import time
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

from zephyr import Dataset, ZephyrContext

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Bundled in repo — no network fetch at runtime
DCLM_WARCS_FILE = Path(__file__).parent / "dclm_400m_1x_warcs.txt"
FINEWEB_EDU_BASE = "gs://marin-us-central2/raw/fineweb-edu"
OUTPUT_PATH = "gs://marin-us-central2/raw/baseline-dataset-collection"


def normalize_warc_path(path: str) -> str:
    """Strip s3://commoncrawl/ or https://data.commoncrawl.org/ prefix."""
    for prefix in ["s3://commoncrawl/", "https://data.commoncrawl.org/"]:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def load_dclm_warcs() -> dict[str, list[str]]:
    """Load DCLM WARC list from bundled file, grouped by snapshot."""
    by_snapshot: dict[str, list[str]] = {}
    with open(DCLM_WARCS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.search(r"CC-MAIN-\d{4}-\d{2}", line)
            if m:
                by_snapshot.setdefault(m.group(0), []).append(line)
    return by_snapshot


def process_snapshot(task: dict) -> list[dict]:
    """Process one snapshot: read FineWeb-Edu file_path column, intersect with DCLM.

    Args:
        task: {"snapshot": str, "dclm_warcs": [str]}

    Returns list of {"warc_path": str, "snapshot": str, "in_fineweb": bool}
    """
    snapshot = task["snapshot"]
    dclm_warcs = task["dclm_warcs"]
    dclm_normalized = {normalize_warc_path(p): p for p in dclm_warcs}
    t_snap = time.time()

    logger.info(f"[START] {snapshot}: checking {len(dclm_warcs)} DCLM WARCs against FineWeb-Edu...")

    fs = fsspec.filesystem("gcs")
    base = FINEWEB_EDU_BASE.replace("gs://", "")
    snap_dir = f"{base}/{snapshot}"

    try:
        files = [f for f in fs.ls(snap_dir, detail=False) if f.endswith(".parquet")]
    except FileNotFoundError:
        logger.warning(f"[MISS] {snapshot}: not found in FineWeb-Edu, marking all {len(dclm_warcs)} as missing")
        return [{"warc_path": w, "snapshot": snapshot, "in_fineweb": False} for w in dclm_warcs]

    logger.info(f"  {snapshot}: found {len(files)} parquet files, reading file_path column...")

    # Read just the file_path column from all parquet files for this snapshot
    fineweb_paths: set[str] = set()
    for i, pf_path in enumerate(files):
        t0 = time.time()
        with fs.open(pf_path, "rb") as f:
            table = pq.read_table(f, columns=["file_path"])
        new_paths = table.column("file_path").to_pylist()
        fineweb_paths.update(new_paths)
        elapsed = time.time() - t0
        logger.info(
            f"  {snapshot}: [{i + 1}/{len(files)}] {pf_path.split('/')[-1]} "
            f"— {len(new_paths):,} rows in {elapsed:.1f}s"
        )

    fineweb_normalized = {normalize_warc_path(p) for p in fineweb_paths}

    results = []
    overlap = 0
    for norm, orig in dclm_normalized.items():
        found = norm in fineweb_normalized
        if found:
            overlap += 1
        results.append({"warc_path": orig, "snapshot": snapshot, "in_fineweb": found})

    total_elapsed = time.time() - t_snap
    logger.info(
        f"[DONE] {snapshot}: DCLM={len(dclm_warcs)}, FineWeb WARCs={len(fineweb_normalized)}, "
        f"overlap={overlap}, miss={len(dclm_warcs) - overlap} — {total_elapsed:.0f}s total"
    )
    return results


def main():
    t_start = time.time()

    dclm_by_snapshot = load_dclm_warcs()
    total_dclm = sum(len(v) for v in dclm_by_snapshot.values())
    logger.info(f"DCLM: {total_dclm:,} WARCs across {len(dclm_by_snapshot)} snapshots")

    # Build task list: one task per snapshot
    tasks = [{"snapshot": snap, "dclm_warcs": warcs} for snap, warcs in sorted(dclm_by_snapshot.items())]

    # Process snapshots in parallel with Zephyr
    logger.info(f"Processing {len(tasks)} snapshots in parallel...")
    pipeline = Dataset.from_list(tasks).flat_map(process_snapshot)

    # One worker per snapshot (89 tasks). CPU-only, I/O-bound parquet reads.
    ctx = ZephyrContext(name="find-overlapping-warcs", max_workers=len(tasks))
    all_results = list(ctx.execute(pipeline))

    # Aggregate results
    overlapping = [r for r in all_results if r["in_fineweb"]]
    missing = [r for r in all_results if not r["in_fineweb"]]

    logger.info("=" * 60)
    logger.info(f"TOTAL DCLM WARCs: {len(all_results):,}")
    logger.info(f"In both DCLM + FineWeb-Edu: {len(overlapping):,}")
    logger.info(f"DCLM only (not in FineWeb-Edu): {len(missing):,}")
    logger.info(f"Overlap rate: {len(overlapping) / len(all_results) * 100:.1f}%")
    logger.info(f"Elapsed: {time.time() - t_start:.0f}s")

    # Write results to GCS
    fs = fsspec.filesystem("gcs")
    out = OUTPUT_PATH.replace("gs://", "")

    # Overlapping WARC list (one per line) — the main artifact
    with fs.open(f"{out}/dclm_fineweb_overlapping_warcs.txt", "w") as f:
        for r in sorted(overlapping, key=lambda x: x["warc_path"]):
            f.write(r["warc_path"] + "\n")

    # Missing WARCs (for debugging)
    if missing:
        with fs.open(f"{out}/dclm_warcs_not_in_fineweb.txt", "w") as f:
            for r in sorted(missing, key=lambda x: x["warc_path"]):
                f.write(r["warc_path"] + "\n")

    # Summary JSON
    summary = {
        "dclm_source": "400m-1x",
        "fineweb_edu_source": FINEWEB_EDU_BASE,
        "total_dclm_warcs": len(all_results),
        "overlapping_warcs": len(overlapping),
        "dclm_only_warcs": len(missing),
        "overlap_rate_pct": round(len(overlapping) / len(all_results) * 100, 2),
        "snapshots_checked": len(tasks),
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    with fs.open(f"{out}/overlap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results written to {OUTPUT_PATH}/")
    logger.info(f"  -> dclm_fineweb_overlapping_warcs.txt ({len(overlapping):,} WARCs)")
    if missing:
        logger.info(f"  -> dclm_warcs_not_in_fineweb.txt ({len(missing):,} WARCs)")
    logger.info("  -> overlap_summary.json")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
