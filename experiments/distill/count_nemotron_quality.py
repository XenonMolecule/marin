"""Count nemotron_quality distribution across filtered baseline datasets on us-central2."""

import gzip
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = {
    "baseline_nemotron (actual-only)": "gs://marin-us-central2/filtered/baseline_nemotron-037958/",
    "baseline_nemotron_full (actual + synthetic)": "gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe/",
}


def count_one_file(gcs_path: str) -> Counter:
    """Read one .jsonl.gz and count quality/kind combos."""
    counts: Counter = Counter()
    try:
        with fsspec.open(gcs_path, "rb") as fh:
            data = gzip.decompress(fh.read())
        for line in data.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("nemotron_quality", "unknown")
            k = rec.get("nemotron_kind", "unknown")
            counts[f"{q}/{k}"] += 1
    except Exception as e:
        logger.warning(f"Failed to read {gcs_path}: {e}")
    return counts


def process_dataset(label: str, base_path: str) -> None:
    fs = fsspec.filesystem("gcs")
    prefix = base_path.replace("gs://", "").rstrip("/")
    files = [f"gs://{f}" for f in fs.ls(prefix, detail=False) if f.endswith(".jsonl.gz")]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"{label}: {len(files)} files")
    logger.info(f"{'=' * 60}")

    total: Counter = Counter()
    done = 0
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(count_one_file, f): f for f in files}
        for fut in as_completed(futures):
            total += fut.result()
            done += 1
            if done % 500 == 0:
                logger.info(f"  ...processed {done}/{len(files)} files")

    grand = sum(total.values())
    logger.info(f"\nTotal documents: {grand:,}")

    logger.info(f"\nBy quality/kind:")
    for key in sorted(total.keys()):
        c = total[key]
        logger.info(f"  {key:30s}: {c:>10,} ({100 * c / grand:5.1f}%)")

    quality_only: Counter = Counter()
    for key, c in total.items():
        q = key.split("/")[0]
        quality_only[q] += c

    logger.info(f"\nBy quality (collapsed):")
    for q in ["high", "medium-high", "medium", "medium-low", "low"]:
        c = quality_only.get(q, 0)
        logger.info(f"  {q:12s}: {c:>10,} ({100 * c / grand:5.1f}%)")


def main():
    for label, path in DATASETS.items():
        process_dataset(label, path)


if __name__ == "__main__":
    main()
