# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explore raw Nemotron-CC: lookup by id/url, plus sample per partition.

All heavy scanning runs on the us-central2 Ray cluster (data is in
gs://marin-us-central2) to avoid cross-region egress. Output is a few MB of
JSON(L) that a laptop can pull directly.

Launch
------
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -- python experiments/baseline_collection/nemotron_explorer.py \\
            --output-root gs://marin-us-central2/scratch/nemotron_explorer \\
            --lookup-ids 8ddb1122-56d7-4eb4-b126-2cd806ca2398 \\
            --lookup-urls http://delta-z.ru/teens/20830-zillah-zebra-loses-her-stripes.html \\
            --sample-per-partition 30 \\
            --files-per-partition 4

The lookup scan fans out across every file in every partition (Nemotron-CC is
~25k files, readable in-region). The partition sample uses a bounded random
subset of files per partition.

Outputs to ``<output-root>/``:
  - lookups.jsonl  — every record matching the requested ids/urls
  - samples.jsonl  — N random records per (quality, kind, kind2)
  - summary.json   — partition stats (total/scanned files, sample counts)
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import re
import sys
from collections import defaultdict

import fsspec
import ray

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("nemotron_explorer")

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"

# (quality, kind, kind2) partitions matching filter_nemotron.py's full-mode scan.
PARTITIONS: list[tuple[str, str, str]] = [
    ("high", "actual", "actual"),
    ("medium-high", "actual", "actual"),
    ("medium", "actual", "actual"),
    ("medium-low", "actual", "actual"),
    ("low", "actual", "actual"),
    ("high", "synthetic", "distill"),
    ("high", "synthetic", "diverse_qa_pairs"),
    ("high", "synthetic", "extract_knowledge"),
    ("high", "synthetic", "knowledge_list"),
    ("high", "synthetic", "wrap_medium"),
    ("low", "synthetic", "wrap_medium"),
]

SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")


def _partition_dir(q: str, k: str, k2: str) -> str:
    return f"{NEMOTRON_BASE}/quality={q}/kind={k}/kind2={k2}"


def _list_partition_files(q: str, k: str, k2: str) -> list[str]:
    fs = fsspec.filesystem("gcs")
    pdir = _partition_dir(q, k, k2).replace("gs://", "")
    try:
        return sorted(f"gs://{p}" for p in fs.ls(pdir, detail=False) if p.endswith(".jsonl.gz"))
    except FileNotFoundError:
        return []


def _iter_records(path: str):
    with fsspec.open(path, "rb") as fh, gzip.open(fh, "rt", encoding="utf-8") as gz:
        for line in gz:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _tag(rec: dict, q: str, k: str, k2: str, path: str) -> dict:
    snap_m = SNAPSHOT_RE.search(path)
    return {
        "id": rec.get("id", ""),
        "url": (rec.get("metadata") or {}).get("nemotron_url", ""),
        "text": rec.get("text", ""),
        "source": rec.get("source"),
        "format": rec.get("format"),
        "metadata": rec.get("metadata") or {},
        "_quality": q,
        "_kind": k,
        "_kind2": k2,
        "_snapshot": snap_m.group(0) if snap_m else "unknown",
        "_file": path.rsplit("/", 1)[-1],
    }


@ray.remote
def lookup_file(
    path: str,
    q: str,
    k: str,
    k2: str,
    lookup_ids: frozenset[str],
    lookup_urls: frozenset[str],
) -> list[dict]:
    """Scan one file for id/url matches. Returns tagged records."""
    hits: list[dict] = []
    for rec in _iter_records(path):
        rid = rec.get("id", "")
        url = (rec.get("metadata") or {}).get("nemotron_url", "")
        if (lookup_ids and rid in lookup_ids) or (lookup_urls and url in lookup_urls):
            hits.append(_tag(rec, q, k, k2, path))
    return hits


@ray.remote
def sample_file(path: str, q: str, k: str, k2: str, n: int, seed: int) -> list[dict]:
    """Reservoir-sample n records from one file (tagged)."""
    rng = random.Random(seed)
    reservoir: list[tuple[float, dict]] = []
    for rec in _iter_records(path):
        key = rng.random()
        tagged = _tag(rec, q, k, k2, path)
        if len(reservoir) < n:
            reservoir.append((key, tagged))
            if len(reservoir) == n:
                reservoir.sort(key=lambda x: x[0])
        elif key > reservoir[0][0]:
            reservoir[0] = (key, tagged)
            reservoir.sort(key=lambda x: x[0])
    return [r for _, r in reservoir]


def _drain(futures: list, label: str) -> list:
    results: list = []
    pending = list(futures)
    total = len(pending)
    while pending:
        ready, pending = ray.wait(pending, num_returns=min(64, len(pending)))
        for r in ray.get(ready):
            results.extend(r)
        done = total - len(pending)
        if done % 100 == 0 or not pending:
            logger.info("[%s] %d / %d tasks done (results=%d)", label, done, total, len(results))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--lookup-ids", nargs="*", default=[])
    parser.add_argument("--lookup-urls", nargs="*", default=[])
    parser.add_argument("--sample-per-partition", type=int, default=30)
    parser.add_argument(
        "--files-per-partition",
        type=int,
        default=4,
        help="Number of random files to sample per partition for --sample-per-partition.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-lookup", action="store_true", help="Skip the id/url lookup scan.")
    parser.add_argument("--skip-sample", action="store_true", help="Skip the per-partition sampling.")
    args = parser.parse_args()

    ray.init(address="auto")

    lookup_ids = frozenset(args.lookup_ids)
    lookup_urls = frozenset(args.lookup_urls)
    rng = random.Random(args.seed)

    partition_files: dict[tuple[str, str, str], list[str]] = {}
    partition_summary: dict[str, dict] = {}
    for q, k, k2 in PARTITIONS:
        files = _list_partition_files(q, k, k2)
        partition_files[(q, k, k2)] = files
        partition_summary[f"{q}/{k}/{k2}"] = {"total_files": len(files)}
        logger.info("partition=%s/%s/%s files=%d", q, k, k2, len(files))

    # --- Lookup: scan every file in every partition ---
    matched: list[dict] = []
    if not args.skip_lookup and (lookup_ids or lookup_urls):
        lookup_futures = [
            lookup_file.remote(f, q, k, k2, lookup_ids, lookup_urls)
            for (q, k, k2), files in partition_files.items()
            for f in files
        ]
        logger.info("[lookup] dispatching %d tasks across %d partitions", len(lookup_futures), len(partition_files))
        matched = _drain(lookup_futures, "lookup")
        logger.info("[lookup] total matched records: %d", len(matched))

    # --- Sample: pick files per partition, reservoir-sample within each file ---
    samples: list[dict] = []
    per_partition_file_counts: dict[str, int] = {}
    if not args.skip_sample and args.sample_per_partition > 0:
        sample_futures = []
        for (q, k, k2), files in partition_files.items():
            if not files:
                per_partition_file_counts[f"{q}/{k}/{k2}"] = 0
                continue
            picks = rng.sample(files, min(args.files_per_partition, len(files)))
            per_partition_file_counts[f"{q}/{k}/{k2}"] = len(picks)
            # Oversample per file; we downsample after collection.
            per_file_n = max(1, args.sample_per_partition // len(picks) + 1)
            for f in picks:
                sample_futures.append(sample_file.remote(f, q, k, k2, per_file_n, rng.randint(0, 2**31)))
        logger.info("[sample] dispatching %d tasks", len(sample_futures))
        collected = _drain(sample_futures, "sample")
        # Downsample to exactly N per partition.
        by_part: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for rec in collected:
            by_part[(rec["_quality"], rec["_kind"], rec["_kind2"])].append(rec)
        for (q, k, k2), recs in by_part.items():
            rng2 = random.Random((hash((q, k, k2)) ^ args.seed) & 0xFFFFFFFF)
            rng2.shuffle(recs)
            samples.extend(recs[: args.sample_per_partition])

    for key, n in per_partition_file_counts.items():
        partition_summary[key]["sampled_files"] = n

    fs = fsspec.filesystem("gcs")
    out_root = args.output_root.rstrip("/")

    def _write_jsonl(path: str, records: list[dict]) -> None:
        with fs.open(path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _write_json(path: str, obj) -> None:
        with fs.open(path, "w") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False, indent=2))

    _write_jsonl(f"{out_root}/lookups.jsonl", matched)
    _write_jsonl(f"{out_root}/samples.jsonl", samples)
    _write_json(
        f"{out_root}/summary.json",
        {
            "partition_summary": partition_summary,
            "lookup_ids": sorted(lookup_ids),
            "lookup_urls": sorted(lookup_urls),
            "total_matched": len(matched),
            "total_samples": len(samples),
            "sample_per_partition": args.sample_per_partition,
            "files_per_partition": args.files_per_partition,
        },
    )
    logger.info("Wrote %s  lookups=%d  samples=%d", out_root, len(matched), len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
