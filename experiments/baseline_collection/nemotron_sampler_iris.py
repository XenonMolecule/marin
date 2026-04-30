# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native sampler for raw Nemotron-CC. No Ray.

Runs as a single CPU Iris job in us-central2 (where the data lives) and uses a
ThreadPoolExecutor to scan a handful of files per (quality, kind, kind2)
partition. Writes ``samples.jsonl`` + ``summary.json`` to GCS.

Launch
------
    uv run iris --cluster marin job run --no-wait --priority batch \\
        --cpu 16 --memory 32GB --region us-central2 \\
        --job-name nemotron-sampler \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -- python experiments/baseline_collection/nemotron_sampler_iris.py \\
            --output-root gs://marin-us-central2/scratch/nemotron_iris_samples \\
            --sample-per-partition 40 --files-per-partition 3

Outputs to ``<output-root>/``:
  - samples.jsonl  -- N random records per (quality, kind, kind2)
  - summary.json   -- partition stats

The sibling ``nemotron_viewer.py`` renders these into an HTML viewer.
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("nemotron_sampler_iris")

NEMOTRON_BASE = "gs://marin-us-central2/raw/nemotro-cc-eeb783/contrib/Nemotron/Nemotron-CC/data-jsonl"

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


def _list_partition_files(fs, q: str, k: str, k2: str) -> list[str]:
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


def sample_file(path: str, q: str, k: str, k2: str, n: int, seed: int) -> list[dict]:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sample-per-partition", type=int, default=40)
    parser.add_argument("--files-per-partition", type=int, default=3)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fs = fsspec.filesystem("gcs")

    partition_files: dict[tuple[str, str, str], list[str]] = {}
    partition_summary: dict[str, dict] = {}
    for q, k, k2 in PARTITIONS:
        files = _list_partition_files(fs, q, k, k2)
        partition_files[(q, k, k2)] = files
        partition_summary[f"{q}/{k}/{k2}"] = {"total_files": len(files)}
        logger.info("partition=%s/%s/%s files=%d", q, k, k2, len(files))

    tasks: list[tuple[str, str, str, str, int]] = []
    per_partition_picks: dict[str, list[str]] = {}
    for (q, k, k2), files in partition_files.items():
        if not files:
            per_partition_picks[f"{q}/{k}/{k2}"] = []
            continue
        picks = rng.sample(files, min(args.files_per_partition, len(files)))
        per_partition_picks[f"{q}/{k}/{k2}"] = [p.rsplit("/", 1)[-1] for p in picks]
        # Oversample per file; downsample per partition after collection.
        per_file_n = max(1, args.sample_per_partition // len(picks) + 1)
        for f in picks:
            tasks.append((f, q, k, k2, per_file_n))

    logger.info("Dispatching %d file-scan tasks across %d threads", len(tasks), args.threads)

    collected: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {
            ex.submit(sample_file, f, q, k, k2, n, rng.randint(0, 2**31)): (q, k, k2, f) for (f, q, k, k2, n) in tasks
        }
        done = 0
        for fut in as_completed(futures):
            q, k, k2, f = futures[fut]
            try:
                recs = fut.result()
            except Exception as e:
                logger.warning("sample failed q=%s k=%s k2=%s f=%s err=%s", q, k, k2, f, e)
                recs = []
            collected.extend(recs)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                logger.info(
                    "progress: %d / %d file scans complete (%d records so far)", done, len(tasks), len(collected)
                )

    # Downsample to exactly sample_per_partition per partition.
    by_part: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for rec in collected:
        by_part[(rec["_quality"], rec["_kind"], rec["_kind2"])].append(rec)
    samples: list[dict] = []
    for (q, k, k2), recs in by_part.items():
        rng2 = random.Random((hash((q, k, k2)) ^ args.seed) & 0xFFFFFFFF)
        rng2.shuffle(recs)
        samples.extend(recs[: args.sample_per_partition])

    for key, picks in per_partition_picks.items():
        partition_summary[key]["sampled_files"] = picks

    out_root = args.output_root.rstrip("/")
    with fs.open(f"{out_root}/samples.jsonl", "w") as fh:
        for r in samples:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with fs.open(f"{out_root}/summary.json", "w") as fh:
        fh.write(
            json.dumps(
                {
                    "partition_summary": partition_summary,
                    "total_samples": len(samples),
                    "sample_per_partition": args.sample_per_partition,
                    "files_per_partition": args.files_per_partition,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    logger.info("Wrote %s  samples=%d", out_root, len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
