# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Docs and text chars of a deduped JSONL output, broken down by source snapshot (run in-region).

``--part I/N`` counts every N-th file starting at I, so N jobs split one pass over the output;
each part writes ``{output_dir}/part-IIIII-of-NNNNN.json``. ``--merge`` sums the parts into
``{output_dir}/summary.json`` with per-family (CC-MAIN, CC-SUPPLEMENTAL, CC-NEWS) and
per-snapshot docs, chars and llama3 token estimates.

    python -m experiments.fast_curation.count_deduped_by_source \\
        --deduped-prefix gs://marin-us-central2/documents/<tree>/deduped_c075/ \\
        --output-dir gs://marin-us-central2/documents/<tree>/stats/by_source --part 0/32
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor

import fsspec
from google.cloud import storage

from experiments.fast_curation.count_deduped_docs import LLAMA3_CHARS_PER_TOKEN

logger = logging.getLogger(__name__)

_FAMILY_RE = re.compile(r"^CC-(MAIN|SUPPLEMENTAL|NEWS)-")


def family(snapshot: str) -> str:
    m = _FAMILY_RE.match(snapshot)
    if not m:
        raise ValueError(f"unknown snapshot {snapshot!r}")
    return f"CC-{m.group(1)}"


def _count_file(path: str) -> dict[str, list[int]]:
    """{snapshot: [docs, chars]} for one jsonl.gz file."""
    counts: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            entry = counts[record["snapshot"]]
            entry[0] += 1
            entry[1] += len(record["text"])
    return dict(counts)


def _list_files(prefix: str) -> list[str]:
    bucket, _, key = prefix.removeprefix("gs://").partition("/")
    blobs = storage.Client().list_blobs(bucket, prefix=key, fields="items(name),nextPageToken", page_size=5000)
    return sorted(f"gs://{bucket}/{b.name}" for b in blobs if b.name.endswith(".jsonl.gz"))


def count_part(prefix: str, output_dir: str, index: int, parts: int, workers: int) -> None:
    files = _list_files(prefix)[index::parts]
    logger.info("part %d/%d: %d files", index, parts, len(files))
    totals: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, counts in enumerate(pool.map(_count_file, files, chunksize=4)):
            for snapshot, (docs, chars) in counts.items():
                totals[snapshot][0] += docs
                totals[snapshot][1] += chars
            if i % 1000 == 0:
                logger.info("%d/%d files", i, len(files))
    with fsspec.open(f"{output_dir}/part-{index:05d}-of-{parts:05d}.json", "w") as f:
        json.dump({"files": len(files), "snapshots": totals}, f)


def merge(output_dir: str, parts: int) -> dict:
    files = 0
    snapshots: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for index in range(parts):
        with fsspec.open(f"{output_dir}/part-{index:05d}-of-{parts:05d}.json") as f:
            part = json.load(f)
        files += part["files"]
        for snapshot, (docs, chars) in part["snapshots"].items():
            snapshots[snapshot][0] += docs
            snapshots[snapshot][1] += chars
    families: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for snapshot, (docs, chars) in snapshots.items():
        families[family(snapshot)][0] += docs
        families[family(snapshot)][1] += chars

    def row(docs: int, chars: int) -> dict:
        return {"docs": docs, "chars": chars, "llama3_tokens_est": chars / LLAMA3_CHARS_PER_TOKEN}

    summary = {
        "files": files,
        "total": row(sum(d for d, _ in snapshots.values()), sum(c for _, c in snapshots.values())),
        "families": {k: row(*v) for k, v in sorted(families.items())},
        "snapshots": {k: row(*v) for k, v in sorted(snapshots.items())},
        "llama3_chars_per_token": LLAMA3_CHARS_PER_TOKEN,
    }
    with fsspec.open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deduped-prefix")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--part", help="I/N: count every N-th file starting at I.")
    ap.add_argument("--merge", type=int, metavar="N", help="Sum N parts into summary.json.")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.merge:
        summary = merge(args.output_dir, args.merge)
        print(json.dumps({k: summary[k] for k in ("files", "total", "families")}, indent=1))
        return
    index, parts = (int(x) for x in args.part.split("/"))
    count_part(args.deduped_prefix, args.output_dir, index, parts, args.workers)


if __name__ == "__main__":
    main()
