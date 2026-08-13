#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate the FineWeb-CC per-shard survivors into the tokenizer's deduped tree.

`filter_fineweb_cc.py` writes one ``data-<snapshot>-<shard>.jsonl.gz`` per FineWeb
parquet shard (~21.5k files, most empty — only shards that touch one of our WARCs
carry rows). The tokenizer (`tokenize_deduped_extracted.py`) expects the canonical
deduped tree::

    gs://marin-{region}/documents/baseline_{spec}_deduped/{n}warcs/deduped/data-*.jsonl.gz

FineWeb-CC gets NEITHER dedup NOR decontam (we trust HF's own dedup, and the
decon asymmetry is negligible — see project notes). So "consolidate" here is just:
drop empty records, project to ``{text}`` (matching every other method's deduped
tree), and reshard the ~21.5k tiny/empty shards into ~940 evenly-sized shards so
the levanter tokenizer reads a sane file count at the same ~58 MB/shard the
high_quality tree uses.

Usage (Iris, CPU, us-central2)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB --priority interactive \\
        --extra cpu --enable-extra-resources --region us-central2 \\
        --job-name consolidate-fineweb-cc-10364 \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/consolidate_fineweb_cc.py \\
           --n 10364 --region us-central2
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator

import fsspec
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl

logger = logging.getLogger(__name__)

_REGION_TO_BUCKET: dict[str, str] = {
    "us-east5": "gs://marin-us-east5",
    "us-central2": "gs://marin-us-central2",
    "us-central1": "gs://marin-us-central1",
    "eu-west4": "gs://marin-eu-west4",
}

# 54.6 GB of survivors / ~58 MB-per-shard (matches the high_quality deduped tree).
NUM_OUTPUT_SHARDS = 940


def _text_records(path: str) -> Iterator[dict]:
    """Yield ``{text}`` records for one FineWeb-CC survivor shard, dropping empties."""
    for record in load_jsonl(path):
        text = record.get("text")
        if not text:
            continue
        yield {"text": text}


def _input_shard_paths(region: str, n: int) -> list[str]:
    raw_glob = f"{_REGION_TO_BUCKET[region]}/documents/baseline_fineweb_cc/{n}warcs/data-*.jsonl.gz"
    fs = fsspec.filesystem("gcs")
    return sorted(f"gs://{p}" for p in fs.glob(raw_glob))


def run_consolidate(n: int, region: str, max_workers: int) -> dict:
    files = _input_shard_paths(region, n)
    if not files:
        raise ValueError(f"No FineWeb-CC survivor shards found for n={n} region={region}")
    logger.info("Consolidating %d FineWeb-CC survivor shards -> %d output shards", len(files), NUM_OUTPUT_SHARDS)

    out_dir = f"{_REGION_TO_BUCKET[region]}/documents/baseline_fineweb_cc_deduped/{n}warcs/deduped"
    template = f"{out_dir}/data-{{shard:05d}}-of-{NUM_OUTPUT_SHARDS:05d}.jsonl.gz"

    pipeline = (
        Dataset.from_iterable(files)
        .flat_map(_text_records)
        .reshard(NUM_OUTPUT_SHARDS)
        .write_jsonl(template, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="consolidate-fineweb-cc",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="14g", disk="10g"),
    )
    ctx.execute(pipeline)
    return {"input_files": len(files), "output_shards": NUM_OUTPUT_SHARDS, "output_dir": out_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--region", default="us-central2")
    parser.add_argument("--max-workers", type=int, default=200)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_consolidate(args.n, args.region, args.max_workers)
    logger.info("FineWeb-CC consolidate complete: %s", result)


if __name__ == "__main__":
    main()
