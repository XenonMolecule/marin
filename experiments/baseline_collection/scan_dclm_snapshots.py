# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off: enumerate which CC snapshots DCLM-baseline-1.0 actually covers.

Reads the FIRST record of every DCLM shard on GCS and extracts the
``isPartOf: CC-MAIN-YYYY-WW`` line from its ``warcinfo`` string. Writes one
row per shard to GCS; aggregate externally with ``jq`` or pandas.

DCLM shards are "globally" mixed, so the first-record snapshot is a sample,
not a ground-truth shard identity. Aggregating across all 27,838 shards gives
a reliable picture of which snapshots DCLM touched (the union of first records).

Usage::

    uv run lib/marin/src/marin/run/ray_run.py --cluster marin-big-run --no_wait \
        -e WANDB_API_KEY ... -e HF_TOKEN ... \
        -- python experiments/baseline_collection/scan_dclm_snapshots.py
"""

import io
import json
import logging

import fsspec
import zstandard

from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)

DCLM_BASE = (
    "gs://marin-us-central2/raw/dclm/a3b142c/huggingface.co/datasets/" "mlfoundations/dclm-baseline-1.0/resolve/a3b142c"
)
OUTPUT_PATH = "gs://marin-us-central2/tmp/dclm_shard_snapshots"


def _list_dclm_shards() -> list[str]:
    fs = fsspec.filesystem("gcs")
    base = DCLM_BASE.replace("gs://", "")
    shards: list[str] = []
    for global_shard in fs.ls(base, detail=False):
        if "global-shard" not in global_shard:
            continue
        for local_shard in fs.ls(global_shard, detail=False):
            for f in fs.ls(local_shard, detail=False):
                if f.endswith(".jsonl.zst"):
                    shards.append(f"gs://{f}")
    return shards


def _extract_is_part_of(warcinfo: str) -> str:
    for line in warcinfo.splitlines():
        line = line.strip()
        if line.startswith("isPartOf:"):
            return line.split(":", 1)[1].strip()
    return ""


def _scan_first_records(shard_path: str) -> list[dict]:
    """Read the first 3 records of a DCLM shard and return their snapshot labels."""
    snapshots: list[str] = []
    try:
        with fsspec.open(shard_path, "rb") as fh:
            dctx = zstandard.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for _ in range(3):
                    line = text_stream.readline()
                    if not line:
                        break
                    record = json.loads(line)
                    warcinfo = record.get("warcinfo", "")
                    snapshots.append(_extract_is_part_of(warcinfo))
    except Exception as e:
        return [{"shard": shard_path, "snapshot": "", "error": str(e)[:200]}]

    return [{"shard": shard_path, "snapshot": s, "error": ""} for s in snapshots]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    shards = _list_dclm_shards()
    logger.info("Found %d DCLM shards", len(shards))

    pipeline = (
        Dataset.from_list(shards)
        .flat_map(_scan_first_records)
        .write_jsonl(f"{OUTPUT_PATH}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )

    ctx = ZephyrContext(name="dclm-snapshot-scan", max_workers=200)
    ctx.execute(pipeline)
    logger.info("Done → %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
