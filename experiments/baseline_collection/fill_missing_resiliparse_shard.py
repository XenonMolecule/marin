# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Surgically fill resiliparse extraction shards lost to preemption.

The 500-worker extract step in ``pipeline.py`` kept getting churned by
us-central2 preemption with only one shard left (2999/3000), so the step never
reached SUCCESS — each relaunch spins up the full pool and the single real-work
task is killed before finishing. This is the endgame pattern (one targeted unit,
not the scaling parent): for exactly the missing shard(s) it runs a single-file
extraction on **non-preemptible** workers, writing to the original
``data-{i:05d}-of-{total:05d}.jsonl.gz`` path so the dedup reads a complete set.

It reuses ``_extract_text`` / ``_is_non_empty`` and the same ``ExtractTextConfig``
defaults the pipeline used, and derives the partition->file mapping with the same
``resolve_glob`` (lexicographic sort) the executor's ``from_files`` uses, so the
output is identical to what the executor would have produced.

Usage (Iris CPU job, in us-central2):
    python experiments/baseline_collection/fill_missing_resiliparse_shard.py \\
        --download gs://marin-us-central2/raw/commoncrawl/baseline_3000_random-34884d \\
        --extract  gs://marin-us-central2/extracted/baseline_resiliparse-5ce413
"""

from __future__ import annotations

import argparse
import logging
import re

from fray import ResourceConfig
from marin.transform.extract_text_from_html import ExtractTextConfig, _extract_text, _is_non_empty
from zephyr import Dataset, ZephyrContext
from zephyr.dataset import GlobSource, resolve_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

SHARD_RE = re.compile(r"data-(\d+)-of-(\d+)\.jsonl\.gz$")


def _sorted_paths(pattern: str) -> list[str]:
    """Same globbing + lexicographic sort the executor's from_files uses."""
    return [e.path for e in resolve_glob(GlobSource(pattern))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--download", required=True, help="download dir of HTML jsonl.gz (one per WARC)")
    p.add_argument("--extract", required=True, help="resiliparse extract output dir")
    p.add_argument("--region", default="us-central2", help="region to run non-preemptible workers in")
    args = p.parse_args()

    download_files = _sorted_paths(f"{args.download}/*.jsonl.gz")
    total = len(download_files)

    present: set[int] = set()
    for path in _sorted_paths(f"{args.extract}/data-*-of-*.jsonl.gz"):
        m = SHARD_RE.search(path)
        if m:
            present.add(int(m.group(1)))

    missing = [i for i in range(total) if i not in present]
    logger.info("total=%d present=%d missing=%s", total, len(present), missing)
    if not missing:
        logger.info("nothing missing — extraction is complete")
        return

    # Identical config to pipeline.py's extract step (only input/output set; rest default).
    config = ExtractTextConfig(input_path=f"{args.download}/*.jsonl.gz", output_path=args.extract)

    for i in missing:
        src = download_files[i]
        out = f"{args.extract}/data-{i:05d}-of-{total:05d}.jsonl.gz"
        logger.info("filling shard %05d: %s -> %s", i, src, out)
        pipeline = (
            Dataset.from_iterable([src])
            .load_file()
            .map(_extract_text)
            .filter(_is_non_empty)
            .write_jsonl(lambda shard, tot, _o=out: _o, skip_existing=True)
        )
        ctx = ZephyrContext(
            name=f"fill-resiliparse-{i:05d}",
            max_workers=2,
            # Non-preemptible is the whole point: the preemptible pool churn is
            # what killed this shard repeatedly. Pin to the data's region.
            resources=ResourceConfig(cpu=4, ram="32g", regions=[args.region], preemptible=False),
        )
        ctx.put("config", config)
        ctx.execute(pipeline)

    logger.info("done — filled %d shard(s)", len(missing))


if __name__ == "__main__":
    main()
