# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate ``c075`` survival from a verify stage that has not written every marker shard.

The verify reduce groups markers by normalized shard, so a marker file that exists is complete for
its shard. Normalized shards are hash-partitioned by document id, so the shards that have a marker
file are a uniform sample of the corpus. Measuring a deterministic subset of them gives an unbiased
survival estimate while the last reduce tasks wait for capacity.

A marker file is only evidence of a finished shard when every shard has removals. That holds for
large pools (the 500k tree loses about 39% of its documents) and fails for a small one, where a
shard with no duplicates also has no file. The estimate carries its own check: ``cc_canonical``
removals come from the fuzzy attributes, which are complete, so the sampled ``cc_canonical``
document survival must land on the exact value the fuzzy artifact implies.

The usual reason to need this is that capacity has dried up, so ``--executor threads`` measures
the sample inside this process and needs no workers at all. 300 shards of the 500k tree is about
28M documents, which puts the sampling error near 0.1%. ``--executor zephyr`` fans out to workers
small enough for the non-preemptible CPU VMs, for a larger sample when those have room::

    uv run iris --cluster marin job run --no-wait --no-preemptible --cpu 2 --memory 8GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources --region us-east1 \\
        -- python -m experiments.fast_curation.estimate_survival_from_partial_markers \\
            --tree gs://marin-us-east1/documents/.../500000warcs --sample-shards 300 --executor threads
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import fsspec
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.baseline_collection.verified_dedup import _SURVIVAL_FIELDS, DedupRule, _measure_shard
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

PARTIAL_SURVIVAL_FILENAME = "survival_partial.json"
_FUZZY_DOCUMENT_COUNTERS = "dedup/fuzzy/document"
# One task holds a normalized shard's ids and text lengths in Arrow: about 1.5 GB for a 400 MB shard.
_WORKER = ResourceConfig(cpu=1, ram="4g", disk="5g", preemptible=False)


class Executor(StrEnum):
    """Where the sampled shards are measured."""

    THREADS = "threads"
    ZEPHYR = "zephyr"


def select_marked_shards(marker_basenames: list[str], sample_shards: int) -> list[str]:
    """A deterministic subset of the shards that have a marker file, in hash order of their names."""
    ranked = sorted(marker_basenames, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    return sorted(ranked[:sample_shards])


def estimate(tree: str, sample_shards: int, executor: Executor, max_workers: int) -> dict:
    tree = tree.rstrip("/")
    normalized = read_artifact(f"{tree}/normalize", NormalizedData)
    components = read_artifact(f"{tree}/fuzzy", FuzzyDupsAttrData)
    component_dir = next(iter(components.sources.values())).attr_dir.rstrip("/")
    main_dir = normalized.main_output_dir.rstrip("/")
    marker_dir = f"{tree}/dedup_{DedupRule.C075}/verified/outputs/source_000"

    marked = [os.path.basename(path) for path in fsspec_glob(f"{marker_dir}/*.parquet")]
    total_shards = len(fsspec_glob(f"{main_dir}/*.parquet"))
    if not marked:
        raise FileNotFoundError(f"no marker shards under {marker_dir}")
    sample = select_marked_shards(marked, sample_shards)
    logger.info("%d of %d shards have markers; measuring %d", len(marked), total_shards, len(sample))
    pairs = [
        {
            "source_shard": f"{main_dir}/{name}",
            "marker_shard": f"{marker_dir}/{name}",
            "component_shard": f"{component_dir}/{name}",
        }
        for name in sample
    ]

    if executor == Executor.THREADS:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_measure_shard, pairs))
    else:
        ctx = ZephyrContext(
            name="estimate-dedup-survival",
            max_workers=max_workers,
            resources=_WORKER,
            coordinator_resources=ResourceConfig(cpu=1, ram="4g", preemptible=False),
        )
        results = ctx.execute(Dataset.from_list(pairs).map(_measure_shard)).results
    totals = {field: sum(result[field] for result in results) for field in _SURVIVAL_FIELDS}

    counters = components.counters
    members = counters[f"{_FUZZY_DOCUMENT_COUNTERS}/cluster_members"]
    canonicals = counters[f"{_FUZZY_DOCUMENT_COUNTERS}/canonicals"]
    all_docs = members + counters[f"{_FUZZY_DOCUMENT_COUNTERS}/singletons_skipped"]
    return {
        "tree": tree,
        "estimate": True,
        "shards_total": total_shards,
        "shards_with_markers": len(marked),
        "shards_sampled": len(sample),
        **totals,
        "cc_canonical_doc_survival": 1 - totals["cc_canonical_removed_docs"] / totals["docs"],
        "cc_canonical_doc_survival_exact": 1 - (members - canonicals) / all_docs,
        "cc_canonical_char_survival": 1 - totals["cc_canonical_removed_chars"] / totals["chars"],
        "c075_doc_survival": 1 - totals["c075_removed_docs"] / totals["docs"],
        "c075_char_survival": 1 - totals["c075_removed_chars"] / totals["chars"],
    }


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--sample-shards", type=int, required=True)
    parser.add_argument("--executor", required=True, type=Executor, choices=list(Executor))
    parser.add_argument("--max-workers", type=int, required=True, help="Threads, or Zephyr workers.")
    args = parser.parse_args()
    payload = estimate(args.tree, args.sample_shards, args.executor, args.max_workers)
    path = f"{args.tree.rstrip('/')}/survival_{DedupRule.C075}/{PARTIAL_SURVIVAL_FILENAME}"
    with fsspec.open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Partial survival -> %s\n%s", path, json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
