# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Containment-verified fuzzy dedup steps shared by the curation dedup drivers.

MinHash connected components chain unrelated documents through transitive closure. The original
``cc_canonical`` rule drops every non-canonical member of a component, so survival collapses as the
corpus grows (90% / 46% / 30% at 10k / 154k / 500k WARCs). The ``c075`` rule keeps the components
as candidate clusters and removes a document only when a longer survivor in its cluster holds at
least 0.75 of its word 3-grams. It is the rule the Harrier hero store was built with, ported from
upstream ``rav/hero_store_dup_075`` at ``c2f981fe3d``; see
``.agents/projects/hero_cluster_dedup_port.md``.

Both rules share every step through ``fuzzy``, so they judge identical candidates. They differ
only after it, and they write to different trees::

    cc_canonical   {bucket}/deduped/                       (every tree the paper reports)
    c075           {bucket}/dedup_c075/{large_clusters,cluster_text,verified}/
                   {bucket}/deduped_c075/

The exact driver code behind the paper's trees is tagged ``paper-dedup-cc-canonical``.
"""

import argparse
import dataclasses
import logging
import os
from collections.abc import Callable, Iterator
from enum import StrEnum

from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.datakit.source_key import datakit_source_key
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.cluster_dedup import ClusterDedupParams
from marin.processing.classification.deduplication.cluster_verify import COUNTER_PREFIX, cluster_verify_step
from marin.processing.classification.deduplication.verify_fuzzy_dups import VerifiedFuzzyDupsAttrData
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

from experiments.datakit.scripts import fuzzy_cluster_text, fuzzy_large_clusters
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)


class DedupRule(StrEnum):
    """Which documents a fuzzy-duplicate component loses."""

    CC_CANONICAL = "cc_canonical"
    C075 = "c075"


# Production values, pinned in full. ``ClusterDedupParams`` defaults ``minimum_containment`` to
# 0.60, which the hero audit rejected: about 4 of 17 sampled removals between 0.60 and 0.75 were
# not duplicates.
C075_PARAMS = ClusterDedupParams(
    rule_version="containment_ngram3_v2",
    ngram_size=3,
    minimum_containment=0.75,
    exact_scan_maximum=256,
    probe_ngrams=32,
    maximum_posting_length=512,
    maximum_candidates=32,
)
MAX_CLUSTER_SIZE = 100_000
LARGE_CLUSTER_STRIDE = 256
# fuzzy_large_clusters.py gives each map task 32g of disk, and Zephyr rejects a task larger than
# its worker.
LARGE_CLUSTERS_WORKER_DISK = "64g"

_APPLY_COUNTER_PREFIX = "dedup/apply_verified"


@dataclasses.dataclass(frozen=True)
class ClusterStageResources:
    """Sizing for the cluster-text shuffle and the cluster solve."""

    max_workers: int
    worker_cpu: int
    worker_ram: str
    worker_disk: str
    task_ram: str
    reduce_task_ram: str
    output_shards: int
    shards_per_task: int


def add_cluster_stage_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the ``c075`` sizing flags. Defaults are the hero run's, which assumed 32-core nodes."""
    group = parser.add_argument_group("c075 cluster stages")
    group.add_argument("--cluster-max-workers", type=int, default=64)
    group.add_argument("--cluster-worker-cpu", type=int, default=32)
    group.add_argument("--cluster-worker-ram", default="128g")
    group.add_argument("--cluster-worker-disk", default="512g")
    group.add_argument("--cluster-task-ram", default="12g", help="One map task: a group of shards or text files.")
    group.add_argument(
        "--cluster-reduce-task-ram",
        default="26g",
        help="One cluster-text reduce task: a partition plus its sort spill, which lands on tmpfs.",
    )
    group.add_argument(
        "--cluster-output-shards",
        type=int,
        default=4096,
        help="Cluster-text reduce tasks. Raise it to shrink partitions instead of raising reduce RAM.",
    )
    group.add_argument("--cluster-shards-per-task", type=int, default=8)


def cluster_stage_resources(args: argparse.Namespace) -> ClusterStageResources:
    return ClusterStageResources(
        max_workers=args.cluster_max_workers,
        worker_cpu=args.cluster_worker_cpu,
        worker_ram=args.cluster_worker_ram,
        worker_disk=args.cluster_worker_disk,
        task_ram=args.cluster_task_ram,
        reduce_task_ram=args.cluster_reduce_task_ram,
        output_shards=args.cluster_output_shards,
        shards_per_task=args.cluster_shards_per_task,
    )


def _marin_prefix_and_key(path: str) -> tuple[str, str]:
    """Split ``gs://bucket/a/b`` into the ``(gs://bucket, a/b)`` pair the hero scripts take."""
    key = datakit_source_key(path)
    stripped = path.rstrip("/")
    if not stripped.endswith(key):
        raise ValueError(f"{path!r} does not end with its datakit source key {key!r}")
    return stripped[: -len(key)].rstrip("/"), key


def _large_clusters(fuzzy: StepSpec, resources: ClusterStageResources, output_path: str) -> dict:
    prefix, candidates = _marin_prefix_and_key(fuzzy.output_path)
    fuzzy_large_clusters.main(
        [
            *("--prefix", prefix),
            *("--candidates", candidates),
            *("--out", output_path),
            *("--stride", str(LARGE_CLUSTER_STRIDE)),
            *("--minimum-size", str(MAX_CLUSTER_SIZE)),
            *("--shards-per-task", str(resources.shards_per_task)),
            *("--max-workers", str(resources.max_workers)),
            *("--worker-cpu", str(resources.worker_cpu)),
            *("--worker-ram", resources.worker_ram),
            *("--worker-disk", LARGE_CLUSTERS_WORKER_DISK),
        ]
    )
    return {"success": True}


def _cluster_text(fuzzy: StepSpec, large_clusters: StepSpec, resources: ClusterStageResources, output_path: str) -> dict:
    prefix, candidates = _marin_prefix_and_key(fuzzy.output_path)
    fuzzy_cluster_text.main(
        [
            *("--prefix", prefix),
            *("--candidates", candidates),
            *("--large-clusters", prefix_join(large_clusters.output_path, "large_clusters.parquet")),
            *("--out", output_path),
            *("--max-cluster-size", str(MAX_CLUSTER_SIZE)),
            *("--output-shards", str(resources.output_shards)),
            *("--shards-per-task", str(resources.shards_per_task)),
            *("--max-workers", str(resources.max_workers)),
            *("--worker-cpu", str(resources.worker_cpu)),
            *("--worker-ram", resources.worker_ram),
            *("--worker-disk", resources.worker_disk),
            *("--task-ram", resources.task_ram),
            *("--reduce-task-ram", resources.reduce_task_ram),
        ]
    )
    return {"success": True}


def verified_dups_steps(*, fuzzy: StepSpec, bucket: str, resources: ClusterStageResources) -> list[StepSpec]:
    """Steps that turn ``fuzzy``'s candidate components into verified duplicate markers.

    Returns ``[large_clusters, cluster_text, verified]``. The last step's artifact is a
    ``VerifiedFuzzyDupsAttrData`` for :func:`apply_verified_dups`.
    """
    root = f"{bucket}/dedup_{DedupRule.C075}"
    large_clusters = StepSpec(
        name="large_clusters",
        override_output_path=f"{root}/large_clusters",
        deps=[fuzzy],
        fn=lambda op: _large_clusters(fuzzy, resources, op),
    )
    cluster_text = StepSpec(
        name="cluster_text",
        override_output_path=f"{root}/cluster_text",
        deps=[fuzzy, large_clusters],
        fn=lambda op: _cluster_text(fuzzy, large_clusters, resources, op),
    )
    task = ResourceConfig(cpu=1, ram=resources.task_ram, disk="48g")
    verified = cluster_verify_step(
        name="verified",
        cluster_text_step=cluster_text,
        params=C075_PARAMS,
        max_workers=resources.max_workers,
        worker_resources=ResourceConfig(
            cpu=resources.worker_cpu, ram=resources.worker_ram, disk=resources.worker_disk
        ),
        map_task_resources=task,
        reduce_task_resources=task,
        override_output_path=f"{root}/verified",
    )
    return [large_clusters, cluster_text, verified]


def apply_verified_dups(
    normalize: StepSpec,
    verified: StepSpec,
    output_path: str,
    carry_provenance: Callable[[dict], dict],
) -> dict:
    """Drop every document the verifier marked and re-emit the rest as jsonl.gz.

    Marker files are sparse: a normalized shard with no removals has none, so an absent file
    cannot be told from a lost one shard by shard. The total can. Every marker row is a removal,
    so a fresh run must drop exactly the ``markers`` count on the verifier's artifact, and this
    raises when it does not.
    """
    norm = read_artifact(normalize.output_path, NormalizedData)
    markers = read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData)
    if len(markers.sources) != 1:
        raise RuntimeError(f"expected 1 source in VerifiedFuzzyDupsAttrData, got {len(markers.sources)}")
    attr_dir = next(iter(markers.sources.values())).attr_dir.rstrip("/")
    expected_drops = int(markers.counters[f"{COUNTER_PREFIX}/markers"])

    main_dir = norm.main_output_dir.rstrip("/")
    source_shards = sorted(fsspec_glob(f"{main_dir}/*.parquet"))
    if not source_shards:
        raise FileNotFoundError(f"no parquet under {main_dir}")
    marker_basenames = {os.path.basename(path) for path in fsspec_glob(f"{attr_dir}/*.parquet")}
    unknown = marker_basenames - {os.path.basename(path) for path in source_shards}
    if unknown:
        raise RuntimeError(f"marker shards with no normalized shard under {main_dir}: {sorted(unknown)[:5]}")
    shard_pairs = [
        {
            "source_shard": path,
            "marker_shard": f"{attr_dir}/{os.path.basename(path)}"
            if os.path.basename(path) in marker_basenames
            else None,
        }
        for path in source_shards
    ]
    resumed = bool(fsspec_glob(f"{output_path}/data-*.jsonl.gz"))
    logger.info(
        "apply: %d source shards under %s, %d marker shards under %s, %d removals expected",
        len(shard_pairs),
        main_dir,
        len(marker_basenames),
        attr_dir,
        expected_drops,
    )

    def _process_one(item: dict) -> Iterator[dict]:
        removed_ids: set[str] = set()
        if item["marker_shard"] is not None:
            removed_ids = {row["id"] for row in load_parquet(item["marker_shard"]) if row["dup_doc"]}
        kept = dropped = 0
        for record in load_parquet(item["source_shard"]):
            if record["id"] in removed_ids:
                dropped += 1
                continue
            kept += 1
            yield carry_provenance(record)
        if dropped != len(removed_ids):
            raise RuntimeError(
                f"{item['marker_shard']}: {len(removed_ids)} marked ids but {dropped} found in {item['source_shard']}"
            )
        counters.pipeline.update_counter(f"{_APPLY_COUNTER_PREFIX}/kept", kept)
        counters.pipeline.update_counter(f"{_APPLY_COUNTER_PREFIX}/dropped", dropped)

    pipeline = (
        Dataset.from_list(shard_pairs)
        .flat_map(_process_one)
        .reshard(len(shard_pairs))
        .write_jsonl(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(
        name="apply-verified-dups",
        max_workers=200,
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=1, ram="8g", preemptible=False),
    )
    outcome = ctx.execute(pipeline)
    kept = int(outcome.counters.get(f"{_APPLY_COUNTER_PREFIX}/kept", 0))
    dropped = int(outcome.counters.get(f"{_APPLY_COUNTER_PREFIX}/dropped", 0))
    if resumed:
        # skip_existing leaves finished shards alone, so this run's counters cover only the rest.
        logger.warning("apply resumed into existing output; removal total not checked (this run dropped %d)", dropped)
    elif dropped != expected_drops:
        raise RuntimeError(f"verifier wrote {expected_drops} markers but apply dropped {dropped}")
    return {"success": True, "shards": len(shard_pairs), "kept": kept, "dropped": dropped, "resumed": resumed}
