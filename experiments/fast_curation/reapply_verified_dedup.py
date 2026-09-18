# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the ``c075`` dedup rule to a fast-curation dedup tree that is already built.

``dedup.py --dedup-rule c075`` rediscovers its inputs from ``kept_text`` and schedules every step
from ``reshape`` on. That cannot serve a tree whose upstream scratch was pruned: the 154k and 500k
fused trees keep only ``normalize/``, ``fuzzy/``, ``deduped/`` and ``stats/``. This starts from the
two steps the rule needs, pins them, and refuses to run if either is not already built::

    {tree}/normalize  {tree}/fuzzy                                   (pinned, never rebuilt)
      -> {tree}/dedup_c075/{large_clusters,cluster_text,verified}
        -> {tree}/deduped_c075/    {tree}/stats_c075/dedup_stats.json

``{tree}/deduped/`` (the ``cc_canonical`` output) is never read or written. Because ``fuzzy`` is
reused, both rules judge the same candidate components, and the survival difference between
``deduped/`` and ``deduped_c075/`` is the rule alone.

Run in the tree's region. Sort spill lands on tmpfs, so size ``--cluster-output-shards`` to keep
one cluster-text partition near 1 GB instead of raising reduce RAM::

    uv run iris --cluster marin job run --no-wait --no-preemptible --cpu 4 --memory 14GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources --region us-east5 \\
        --job-name reapply-c075-v2-10k -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.fast_curation.reapply_verified_dedup \\
            --tree gs://marin-us-east5/documents/baseline_lpv11_fastpipe_v2_deduped/10364warcs \\
            --cluster-output-shards 512 ...
"""

import argparse
import json
import logging
import sys

import fsspec
from marin.execution.artifact import read_artifact, read_record
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.verify_fuzzy_dups import VerifiedFuzzyDupsAttrData
from rigging.log_setup import configure_logging

from experiments.baseline_collection.verified_dedup import (
    C075_PARAMS,
    MAX_CLUSTER_SIZE,
    ClusterStageResources,
    DedupRule,
    add_cluster_stage_arguments,
    apply_verified_dups,
    cluster_stage_resources,
    verified_dups_steps,
)
from experiments.fast_curation.dedup import _carry_provenance

logger = logging.getLogger(__name__)

_FUZZY_DOCUMENT_COUNTERS = "dedup/fuzzy/document"


def _pinned_step(name: str, output_path: str) -> StepSpec:
    """A step served from ``output_path`` that raises if a runner ever executes it."""

    def refuse(_: str) -> None:
        raise RuntimeError(f"{name} at {output_path} is pinned input to a re-apply and must not be rebuilt")

    step = StepSpec(name=name, override_output_path=output_path, deps=[], fn=refuse)
    if not step_is_built(step):
        raise FileNotFoundError(f"{output_path} has no SUCCESS status; a re-apply needs a finished {name} step")
    return step


def _write_stats(tree: str, fuzzy: StepSpec, verified: StepSpec, deduped: StepSpec, output_path: str) -> dict:
    components = read_artifact(fuzzy.output_path, FuzzyDupsAttrData).counters
    markers = read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData)
    applied = read_record(deduped.output_path)
    if applied is None or applied.result is None:
        raise FileNotFoundError(f"no apply result recorded at {deduped.output_path}")
    members = int(components[f"{_FUZZY_DOCUMENT_COUNTERS}/cluster_members"])
    canonicals = int(components[f"{_FUZZY_DOCUMENT_COUNTERS}/canonicals"])
    payload = {
        "tree": tree,
        "dedup_rule": {
            "rule": DedupRule.C075,
            "max_cluster_size": MAX_CLUSTER_SIZE,
            **C075_PARAMS.model_dump(mode="json"),
        },
        "deduped_path": deduped.output_path,
        "verified_path": verified.output_path,
        "component_members": members,
        "removed_by_cc_canonical": members - canonicals,
        "removed_by_c075": applied.result["dropped"],
        "kept_by_c075": applied.result["kept"],
        "apply_resumed": applied.result["resumed"],
        "verifier_counters": dict(sorted(markers.counters.items())),
    }
    stats_path = f"{output_path}/dedup_stats.json"
    with fsspec.open(stats_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Stats -> %s\n%s", stats_path, json.dumps(payload, indent=2))
    return {"success": True, "path": stats_path}


def build_steps(tree: str, resources: ClusterStageResources) -> list[StepSpec]:
    tree = tree.rstrip("/")
    normalize = _pinned_step("normalize", f"{tree}/normalize")
    fuzzy = _pinned_step("fuzzy", f"{tree}/fuzzy")
    verification = verified_dups_steps(fuzzy=fuzzy, bucket=tree, resources=resources)
    verified = verification[-1]
    deduped = StepSpec(
        name="deduped",
        override_output_path=f"{tree}/deduped_{DedupRule.C075}",
        deps=[normalize, verified],
        fn=lambda op: apply_verified_dups(normalize, verified, op, _carry_provenance),
    )
    stats = StepSpec(
        name="stats",
        override_output_path=f"{tree}/stats_{DedupRule.C075}",
        deps=[fuzzy, verified, deduped],
        fn=lambda op: _write_stats(tree, fuzzy, verified, deduped, op),
    )
    return [*verification, deduped, stats]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, help="Dedup tree root holding built normalize/ and fuzzy/ steps.")
    add_cluster_stage_arguments(parser)
    args = parser.parse_args()
    StepRunner().run(build_steps(args.tree, cluster_stage_resources(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
