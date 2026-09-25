# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the ``c075`` dedup rule to a fast-curation dedup tree that is already built.

``dedup.py --dedup-rule c075`` rediscovers its inputs from ``kept_text`` and schedules every step
from ``reshape`` on. That cannot serve a tree whose upstream scratch was pruned: the 154k and 500k
fused trees keep only ``normalize/``, ``fuzzy/``, ``deduped/`` and ``stats/``. This starts from the
two steps the rule needs, pins them, and refuses to run if either is not already built::

    {tree}/normalize  {tree}/fuzzy                                   (pinned, never rebuilt)
      -> {tree}/dedup_c075/{large_clusters,cluster_text,verified}
        -> {tree}/survival_c075/survival.json   (--output counts and tree)
        -> {tree}/deduped_c075/                 (--output tree only)

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
            --output tree --cluster-output-shards 512 ...

``survival.json`` counts documents and characters removed under ``cc_canonical`` and under ``c075``
from one pass over the normalized shards. A scaling curve needs only that, so ``--output counts``
skips writing a second copy of a multi-terabyte corpus.
"""

import argparse
import json
import logging
import sys
from enum import StrEnum

import fsspec
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
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
    measure_survival,
    verified_dups_steps,
)
from experiments.fast_curation.dedup import _carry_provenance

logger = logging.getLogger(__name__)

SURVIVAL_FILENAME = "survival.json"


class ReapplyOutput(StrEnum):
    """What a re-apply leaves behind besides the verified markers."""

    COUNTS = "counts"
    TREE = "tree"


def _pinned_step(name: str, output_path: str) -> StepSpec:
    """A step served from ``output_path`` that raises if a runner ever executes it."""

    def refuse(_: str) -> None:
        raise RuntimeError(f"{name} at {output_path} is pinned input to a re-apply and must not be rebuilt")

    step = StepSpec(name=name, override_output_path=output_path, deps=[], fn=refuse)
    if not step_is_built(step):
        raise FileNotFoundError(f"{output_path} has no SUCCESS status; a re-apply needs a finished {name} step")
    return step


def _write_survival(tree: str, normalize: StepSpec, fuzzy: StepSpec, verified: StepSpec, output_path: str) -> dict:
    totals = measure_survival(normalize, fuzzy, verified)
    markers = read_artifact(verified.output_path, VerifiedFuzzyDupsAttrData)
    payload = {
        "tree": tree,
        "dedup_rule": {
            "rule": DedupRule.C075,
            "max_cluster_size": MAX_CLUSTER_SIZE,
            **C075_PARAMS.model_dump(mode="json"),
        },
        "verified_path": verified.output_path,
        # Counts are over the normalized documents, which exact dedup has already thinned.
        **totals,
        "cc_canonical_doc_survival": 1 - totals["cc_canonical_removed_docs"] / totals["docs"],
        "cc_canonical_char_survival": 1 - totals["cc_canonical_removed_chars"] / totals["chars"],
        "c075_doc_survival": 1 - totals["c075_removed_docs"] / totals["docs"],
        "c075_char_survival": 1 - totals["c075_removed_chars"] / totals["chars"],
        "verifier_counters": dict(sorted(markers.counters.items())),
    }
    survival_path = f"{output_path}/{SURVIVAL_FILENAME}"
    with fsspec.open(survival_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Survival -> %s\n%s", survival_path, json.dumps(payload, indent=2))
    return {"success": True, "path": survival_path}


def build_steps(tree: str, output: ReapplyOutput, resources: ClusterStageResources) -> list[StepSpec]:
    tree = tree.rstrip("/")
    normalize = _pinned_step("normalize", f"{tree}/normalize")
    fuzzy = _pinned_step("fuzzy", f"{tree}/fuzzy")
    verification = verified_dups_steps(fuzzy=fuzzy, bucket=tree, resources=resources)
    verified = verification[-1]
    # Not stats_{rule}: the dedup drivers' own stats step owns that directory and marks it SUCCESS, and
    # StepRunner would then prune this step as already built.
    survival = StepSpec(
        name="survival",
        override_output_path=f"{tree}/survival_{DedupRule.C075}",
        deps=[normalize, fuzzy, verified],
        fn=lambda op: _write_survival(tree, normalize, fuzzy, verified, op),
    )
    if output == ReapplyOutput.COUNTS:
        return [*verification, survival]
    deduped = StepSpec(
        name="deduped",
        override_output_path=f"{tree}/deduped_{DedupRule.C075}",
        deps=[normalize, verified],
        fn=lambda op: apply_verified_dups(normalize, verified, op, _carry_provenance),
    )
    return [*verification, survival, deduped]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, help="Dedup tree root holding built normalize/ and fuzzy/ steps.")
    parser.add_argument(
        "--output",
        required=True,
        type=ReapplyOutput,
        choices=list(ReapplyOutput),
        help="counts: verified markers plus survival_c075/survival.json, which holds documents and characters "
        "removed under both rules. tree: also write deduped_c075/, a second full copy of the survivors.",
    )
    add_cluster_stage_arguments(parser)
    args = parser.parse_args()
    StepRunner().run(build_steps(args.tree, args.output, cluster_stage_resources(args)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
