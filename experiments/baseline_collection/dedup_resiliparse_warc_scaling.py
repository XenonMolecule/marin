# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-N Marin fuzzy dedup for a flat-shard `{text}` extraction (resiliparse, etc.).

Despite the name, this is the GENERAL flat-shard dedup driver: it works for any extraction
that lands as a flat per-WARC/per-shard tree with a `text` field (resiliparse, raw HTML->text,
the fast_curation cascade, ...). The real dedup engine is the shared `lib/marin` primitives
(`normalize_step` + `compute_minhash_attrs_step` + `compute_fuzzy_dups_attrs_step`); this file
and `dedup_extracted.py` are thin wrappers that differ ONLY in their reshape/input-discovery.
`dedup_extracted.py` is the canonical reference (it reads the consolidated LLM-extraction archive
via a `resolved_{spec}.jsonl.gz` manifest); this one reads raw flat shards instead.

Mirrors `dedup_extracted.py` (LLM-extracted quality bands) but reads from
the raw resiliparse output instead of the consolidated LLM-extraction archive.

Why per-N (and not slice the existing 3000-WARC deduped corpus)::

    Fuzzy dedup is N-dependent. At N=100 we only drop duplicates *within*
    those 100 WARCs. Slicing the 3000-WARC deduped output to first-100 would
    remove docs that got dedup'd against the OTHER 2900 WARCs — those docs
    would have survived a true 100-WARC dedup. That'd give resiliparse an
    artificial token disadvantage relative to LLM-extracted specs (which
    DO get per-N dedup via `dedup_extracted.py`).

Input::

    gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa/
        data-00000-of-03000.jsonl.gz  ← one shard per WARC, in WARC-list order
        data-00001-of-03000.jsonl.gz
        ...

`--n` selects the first N shards (lexicographic), matching the convention
in `dedup_extracted.py`: N is a stable prefix of
`experiments/distill/baseline_warcs_3000.txt`.

Output::

    gs://marin-us-central2/documents/baseline_resiliparse_deduped/{n}warcs/
        reshape/data-XXXXX-of-00200.jsonl.gz
        normalize/outputs/main/part-XXXXX-of-YYYYY.parquet
        minhash/outputs/<basename>.parquet
        fuzzy/outputs/source_000/<basename>.parquet
        deduped/data-XXXXX-of-YYYYY.jsonl.gz   ← tokenize reads here
        stats/dedup_stats.json

NOTE: dedup output lands in the SAME region the job runs in (see ``--region``).
Tokenize and downstream training then either pin to that region or pay a one-
time mirror. For N=500/1000/2000 the recommended path is us-east5 preemptible
with the raw extraction pre-mirrored from us-central2 to us-east5 (one-time
$3.56 for ~178 GB) so every step is intra-region.

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive \\
        --extra cpu \\
        --enable-extra-resources \\
        --region us-east5 \\
        --job-name dedup-resiliparse-{n}warcs-$(date +%s) \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/dedup_resiliparse_warc_scaling.py \\
           --n {n} --region us-east5 --target-partition-bytes 33554432

`--target-partition-bytes` cheat sheet (see also curation_playbook.md):
    N=100   → 16 MB (16777216)
    N=500   → 32 MB (33554432)
    N=1000+ → 64 MB (67108864, the marin default)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections.abc import Iterator
from enum import StrEnum

import fsspec
from fray.types import ResourceConfig
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import read_artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs_step
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl, load_parquet

from experiments.baseline_collection.verified_dedup import (
    C075_PARAMS,
    ClusterStageResources,
    DedupRule,
    add_cluster_stage_arguments,
    apply_verified_dups,
    cluster_stage_resources,
    verified_dups_steps,
)

logger = logging.getLogger(__name__)

# Region → marin-* bucket. Raw input AND dedup output both follow this map;
# pass --region to pick one. For N=500/1000/2000 the raw extraction must be
# pre-mirrored into the chosen region (us-east5 mirror is a one-time $3.56).
_REGION_TO_BUCKET: dict[str, str] = {
    "us-east5": "gs://marin-us-east5",
    "us-central2": "gs://marin-us-central2",
    "eu-west4": "gs://marin-eu-west4",
}


# The head-based 3000-WARC extraction these defaults reproduce. Override
# --extraction-name / --output-name to dedup a different extraction (e.g. the
# random-sample batch) without colliding with these head-based outputs.
DEFAULT_EXTRACTION_NAME = "baseline_resiliparse-19bdaa"
DEFAULT_OUTPUT_NAME = "baseline_resiliparse_deduped"
DEFAULT_TOTAL_WARC_SHARDS = 3000
NUM_RESHAPE_SHARDS = 200
# Seed of the nested random shard order. Every ``--shard-order random`` run of one extraction
# draws the same permutation, so its N-WARC subsets nest: 300 is a subset of 3000 of 10364.
SHARD_SAMPLE_SEED = 0


class ShardOrder(StrEnum):
    """Which N of an extraction's shards a run reads."""

    PREFIX = "prefix"
    RANDOM = "random"


def _raw_path(region: str, extraction_name: str) -> str:
    return f"{_REGION_TO_BUCKET[region]}/extracted/{extraction_name}"


def _output_prefix(region: str, output_name: str) -> str:
    return f"{_REGION_TO_BUCKET[region]}/documents/{output_name}"


def shard_indices(n: int, total_shards: int, shard_order: ShardOrder) -> list[int]:
    """The N shards a run reads: the first N in WARC order, or a seeded nested random draw."""
    if n > total_shards:
        raise ValueError(f"--n {n} exceeds available {total_shards} resiliparse shards")
    if shard_order == ShardOrder.PREFIX:
        return list(range(n))
    order = list(range(total_shards))
    random.Random(SHARD_SAMPLE_SEED).shuffle(order)
    return sorted(order[:n])


def _input_shard_paths(
    n: int, region: str, extraction_name: str, total_shards: int, shard_order: ShardOrder
) -> list[str]:
    raw_path = _raw_path(region, extraction_name)
    return [
        f"{raw_path}/data-{i:05d}-of-{total_shards:05d}.jsonl.gz" for i in shard_indices(n, total_shards, shard_order)
    ]


# --------------------------------------------------------------------------
# Reshape step — read raw resiliparse, project to {text}, write 200-shard tree.
# --------------------------------------------------------------------------


def _reshape_records(path: str) -> Iterator[dict]:
    """Yield ``{text}`` records for one raw resiliparse shard."""
    for record in load_jsonl(path):
        text = record.get("text")
        if not text:
            continue
        yield {"text": text}


def _reshape(
    n: int,
    region: str,
    output_path: str,
    extraction_name: str,
    total_shards: int,
    shard_order: ShardOrder,
    n_out: int,
) -> dict:
    files = _input_shard_paths(n, region, extraction_name, total_shards, shard_order)
    logger.info("Reshape input: %d raw resiliparse shards -> %d output shards", len(files), n_out)

    template = f"{output_path}/data-{{shard:05d}}-of-{n_out:05d}.jsonl.gz"
    pipeline = (
        Dataset.from_iterable(files).flat_map(_reshape_records).reshard(n_out).write_jsonl(template, skip_existing=True)
    )
    ctx = ZephyrContext(
        name="reshape-resiliparse",
        max_workers=200,
        # cpu=1/ram=14g fits on e2-highmem-2 ondemand (1.9 cpu, 14.1g
        # usable) so reshape workers run on the fast CPU VM pool instead
        # of queueing against scarce us-central2 v4 host autoscale.
        resources=ResourceConfig(cpu=1, ram="14g", disk="10g"),
    )
    ctx.execute(pipeline)
    return {"success": True, "input_files": len(files), "output_shards": n_out}


# --------------------------------------------------------------------------
# Apply step — copied verbatim from dedup_extracted.py. Joins NormalizedData
# parquet with FuzzyDupsAttrData markers, drops non-canonical cluster members.
# --------------------------------------------------------------------------


def _apply_fuzzy_dups(
    normalize_step_obj: StepSpec,
    fuzzy_step_obj: StepSpec,
    output_path: str,
) -> dict:
    # Post-merge artifact API: Artifact.load no longer exists.
    norm = read_artifact(normalize_step_obj.output_path, NormalizedData)
    fuzzy = read_artifact(fuzzy_step_obj.output_path, FuzzyDupsAttrData)

    if len(fuzzy.sources) != 1:
        raise RuntimeError(f"expected 1 source in FuzzyDupsAttrData, got {len(fuzzy.sources)}")
    attr_dir = next(iter(fuzzy.sources.values())).attr_dir
    main_dir = norm.main_output_dir

    from experiments.fsspec_paths import fsspec_glob

    src_shards = sorted(fsspec_glob(f"{main_dir.rstrip('/')}/*.parquet"))
    if not src_shards:
        raise FileNotFoundError(f"no parquet under {main_dir}")

    import os as _os

    shard_pairs = []
    for s in src_shards:
        bn = _os.path.basename(s)
        attr = f"{attr_dir.rstrip('/')}/{bn}"
        shard_pairs.append({"source_shard": s, "attr_shard": attr, "basename": bn})

    logger.info(
        "apply: %d source shards under %s, joining with attrs under %s",
        len(shard_pairs),
        main_dir,
        attr_dir,
    )

    def _process_one(item: dict) -> Iterator[dict]:
        non_canonical_ids: set[str] = set()
        try:
            for r in load_parquet(item["attr_shard"]):
                # Post-merge attrs are FLAT: {id, dup_cluster_id, is_cluster_canonical}. Reading the
                # old nested "attributes" shape silently matches nothing, so fail loudly instead.
                if "is_cluster_canonical" not in r:
                    raise KeyError(
                        f"{item['attr_shard']}: expected flat fuzzy attrs with 'is_cluster_canonical', "
                        f"got columns {sorted(r)}"
                    )
                if r["is_cluster_canonical"] is False:
                    non_canonical_ids.add(r["id"])
        except FileNotFoundError:
            pass
        kept = 0
        dropped = 0
        for r in load_parquet(item["source_shard"]):
            if r["id"] in non_canonical_ids:
                dropped += 1
                continue
            kept += 1
            yield {"text": r["text"]}
        logger.info("  %s: kept=%d dropped=%d", item["basename"], kept, dropped)

    pipeline = (
        Dataset.from_list(shard_pairs)
        .flat_map(_process_one)
        .reshard(len(shard_pairs))
        .write_jsonl(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz",
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(
        name="apply-fuzzy-dups",
        max_workers=200,
        # cpu=1/ram=14g fits on e2-highmem-2 ondemand — same reasoning as
        # reshape; apply is light (load_parquet → set membership → write).
        resources=ResourceConfig(cpu=1, ram="14g", disk="10g"),
    )
    ctx.execute(pipeline)
    return {"success": True, "shards": len(shard_pairs)}


# --------------------------------------------------------------------------
# Pipeline builder
# --------------------------------------------------------------------------


def _write_stats(stats_path: str, payload: dict) -> dict:
    with fsspec.open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Stats → %s", stats_path)
    logger.info("%s", json.dumps(payload, indent=2))
    return {"success": True, "path": stats_path}


def build_steps(
    n: int,
    region: str,
    target_partition_bytes: int = 64 * 1024 * 1024,
    extraction_name: str = DEFAULT_EXTRACTION_NAME,
    output_name: str = DEFAULT_OUTPUT_NAME,
    total_shards: int = DEFAULT_TOTAL_WARC_SHARDS,
    *,
    dedup_rule: DedupRule,
    cluster_resources: ClusterStageResources,
    shard_order: ShardOrder = ShardOrder.PREFIX,
    reshape_shards: int = NUM_RESHAPE_SHARDS,
    normalize_ram: str = "14g",
) -> list[StepSpec]:
    logger.info(
        "Building Marin FUZZY dedup pipeline for resiliparse n=%d region=%s extraction=%s",
        n,
        region,
        extraction_name,
    )

    # A random draw never shares a tree with the WARC-order prefix of the same N.
    suffix = "" if shard_order == ShardOrder.PREFIX else f"_{shard_order}"
    bucket = f"{_output_prefix(region, output_name)}/{n}warcs{suffix}"

    reshape = StepSpec(
        name="reshape",
        override_output_path=f"{bucket}/reshape",
        deps=[],
        fn=lambda op: _reshape(n, region, op, extraction_name, total_shards, shard_order, reshape_shards),
    )

    normalize = normalize_step(
        name="normalize",
        download=reshape,
        text_field="text",
        target_partition_bytes=target_partition_bytes,
        # cpu=1/ram=14g fits on e2-highmem-2 ondemand — default
        # (cpu=2/ram=16g) was queueing against scarce us-central2 v4 hosts.
        # The content-hash reduce concentrates mega-duplicated docs into one "whale" shard whose
        # size grows with the corpus; fast_curation/dedup.py needed 96g at 154k WARCs.
        worker_resources=ResourceConfig(cpu=1, ram=normalize_ram, disk="10g"),
        override_output_path=f"{bucket}/normalize",
    )

    minhash = compute_minhash_attrs_step(
        name="minhash",
        normalize=normalize,
        num_perms=286,
        num_bands=26,
        ngram_size=5,
        seed=42,
        # preemptible=True for us-east5 (no reserved capacity exists there).
        # us-east5 has abundant Ready preemptible v5p, so retries land
        # instantly — unlike us-central2 where each preemption cost ~2 hr
        # of wait_ready timeout because the pool was exhausted.
        worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g", preemptible=True),
        override_output_path=f"{bucket}/minhash",
    )

    fuzzy = compute_fuzzy_dups_attrs_step(
        name="fuzzy",
        minhash_steps=[minhash],
        max_parallelism=1024,
        # 192g RAM bumped from 128g after still OOMing at p0 of multi-pool
        # CC at N=100. Resiliparse LSH key space is much bigger than the
        # LLM-extracted spec at the same N. v6e-8 host has 720g so 192g
        # fits 3 concurrent; v5p-8 host has 448g so 192g fits 2 concurrent.
        worker_resources=ResourceConfig(cpu=1, ram="192g", disk="10g", preemptible=True),
        override_output_path=f"{bucket}/fuzzy",
    )

    # Both rules share every step above, so they judge the same candidate components. They write
    # separate trees: `deduped/` holds what the paper reports and must never change under it.
    if dedup_rule == DedupRule.CC_CANONICAL:
        verification: list[StepSpec] = []
        deduped = StepSpec(
            name="deduped",
            override_output_path=f"{bucket}/deduped",
            deps=[normalize, fuzzy],
            fn=lambda op: _apply_fuzzy_dups(normalize, fuzzy, op),
        )
        stats_dir = f"{bucket}/stats"
        rule_params: dict = {"rule": dedup_rule}
    else:
        verification = verified_dups_steps(fuzzy=fuzzy, bucket=bucket, resources=cluster_resources)
        verified = verification[-1]
        deduped = StepSpec(
            name="deduped",
            override_output_path=f"{bucket}/deduped_{dedup_rule}",
            deps=[normalize, verified],
            fn=lambda op: apply_verified_dups(normalize, verified, op, lambda record: {"text": record["text"]}),
        )
        stats_dir = f"{bucket}/stats_{dedup_rule}"
        rule_params = {"rule": dedup_rule, "verified_path": verified.output_path, **C075_PARAMS.model_dump(mode="json")}

    stats = StepSpec(
        name="stats",
        override_output_path=stats_dir,
        deps=[reshape, normalize, minhash, fuzzy, *verification, deduped],
        fn=lambda op: _write_stats(
            f"{op}/dedup_stats.json",
            {
                "source": "resiliparse",
                "n_warcs": n,
                "region": region,
                "extraction_name": extraction_name,
                "output_name": output_name,
                "raw_path": _raw_path(region, extraction_name),
                "reshape_path": reshape.output_path,
                "normalize_path": normalize.output_path,
                "minhash_path": minhash.output_path,
                "fuzzy_path": fuzzy.output_path,
                "deduped_path": deduped.output_path,
                "fuzzy_params": {
                    "num_perms": 286,
                    "num_bands": 26,
                    "ngram_size_chars": 5,
                    "seed": 42,
                    "approx_jaccard_threshold": 0.75,
                },
                "target_partition_bytes": target_partition_bytes,
                "shard_order": shard_order,
                "reshape_shards": reshape_shards,
                "dedup_rule": rule_params,
            },
        ),
    )

    return [reshape, normalize, minhash, fuzzy, *verification, deduped, stats]


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, required=True, help="WARC count (first-N of resiliparse extraction)")
    parser.add_argument(
        "--region",
        default="us-east5",
        choices=sorted(_REGION_TO_BUCKET),
        help="Region whose marin-* bucket holds raw input AND dedup outputs. "
        "Raw extraction must already be present in this region (pre-mirrored).",
    )
    parser.add_argument(
        "--target-partition-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Normalize parquet target partition size; controls downstream parallelism width.",
    )
    parser.add_argument(
        "--extraction-name",
        default=DEFAULT_EXTRACTION_NAME,
        help="Raw resiliparse extraction dir under <bucket>/extracted/ "
        f"(default: {DEFAULT_EXTRACTION_NAME}, the head-based 3000 batch). "
        "For the random batch pass baseline_resiliparse-<hash> from pipeline.py output.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Deduped output dir under <bucket>/documents/ "
        f"(default: {DEFAULT_OUTPUT_NAME}). Use a distinct name for the random "
        "batch so it does not collide with the head-based deduped outputs.",
    )
    parser.add_argument(
        "--total-shards",
        type=int,
        default=DEFAULT_TOTAL_WARC_SHARDS,
        help="Shard count in the extraction's data-XXXXX-of-NNNNN filenames " f"(default: {DEFAULT_TOTAL_WARC_SHARDS}).",
    )
    parser.add_argument(
        "--shard-order",
        type=ShardOrder,
        choices=list(ShardOrder),
        default=ShardOrder.PREFIX,
        help="prefix: the first N shards in WARC order (the paper's per-N trees). random: a seeded "
        "nested draw of N shards, written to a separate {n}warcs_random tree.",
    )
    parser.add_argument(
        "--reshape-shards",
        type=int,
        default=NUM_RESHAPE_SHARDS,
        help="Reshape output shard count. Scale with corpus size: the normalize merge opens one reader "
        "per shard, so a 400M-document extraction wants ~800.",
    )
    parser.add_argument("--normalize-ram", default="14g", help="Normalize worker RAM; the whale shard grows with N.")
    parser.add_argument(
        "--dedup-rule",
        required=True,
        type=DedupRule,
        choices=list(DedupRule),
        help="cc_canonical: drop every non-canonical member of a MinHash component, into deduped/ "
        "(the paper's rule). c075: drop only documents a longer survivor in the component contains "
        "at >= 0.75 word 3-gram containment, into deduped_c075/ (the Harrier hero rule).",
    )
    add_cluster_stage_arguments(parser)
    args = parser.parse_args()

    steps = build_steps(
        args.n,
        args.region,
        target_partition_bytes=args.target_partition_bytes,
        extraction_name=args.extraction_name,
        output_name=args.output_name,
        total_shards=args.total_shards,
        dedup_rule=args.dedup_rule,
        cluster_resources=cluster_stage_resources(args),
        shard_order=args.shard_order,
        reshape_shards=args.reshape_shards,
        normalize_ram=args.normalize_ram,
    )
    StepRunner().run(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
