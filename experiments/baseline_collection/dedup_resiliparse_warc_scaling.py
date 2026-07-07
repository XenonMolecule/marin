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
import sys
from collections.abc import Iterator

import fsspec
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData, normalize_step
from marin.execution.artifact import Artifact
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.classification.deduplication.fuzzy_dups import (
    FuzzyDupsAttrData,
    compute_fuzzy_dups_attrs_step,
)
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs_step
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_jsonl, load_parquet

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


def _raw_path(region: str, extraction_name: str) -> str:
    return f"{_REGION_TO_BUCKET[region]}/extracted/{extraction_name}"


def _output_prefix(region: str, output_name: str) -> str:
    return f"{_REGION_TO_BUCKET[region]}/documents/{output_name}"


def _input_shard_paths(n: int, region: str, extraction_name: str, total_shards: int) -> list[str]:
    """Return the first-N raw resiliparse shard paths (lexicographic = WARC order)."""
    if n > total_shards:
        raise ValueError(f"--n {n} exceeds available {total_shards} resiliparse shards")
    raw_path = _raw_path(region, extraction_name)
    return [f"{raw_path}/data-{i:05d}-of-{total_shards:05d}.jsonl.gz" for i in range(n)]


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


def _reshape(n: int, region: str, output_path: str, extraction_name: str, total_shards: int) -> dict:
    files = _input_shard_paths(n, region, extraction_name, total_shards)
    logger.info("Reshape input: %d raw resiliparse shards", len(files))

    n_out = NUM_RESHAPE_SHARDS
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
    norm = Artifact.load(normalize_step_obj, NormalizedData)
    fuzzy = Artifact.load(fuzzy_step_obj, FuzzyDupsAttrData)

    if len(fuzzy.sources) != 1:
        raise RuntimeError(f"expected 1 source in FuzzyDupsAttrData, got {len(fuzzy.sources)}")
    attr_dir = next(iter(fuzzy.sources.values())).attr_dir
    main_dir = norm.main_output_dir

    from marin.utils import fsspec_glob

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
                attrs = r.get("attributes") or {}
                if attrs.get("is_cluster_canonical") is False:
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
) -> list[StepSpec]:
    logger.info(
        "Building Marin FUZZY dedup pipeline for resiliparse n=%d region=%s extraction=%s",
        n,
        region,
        extraction_name,
    )

    bucket = f"{_output_prefix(region, output_name)}/{n}warcs"

    reshape = StepSpec(
        name="reshape",
        override_output_path=f"{bucket}/reshape",
        deps=[],
        fn=lambda op: _reshape(n, region, op, extraction_name, total_shards),
    )

    normalize = normalize_step(
        name="normalize",
        download=reshape,
        text_field="text",
        target_partition_bytes=target_partition_bytes,
        # cpu=1/ram=14g fits on e2-highmem-2 ondemand — default
        # (cpu=2/ram=16g) was queueing against scarce us-central2 v4 hosts.
        worker_resources=ResourceConfig(cpu=1, ram="14g", disk="10g"),
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

    deduped = StepSpec(
        name="deduped",
        override_output_path=f"{bucket}/deduped",
        deps=[normalize, fuzzy],
        fn=lambda op: _apply_fuzzy_dups(normalize, fuzzy, op),
    )

    stats = StepSpec(
        name="stats",
        override_output_path=f"{bucket}/stats",
        deps=[reshape, normalize, minhash, fuzzy, deduped],
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
            },
        ),
    )

    return [reshape, normalize, minhash, fuzzy, deduped, stats]


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
    args = parser.parse_args()

    steps = build_steps(
        args.n,
        args.region,
        target_partition_bytes=args.target_partition_bytes,
        extraction_name=args.extraction_name,
        output_name=args.output_name,
        total_shards=args.total_shards,
    )
    StepRunner().run(steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
