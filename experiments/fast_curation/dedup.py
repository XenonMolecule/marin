# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin fuzzy dedup over the fast-curation KEPT corpus, PRESERVING every provenance column.

Dedup stage of the downstream curation path. It mirrors the step DAG of ``dedup_extracted.py`` (the
canonical reference) and reuses the SAME shared ``lib/marin`` engine (``normalize_step`` +
``compute_minhash_attrs_step`` + ``compute_fuzzy_dups_attrs_step``) with IDENTICAL params (286 perms
/ 26 bands / 5-char ngram / seed 42 / Jaccard ~0.75 — same as every other curation method).

Two differences from the reference:
  1. Reads the CONSOLIDATED ``kept_text/`` parquet, not ``kept/`` (which in any single region holds
     only that region's WARCs).
  2. The reshape and apply steps carry EVERY non-text column of ``kept_text`` through — ``url``,
     ``doc_id``, ``warc_hash``, ``snapshot``, ``fasttext_score``, ``modernbert_prob`` — via
     :func:`_carry_provenance`. ``normalize_to_parquet`` keeps non id/text columns as passthrough,
     so they survive normalize -> apply untouched. Earlier versions projected to
     ``{text, modernbert_prob}``, which made every deduped tree URL-less and forced a hash-join
     recovery (``recover_lpv11_urls.py``) before WebOrganizer could label it. ``modernbert_prob`` is
     what lets the corpus be re-thresholded into quality bands; ``url`` is what WebOrganizer needs;
     ``warc_hash`` is what an N-of-pool subset needs.

Stages::

    reshape (kept_text/*.parquet -> {text, <provenance>} 200-shard tree)
      -> normalize (parquet + exact-doc dedup; provenance passthrough)
        -> minhash -> fuzzy CC -> apply (drop non-canonical near-dups, re-emit {text, <provenance>})
          -> deduped/data-*.jsonl.gz   <- decon (preserves columns) then the percentile split read here

Output::

    {bucket}/documents/baseline_{spec_id}_deduped/{n}warcs/deduped/data-*.jsonl.gz

Run inside an Iris CPU job in the region holding the consolidated kept_text tree::

    uv run iris --cluster marin job run --no-wait --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources --region us-east5 \\
        --job-name fastcur-dedup-$(date +%s) -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python -m experiments.fast_curation.dedup --spec fastpipe_v3 --region us-east5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterator

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
from zephyr.readers import load_parquet

from experiments.fast_curation.spec import get_spec
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

_REGION_TO_BUCKET: dict[str, str] = {
    "us-east5": "gs://marin-us-east5",
    "us-east1": "gs://marin-us-east1",
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-west4": "gs://marin-us-west4",
    "eu-west4": "gs://marin-eu-west4",
}

NUM_RESHAPE_SHARDS = 200


def _shard_hash(path: str) -> str:
    """``.../kept_text/data-{warc_hash}.parquet`` -> ``{warc_hash}``."""
    return os.path.basename(path).removeprefix("data-").removesuffix(".parquet")


def _kept_text_shard_paths(spec_id: str, region: str, warc_hashes: set[str] | None = None) -> list[str]:
    """All CONSOLIDATED kept_text parquet files (text + modernbert_prob) for this spec.

    ``kept_text`` is sharded one parquet per WARC (``data-{warc_hash}.parquet``). When
    ``warc_hashes`` is given, keep only those shards — the file-level subset that makes an
    N-of-pool run. Raises if any requested WARC is absent (no silent partial subset)."""
    spec = get_spec(spec_id)
    glob = f"{spec.namespace(_REGION_TO_BUCKET[region])}/kept_text/data-*.parquet"
    paths = sorted(fsspec_glob(glob))
    if warc_hashes is None:
        return paths
    kept = [p for p in paths if _shard_hash(p) in warc_hashes]
    missing = warc_hashes - {_shard_hash(p) for p in paths}
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(warc_hashes)} requested WARCs absent from kept_text "
            f"for spec {spec_id!r}; sample {sorted(missing)[:5]}"
        )
    return kept


# datakit injects these onto the normalized parquet (``id`` = content hash; ``source_id`` = original
# id when an ``id_field`` was present). Neither is provenance, so they must not leak into the output.
_DATAKIT_INTERNAL_FIELDS = ("id", "source_id")


def _carry_provenance(record: dict) -> dict:
    """Project ``record`` to ``{text, <every other column>}`` minus datakit-internal ids.

    ``modernbert_prob`` is coerced to float so the reshape and apply outputs agree on its type
    regardless of how the source parquet stored it. Everything else is carried verbatim.
    """
    out = {k: v for k, v in record.items() if k not in _DATAKIT_INTERNAL_FIELDS}
    out["modernbert_prob"] = float(record["modernbert_prob"])
    return out


def _reshape_records(path: str) -> Iterator[dict]:
    """Yield ``{text, <provenance>}`` for one kept_text PARQUET shard (skip empty-text rows)."""
    for record in load_parquet(path):
        if not record.get("text"):
            continue
        yield _carry_provenance(record)


def _reshape_bucket(bucket: list[str]) -> Iterator[dict]:
    """Yield ``{text, <provenance>}`` for every kept_text shard in one output shard's bucket."""
    for path in bucket:
        yield from _reshape_records(path)


def _reshape(files: list[str], output_path: str, n_out: int = NUM_RESHAPE_SHARDS) -> dict:
    logger.info("Reshape input: %d kept_text parquet shards -> %d output shards", len(files), n_out)
    # Per-shard checkpointing: pre-bucket inputs by output shard (round-robin over the sorted
    # kept_text ordering) so each Zephyr task reads its own bucket and writes exactly one output
    # shard via write_jsonl(skip_existing=True). Each shard is durable in GCS the moment its task
    # finishes, so a coordinator preemption loses only in-flight shards.
    #
    # This mirrors the fix already made in dedup_extracted.py: the previous `.reshard(n_out)` path
    # required ALL 10,364 input tasks to finish before any output shard could land, so nothing was
    # checkpointed and a preemption discarded the entire reshape.
    #
    # Equivalence: same n_out shards holding the same total record set. Only the record-to-shard
    # assignment differs, and normalize -> minhash -> fuzzy re-hash records globally, so shard
    # membership is observably irrelevant.
    buckets = [files[i::n_out] for i in range(n_out)]
    template = f"{output_path}/data-{{shard:05d}}-of-{n_out:05d}.jsonl.gz"
    pipeline = Dataset.from_iterable(buckets).flat_map(_reshape_bucket).write_jsonl(template, skip_existing=True)
    ctx = ZephyrContext(
        name="reshape-fastcur",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="14g", disk="10g"),
    )
    ctx.execute(pipeline)
    return {"success": True, "input_files": len(files), "output_shards": n_out}


def _apply_fuzzy_dups(normalize_step_obj: StepSpec, fuzzy_step_obj: StepSpec, output_path: str) -> dict:
    """Join NormalizedData with fuzzy markers, drop non-canonical near-dups, re-emit
    ``{text, <provenance>}`` (every passthrough column of the normalized parquet)."""
    # Post-merge artifact API: the typed payload is read off the step's record via read_artifact
    # (Artifact.load no longer exists).
    norm = read_artifact(normalize_step_obj.output_path, NormalizedData)
    fuzzy = read_artifact(fuzzy_step_obj.output_path, FuzzyDupsAttrData)

    if len(fuzzy.sources) != 1:
        raise RuntimeError(f"expected 1 source in FuzzyDupsAttrData, got {len(fuzzy.sources)}")
    attr_dir = next(iter(fuzzy.sources.values())).attr_dir
    main_dir = norm.main_output_dir

    src_shards = sorted(fsspec_glob(f"{main_dir.rstrip('/')}/*.parquet"))
    if not src_shards:
        raise FileNotFoundError(f"no parquet under {main_dir}")

    shard_pairs = []
    for s in src_shards:
        bn = os.path.basename(s)
        shard_pairs.append({"source_shard": s, "attr_shard": f"{attr_dir.rstrip('/')}/{bn}", "basename": bn})
    logger.info("apply: %d source shards under %s, joining attrs under %s", len(shard_pairs), main_dir, attr_dir)

    def _process_one(item: dict) -> Iterator[dict]:
        # Post-merge attr schema is FLAT: {id, dup_cluster_id, is_cluster_canonical} — one row per
        # doc that is in a dup cluster. The old nested r["attributes"] form read as {} on every row
        # here, which silently marked every doc canonical (dropped=0 across the whole corpus, no
        # error) before this was caught in the apply logs. No FileNotFoundError guard: the fuzzy
        # step co-partitions attrs 1:1 with the normalize shards, so a missing attr shard is
        # corruption and must raise, not quietly dedup nothing.
        non_canonical_ids: set[str] = set()
        for r in load_parquet(item["attr_shard"]):
            if r.get("is_cluster_canonical") is False:
                non_canonical_ids.add(r["id"])
        kept = dropped = 0
        for r in load_parquet(item["source_shard"]):
            if r["id"] in non_canonical_ids:
                dropped += 1
                continue
            kept += 1
            yield _carry_provenance(r)
        logger.info("  %s: kept=%d dropped=%d", item["basename"], kept, dropped)

    pipeline = (
        Dataset.from_list(shard_pairs)
        .flat_map(_process_one)
        .reshard(len(shard_pairs))
        .write_jsonl(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(name="apply-fuzzy-dups", max_workers=200, resources=ResourceConfig(cpu=1, ram="14g", disk="10g"))
    ctx.execute(pipeline)
    return {"success": True, "shards": len(shard_pairs)}


def _write_stats(stats_path: str, payload: dict) -> dict:
    with fsspec.open(stats_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Stats -> %s\n%s", stats_path, json.dumps(payload, indent=2))
    return {"success": True, "path": stats_path}


def build_steps(
    spec_id: str,
    region: str,
    target_partition_bytes: int,
    warc_hashes: set[str] | None = None,
    reshape_shards: int = NUM_RESHAPE_SHARDS,
) -> list[StepSpec]:
    kept_files = _kept_text_shard_paths(spec_id, region, warc_hashes)
    n = len(kept_files)
    if n == 0:
        raise FileNotFoundError(f"no kept_text parquet for spec {spec_id!r} in {region}; consolidate/project first")
    bucket = f"{_REGION_TO_BUCKET[region]}/documents/baseline_{spec_id}_deduped/{n}warcs"
    logger.info("Building fuzzy dedup for spec=%s region=%s over %d kept_text shards", spec_id, region, n)

    reshape = StepSpec(
        name="reshape",
        override_output_path=f"{bucket}/reshape",
        deps=[],
        fn=lambda op: _reshape(kept_files, op, reshape_shards),
    )
    normalize = normalize_step(
        name="normalize",
        download=reshape,
        text_field="text",
        target_partition_bytes=target_partition_bytes,
        # 48g: normalize's reduce side sorts a full output partition and Zephyr's external sort
        # spills to tmpfs (=RAM), so a worker pays ~2x the decompressed partition against its
        # cgroup. 14g and 28g both OOM-killed at 154k-WARC scale (2026-09-04); pair this with
        # --target-partition-bytes <= 256MB so partitions stay comfortably under the ceiling.
        worker_resources=ResourceConfig(cpu=1, ram="48g", disk="10g"),
        override_output_path=f"{bucket}/normalize",
    )
    minhash = compute_minhash_attrs_step(
        name="minhash",
        normalize=normalize,
        num_perms=286,
        num_bands=26,
        ngram_size=5,
        seed=42,
        # EXPLICIT None = no truncation, matching every other curation method. Upstream added
        # `text_cap_chars` with a 500_000 default AFTER dclm/nemotron/high_quality/fastpipe_v3 were
        # deduped, so taking the default would silently compute this corpus's MinHash signatures
        # from a prefix while theirs used full text -- a comparability break in the one step whose
        # whole point is being parameter-identical. Measured exposure is tiny (1 doc in 22,510
        # sampled exceeds 500k chars), which is exactly why it would never have shown up in the
        # results and would instead sit as an unexplained asymmetry.
        text_cap_chars=None,
        worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g", preemptible=True),
        override_output_path=f"{bucket}/minhash",
    )
    fuzzy = compute_fuzzy_dups_attrs_step(
        name="fuzzy",
        minhash_steps=[minhash],
        max_parallelism=1024,
        # 128g: this corpus (~10k WARCs) is larger than the 3k-WARC methods, so bump above the 64g
        # reference default; cc_resume protects progress if it still OOMs and has to be relaunched at
        # 192g. preemptible widens the schedulable pool ~10x.
        worker_resources=ResourceConfig(cpu=1, ram="128g", disk="10g", preemptible=True),
        cc_resume=True,
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
                "source": f"fast_curation:{spec_id}",
                "n_warcs": n,
                "region": region,
                "kept_text_glob": f"{get_spec(spec_id).namespace(_REGION_TO_BUCKET[region])}/kept_text/data-*.parquet",
                "deduped_path": deduped.output_path,
                "preserved_columns": (
                    "all kept_text columns (text, url, doc_id, warc_hash, snapshot, " "fasttext_score, modernbert_prob)"
                ),
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
    parser.add_argument("--spec", default="fastpipe_v3")
    parser.add_argument("--region", default="us-east5", choices=sorted(_REGION_TO_BUCKET))
    parser.add_argument("--target-partition-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--reshape-shards",
        type=int,
        default=NUM_RESHAPE_SHARDS,
        help="Reshape output shard count. Scale with corpus size: 14g normalize workers OOM on "
        "shards much beyond ~2GB gz (200 was sized for ~10k WARCs; a 154k-WARC corpus needs ~4000).",
    )
    parser.add_argument(
        "--warc-manifest",
        default=None,
        help="Optional WARC manifest; subset kept_text to these WARCs (sha256(line)[:12]) for an "
        "N-of-pool run. Output tree becomes baseline_{spec}_deduped/{len(subset)}warcs/. "
        "Manifest must contain only WARC paths (no comment lines beyond '#').",
    )
    args = parser.parse_args()
    warc_hashes = None
    if args.warc_manifest:
        with open(args.warc_manifest) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        warc_hashes = {hashlib.sha256(ln.encode()).hexdigest()[:12] for ln in lines}
        logger.info("Subsetting kept_text to %d WARCs from %s", len(warc_hashes), args.warc_manifest)
    StepRunner().run(build_steps(args.spec, args.region, args.target_partition_bytes, warc_hashes, args.reshape_shards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
