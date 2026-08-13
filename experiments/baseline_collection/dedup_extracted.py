# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Marin doc-level FUZZY dedup over the first-N WARCs of an LLM extraction spec.

LLM extraction is stochastic, so exact-match dedup catches almost nothing —
two extractions of "the same page" come out as different strings. We need
MinHash-LSH-based fuzzy dedup (~0.75 Jaccard threshold over 5-char ngrams) to
collapse near-duplicate pages into a single canonical record.

Pipeline (StepRunner)::

    reshape   — read first-N done batches from resolved_{spec}.jsonl.gz (paths
                already point at the regional sources, remapped here to the
                us-central1 consolidated archive). Emit a 200-shard flat
                ``data-XXXXX-of-00200.jsonl.gz`` tree carrying ``text`` plus all
                provenance columns (``url``, ``warc_record_id``, ``warc_file``,
                ``snapshot``, and any spec-specific columns).
    normalize — datakit normalize_to_parquet: jsonl.gz → NormalizedData parquet
                with xxh3_128 ids and DedupMode.EXACT (bonus exact-doc dedup).
                Output goes under ``normalize_<hash>/outputs/main/``.
    minhash   — per-shard MinHash bucket attrs (286 perms / 26 bands / 5-char
                ngram / seed 42). Co-partitioned 1:1 with normalize output.
    fuzzy     — global LSH + connected-components across all minhash shards.
                Emits per-doc cluster markers
                ``{id, attributes: {dup_cluster_id, is_cluster_canonical}}``;
                singletons get no row, exactly one row per cluster has
                ``is_cluster_canonical=True``.
    deduped   — apply step: join normalize parquet with fuzzy attr parquet,
                drop rows where ``is_cluster_canonical=False``; keep canonicals
                and singletons. Write deduped jsonl.gz carrying text + all
                provenance columns, ready to tokenize (decon preserves columns).
    stats     — JSON summary with all step output paths and counters.

Output paths::

    gs://marin-us-central1/documents/baseline_{spec}_deduped/{n}warcs/
        reshape_<hash>/data-XXXXX-of-00200.jsonl.gz
        normalize_<hash>/outputs/main/part-XXXXX-of-YYYYY.parquet
        minhash_<hash>/outputs/<basename>.parquet
        fuzzy_<hash>/outputs/source_000/<basename>.parquet
        deduped_<hash>/data-XXXXX-of-YYYYY.jsonl.gz   <-- tokenize reads here
        stats_<hash>/dedup_stats.json

Usage::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive \\
        --extra cpu \\
        --enable-extra-resources \\
        --region us-central1 \\
        --job-name dedup-{spec}-{n}warcs \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/baseline_collection/dedup_extracted.py \\
           --spec {spec} --n {N}
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
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

CONSOLIDATED_ROOT = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated"
LEGACY_SPEC = "low_quality"
DEFAULT_MANIFEST = "experiments/distill/baseline_warcs_3000.txt"
NUM_RESHAPE_SHARDS = 200

# Region-source URI prefix → consolidated-archive URI prefix.
_REGIONAL_BUCKETS: dict[str, str] = {
    "us-central1": "gs://marin-us-central1",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "us-west4": "gs://marin-us-west4",
    "europe-west4": "gs://marin-eu-west4",
}


def _source_prefix(region: str, spec: str) -> str:
    base = f"{_REGIONAL_BUCKETS[region]}/documents/baseline_llm_extraction"
    return base if spec == LEGACY_SPEC else f"{base}/{spec}"


def _archive_prefix(region: str, spec: str) -> str:
    base = f"{CONSOLIDATED_ROOT}/by_region/{region}"
    return base if spec == LEGACY_SPEC else f"{base}/{spec}"


def _remap_to_archive(source_path: str, spec: str) -> str:
    for region in _REGIONAL_BUCKETS:
        src = _source_prefix(region, spec) + "/"
        if source_path.startswith(src):
            return _archive_prefix(region, spec) + "/" + source_path[len(src) :]
    raise ValueError(f"path {source_path!r} does not match any known source prefix for spec={spec!r}")


def _warc_path_hash(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()[:12]


def _load_manifest_hashes(manifest_path: str, n: int) -> list[str]:
    hashes: list[str] = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            hashes.append(_warc_path_hash(line))
            if len(hashes) >= n:
                break
    if len(hashes) < n:
        raise ValueError(f"manifest {manifest_path!r} has only {len(hashes)} lines but N={n}")
    return hashes


def _resolved_manifest_path(spec: str) -> str:
    fn = "resolved.jsonl.gz" if spec == LEGACY_SPEC else f"resolved_{spec}.jsonl.gz"
    return f"{CONSOLIDATED_ROOT}/resolved/{fn}"


def _load_canonical_paths(spec: str, hashes: set[str]) -> list[str]:
    """Read resolved_{spec}.jsonl.gz, filter to hashes + non-empty, remap paths."""
    from google.cloud import storage as gcs_storage

    path = _resolved_manifest_path(spec)
    logger.info("Loading resolved manifest: %s", path)
    assert path.startswith("gs://"), path
    bucket_name, _, blob_path = path[len("gs://") :].partition("/")
    client = gcs_storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    raw = blob.download_as_bytes()

    paths_by_hash: dict[str, list[str]] = {}
    skipped_empty = 0
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        h = row.get("warc_hash")
        if h not in hashes:
            continue
        if (row.get("num_records") or 0) == 0:
            skipped_empty += 1
            continue
        paths_by_hash.setdefault(h, []).append(_remap_to_archive(row["path"], spec))

    missing = sorted(h for h in hashes if h not in paths_by_hash)
    if missing:
        logger.error(
            "STRICT-DONE FAILURE: %d of %d requested WARC hashes are absent from %s",
            len(missing),
            len(hashes),
            path,
        )
        logger.error("Sample missing: %s", missing[:10])
        sys.exit(2)

    files: list[str] = []
    for h in sorted(paths_by_hash):
        files.extend(sorted(paths_by_hash[h]))
    logger.info(
        "Resolved %d canonical paths across %d WARCs (avg %.1f batches/WARC); skipped %d empty batches.",
        len(files),
        len(hashes),
        len(files) / max(len(hashes), 1),
        skipped_empty,
    )
    return files


# --------------------------------------------------------------------------
# Reshape step
# --------------------------------------------------------------------------


# Source fields that hold the primary text. The canonical ``text`` key is set
# from whichever is present, so neither is re-emitted verbatim.
_TEXT_SOURCE_FIELDS = ("text", "generated_text")
# datakit injects these onto the normalized parquet (``id`` = content hash;
# ``source_id`` = original id when an ``id_field`` was present). Neither is
# provenance, so they must not leak into the deduped output.
_DATAKIT_INTERNAL_FIELDS = ("id", "source_id")


def _carry_provenance(record: dict, text: str) -> dict:
    """Project ``record`` to ``{text, <all provenance columns>}``.

    Every column except the text-source fields and datakit-internal ids is
    carried through verbatim, so provenance (``url``, ``warc_record_id``,
    ``warc_file``, ``snapshot``, and any spec-specific columns such as
    ``pipeline_id`` / ``num_chunks``) survives dedup + decon end-to-end.
    ``text`` remains the sole dedup key.
    """
    out = {k: v for k, v in record.items() if k not in _TEXT_SOURCE_FIELDS and k not in _DATAKIT_INTERNAL_FIELDS}
    out["text"] = text
    return out


def _reshape_records(path: str) -> Iterator[dict]:
    """Yield ``{text, <provenance...>}`` records for one canonical batch.

    Carries every source column through (see :func:`_carry_provenance`) so
    provenance survives dedup. ``text`` is the canonical dedup key.
    """
    for record in load_jsonl(path):
        text = record.get("text") or record.get("generated_text")
        if not text:
            continue
        yield _carry_provenance(record, text)


def _reshape_bucket(bucket: list[str]) -> Iterator[dict]:
    """Yield records for every canonical batch in one shard's bucket."""
    for path in bucket:
        yield from _reshape_records(path)


def _reshape(spec: str, hashes_ordered: list[str], output_path: str) -> dict:
    files = _load_canonical_paths(spec, set(hashes_ordered))
    if not files:
        raise RuntimeError("no canonical input files for reshape")

    n = NUM_RESHAPE_SHARDS
    # Per-shard checkpointing: pre-bucket inputs by output shard (round-robin
    # over the deterministic _load_canonical_paths ordering) so each Zephyr
    # task reads its bucket and writes exactly one output shard via
    # write_jsonl(skip_existing=True). Each shard is durable in GCS as soon
    # as its task finishes (write_jsonl_file uses atomic_rename), so an
    # iris-coord preemption only loses currently in-flight shards instead of
    # all reshape progress.
    #
    # Replaces the previous .reshard(n) path, which required ALL 9508+ input
    # tasks to finish before any output shard could land — catastrophic when
    # 500W+ reshape crosses preemption intervals on the iris coord.
    #
    # Equivalence to old reshape: same NUM_RESHAPE_SHARDS output shards with
    # the same total set of {"text": ...} records. The records-per-shard
    # assignment differs (round-robin over input files instead of zephyr's
    # internal reshard regroup), but downstream stages (normalize → minhash →
    # fuzzy → deduped) re-hash records globally, so shard membership is
    # observably irrelevant.
    buckets = [files[i::n] for i in range(n)]
    template = f"{output_path}/data-{{shard:05d}}-of-{n:05d}.jsonl.gz"
    pipeline = Dataset.from_iterable(buckets).flat_map(_reshape_bucket).write_jsonl(template, skip_existing=True)
    ctx = ZephyrContext(
        name="reshape-extracted",
        max_workers=200,
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
    )
    ctx.execute(pipeline)
    return {"success": True, "input_files": len(files), "output_shards": n}


# --------------------------------------------------------------------------
# Apply step — join NormalizedData parquet with FuzzyDupsAttrData markers,
# drop non-canonical cluster members.
# --------------------------------------------------------------------------


def _apply_fuzzy_dups(
    normalize_step_obj: StepSpec,
    fuzzy_step_obj: StepSpec,
    output_path: str,
) -> dict:
    """Filter normalized parquet by fuzzy-dup attrs; write jsonl.gz tree."""
    norm = Artifact.load(normalize_step_obj, NormalizedData)
    fuzzy = Artifact.load(fuzzy_step_obj, FuzzyDupsAttrData)

    # There should be exactly one source (our normalize step).
    if len(fuzzy.sources) != 1:
        raise RuntimeError(f"expected 1 source in FuzzyDupsAttrData, got {len(fuzzy.sources)}")
    attr_dir = next(iter(fuzzy.sources.values())).attr_dir
    main_dir = norm.main_output_dir

    from marin.utils import fsspec_glob

    src_shards = sorted(fsspec_glob(f"{main_dir.rstrip('/')}/*.parquet"))
    if not src_shards:
        raise FileNotFoundError(f"no parquet under {main_dir}")

    # Build a (shard_basename → attr_basename) mapping. The attr file for a
    # given source shard shares its basename per the datakit invariant.
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
        """Read one source parquet + its attr parquet, drop non-canonical, yield kept records."""
        # Load non-canonical IDs from attr file. Missing attr file => all
        # records in this shard are singletons (no rows emitted).
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
            # Re-emit text + all provenance columns (carried through normalize
            # as passthrough), dropping only the datakit-internal id.
            yield _carry_provenance(r, r["text"])
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
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
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
    spec: str,
    n: int,
    manifest_path: str,
    target_partition_bytes: int = 64 * 1024 * 1024,
    region: str = "us-central1",
    fuzzy_ram_gb: int = 64,
    fuzzy_preemptible: bool = True,
) -> list[StepSpec]:
    hashes_ordered = _load_manifest_hashes(manifest_path, n)
    logger.info("Building Marin FUZZY dedup pipeline for spec=%s n=%d region=%s", spec, n, region)

    if region not in _REGIONAL_BUCKETS:
        raise ValueError(f"unknown region {region!r}; expected one of {sorted(_REGIONAL_BUCKETS)}")
    bucket = f"{_REGIONAL_BUCKETS[region]}/documents/baseline_{spec}_deduped/{n}warcs"

    # Fixed override_output_paths give us predictable layout under {bucket}/.
    # We lose content-hashed cache invalidation, but for this pipeline we don't
    # rerun with different params — only different (spec, n) which already
    # changes the bucket.

    reshape = StepSpec(
        name="reshape",
        override_output_path=f"{bucket}/reshape",
        deps=[],
        fn=lambda op: _reshape(spec, hashes_ordered, op),
    )

    # 64 MB default partitions (vs the marin 256 MB default) → ~4x more parquet
    # shards, which gives the downstream MinHash + fuzzy CC stages ~4x more
    # parallelism width. Zephyr's actor pool sizes to the upstream `from_list`
    # element count, so `max_parallelism=1024` on the fuzzy step is otherwise
    # capped by N (the normalize shard count). Outputs are identical regardless
    # of partition size; this is a pure parallelism knob. Override via the
    # --target-partition-bytes CLI flag.
    normalize = normalize_step(
        name="normalize",
        download=reshape,
        text_field="text",
        target_partition_bytes=target_partition_bytes,
        override_output_path=f"{bucket}/normalize",
    )

    minhash = compute_minhash_attrs_step(
        name="minhash",
        normalize=normalize,
        num_perms=286,
        num_bands=26,
        ngram_size=5,
        seed=42,
        worker_resources=ResourceConfig(cpu=5, ram="32g", disk="5g"),
        override_output_path=f"{bucket}/minhash",
    )

    fuzzy = compute_fuzzy_dups_attrs_step(
        name="fuzzy",
        minhash_steps=[minhash],
        max_parallelism=1024,
        # ram 32g→64g 2026-05-11 after med_low_quality OOM'd in stage1-Reduce.
        # 64g→96g attempted for low_quality 500W; iter_3 landed at 96g but
        # subsequent iters churned (workers preempting, Zephyr exhausting per-
        # shard infra-retries on iter_4) because 96GB slots are too rare on the
        # current cluster (3.8GB free across 28 candidate hosts mid-day
        # 2026-05-13). Reverting to 64g: smaller slots schedule far more
        # easily, lower churn, faster CC convergence. cc_resume=True preserves
        # iter_3 cache if 64g does OOM and we have to bump back.
        #
        # preemptible toggled back on 2026-05-15: non-preemptible 64GB pool is
        # cluster-wide tight (us-central1 AND us-east5 both showing ~10GB free
        # vs 64GB need on the reservation pool). Preemptible widens the
        # schedulable host pool by ~10x. cc_resume is the safety net — if rav-
        # style sync jobs evict a worker mid-iteration, the next worker picks
        # up from the last complete CC iter on disk. Trade-off: more churn,
        # but progress > paralysis.
        # RAM is a CLI knob (--fuzzy-ram-gb, default 64). The CC graph density —
        # not corpus size — drives peak memory: highly-templated corpora (e.g. the
        # one-call llm_simple_v1, which fails at 64g with a MemoryError/exit-1 while
        # same-size llm_pipeline_v1 succeeds) build a far denser near-dup graph and
        # need more. Bump this, not the default, for those.
        worker_resources=ResourceConfig(cpu=1, ram=f"{fuzzy_ram_gb}g", disk="10g", preemptible=fuzzy_preemptible),
        # Resume from the last complete CC iteration on disk. Added 2026-05-12
        # after med_low_quality fuzzy lost 5 CC iterations to cross-user
        # preemption — every subsequent relaunch had to redo iters 0..N.
        # `_find_last_complete_iteration` scans `{output_path}/metadata/cc/it_*/`
        # and skips ahead. Safe to leave on always; degrades to from-scratch
        # if no usable prior state exists.
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
        deps=[normalize, minhash, fuzzy, deduped],
        fn=lambda op: _write_stats(
            f"{op}/dedup_stats.json",
            {
                "spec": spec,
                "n": n,
                "manifest": manifest_path,
                "reshape_path": reshape.output_path,
                "normalize_path": normalize.output_path,
                "minhash_path": minhash.output_path,
                "fuzzy_path": fuzzy.output_path,
                "deduped_path": deduped.output_path,
                "params": {
                    "num_perms": 286,
                    "num_bands": 26,
                    "ngram_size_chars": 5,
                    "seed": 42,
                    "approx_jaccard_threshold": 0.75,
                },
            },
        ),
    )

    return [reshape, normalize, minhash, fuzzy, deduped, stats]


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True, help="Extraction spec.")
    parser.add_argument("--n", type=int, required=True, help="Number of priority WARCs (first-N).")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="WARC manifest path.")
    parser.add_argument(
        "--target-partition-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help=(
            "Bytes per normalize parquet partition. Smaller → more shards → more "
            "parallelism width for MinHash + fuzzy CC. Output records identical "
            "regardless. Default 64 MB; lower for very small corpora that "
            "otherwise get N<10 minhash workers."
        ),
    )
    parser.add_argument(
        "--region",
        default="us-central1",
        choices=sorted(_REGIONAL_BUCKETS),
        help=(
            "Region whose marin-* bucket holds the dedup outputs (override_output_paths). "
            "Set this to match the --region scheduling constraint to avoid cross-region "
            "reads/writes. Cache lookups key on the literal path, so changing region "
            "without copying cached state will rerun from scratch."
        ),
    )
    parser.add_argument(
        "--fuzzy-ram-gb",
        type=int,
        default=64,
        help=(
            "Host RAM (GB) for the fuzzy connected-components worker. Default 64 is fine for "
            "most corpora; bump (e.g. 192) for highly-templated corpora whose dense near-dup "
            "graph OOMs the CC pass (llm_simple_v1 / one-call outputs)."
        ),
    )
    parser.add_argument(
        "--fuzzy-preemptible",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Whether the fuzzy CC worker runs preemptible (default True). Use --no-fuzzy-preemptible "
            "for corpora where preemption mid-CC corrupts an it_N iteration: cc_resume then re-reads the "
            "corrupt iteration and fails deterministically ('Lost/corrupted structure for node'). Delete "
            "the fuzzy/ output first so it restarts from a clean it_0."
        ),
    )
    args = parser.parse_args()

    # CONSOLIDATED_ROOT is used by _archive_prefix / _resolved_manifest_path /
    # _load_canonical_paths to find the staged source. When running in a
    # non-central region, the source has been mirrored there to keep dedup
    # reads/writes intra-region. Mutate the module global to track.
    global CONSOLIDATED_ROOT
    CONSOLIDATED_ROOT = f"{_REGIONAL_BUCKETS[args.region]}/documents/baseline_llm_extraction_consolidated"

    steps = build_steps(
        args.spec,
        args.n,
        args.manifest,
        args.target_partition_bytes,
        args.region,
        args.fuzzy_ram_gb,
        args.fuzzy_preemptible,
    )
    StepRunner().run(steps)
    logger.info("Done. Deduped tree at %s/deduped_*/", steps[-1].output_path.rsplit("/", 1)[0])


if __name__ == "__main__":
    main()
