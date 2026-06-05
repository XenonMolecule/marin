# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# DEPRECATED — DO NOT USE FOR NEW WORK.
# ----------------------------------------------------------------------------
# Use ``dedup_extracted.py`` + ``tokenize_deduped_extracted.py`` instead.
# Those run Marin's global doc-level MinHash+LSH dedup (286 perms / 26 bands /
# 5-char ngram / 0.75 Jaccard), which is the project's canonical curation
# pipeline and scales smoothly from N=100 → N=3000 without per-N
# bloom-sharding hyperparameter tuning.
#
# This BFF script remains only so existing in-flight jobs can finish and
# their token counts can be compared against the Marin pipeline. Outputs from
# any run of this file should be moved to ``gs://.../_deprecated/`` and must
# NOT be replicated to training regions.
#
# History: written 2026-05-10 as the first manifest-first curation pipeline;
# replaced same day after we recognized that BFF's sharded bloom filter forces
# a tradeoff between memory and global-dedup recall as N scales, while Marin's
# LSH+connected-components is globally consistent at any N.
# ============================================================================

"""Manifest-first BFF dedup + tokenize for LLM-extracted specs. **DEPRECATED.**

For a given ``(spec, N)`` pair this script:

1. Loads the first N WARC paths from ``--manifest`` and hashes each to its
   12-char ``warc_hash``.
2. Reads ``resolved_{spec}.jsonl.gz`` produced by the consolidation pipeline
   (``experiments/baseline_collection/consolidate/resolve_duplicates.py``).
3. STRICT GATE: every hash must appear in the resolved manifest with at least
   one non-empty batch. If any are missing the script exits non-zero with the
   offending hashes printed, so the user can re-run consolidation after
   waiting for extraction to land.
4. Filters resolved entries to those N hashes, remaps each batch's source-
   region path to its us-central1 ``by_region/{region}/[{spec}/]data-{h}/``
   archive path, and produces an explicit ``input_files`` list (no glob).
5. Defines two ExecutorSteps: ``bff_dedup(input_files=...)`` → ``default_tokenize``.

The output dataset is named ``{spec}_{n}warcs`` and is registered alongside
other curation datasets via ``experiments/scaling_law_sweeps/curation_plan.py``.

This is "manifest-first dedup": the bloom filter sees ONLY the records from
the first N WARCs. Different N's yield independent dedup passes (a 100W and
500W dataset are NOT strict subsets of each other; that's intentional).

Usage::

    uv run iris --cluster marin job run --no-wait \\
        --cpu 4 --memory 4GB --disk 10GB \\
        --priority interactive \\
        --extra cpu --extra dclm \\
        --enable-extra-resources \\
        --region us-central1 \\
        --job-name curate-high_quality-100 \\
        -e WANDB_API_KEY $WANDB_API_KEY \\
        -e HF_TOKEN $HF_TOKEN \\
        -- python experiments/baseline_collection/curate_extracted.py \\
        --spec high_quality --n 100
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    this_output_path,
    versioned,
)
from marin.transform.bff_dedup import BffDedupConfig, bff_dedup

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

CONSOLIDATED_ROOT = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated"
LEGACY_SPEC = "low_quality"
DEFAULT_MANIFEST = "experiments/distill/baseline_warcs_3000.txt"

# Source-region URI prefix → consolidated-archive URI prefix.
# These mirror the entries in pipeline_llm_curated.py but adapt to the spec
# subdir layout (for non-legacy specs, ``by_region/{region}/{spec}/...``).
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
    """Rewrite a regional-source path to its us-central1 consolidated path."""
    for region in _REGIONAL_BUCKETS:
        src = _source_prefix(region, spec) + "/"
        if source_path.startswith(src):
            return _archive_prefix(region, spec) + "/" + source_path[len(src) :]
    raise ValueError(f"path {source_path!r} does not match any known source prefix for spec={spec!r}")


def _warc_path_hash(line: str) -> str:
    """Match the 12-char hashing convention used during extraction."""
    return hashlib.sha256(line.encode()).hexdigest()[:12]


def _load_manifest_hashes(manifest_path: str, n: int) -> list[str]:
    """Return the first N WARC hashes from the manifest (in manifest order)."""
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
        raise ValueError(f"manifest {manifest_path!r} has only {len(hashes)} lines but N={n} requested")
    return hashes


def _resolved_manifest_path(spec: str) -> str:
    fn = "resolved.jsonl.gz" if spec == LEGACY_SPEC else f"resolved_{spec}.jsonl.gz"
    return f"{CONSOLIDATED_ROOT}/resolved/{fn}"


def _load_resolved(spec: str) -> list[dict]:
    """Read resolved_{spec}.jsonl.gz; return list of dict rows.

    Uses google-cloud-storage (not fsspec) so the script also works from a
    local laptop where fsspec's aiohttp transport hits SSL certificate
    verification issues on some Pythons.
    """
    path = _resolved_manifest_path(spec)
    logger.info("Loading resolved manifest: %s", path)
    assert path.startswith("gs://"), path
    bucket_name, _, blob_path = path[len("gs://") :].partition("/")
    from google.cloud import storage as gcs_storage  # local import keeps the CLI light

    client = gcs_storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    raw = blob.download_as_bytes()
    rows: list[dict] = []
    for line in gzip.decompress(raw).decode("utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    logger.info("Loaded %d resolved entries", len(rows))
    return rows


def _filter_to_subset(resolved: list[dict], hashes: set[str], spec: str) -> tuple[list[str], list[str]]:
    """Return (input_files, missing_hashes).

    input_files is the remapped consolidated-archive URIs for non-empty batches
    of the requested hashes. missing_hashes is the set of requested hashes that
    had no non-empty rows in the resolved manifest.
    """
    by_hash: dict[str, list[dict]] = {}
    for row in resolved:
        h = row.get("warc_hash")
        if h in hashes and (row.get("num_records") or 0) > 0:
            by_hash.setdefault(h, []).append(row)

    input_files: list[str] = []
    for h in sorted(hashes):
        rows = by_hash.get(h, [])
        for row in rows:
            input_files.append(_remap_to_archive(row["path"], spec))

    missing = sorted(h for h in hashes if h not in by_hash)
    return input_files, missing


def build_steps(
    spec: str,
    n: int,
    manifest_path: str,
    output_name: str | None = None,
    shards_per_group: int = 10,
    dry_run: bool = False,
) -> list[ExecutorStep]:
    """Resolve the subset and build the dedup+tokenize ExecutorStep pair.

    On --dry-run, returns an empty list after printing the plan.
    """
    name = output_name or f"{spec}_{n}warcs"
    logger.info("Building curation plan for spec=%s n=%d (dataset name=%s)", spec, n, name)

    hashes_ordered = _load_manifest_hashes(manifest_path, n)
    hashes = set(hashes_ordered)
    logger.info("Manifest gave us %d WARC hashes (first-N).", len(hashes))

    resolved = _load_resolved(spec)
    input_files, missing = _filter_to_subset(resolved, hashes, spec)

    if missing:
        logger.error(
            "STRICT-DONE FAILURE: %d of %d requested WARC hashes have no non-empty " "rows in %s.",
            len(missing),
            n,
            _resolved_manifest_path(spec),
        )
        logger.error("Missing hashes (up to 20): %s", missing[:20])
        logger.error(
            "Re-run consolidation (inventory + resolve) once those WARCs land. "
            "Or pass a smaller --n to match what's complete."
        )
        sys.exit(2)

    logger.info(
        "Resolved %d input files across %d WARCs (avg %.1f batches/WARC).",
        len(input_files),
        len(hashes),
        len(input_files) / max(len(hashes), 1),
    )

    if dry_run:
        sample = input_files[:5]
        print(
            json.dumps(
                {
                    "spec": spec,
                    "n": n,
                    "dataset_name": name,
                    "manifest": manifest_path,
                    "resolved_manifest": _resolved_manifest_path(spec),
                    "warc_hashes": len(hashes),
                    "input_files": len(input_files),
                    "sample_input_files": sample,
                    "shards_per_group": shards_per_group,
                },
                indent=2,
            )
        )
        return []

    dedup_step = ExecutorStep(
        name=f"deduped/bff_{name}",
        description=(
            f"BFF 13-gram dedup over the first-{n} WARCs of spec={spec!r}. "
            f"Manifest-first: bloom filter only sees these records."
        ),
        fn=bff_dedup,
        config=BffDedupConfig(
            input_files=input_files,
            output_path=this_output_path(),
            shards_per_group=versioned(shards_per_group),
            expected_ngram_count_per_group=versioned(5_000_000_000),
            fp_rate=versioned(0.01),
            min_ngram_size=versioned(13),
            max_ngram_size=versioned(13),
            filtering_threshold=versioned(0.8),
            remove_type=versioned("old-both"),
        ),
    )

    tokenize_step = default_tokenize(
        name=name,
        dataset=dedup_step / "*.jsonl.gz",
        tokenizer=llama3_tokenizer,
    )

    return [tokenize_step]


def _parse_args() -> argparse.Namespace:
    """Consume our flags from sys.argv via parse_known_args.

    executor_main is wrapped in ``@draccus.wrap()`` which re-reads sys.argv
    into an ``ExecutorMainConfig``. Leaving our flags in sys.argv crashes
    that downstream parse with "unrecognized arguments". So we use
    ``parse_known_args`` to split, then rewrite sys.argv to retain only the
    unknowns (which are draccus's to interpret).
    """
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--spec", required=True, help="Extraction spec to curate.")
    p.add_argument("--n", type=int, required=True, help="Number of priority WARCs to include (first-N).")
    p.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help=f"WARC manifest path (default: {DEFAULT_MANIFEST}).",
    )
    p.add_argument(
        "--output-name",
        default=None,
        help="Override dataset name. Default: {spec}_{n}warcs.",
    )
    p.add_argument(
        "--shards-per-group",
        type=int,
        default=10,
        help="BFF group size. Each group gets its own bloom filter.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve + filter only; print plan as JSON. No ExecutorStep submission.",
    )
    args, unknown = p.parse_known_args()
    # Hand the remainder back to draccus by overwriting sys.argv.
    sys.argv = [sys.argv[0]] + unknown
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = _parse_args()

    if not Path(args.manifest).is_file():
        raise SystemExit(f"manifest not found: {args.manifest}")

    steps = build_steps(
        spec=args.spec,
        n=args.n,
        manifest_path=args.manifest,
        output_name=args.output_name,
        shards_per_group=args.shards_per_group,
        dry_run=args.dry_run,
    )
    if not steps:
        return  # --dry-run

    executor_main(
        steps=steps,
        description=(f"Curate first-{args.n} WARCs of spec={args.spec!r}: BFF dedup + llama3 tokenize."),
    )


if __name__ == "__main__":
    main()
