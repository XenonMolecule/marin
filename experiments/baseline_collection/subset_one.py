# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone child: filter ONE method's filtered output to a WARC subset and tokenize.

Runs inside a region-pinned Iris CPU job. Calls Zephyr directly (no Ray
executor) — the parent Iris job's worker pool runs both the filter pipeline
and the tokenizer pipeline.

Usage (typically invoked by ``launch_subsets.py``)::

    iris --config lib/iris/examples/marin.yaml job run --memory 64GB --cpu 4 \\
        --no-wait --job-name subset-dclm-100warcs \\
        -- python experiments/baseline_collection/subset_one.py \\
        --method dclm --n 100

Output paths (deterministic, NOT executor-hashed — written verbatim to skip
the Ray executor entirely)::

    gs://marin-{region}/filtered_subsets/baseline_{method}_{n}warcs/
        data-{shard:05d}-of-{total:05d}.jsonl.gz
    gs://marin-{region}/tokenized/baseline_{method}_{n}warcs/
        train/<levanter cache>
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import fsspec
from fray.types import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from rigging.filesystem import marin_prefix
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, zephyr_worker_ctx

logger = logging.getLogger(__name__)


# --- Path SUFFIXES (joined with marin_prefix() at run time) ---
# Every Iris child is region-pinned to a region whose primary bucket already
# holds the source data, so we never need to hardcode the bucket prefix.
# DCLM: -23e9be is the *tokenized* cache hash; -8fff14 is the raw filter;
# -1ac313 is the resharded output that feeds the standard tokenize step.
DCLM_FILTERED_SUFFIX = "filtered/baseline_dclm_resharded-1ac313"
FINEWEB_EDU_FILTERED_SUFFIX = "filtered/baseline_fineweb_edu-72c2c7"
NEMOTRON_FULL_FILTERED_SUFFIX = "filtered/baseline_nemotron_full-347dfe"
# Nemotron-CC-HQ — quality=high subset of nemotron_full. Filter logic in
# experiments/baseline_collection/filter_nemotron_quality.py preserves the
# `url` field that subset_one filters on. Currently only exists on
# us-central1; not mirrored.
NEMOTRON_QHIGH_FILTERED_SUFFIX = "filtered/baseline_nemotron_qhigh-v1"
LLM_CURATED_DOCUMENTS_SUFFIX = "documents/baseline_llm_curated-050243"
LLM_CURATED_DCLM_FILTERED_DEDUPED_SUFFIX = "deduped/bff_llm_curated_dclm_filtered_v1"
WARC_HTML_DOWNLOAD_SUFFIX = "raw/commoncrawl/baseline_3000-265ff5"

# Canonical metadata: us-central2 primary, mirrored to us-central1. Either
# region's prefix resolves to the same hash directory under metadata/.
WARC_METADATA_SUFFIX = "metadata/baseline_warc_metadata-d671da"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSET_MANIFEST_TEMPLATE = str(REPO_ROOT / "experiments/distill/subsets/baseline_warcs_{N}.txt")

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Per-method source suffix + filter mode. Final source/output/metadata paths
# are built from these by prepending marin_prefix() at run time, which
# resolves to gs://marin-{worker_region}. Every method is region-pinned so
# the source data is already in the worker's local bucket.
METHOD_SPEC: dict[str, dict] = {
    "dclm": {
        "region": "us-central2",
        "source_suffix": DCLM_FILTERED_SUFFIX,
        "source_pattern": "data-*.jsonl.gz",
        "mode": "metadata_record_id",
        "use_metadata": True,
        "filter_name": "baseline_dclm",
        "tokenize_name": "baseline_dclm",
    },
    "fineweb_edu": {
        "region": "us-central2",
        "source_suffix": FINEWEB_EDU_FILTERED_SUFFIX,
        "source_pattern": "data-*.jsonl.gz",
        "mode": "manifest_file_path",
        "use_metadata": False,
        "filter_name": "baseline_fineweb_edu",
        "tokenize_name": "baseline_fineweb_edu",
    },
    "nemotron_full": {
        "region": "us-central1",
        "source_suffix": NEMOTRON_FULL_FILTERED_SUFFIX,
        "source_pattern": "*.jsonl.gz",
        "mode": "metadata_url",
        "use_metadata": True,
        "filter_name": "baseline_nemotron_full_bos_fixed",
        "tokenize_name": "baseline_nemotron_full_bos_fixed",
    },
    # Nemotron-CC-HQ (quality=high only, both kind=actual + kind=synthetic).
    # Same filter mode/keys as nemotron_full since the qhigh JSONL is a
    # row-subset of the nemotron_full filtered output and preserves the `url`
    # field.
    "nemotron_qhigh": {
        "region": "us-central1",
        "source_suffix": NEMOTRON_QHIGH_FILTERED_SUFFIX,
        "source_pattern": "*.jsonl.gz",
        "mode": "metadata_url",
        "use_metadata": True,
        "filter_name": "baseline_nemotron_qhigh",
        "tokenize_name": "baseline_nemotron_qhigh",
    },
    "llm_curated": {
        "region": "us-central1",
        "source_suffix": LLM_CURATED_DOCUMENTS_SUFFIX,
        "source_pattern": "data-*.jsonl.gz",
        "mode": "manifest_warc_file",
        "use_metadata": False,
        "filter_name": "baseline_llm_curated_bos_fixed",
        "tokenize_name": "baseline_llm_curated_bos_fixed",
    },
    "llm_curated_dclm_filtered": {
        "region": "us-central1",
        "source_suffix": LLM_CURATED_DCLM_FILTERED_DEDUPED_SUFFIX,
        "source_pattern": "data-*.jsonl.gz",
        "mode": "manifest_warc_file",
        "use_metadata": False,
        "filter_name": "baseline_llm_curated_dclm_filtered",
        "tokenize_name": "baseline_llm_curated_dclm_filtered",
    },
    # Resiliparse needs re-extraction from HTML — its existing extracted output
    # only carries (text, url) and URLs can collide across snapshots within the
    # 3000-WARC pool. Filter HTML records by metadata.warc_file, then run
    # resiliparse extract_plain_text on the filtered HTML.
    "resiliparse": {
        "region": "us-central2",
        "source_suffix": WARC_HTML_DOWNLOAD_SUFFIX,
        "source_pattern": "data-*.jsonl.gz",
        "mode": "resiliparse_html",
        "use_metadata": False,
        "filter_name": "baseline_resiliparse",
        "tokenize_name": "baseline_resiliparse",
    },
}


# --- Helpers ----------------------------------------------------------------


def _normalize_warc_path(s: str) -> str:
    s = s.strip()
    for prefix in ("s3://commoncrawl/", "gs://commoncrawl/", "https://data.commoncrawl.org/"):
        if s.startswith(prefix):
            return s[len(prefix) :]
    return s


def _load_subset_warc_set(manifest_path: str, normalize: bool = False) -> set[str]:
    out: set[str] = set()
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.add(_normalize_warc_path(line) if normalize else line)
    return out


def _glob_files(pattern: str) -> list[str]:
    fs = fsspec.filesystem("gcs") if pattern.startswith("gs://") else fsspec.filesystem("file")
    raw = fs.glob(pattern)
    if pattern.startswith("gs://"):
        return [p if p.startswith("gs://") else f"gs://{p}" for p in raw]
    return list(raw)


def _build_key_set_from_metadata(metadata_glob: str, subset_warc_set: set[str], key_field: str) -> set[str]:
    keys: set[str] = set()
    files = _glob_files(metadata_glob)
    logger.info("Loading metadata key set from %d shards (key_field=%s)", len(files), key_field)
    for path in files:
        with fsspec.open(path, "rb") as fh, gzip.open(fh, "rt") as gz:
            for line in gz:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("warc_file") in subset_warc_set:
                    val = rec.get(key_field, "")
                    if val:
                        keys.add(val)
    logger.info(
        "Built key set: %d unique %s values from %d subset WARCs",
        len(keys),
        key_field,
        len(subset_warc_set),
    )
    return keys


# --- Filter step ------------------------------------------------------------


@dataclass(frozen=True)
class FilterSubsetConfig:
    input_glob: str
    output_path: str
    subset_manifest_path: str
    metadata_glob: str
    mode: str  # metadata_record_id | metadata_url | manifest_file_path | manifest_warc_file


def _filter_one_shard(file_path: str) -> list[dict]:
    ctx = zephyr_worker_ctx()
    key_set: set[str] = ctx.get_shared("key_set")
    filter_field: str = ctx.get_shared("filter_field")
    normalize_value: bool = ctx.get_shared("normalize_value")

    out: list[dict] = []
    matched = total = 0
    with fsspec.open(file_path, "rb") as fh, gzip.open(fh, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            val = rec.get(filter_field, "")
            if normalize_value:
                val = _normalize_warc_path(val)
            if val in key_set:
                matched += 1
                out.append(rec)
    if total > 0:
        logger.info("  %s: %d/%d matched", file_path.split("/")[-1], matched, total)
    return out


def _process_html_shard(file_path: str) -> list[dict]:
    """Worker: filter HTML records by metadata.warc_file ∈ subset, run resiliparse, return (text, url)."""
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    ctx = zephyr_worker_ctx()
    warc_file_set: set[str] = ctx.get_shared("key_set")

    out: list[dict] = []
    matched = total = kept = 0
    with fsspec.open(file_path, "rb") as fh, gzip.open(fh, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            total += 1
            rec = json.loads(line)
            warc_file = rec.get("metadata", {}).get("warc_file", "")
            if warc_file not in warc_file_set:
                continue
            matched += 1
            html = rec.get("html", "")
            url = rec.get("url", "")
            if not html:
                continue
            try:
                tree = HTMLTree.parse(html)
                text = extract_plain_text(tree, main_content=True, alt_texts=False, noscript=False).strip()
            except Exception:
                text = ""
            if len(text) >= 50:
                out.append({"text": text, "url": url})
                kept += 1
    if total > 0:
        logger.info(
            "  %s: matched=%d/%d warc-file, kept=%d after resiliparse", file_path.split("/")[-1], matched, total, kept
        )
    return out


def filter_subset(config: FilterSubsetConfig) -> None:
    if config.mode == "metadata_record_id":
        subset_warc_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        key_set = _build_key_set_from_metadata(f"{config.metadata_glob}/*.jsonl.gz", subset_warc_set, "warc_record_id")
        filter_field, normalize_value = "warc_record_id", False
        worker_fn = _filter_one_shard
    elif config.mode == "metadata_url":
        subset_warc_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        key_set = _build_key_set_from_metadata(f"{config.metadata_glob}/*.jsonl.gz", subset_warc_set, "url")
        filter_field, normalize_value = "url", False
        worker_fn = _filter_one_shard
    elif config.mode == "manifest_file_path":
        key_set = _load_subset_warc_set(config.subset_manifest_path, normalize=True)
        filter_field, normalize_value = "file_path", True
        worker_fn = _filter_one_shard
    elif config.mode == "manifest_warc_file":
        key_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        filter_field, normalize_value = "warc_file", False
        worker_fn = _filter_one_shard
    elif config.mode == "resiliparse_html":
        # HTML re-extraction: the key set IS the subset WARC paths; we'll match
        # records against metadata.warc_file directly inside the worker.
        key_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        filter_field, normalize_value = "_html_resiliparse", False
        worker_fn = _process_html_shard
    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    if not key_set:
        raise RuntimeError(f"Empty key set for {config.subset_manifest_path} (mode={config.mode})")

    input_files = _glob_files(config.input_glob)
    if not input_files:
        raise RuntimeError(f"No input files matched {config.input_glob}")
    logger.info(
        "Filtering %d input shards from %s with %d keys (field=%s)",
        len(input_files),
        config.input_glob,
        len(key_set),
        filter_field,
    )

    # Reshard to consolidate output into ~min(N_input, max_output_shards) non-empty
    # shards. Without this, flat_map writes one output partition per input partition,
    # producing many empty .jsonl.gz files (esp. for resiliparse where N_input=3000
    # but only ~N_subset records survive). Levanter's tokenize step picks the first
    # file in its first group of 64 sorted-by-name files for the exemplar; if that
    # file is empty, tokenize raises IndexError.
    output_shards = min(len(input_files), 200)
    template = f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(worker_fn)
        .reshard(output_shards)
        .write_jsonl(template, skip_existing=True)
    )

    # Worker RAM has to hold the entire URL key_set in-process plus per-shard
    # record buffers. For nemotron at N=2000, the key_set unpickles to ~18GB+
    # (100M+ URLs), so 32g workers OOM. 96g gives headroom; max_workers cut to
    # 64 caps cluster RAM at ~6TB and matches realistic us-central1 capacity.
    ctx = ZephyrContext(
        name="filter-subset",
        max_workers=64,
        resources=ResourceConfig(cpu=1, ram="96g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="64g"),
    )
    ctx.put("key_set", key_set)
    ctx.put("filter_field", filter_field)
    ctx.put("normalize_value", normalize_value)
    ctx.execute(pipeline)
    logger.info("Subset filter complete → %s", config.output_path)


# --- Entry point ------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHOD_SPEC.keys()))
    parser.add_argument("--n", type=int, required=True, help="Number of WARCs to subset to.")
    parser.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Only run filter, skip tokenization (useful when tokenizer side is launched separately).",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filter, only tokenize (assumes filtered output exists).",
    )
    args = parser.parse_args()

    spec = METHOD_SPEC[args.method]
    n = args.n
    manifest = SUBSET_MANIFEST_TEMPLATE.format(N=n)
    if not Path(manifest).exists():
        raise FileNotFoundError(f"Missing subset manifest: {manifest}")

    # marin_prefix() resolves to gs://marin-{worker_region}. The Iris region
    # constraint guarantees the worker is in the right region for this method.
    prefix = marin_prefix()
    source_glob = f"{prefix}/{spec['source_suffix']}/{spec['source_pattern']}"
    metadata_glob = f"{prefix}/{WARC_METADATA_SUFFIX}" if spec["use_metadata"] else ""
    filter_out = f"{prefix}/filtered_subsets/{spec['filter_name']}_{n}warcs"
    tokenize_out = f"{prefix}/tokenized/{spec['tokenize_name']}_{n}warcs"

    print(f"=== Subset job: method={args.method} n={n} ===")
    print(f"  marin_prefix  : {prefix}  (worker region: {spec['region']})")
    print(f"  source_glob   : {source_glob}")
    print(f"  metadata_glob : {metadata_glob or '(unused)'}")
    print(f"  filter output : {filter_out}")
    print(f"  tokenize cache: {tokenize_out}")

    if not args.skip_filter:
        filter_subset(
            FilterSubsetConfig(
                input_glob=source_glob,
                output_path=filter_out,
                subset_manifest_path=manifest,
                metadata_glob=metadata_glob,
                mode=spec["mode"],
            )
        )

    if not args.skip_tokenize:
        os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        os.environ.setdefault("USE_TORCH", "0")
        os.environ.setdefault("TORCH_DISABLE_GLOBAL_DEPS", "1")

        tokenize_config = TokenizeConfig(
            train_paths=[f"{filter_out}/*.jsonl.gz"],
            validation_paths=[],
            cache_path=tokenize_out,
            tokenizer=TOKENIZER,
            format=TextLmDatasetFormat(),
        )
        logger.info("Starting tokenize → %s", tokenize_out)
        tokenize(tokenize_config)
        logger.info("Tokenize complete → %s", tokenize_out)

    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
