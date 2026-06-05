# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subset baseline curated outputs to {100, 500, 1k, 2k} WARCs and re-tokenize.

Source-of-truth WARC manifest is ``experiments/distill/baseline_warcs_3000.txt``.
Subsets are deterministic prefixes (``head -N``) so 100 ⊂ 500 ⊂ 1k ⊂ 2k ⊂ 3k.

Four reliable methods supported here. Each filters an EXISTING filtered/extracted
output in-place by a per-record key that uniquely identifies a (WARC, record):

    method            | join field           | key set source
    ------------------+----------------------+--------------------------------
    dclm              | record.warc_record_id| metadata records in subset WARCs
    nemotron_full     | record.url           | metadata records in subset WARCs
    fineweb_edu       | record.file_path     | subset manifest (normalized)
    llm_curated       | record.warc_file     | subset manifest (direct)

Resiliparse is intentionally NOT here — its extracted output keeps only ``text``
and ``url``, and within the 3000-WARC pool URLs can collide across snapshots,
which would over-include records. Subset Resiliparse is built by re-extracting
from the downloaded HTML JSONL filtered by ``warc_file`` (see
``subset_resiliparse.py``).

Region pinning (sources differ across us-central1/us-central2):

    us-central2  : dclm, fineweb_edu
    us-central1  : nemotron_full, llm_curated

Launch on each region's Ray cluster::

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -e HF_TOKEN <YOUR_HF_TOKEN> \\
        -- python experiments/baseline_collection/subset_baselines.py \\
        --region us-central2

    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central1 --no_wait \\
        -e WANDB_API_KEY <YOUR_WANDB_API_KEY> \\
        -e HF_TOKEN <YOUR_HF_TOKEN> \\
        -- python experiments/baseline_collection/subset_baselines.py \\
        --region us-central1
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import fsspec
from fray import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from zephyr import Dataset, ZephyrContext
from zephyr.execution import zephyr_worker_ctx

from experiments.defaults import default_tokenize

logger = logging.getLogger(__name__)


# --- Source paths (existing filtered/extracted outputs as of 2026-04-28) ---
DCLM_FILTERED = "gs://marin-us-central2/filtered/baseline_dclm-23e9be"
NEMOTRON_FULL_FILTERED = "gs://marin-us-central1/filtered/baseline_nemotron_full-347dfe"
NEMOTRON_QHIGH_FILTERED = "gs://marin-us-central1/filtered/baseline_nemotron_qhigh-v1"
FINEWEB_EDU_FILTERED = "gs://marin-us-central2/filtered/baseline_fineweb_edu-7a3bc5"
RESILIPARSE_EXTRACTED = "gs://marin-us-central2/extracted/baseline_resiliparse-19bdaa"
LLM_CURATED_DOCUMENTS = "gs://marin-us-central1/documents/baseline_llm_curated-050243"

# Metadata extraction output for the 3000-WARC pool. Carries per-record
# (warc_record_id, url, warc_file, snapshot) — the universal join lookup.
WARC_METADATA = "gs://marin-us-central2/metadata/baseline_warc_metadata-d671da"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SUBSET_MANIFEST_TEMPLATE = str(REPO_ROOT / "experiments/distill/subsets/baseline_warcs_{N}.txt")
SUBSET_SIZES_DEFAULT = (100, 500, 1000, 2000)

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# Methods grouped by source-bucket region. Pinning each method to its source
# region means every read is in-region (free).
METHODS_BY_REGION: dict[str, tuple[str, ...]] = {
    "us-central2": ("dclm", "fineweb_edu"),
    "us-central1": ("nemotron_full", "nemotron_qhigh", "llm_curated"),
}
ALL_METHODS = tuple(m for ms in METHODS_BY_REGION.values() for m in ms)


# --- Helpers ----------------------------------------------------------------


def _normalize_warc_path(s: str) -> str:
    """Drop scheme/host prefix; FineWeb-Edu file_path is the relative form."""
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
    """fsspec glob with gs:// prefix preserved."""
    fs = fsspec.filesystem("gcs") if pattern.startswith("gs://") else fsspec.filesystem("file")
    raw = fs.glob(pattern)
    if pattern.startswith("gs://"):
        return [p if p.startswith("gs://") else f"gs://{p}" for p in raw]
    return list(raw)


def _build_key_set_from_metadata(metadata_glob: str, subset_warc_set: set[str], key_field: str) -> set[str]:
    """Stream metadata shards, return the set of `key_field` values for records in subset WARCs."""
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
    logger.info("Built key set: %d unique %s values from %d subset WARCs", len(keys), key_field, len(subset_warc_set))
    return keys


# --- Filter step ------------------------------------------------------------


@dataclass(frozen=True)
class FilterSubsetConfig:
    """Filter an existing filtered/extracted output by a WARC subset.

    Modes:
      - ``metadata_record_id`` : key set built from metadata, filter on ``warc_record_id``
      - ``metadata_url``       : key set built from metadata, filter on ``url``
      - ``manifest_file_path`` : key set is normalized WARC paths, filter on ``file_path``
      - ``manifest_warc_file`` : key set is raw WARC paths, filter on ``warc_file``
    """

    input_glob: str
    output_path: str
    subset_manifest_path: str
    metadata_glob: str
    mode: str


def _filter_one_shard(file_path: str) -> list[dict]:
    """Worker: filter one input shard by the shared key_set on the configured field."""
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


def filter_subset(config: FilterSubsetConfig) -> None:
    if config.mode == "metadata_record_id":
        subset_warc_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        key_set = _build_key_set_from_metadata(f"{config.metadata_glob}/*.jsonl.gz", subset_warc_set, "warc_record_id")
        filter_field, normalize_value = "warc_record_id", False
    elif config.mode == "metadata_url":
        subset_warc_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        key_set = _build_key_set_from_metadata(f"{config.metadata_glob}/*.jsonl.gz", subset_warc_set, "url")
        filter_field, normalize_value = "url", False
    elif config.mode == "manifest_file_path":
        key_set = _load_subset_warc_set(config.subset_manifest_path, normalize=True)
        filter_field, normalize_value = "file_path", True
    elif config.mode == "manifest_warc_file":
        key_set = _load_subset_warc_set(config.subset_manifest_path, normalize=False)
        filter_field, normalize_value = "warc_file", False
    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    if not key_set:
        raise RuntimeError(f"Empty key set for {config.subset_manifest_path} (mode={config.mode})")

    input_files = _glob_files(config.input_glob)
    if not input_files:
        raise RuntimeError(f"No input files matched {config.input_glob}")
    logger.info(
        "Filtering %d input shards from %s with %d keys (field=%s, normalize=%s)",
        len(input_files),
        config.input_glob,
        len(key_set),
        filter_field,
        normalize_value,
    )

    template = f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_list(input_files).flat_map(_filter_one_shard).write_jsonl(template, skip_existing=True)

    # 32 GB worker RAM is comfortable for both the streaming JSONL read and a
    # broadcast key_set of up to a few hundred MB (URL sets for 2k WARCs run
    # ~50-100 MB in Python set form). Driver gets 64 GB to assemble the key
    # set without thrashing.
    ctx = ZephyrContext(
        name="filter-subset",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="32g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="64g"),
    )
    ctx.put("key_set", key_set)
    ctx.put("filter_field", filter_field)
    ctx.put("normalize_value", normalize_value)
    ctx.execute(pipeline)
    logger.info("Subset filter complete → %s", config.output_path)


# --- Per-method step builders ----------------------------------------------


def _filter_step(method: str, n: int, input_glob: str, mode: str, metadata_glob: str = "") -> ExecutorStep:
    return ExecutorStep(
        name=f"filtered_subsets/baseline_{method}_{n}warcs",
        description=f"{method} filtered subset to first {n} WARCs (mode={mode}).",
        fn=remote(filter_subset, resources=ResourceConfig(cpu=2, ram="64g")),
        config=FilterSubsetConfig(
            input_glob=input_glob,
            output_path=this_output_path(),
            subset_manifest_path=SUBSET_MANIFEST_TEMPLATE.format(N=n),
            metadata_glob=metadata_glob,
            mode=mode,
        ),
    )


def make_steps(method: str, n: int) -> ExecutorStep:
    """Build the (filter, tokenize) pair for one method/size; return the tokenize step."""
    if method == "dclm":
        f = _filter_step("dclm", n, f"{DCLM_FILTERED}/data-*.jsonl.gz", "metadata_record_id", WARC_METADATA)
        tok_name = f"baseline_dclm_{n}warcs"
    elif method == "nemotron_full":
        f = _filter_step(
            "nemotron_full_bos_fixed", n, f"{NEMOTRON_FULL_FILTERED}/*.jsonl.gz", "metadata_url", WARC_METADATA
        )
        tok_name = f"baseline_nemotron_full_bos_fixed_{n}warcs"
    elif method == "nemotron_qhigh":
        f = _filter_step("nemotron_qhigh", n, f"{NEMOTRON_QHIGH_FILTERED}/*.jsonl.gz", "metadata_url", WARC_METADATA)
        tok_name = f"baseline_nemotron_qhigh_{n}warcs"
    elif method == "fineweb_edu":
        f = _filter_step("fineweb_edu", n, f"{FINEWEB_EDU_FILTERED}/data-*.jsonl.gz", "manifest_file_path")
        tok_name = f"baseline_fineweb_edu_{n}warcs"
    elif method == "llm_curated":
        f = _filter_step("llm_curated_bos_fixed", n, f"{LLM_CURATED_DOCUMENTS}/data-*.jsonl.gz", "manifest_warc_file")
        tok_name = f"baseline_llm_curated_bos_fixed_{n}warcs"
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return default_tokenize(name=tok_name, dataset=f / "*.jsonl.gz", tokenizer=TOKENIZER)


# --- Entry point ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region",
        choices=["us-central1", "us-central2"],
        required=True,
        help="Source region — only methods whose source bucket is in this region are launched.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Override which methods to run (must all be in --region). Defaults to all methods for the region.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(SUBSET_SIZES_DEFAULT),
        help="Subset sizes (number of WARCs).",
    )
    args, executor_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + executor_args

    region_methods = METHODS_BY_REGION[args.region]
    methods = tuple(args.methods) if args.methods else region_methods
    bad = [m for m in methods if m not in region_methods]
    if bad:
        raise SystemExit(
            f"Methods {bad} are not pinned to {args.region}. "
            f"Allowed in {args.region}: {region_methods}. Launch others on their region."
        )

    steps: list[ExecutorStep] = []
    for n in args.sizes:
        manifest = SUBSET_MANIFEST_TEMPLATE.format(N=n)
        if not Path(manifest).exists():
            raise FileNotFoundError(f"Missing subset manifest: {manifest}. Run head -n N to generate.")
        for m in methods:
            steps.append(make_steps(m, n))

    print(f"Launching {len(steps)} tokenize steps in {args.region}: methods={methods} sizes={args.sizes}")
    executor_main(
        steps=steps,
        description=(
            f"Subset baselines to {args.sizes} WARCs and tokenize with {TOKENIZER}. "
            f"Region={args.region}, methods={methods}."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
