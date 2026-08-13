# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subset the 10k-WARC extractions to a RANDOM 100-WARC sample and re-tokenize.

WHY: the existing ``*_100`` methods are ``head -100`` of the date-SORTED
``baseline_warcs_3000.txt`` (see ``subset_baselines.py``: "Subsets are
deterministic prefixes"). That head slice covers only 2 crawls, BOTH from 2013
(CC-MAIN-2013-20 x65, 2013-48 x35), while the DCLM 400m-1x pool spans 89 crawls
across 2013-2022. So N=100 was a severe TEMPORAL bias, not a random sample.

This builds the random-sample counterpart: manifest
``experiments/distill/subsets/baseline_warcs_100_random.txt`` (uniform draw over
the whole sorted pool, seed 0, via ``sample_random_warcs.py`` — the same method
used for the 3000-random manifest). It covers 60 crawls spanning 2013-2022 and
is disjoint from the biased head (1 WARC of overlap).

Sources are the EXISTING 10,364-WARC extractions — no re-extraction (HQ would be
~5h/WARC of LLM work, and the raw WARCs may be cleaned up). Each method filters
its existing output by a per-record key identifying the (WARC, record):

    method         | source (region)                        | join field
    ---------------+----------------------------------------+---------------------
    dclm           | filtered/..._10k_dclm_resharded (c2)   | warc_record_id (meta)
    nemotron_full  | filtered/..._10k_nemotron_full (c2)    | url (meta)
    high_quality   | ..._hf_export/10364warcs/joined (c1)   | warc_file (direct)

``high_quality`` needs NO metadata join: the joined parquet export already carries
``warc_file``. dclm/nemotron need the metadata lookup
(``metadata/dclm_400m_1x_10k_warc_metadata-79158f``: warc_record_id, url,
warc_file, snapshot). That metadata has 10,364 shards, so unlike
``subset_baselines._build_key_set_from_metadata`` (driver-serial, fine for 3k)
the key set here is built with a PARALLEL Zephyr pass.

Region pinning (every read in-region, no egress):
    us-central2 : dclm, nemotron_full   (+ the metadata keyset pass)
    us-central1 : high_quality

Usage (one Iris CPU job per region)::

    uv run iris --cluster marin job run --no-wait --region us-central2 \\
        --cpu 8 --memory 64GB --priority interactive --extra cpu --enable-extra-resources \\
        -e MARIN_PREFIX gs://marin-us-central2 \\
        -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -- python experiments/baseline_collection/subset_random100_10k.py --region us-central2

    uv run iris --cluster marin job run --no-wait --region us-central1 \\
        --cpu 8 --memory 64GB --priority interactive --extra cpu --enable-extra-resources \\
        -e MARIN_PREFIX gs://marin-us-central1 \\
        -e WANDB_API_KEY <k> -e HF_TOKEN <t> \\
        -- python experiments/baseline_collection/subset_random100_10k.py --region us-central1

After completion read each cache's ``train/.stats.json:total_tokens`` and register
in ``curation_plan`` as ``{dclm,nemotron_full,high_quality}_random_100``
(sampled_warcs=100), mirroring the ``*_random_3000`` naming.
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
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext, zephyr_worker_ctx

from experiments.baseline_collection.subset_baselines import _glob_files
from experiments.defaults import default_tokenize

logger = logging.getLogger(__name__)


def _load_warc_manifest(manifest_path: str) -> set[str]:
    """Load WARC paths, stripping blanks AND ``#`` comments.

    NOT ``subset_baselines._load_subset_warc_set``: that one only strips blank
    lines, so the 4-line ``#`` provenance header that ``sample_random_warcs.py``
    writes would be loaded as 4 bogus WARC paths (they never match any record, but
    they corrupt the counts/asserts). Mirrors ``sample_random_warcs.load_pool``.
    """
    with open(manifest_path) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


# --- Sources: the EXISTING 10,364-WARC extractions ---
DCLM_10K = "gs://marin-us-central2/filtered/dclm_400m_1x_10k_dclm_resharded-1fe977"
NEMOTRON_FULL_10K = "gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9"
HQ_10K_JOINED = "gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined"
# Universal join lookup for the 10k pool: warc_record_id, url, warc_file, snapshot.
METADATA_10K = "gs://marin-us-central2/metadata/dclm_400m_1x_10k_warc_metadata-79158f"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# N is set by the WARC_N env var (default 100); the draw + downstream names all key
# off it, so the SAME script builds random-100/300/500/... from the nested seed-0
# manifests (100 subset of 300 subset of 500).
WARC_N = int(os.environ.get("WARC_N", "100"))
RANDOM_MANIFEST = str(REPO_ROOT / f"experiments/distill/subsets/baseline_warcs_{WARC_N}_random.txt")

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

METHODS_BY_REGION: dict[str, tuple[str, ...]] = {
    "us-central2": ("dclm", "nemotron_full"),
    "us-central1": ("high_quality",),
}


# --- Parallel key-set build (10,364 metadata shards) ------------------------


def _keys_from_metadata_shard(file_path: str) -> list[dict]:
    """Worker: emit {warc_record_id, url} for metadata records in the subset WARCs."""
    ctx = zephyr_worker_ctx()
    warc_set: set[str] = ctx.get_shared("warc_set")
    out: list[dict] = []
    with fsspec.open(file_path, "rb") as fh, gzip.open(fh, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("warc_file") in warc_set:
                out.append({"warc_record_id": rec.get("warc_record_id", ""), "url": rec.get("url", "")})
    return out


@dataclass(frozen=True)
class KeySetConfig:
    output_path: str
    manifest_path: str
    metadata_glob: str


def build_key_set(config: KeySetConfig) -> None:
    """Parallel pass over the 10k metadata → the (warc_record_id, url) rows of the subset WARCs."""
    warc_set = _load_warc_manifest(config.manifest_path)
    assert warc_set, f"empty manifest {config.manifest_path}"
    files = _glob_files(f"{config.metadata_glob}/*.jsonl.gz")
    assert files, f"no metadata shards at {config.metadata_glob}"
    logger.info("Key-set pass: %d metadata shards, %d subset WARCs", len(files), len(warc_set))

    pipeline = (
        Dataset.from_list(files)
        .flat_map(_keys_from_metadata_shard)
        .write_jsonl(f"{config.output_path}/keys-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(
        name=f"random{WARC_N}-keyset",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="4g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g"),
    )
    ctx.put("warc_set", warc_set)
    ctx.execute(pipeline)
    logger.info("Key set written → %s", config.output_path)


def _load_keys(keyset_glob: str, field: str) -> set[str]:
    keys: set[str] = set()
    for path in _glob_files(keyset_glob):
        with fsspec.open(path, "rb") as fh, gzip.open(fh, "rt") as gz:
            for line in gz:
                if not line.strip():
                    continue
                v = json.loads(line).get(field, "")
                if v:
                    keys.add(v)
    logger.info("Loaded %d unique %s keys", len(keys), field)
    return keys


# --- Filter steps -----------------------------------------------------------


@dataclass(frozen=True)
class FilterConfig:
    input_glob: str
    output_path: str
    keyset_glob: str  # glob over the keyset step's output ("" for warc_file mode)
    manifest_path: str
    field: str  # warc_record_id | url | warc_file
    fmt: str  # jsonl | parquet


def _filter_jsonl_shard(file_path: str) -> list[dict]:
    ctx = zephyr_worker_ctx()
    key_set: set[str] = ctx.get_shared("key_set")
    field: str = ctx.get_shared("field")
    out: list[dict] = []
    with fsspec.open(file_path, "rb") as fh, gzip.open(fh, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get(field, "") in key_set:
                out.append(rec)
    return out


def _filter_parquet_shard(file_path: str) -> list[dict]:
    """HQ: joined parquet carries warc_file; keep text + provenance for kept rows."""
    import pyarrow.parquet as pq

    ctx = zephyr_worker_ctx()
    key_set: set[str] = ctx.get_shared("key_set")
    field: str = ctx.get_shared("field")
    out: list[dict] = []
    with fsspec.open(file_path, "rb") as fh:
        table = pq.read_table(fh)
    cols = table.column_names
    keep = [c for c in ("text", "url", "warc_record_id", "warc_file", "snapshot") if c in cols]
    for batch in table.select(keep).to_batches():
        for rec in batch.to_pylist():
            if rec.get(field, "") in key_set:
                out.append(rec)
    return out


def filter_subset(config: FilterConfig) -> None:
    if config.field == "warc_file":
        key_set = _load_warc_manifest(config.manifest_path)
    else:
        key_set = _load_keys(config.keyset_glob, config.field)
    if not key_set:
        raise RuntimeError(f"Empty key set (field={config.field})")

    files = _glob_files(config.input_glob)
    if not files:
        raise RuntimeError(f"No inputs matched {config.input_glob}")
    logger.info("Filtering %d shards on %s (%d keys, fmt=%s)", len(files), config.field, len(key_set), config.fmt)

    worker = _filter_parquet_shard if config.fmt == "parquet" else _filter_jsonl_shard
    pipeline = (
        Dataset.from_list(files)
        .flat_map(worker)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
    )
    ctx = ZephyrContext(
        name=f"random{WARC_N}-filter",
        max_workers=200,
        resources=ResourceConfig(cpu=1, ram="32g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="64g"),
    )
    ctx.put("key_set", key_set)
    ctx.put("field", config.field)
    ctx.execute(pipeline)
    logger.info("Subset written → %s", config.output_path)


# --- Step builders ----------------------------------------------------------


def _keyset_step() -> ExecutorStep:
    return ExecutorStep(
        name=f"metadata/random{WARC_N}_10k_keyset",
        description=f"(warc_record_id,url) rows for the random-{WARC_N} WARC sample of the 10k pool.",
        fn=remote(build_key_set, resources=ResourceConfig(cpu=2, ram="16g")),
        config=KeySetConfig(
            output_path=this_output_path(),
            manifest_path=RANDOM_MANIFEST,
            metadata_glob=METADATA_10K,
        ),
    )


def _filter_step(method: str, input_glob: str, field: str, fmt: str, keyset: ExecutorStep | None) -> ExecutorStep:
    return ExecutorStep(
        name=f"filtered_subsets/{method}_random_{WARC_N}warcs",
        description=f"{method} 10k extraction subset to the RANDOM {WARC_N}-WARC sample (join={field}).",
        fn=remote(filter_subset, resources=ResourceConfig(cpu=2, ram="64g")),
        config=FilterConfig(
            input_glob=input_glob,
            output_path=this_output_path(),
            keyset_glob=(keyset / "*.jsonl.gz") if keyset is not None else "",
            manifest_path=RANDOM_MANIFEST,
            field=field,
            fmt=fmt,
        ),
    )


def make_tokenize_step(method: str) -> ExecutorStep:
    if method == "dclm":
        ks = _keyset_step()
        f = _filter_step("dclm", f"{DCLM_10K}/data-*.jsonl.gz", "warc_record_id", "jsonl", ks)
        name = f"dclm_random_{WARC_N}warcs"
    elif method == "nemotron_full":
        ks = _keyset_step()
        f = _filter_step("nemotron_full", f"{NEMOTRON_FULL_10K}/*.jsonl.gz", "url", "jsonl", ks)
        name = f"nemotron_full_random_{WARC_N}warcs"
    elif method == "high_quality":
        # joined parquet already carries warc_file → no metadata join needed
        f = _filter_step("high_quality", f"{HQ_10K_JOINED}/*.parquet", "warc_file", "parquet", None)
        name = f"high_quality_random_{WARC_N}warcs"
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return default_tokenize(name=name, dataset=f / "*.jsonl.gz", tokenizer=TOKENIZER)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--region", choices=sorted(METHODS_BY_REGION), required=True)
    p.add_argument("--methods", nargs="+", default=None, help="Subset of this region's methods.")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0], *rest]

    allowed = METHODS_BY_REGION[args.region]
    methods = tuple(args.methods) if args.methods else allowed
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise SystemExit(f"methods {bad} are not sourced in {args.region} (allowed: {allowed})")

    steps = [make_tokenize_step(m) for m in methods]
    executor_main(
        steps=steps, description=f"Random-{WARC_N} (10k pool) subsets + tokenize for {methods} in {args.region}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
