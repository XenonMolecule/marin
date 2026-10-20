# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontaminate the 8M c075 pool with the hero run's method and evals, and track what was removed.

Same method as ``experiments/datakit/reference_pipeline.py`` on the hero branch: the
``aa-index-v4.1.1-v3`` eval corpus (9 Artificial Analysis benchmarks required, lm-eval-harness
tasks best effort, hero exclusions), word 13-grams, a doc is contaminated when >= 50% of some
paragraph's 13-grams hit the bloom (``lib/marin/src/marin/datakit/decon.py``), with a source-local
boilerplate drop set. The pool is one source, as each hero dataset was; the cross-source drop
set needs >= 3 sources and does not apply.

Outputs, all under the pool root ``POOL``:

- ``deduped_c075_decon/``: the decontaminated plain text, same files and fields as
  ``deduped_c075/`` minus contaminated docs. This is the version to export or tokenize.
- ``decon/``: the bloom, drop set, per-doc attributes (``attributes/outputs/main/``), per-file
  removed-doc sidecars, and ``removed_docs.parquet`` (one row per removed doc with the evals it
  matched).
- ``EXPORT.json``: names ``deduped_c075_decon/`` as the dataset to share.

Steps run in order as in-region Iris jobs::

    python -m experiments.fast_curation.decon_8m sample|bloom|dropsets|mark|filter|manifest
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import logging
import os
import random
import re
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.decon import (
    NGramConfig,
    build_all_source_drop_sets,
    build_eval_bloom,
    decon_to_parquet,
)
from marin.datakit.normalize import NormalizedData
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext

from experiments.datakit.decontam.config import (
    GLOBAL_DF_COMMON_MIN_ABS,
    GLOBAL_DF_COMMON_MIN_SOURCES,
    SOURCE_DF_COMMON_FRAC,
    SOURCE_DF_COMMON_MIN_ABS,
    SOURCE_DF_SAMPLE_DOCS,
)
from experiments.datakit.decontam.prepare_eval_corpus import (
    AA_BENCHMARK_NAMES,
    AA_MANIFEST_RELATIVE,
    DECON_EXCLUDED_EVAL_TASKS,
    EVAL_CORPUS_VERSION,
    LMH_MANIFEST_RELATIVE,
)

logger = logging.getLogger(__name__)

POOL = "gs://marin-us-central2/documents/baseline_lpv11_fastpipe_v2_1_fused_deduped/8582115warcs"
EVAL_ROOT = f"gs://marin-us-central2/datakit/decontam/evals/{EVAL_CORPUS_VERSION}"
SAMPLE_DIR = "gs://marin-us-central2/tmp/ttl=14d/decon8m/source_sample"
SOURCE_NAME = "lpv11_8m_c075"

# Hero constants (experiments/datakit/reference_pipeline.py).
NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5)
ESTIMATED_DOC_COUNT = 50_000_000
FALSE_POSITIVE_RATE = 1e-9
# The drop-set builder requires a global sample at least as large as the source sample; with one
# source no global set can form, so this only bounds how many sampled docs are read.
GLOBAL_SAMPLE_DOCS = 1_000_000
SAMPLE_FILES = 120
FLAGGED_SAMPLE_PER_FILE = 8
ID_FIELD = "doc_id"
INPUT_SUFFIX = ".jsonl.gz"


def _paths(pool: str) -> dict[str, str]:
    decon = f"{pool}/decon"
    return {
        "input": f"{pool}/deduped_c075",
        "output": f"{pool}/deduped_c075_decon",
        "bloom": f"{decon}/bloom",
        "dropsets": f"{decon}/dropsets",
        "attributes": f"{decon}/attributes",
        "removed_parts": f"{decon}/removed_parts",
        "removed": f"{decon}/removed_docs.parquet",
        "summary": f"{decon}/summary.json",
        "export": f"{pool}/EXPORT.json",
    }


def _input_files(input_dir: str) -> list[str]:
    return sorted(str(m) for m in StoragePath(f"{input_dir.rstrip('/')}/**/*{INPUT_SUFFIX}").glob())


def _read_jsonl(path: str) -> Iterator[dict]:
    with StoragePath(path).open("rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _sample_file(path: str, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keep: list[dict] = []
    for n, record in enumerate(_read_jsonl(path)):
        row = {"id": record[ID_FIELD], "text": record["text"]}
        if len(keep) < k:
            keep.append(row)
        elif (j := rng.randrange(n + 1)) < k:
            keep[j] = row
    return keep


def sample(pool: str, sample_dir: str, n_docs: int, n_files: int) -> None:
    """Write a shuffled random sample of the pool as the drop-set source (a prefix is representative)."""
    rng = random.Random(20261019)
    files = rng.sample(_input_files(_paths(pool)["input"]), n_files)
    per_file = -(-n_docs // n_files)
    with concurrent.futures.ProcessPoolExecutor(16) as ex:
        rows = [r for part in ex.map(_sample_file, files, [per_file] * n_files, range(n_files)) for r in part]
    rng.shuffle(rows)
    rows = rows[:n_docs]
    part_size = 100_000
    for i in range(0, len(rows), part_size):
        with StoragePath(f"{sample_dir}/part-{i // part_size:05d}.parquet").open("wb") as fh:
            pq.write_table(pa.Table.from_pylist(rows[i : i + part_size]), fh, compression="zstd")
    logger.info("sampled %d docs from %d files into %s", len(rows), n_files, sample_dir)


def bloom(pool: str) -> None:
    result = build_eval_bloom(
        eval_data_sources=[EVAL_ROOT],
        output_path=_paths(pool)["bloom"],
        ngram=NGRAM,
        estimated_doc_count=ESTIMATED_DOC_COUNT,
        false_positive_rate=FALSE_POSITIVE_RATE,
        exclude_eval_dirs=DECON_EXCLUDED_EVAL_TASKS,
        required_eval_manifest_path=f"{EVAL_ROOT}/{AA_MANIFEST_RELATIVE}",
        required_eval_corpus_version=EVAL_CORPUS_VERSION,
        required_eval_names=AA_BENCHMARK_NAMES,
        best_effort_eval_manifest_path=f"{EVAL_ROOT}/{LMH_MANIFEST_RELATIVE}",
        best_effort_eval_corpus_version=EVAL_CORPUS_VERSION,
    )
    logger.info("bloom: %s", result)


def dropsets(pool: str, sample_dir: str) -> None:
    result = build_all_source_drop_sets(
        sources=[(SOURCE_NAME, sample_dir)],
        prebuilt_bloom_dir=_paths(pool)["bloom"],
        output_path=_paths(pool)["dropsets"],
        ngram=NGRAM,
        sample_docs=SOURCE_DF_SAMPLE_DOCS,
        common_frac=SOURCE_DF_COMMON_FRAC,
        common_min_abs=SOURCE_DF_COMMON_MIN_ABS,
        global_sample_docs=GLOBAL_SAMPLE_DOCS,
        global_common_min_abs=GLOBAL_DF_COMMON_MIN_ABS,
        global_common_min_sources=GLOBAL_DF_COMMON_MIN_SOURCES,
    )
    logger.info("drop sets: %s", result)


def _drop_set_dirs(pool: str) -> list[str]:
    return [f"{_paths(pool)['dropsets']}/{SOURCE_NAME}"]


def mark(pool: str, max_workers: int) -> None:
    p = _paths(pool)
    attrs = decon_to_parquet(
        normalized_data=NormalizedData(main_output_dir=p["input"], dup_output_dir="", counters={}),
        prebuilt_bloom_dir=p["bloom"],
        output_path=p["attributes"],
        id_field=ID_FIELD,
        input_suffix=INPUT_SUFFIX,
        ngram=NGRAM,
        drop_set_dirs=_drop_set_dirs(pool),
        flagged_sample_size=FLAGGED_SAMPLE_PER_FILE,
        max_workers=max_workers,
    )
    logger.info("mark: %s", attrs)


def _attributes_path(p: dict[str, str], input_path: str) -> str:
    name = os.path.basename(input_path).removesuffix(INPUT_SUFFIX) + ".parquet"
    return f"{p['attributes']}/outputs/main/{name}"


def _filter_shard(p: dict[str, str]):
    def run(paths: Iterator[str], shard: ShardInfo) -> Iterator[dict]:
        for input_path in paths:
            with StoragePath(_attributes_path(p, input_path)).open("rb") as fh:
                table = pq.read_table(fh, columns=["id", "contaminated", "max_overlap", "matched_hashes"])
            flagged = {r["id"]: r for r in table.to_pylist() if r["contaminated"]}
            name = os.path.basename(input_path)
            out_path = f"{p['output']}/{name}"
            removed = []
            kept = n = kept_chars = removed_chars = 0
            with StoragePath(out_path).open("wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as out:
                for record in _read_jsonl(input_path):
                    n += 1
                    hit = flagged.get(record[ID_FIELD])
                    if hit is None:
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        kept += 1
                        kept_chars += len(record["text"])
                        continue
                    removed_chars += len(record["text"])
                    removed.append(
                        {
                            "doc_id": record[ID_FIELD],
                            "url": record.get("url"),
                            "snapshot": record.get("snapshot"),
                            "chars": len(record["text"]),
                            "max_overlap": hit["max_overlap"],
                            "matched_hashes": hit["matched_hashes"],
                            "source_file": name,
                        }
                    )
            if n != table.num_rows:
                raise RuntimeError(f"{input_path}: {n} docs but {table.num_rows} attribute rows")
            if len(removed) != len(flagged):
                raise RuntimeError(f"{input_path}: {len(flagged)} flagged ids but {len(removed)} removed")
            part = f"{p['removed_parts']}/{name.removesuffix(INPUT_SUFFIX)}.parquet"
            with StoragePath(part).open("wb") as fh:
                pq.write_table(_removed_table(removed), fh, compression="zstd")
            for key, value in (
                ("decon/docs_in", n),
                ("decon/docs_kept", kept),
                ("decon/docs_removed", len(removed)),
                ("decon/chars_kept", kept_chars),
                ("decon/chars_removed", removed_chars),
            ):
                counters.pipeline.update_counter(key, value)
            yield {"file": name, "docs": n, "removed": len(removed)}

    return run


_REMOVED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("snapshot", pa.string()),
        ("chars", pa.int64()),
        ("max_overlap", pa.float64()),
        ("matched_hashes", pa.list_(pa.uint64())),
        ("source_file", pa.string()),
    ]
)


def _removed_table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=_REMOVED_SCHEMA)


def filter_pool(pool: str, max_workers: int) -> None:
    p = _paths(pool)
    files = _input_files(p["input"])
    ctx = ZephyrContext(name="decon-8m-filter", resources=ResourceConfig(cpu=1, ram="4g"), max_workers=max_workers)
    outcome = ctx.execute(Dataset.from_list(files).map_shard(_filter_shard(p)))
    totals = {k: v for k, v in outcome.counters.items() if k.startswith("decon/")}
    totals["files"] = len(files)
    with StoragePath(p["summary"]).open("w") as fh:
        json.dump(totals, fh, indent=1)
    logger.info("filter: %s", totals)


_EVAL_ID_RE = re.compile(r"^(?P<name>.+?)(?:-(?:test|validation|train|dev))?-\d+$")


def eval_name(eval_id: str) -> str:
    """Eval name of a corpus record id: ``aa_lcr-1`` -> ``aa_lcr``, ``mmlu-test-3`` -> ``mmlu``."""
    m = _EVAL_ID_RE.match(eval_id)
    return m.group("name") if m else eval_id


def manifest(pool: str) -> None:
    """Concatenate the removed-doc sidecars, name the matched evals, and point EXPORT.json at the clean tree."""
    p = _paths(pool)
    parts = sorted(str(m) for m in StoragePath(f"{p['removed_parts']}/*.parquet").glob())
    tables = []
    for part in parts:
        with StoragePath(part).open("rb") as fh:
            tables.append(pq.read_table(fh))
    removed = pa.concat_tables(tables) if tables else _REMOVED_SCHEMA.empty_table()
    with StoragePath(f"{p['bloom']}/_bloom/eval_hash_index.parquet").open("rb") as fh:
        index = pq.read_table(fh, columns=["hash", "eval_id"]).to_pylist()
    evals_of: dict[int, set[str]] = {}
    for row in index:
        evals_of.setdefault(row["hash"], set()).add(eval_name(row["eval_id"]))
    matched_evals = [
        sorted({e for h in hashes for e in evals_of.get(h, ())})
        for hashes in removed.column("matched_hashes").to_pylist()
    ]
    removed = removed.append_column("matched_evals", pa.array(matched_evals, pa.list_(pa.string())))
    with StoragePath(p["removed"]).open("wb") as fh:
        pq.write_table(removed, fh, compression="zstd")
    with StoragePath(p["summary"]).open("r") as fh:
        summary = json.load(fh)
    export = {
        "export_dataset": p["output"],
        "do_not_export": p["input"],
        "removed_docs": p["removed"],
        "method": "hero-run decontamination: aa-index-v4.1.1-v3 evals, 13-gram, paragraph overlap >= 0.5",
        "summary": summary,
    }
    with StoragePath(p["export"]).open("w") as fh:
        json.dump(export, fh, indent=1)
    logger.info("manifest: %d removed docs -> %s; %s", removed.num_rows, p["removed"], p["export"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["sample", "bloom", "dropsets", "mark", "filter", "manifest"])
    ap.add_argument("--pool", default=POOL, help="Pool root (override for smoke runs).")
    ap.add_argument("--sample-dir", default=SAMPLE_DIR)
    ap.add_argument("--sample-docs", type=int, default=GLOBAL_SAMPLE_DOCS)
    ap.add_argument("--sample-files", type=int, default=SAMPLE_FILES)
    ap.add_argument("--max-workers", type=int, default=1000)
    args = ap.parse_args()
    if args.step == "sample":
        sample(args.pool, args.sample_dir, args.sample_docs, args.sample_files)
    elif args.step == "bloom":
        bloom(args.pool)
    elif args.step == "dropsets":
        dropsets(args.pool, args.sample_dir)
    elif args.step == "mark":
        mark(args.pool, args.max_workers)
    elif args.step == "filter":
        filter_pool(args.pool, args.max_workers)
    else:
        manifest(args.pool)


if __name__ == "__main__":
    main()
