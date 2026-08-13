# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Consolidate the 200-WARC head-to-head into one comparison parquet dataset.

For every page the high_quality extractor *judged* across the 200 WARCs, emit one
row pairing the original HTML with all three models' verdicts:

    warc_record_id, url, warc_file, snapshot   — provenance / join key
    fed_to_model                               — page was within the length cap
    stripped_html                              — body_strip(raw)  (every row)
    raw_html                                   — full page HTML   (only where >=1 model kept it)
    text_8b   / text_1p7b   / text_0p6b        — each model's extraction (null = abstained)
    label_8b  / label_1p7b  / label_0p6b       — "useful" if kept, else "not_useful"

The negatives (pages a model rejected with [NO_USEFUL_CONTENT]) are recovered by
set difference: a fed page NOT in a model's kept set is a "not_useful" for that
model. This is the substrate for F1 of useful-vs-not classification and for later
columns (e.g. fastText scores).

Each model only persists its KEPT docs, so this is an OUTER join over the union of
warc_record_ids keyed per page from a fresh WARC decode (the only source of the
full fed universe + the rejected pages' HTML).

``build`` — one task per WARC (map-only, one durable parquet shard each,
    skip_existing). Decodes the WARC once in-region (free CommonCrawl ingress),
    reads each model's kept batches (8B is scattered across 5 regions by
    cross-region resume → globbed across all buckets; reads are tiny, kept-only),
    outer-joins on the normalized WARC-Record-ID, writes the shard.

``sample`` — draw ~100k rows uniformly at random across all shards (proportional
    per-shard allocation, seeded) for analysis.

Run in us-east5 (where 1.7B + most of 0.6B live; CC decode is free anywhere)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 28GB --disk 30GB --priority interactive --extra cpu \\
        --enable-extra-resources --region us-east5 --job-name extractor-compare-build \\
        -- python -m experiments.baseline_collection.build_extractor_comparison_dataset build

    # then, after build completes:
    ... --job-name extractor-compare-sample \\
        -- python -m experiments.baseline_collection.build_extractor_comparison_dataset sample
"""

from __future__ import annotations

import argparse
import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass

from fray.types import ResourceConfig
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl

from experiments.baseline_collection.decode_warcs_clean import (
    _decode_one_warc,
    _load_manifest,
    _normalize_record_id,
    _warc_path_hash,
    body_strip,
)
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# --- Inputs ----------------------------------------------------------------

MANIFEST = "experiments/distill/dclm_1p7b_completed_sample200_warcs.txt"

# A page was shown to a model only if its HTML fit the length fast-path filter
# (_filter_by_length: MAX_DOC_TOKENS * 6 chars) in run_extract_standalone.py.
# Longer pages were dropped *before* the model, so they are not abstentions.
MAX_FED_HTML_CHARS = 26624 * 6  # run_extract_standalone.MAX_DOC_TOKENS * 6


@dataclass(frozen=True)
class ModelSource:
    """Where one model's kept (useful) extraction batches live.

    ``buckets`` lists every regional bucket the WARC's batches may be in —
    cross-region resume scatters a single WARC's batch indices across regions,
    so all must be unioned. ``namespace`` is the path under the bucket.
    """

    column: str
    buckets: tuple[str, ...]
    namespace: str


MODELS = (
    ModelSource(
        "8b",
        ("marin-us-central1", "marin-us-east5", "marin-us-east1", "marin-us-west4", "marin-eu-west4"),
        "documents/baseline_llm_extraction/high_quality",
    ),
    ModelSource(
        "1p7b",
        ("marin-us-east5",),
        "documents/benchmark_extraction/high_quality_qwen3_1p7b_s5063",
    ),
    ModelSource(
        "0p6b",
        ("marin-us-east5", "marin-eu-west4", "marin-us-west4"),
        "documents/benchmark_extraction/high_quality_qwen3_0p6b_s10127",
    ),
)

# --- Outputs ---------------------------------------------------------------

OUTPUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SAMPLE_SIZE = 100_000
SAMPLE_SEED = 42

_COLUMNS = (
    "warc_record_id",
    "url",
    "warc_file",
    "snapshot",
    "fed_to_model",
    "stripped_html",
    "raw_html",
    "text_8b",
    "text_1p7b",
    "text_0p6b",
    "label_8b",
    "label_1p7b",
    "label_0p6b",
)


def _dataset_root(tag: str | None) -> str:
    return OUTPUT_ROOT if not tag else f"{OUTPUT_ROOT}_{tag}"


def _data_dir(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/data"


def _sample_dir(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/sample_100k"


def _schema():
    import pyarrow as pa

    # Explicit schema so every shard types identically (an all-null column on one
    # shard would otherwise infer a null type and break cross-shard loading).
    return pa.schema(
        [
            ("warc_record_id", pa.string()),
            ("url", pa.string()),
            ("warc_file", pa.string()),
            ("snapshot", pa.string()),
            ("fed_to_model", pa.bool_()),
            ("stripped_html", pa.string()),
            ("raw_html", pa.string()),
            ("text_8b", pa.string()),
            ("text_1p7b", pa.string()),
            ("text_0p6b", pa.string()),
            ("label_8b", pa.string()),
            ("label_1p7b", pa.string()),
            ("label_0p6b", pa.string()),
        ]
    )


# --- Stage: build ----------------------------------------------------------


def _kept_text_by_id(warc_hash: str, source: ModelSource) -> dict[str, str]:
    """Map normalized WARC-Record-ID -> extracted ``text`` for one model + WARC.

    Globs ``data-<hash>/batch_*.jsonl.gz`` across all of the model's regional
    buckets (cross-region resume splits a WARC's batches across regions). Only
    kept (useful) records are persisted, so absence from this map == abstention.
    """
    out: dict[str, str] = {}
    for bucket in source.buckets:
        pattern = f"gs://{bucket}/{source.namespace}/data-{warc_hash}/batch_*.jsonl.gz"
        for path in fsspec_glob(pattern):
            for r in load_jsonl(path):
                rid = _normalize_record_id(r.get("warc_record_id") or "")
                text = r.get("text") or ""
                if rid and text:
                    out[rid] = text  # cross-region duplicate batch -> identical text, idempotent
    return out


def _build_one_warc(warc_path: str) -> Iterator[dict]:
    """Decode one WARC and outer-join the three models' verdicts onto every page."""
    warc_hash = _warc_path_hash(warc_path)
    kept = {m.column: _kept_text_by_id(warc_hash, m) for m in MODELS}

    n_pages = 0
    for page in _decode_one_warc(warc_path):
        n_pages += 1
        rid = page["doc_id"]
        html = page["html"]
        texts = {col: kept[col].get(rid) for col in kept}
        kept_any = any(t is not None for t in texts.values())
        # A page is "fed" if it fit the length cap, OR a model kept it (a kept page
        # was necessarily fed — guards against a slightly-off cap dropping positives).
        fed = len(html) <= MAX_FED_HTML_CHARS or kept_any
        yield {
            "warc_record_id": rid,
            "url": page["url"],
            "warc_file": warc_path,
            "snapshot": page["snapshot"],
            "fed_to_model": fed,
            "stripped_html": body_strip(html),
            "raw_html": html if kept_any else None,  # raw only where >=1 model found it useful
            "text_8b": texts["8b"],
            "text_1p7b": texts["1p7b"],
            "text_0p6b": texts["0p6b"],
            "label_8b": "useful" if texts["8b"] is not None else "not_useful",
            "label_1p7b": "useful" if texts["1p7b"] is not None else "not_useful",
            "label_0p6b": "useful" if texts["0p6b"] is not None else "not_useful",
        }
    logger.info("WARC %s: %d pages judged", warc_hash, n_pages)


def run_build(limit_warcs: int | None, tag: str | None, max_workers: int) -> None:
    warc_paths = _load_manifest(MANIFEST)
    if limit_warcs is not None:
        warc_paths = warc_paths[:limit_warcs]
    logger.info("Building comparison dataset over %d WARCs -> %s", len(warc_paths), _data_dir(tag))

    def _output_path_fn(shard_idx: int, total_shards: int) -> str:
        return f"{_data_dir(tag)}/data-{_warc_path_hash(warc_paths[shard_idx])}.parquet"

    pipeline = (
        Dataset.from_list(warc_paths)
        .reshard(len(warc_paths))  # one WARC per shard -> one durable output file per WARC
        .flat_map(_build_one_warc)
        .write_parquet(_output_path_fn, schema=_schema(), skip_existing=True)
    )
    # Each worker holds a full WARC body (~1 GB compressed) + 5-10x decompressed
    # HTML; 28 GiB matches decode_warcs_clean's headroom for adversarial WARCs.
    ctx = ZephyrContext(
        name="extractor-compare-build",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="28g"),
    )
    ctx.execute(pipeline)
    logger.info("build done -> %s", _data_dir(tag))


# --- Stage: sample ---------------------------------------------------------
#
# Uniform random sample across all shards: draw round(shard_rows/total * SIZE)
# from each shard (uniform within shard) so the union is uniform overall. Set as
# module globals before execute (zephyr maps a top-level function).

_SAMPLE_TOTAL = 0
_SAMPLE_SIZE = 0


def _sample_one_shard(path: str) -> Iterator[dict]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    if n_rows == 0 or _SAMPLE_TOTAL == 0:
        return
    n_draw = round(n_rows * _SAMPLE_SIZE / _SAMPLE_TOTAL)
    if n_draw <= 0:
        return
    n_draw = min(n_draw, n_rows)
    rng = random.Random(f"{SAMPLE_SEED}:{path}")
    take = set(rng.sample(range(n_rows), n_draw))
    table = pf.read()
    cols = {c: table.column(c).to_pylist() for c in _COLUMNS}
    for i in sorted(take):
        yield {c: cols[c][i] for c in _COLUMNS}


def run_sample(tag: str | None, max_workers: int) -> None:
    import pyarrow.parquet as pq

    shards = sorted(fsspec_glob(f"{_data_dir(tag)}/*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards under {_data_dir(tag)}; run build first")
    total = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)
    logger.info("sample: %d shards, %d total rows -> drawing ~%d", len(shards), total, SAMPLE_SIZE)

    global _SAMPLE_TOTAL, _SAMPLE_SIZE
    _SAMPLE_TOTAL, _SAMPLE_SIZE = total, SAMPLE_SIZE

    out_template = f"{_sample_dir(tag)}/sample-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = Dataset.from_list(shards).flat_map(_sample_one_shard).write_parquet(out_template, schema=_schema())
    ctx = ZephyrContext(
        name="extractor-compare-sample",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="16g"),
    )
    ctx.execute(pipeline)
    logger.info("sample done -> %s (~%d rows)", _sample_dir(tag), SAMPLE_SIZE)


# --- Stage: metrics --------------------------------------------------------
#
# F1 of each small model treating the 8B as gold (useful vs [NO_USEFUL_CONTENT]),
# over fed pages, and mean normalized-Levenshtein similarity between each pair of
# extractions — both over the union (>=1 model extracted; non-extraction == "")
# and over the intersection (both extracted). Map shards -> partial counts/sums,
# reduce, write a results JSON. Reads only the small label/text columns (skips the
# 161 GB of HTML).

_GOLD = "8b"
_SMALL = ("1p7b", "0p6b")
_PAIRS = (("8b", "1p7b"), ("8b", "0p6b"), ("1p7b", "0p6b"))
_METRIC_COLUMNS = ["fed_to_model", "label_8b", "label_1p7b", "label_0p6b", "text_8b", "text_1p7b", "text_0p6b"]


def _shard_partial(path: str) -> Iterator[dict]:
    """Per-shard confusion counts (vs 8B gold) + Levenshtein sums for each pair."""
    import pyarrow.parquet as pq
    from rapidfuzz.distance import Levenshtein

    t = pq.read_table(path, columns=_METRIC_COLUMNS).to_pydict()
    n = len(t["fed_to_model"])
    p: dict[str, float] = {"n_rows": n, "n_fed": 0}
    for m in _SMALL:
        for k in ("tp", "fp", "fn", "tn"):
            p[f"f1_{m}_{k}"] = 0
    for a, b in _PAIRS:
        for v in ("union_sum", "union_cnt", "both_sum", "both_cnt"):
            p[f"lev_{a}_{b}_{v}"] = 0.0

    for i in range(n):
        if t["fed_to_model"][i]:
            p["n_fed"] += 1
            gold = t[f"label_{_GOLD}"][i] == "useful"
            for m in _SMALL:
                pred = t[f"label_{m}"][i] == "useful"
                key = "tp" if pred and gold else "fp" if pred else "fn" if gold else "tn"
                p[f"f1_{m}_{key}"] += 1
        for a, b in _PAIRS:
            ta, tb = t[f"text_{a}"][i], t[f"text_{b}"][i]
            if ta is None and tb is None:
                continue  # neither extracted — nothing to compare
            sim = Levenshtein.normalized_similarity(ta or "", tb or "")
            p[f"lev_{a}_{b}_union_sum"] += sim
            p[f"lev_{a}_{b}_union_cnt"] += 1
            if ta is not None and tb is not None:
                p[f"lev_{a}_{b}_both_sum"] += sim
                p[f"lev_{a}_{b}_both_cnt"] += 1
    logger.info(
        "metrics shard done: %s (%d rows, %d fed, 8b-both=%d 0p6b-both=%d)",
        path.rsplit("/", 1)[-1],
        n,
        p["n_fed"],
        int(p["lev_8b_1p7b_both_cnt"]),
        int(p["lev_8b_0p6b_both_cnt"]),
    )
    yield p


def run_metrics(subdir: str, tag: str, max_workers: int) -> None:
    import json
    from collections import defaultdict

    import fsspec

    # Fail fast IN THE COORDINATOR if the Levenshtein dep is missing (it lives in the
    # marin 'extraction-bakeoff' extra — pass --extra extraction-bakeoff). Otherwise
    # every shard ModuleNotFound-retries 3x before the job dies minutes later.
    try:
        from rapidfuzz.distance import Levenshtein

        Levenshtein.normalized_similarity("ab", "ac")
    except ImportError as e:
        raise RuntimeError(
            "rapidfuzz missing on the worker env — launch with `--extra cpu --extra extraction-bakeoff`"
        ) from e
    logger.info("rapidfuzz OK; computing metrics over subdir=%s", subdir)

    shards = sorted(fsspec_glob(f"{_dataset_root(tag)}/{subdir}/*.parquet"))
    if not shards:
        raise RuntimeError(f"no parquet shards under {_dataset_root(tag)}/{subdir}")
    logger.info("metrics: %d shards to process", len(shards))
    partials_dir = f"{_dataset_root(tag)}/_metrics_partials/{subdir.replace('/', '_')}"
    out_template = f"{partials_dir}/part-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_list(shards).flat_map(_shard_partial).write_jsonl(out_template)
    ctx = ZephyrContext(
        name="extractor-compare-metrics",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="8g"),
    )
    ctx.execute(pipeline)

    agg: dict[str, float] = defaultdict(float)
    for f in sorted(fsspec_glob(f"{partials_dir}/*.jsonl.gz")):
        for row in load_jsonl(f):
            for k, v in row.items():
                agg[k] += v

    results: dict = {
        "subdir": subdir,
        "n_rows": int(agg["n_rows"]),
        "n_fed": int(agg["n_fed"]),
        "f1": {},
        "levenshtein": {},
    }
    for m in _SMALL:
        tp, fp, fn, tn = (agg[f"f1_{m}_{k}"] for k in ("tp", "fp", "fn", "tn"))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        results["f1"][f"{m}_vs_8b"] = {
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    for a, b in _PAIRS:
        us, uc = agg[f"lev_{a}_{b}_union_sum"], agg[f"lev_{a}_{b}_union_cnt"]
        bs, bc = agg[f"lev_{a}_{b}_both_sum"], agg[f"lev_{a}_{b}_both_cnt"]
        results["levenshtein"][f"{a}_vs_{b}"] = {
            "all_union_mean": us / uc if uc else 0.0,
            "all_union_n": int(uc),
            "both_extracted_mean": bs / bc if bc else 0.0,
            "both_extracted_n": int(bc),
        }

    out_json = f"{_dataset_root(tag)}/metrics_{subdir.replace('/', '_')}.json"
    with fsspec.open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("metrics (%s): %s", subdir, json.dumps(results, indent=2))
    logger.info("metrics written -> %s", out_json)


# --- Entry point -----------------------------------------------------------


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_build = sub.add_parser("build", help="Decode + 3-model outer-join -> per-WARC parquet shards.")
    p_build.add_argument("--limit-warcs", type=int, default=None, help="Smoke test: only the first N WARCs.")
    p_build.add_argument("--max-workers", type=int, default=200)

    p_sample = sub.add_parser("sample", help=f"Draw ~{SAMPLE_SIZE} rows uniformly at random.")
    p_sample.add_argument("--max-workers", type=int, default=64)

    p_metrics = sub.add_parser("metrics", help="F1 (vs 8B gold) + Levenshtein similarity over a data dir.")
    p_metrics.add_argument("--subdir", default="data", help="Which dir under the dataset root (data | sample_100k).")
    p_metrics.add_argument("--max-workers", type=int, default=200)

    for p in (p_build, p_sample, p_metrics):
        p.add_argument("--tag", default=None, help="Suffix the output dir to isolate smoke runs.")
    args = parser.parse_args()

    if args.stage == "build":
        run_build(args.limit_warcs, args.tag, args.max_workers)
    elif args.stage == "sample":
        run_sample(args.tag, args.max_workers)
    elif args.stage == "metrics":
        run_metrics(args.subdir, args.tag, args.max_workers)


if __name__ == "__main__":
    main()
