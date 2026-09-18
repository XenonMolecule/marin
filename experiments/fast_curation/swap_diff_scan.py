# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scan a sample of arm A vs arm B documents for the largest extraction differences, by domain.

Samples ``--n-a-shards`` of A's post-decon shards (all of their documents, with A's topic and
quality labels read from the co-partitioned tables), pulls the same documents' B text out of B's
per-WARC shards by ``doc_id``, joins the two by ``doc_id`` and scores every pair:

* ``jaccard``   -- word-set Jaccard similarity of the two texts (lower = more different)
* ``la``/``lb`` -- text lengths in chars
* ``code``/``math``/``table`` -- heuristic content flags on the union of both texts

Writes ``{root}/diff_scan/scored/`` (every pair; texts kept only for pairs under
``KEEP_TEXT_BELOW``) and ``{root}/diff_scan/selection.json`` with the most different pairs
overall and within each domain of interest, texts included, for the side-by-side viewer.

    python -m experiments.fast_curation.swap_diff_scan --n-a-shards 16 --seed 0
"""

from __future__ import annotations

import argparse
import functools
import gzip
import io
import json
import logging
import random
import re
from collections.abc import Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet
from zephyr.worker_context import zephyr_worker_ctx

from experiments.baseline_collection.grid_corpora import output_stem
from experiments.fast_curation.build_swap_allowlist import A_TREE, quality_table_path, topic_table_path
from experiments.fast_curation.swap_extractor import LABEL_BASE, OUTPUT_ROOT, SOURCE_DATASET
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

KEEP_TEXT_BELOW = 0.7  # keep both texts for pairs less similar than this
TEXT_CAP = 30_000
MIN_CHARS = 400  # both texts at least this long to count as a "major difference" candidate
DOMAINS = ("Software Dev.", "Science & Tech.", "Health", "Finance & Business", "Crime & Law", "Education & Jobs")
PER_DOMAIN = 4
OVERALL = 10

_WORD = re.compile(r"\w+")
_MATH = re.compile(
    r"\\(?:frac|sum|int|alpha|beta|theta|mathbb|begin\{|left|right|sqrt)"
    "|[\u2211\u222b\u221a\u2264\u2265\u2260\u00b1\u00d7\u00f7]"
    r"|\$\S[^$\n]{2,}\$|\^\{|_\{"
)
_CODE_LINE = re.compile(
    r"^\s*(?:def |class |import |from \S+ import|#include|function\s*\w*\(|var |let |const |return\b"
    r"|if \(|for \(|while \(|\}|\{|</?\w+>)|;\s*$"
)
_FENCE = re.compile(r"^```", re.M)
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.M)


def _read_jsonl(path: str) -> Iterator[dict]:
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _read_parquet(path: str, columns: list[str]) -> pa.Table:
    with fsspec.open(path, "rb") as fh:
        return pq.read_table(io.BytesIO(fh.read()), columns=columns)


A_SAMPLE_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("a_text", pa.large_string()),
        ("topic", pa.string()),
        ("quality_bucket", pa.int8()),
    ]
)
B_SAMPLE_SCHEMA = pa.schema([("doc_id", pa.string()), ("b_text", pa.large_string())])
SCORED_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("url", pa.string()),
        ("topic", pa.string()),
        ("quality_bucket", pa.int8()),
        ("la", pa.int32()),
        ("lb", pa.int32()),
        ("jaccard", pa.float32()),
        ("code", pa.bool_()),
        ("math", pa.bool_()),
        ("table", pa.bool_()),
        ("a_text", pa.large_string()),
        ("b_text", pa.large_string()),
    ]
)


def a_sample_rows(shard_path: str, label_base: str, dataset: str, label_names: list[str]) -> Iterator[dict]:
    stem = output_stem(shard_path)
    topic = _read_parquet(topic_table_path(label_base, dataset, stem), ["native_id", "cluster_24"])
    quality = _read_parquet(quality_table_path(label_base, dataset, stem), ["quality_bucket"])
    native = topic.column("native_id").to_pylist()
    clusters = topic.column("cluster_24").to_pylist()
    buckets = quality.column("quality_bucket").to_pylist()
    for i, rec in enumerate(_read_jsonl(shard_path)):
        if native[i] != rec["doc_id"]:
            raise RuntimeError(f"{stem} row {i}: label native_id != doc_id")
        yield {
            "doc_id": rec["doc_id"],
            "url": rec["url"],
            "a_text": rec["text"],
            "topic": label_names[clusters[i]],
            "quality_bucket": buckets[i],
        }


def b_sample_rows(shard_path: str) -> Iterator[dict]:
    ids = zephyr_worker_ctx().get_shared("ids")
    for rec in _read_jsonl(shard_path):
        if rec["doc_id"] in ids:
            yield {"doc_id": rec["doc_id"], "b_text": rec["text"]}


def _tagged_rows(path: str) -> Iterator[dict]:
    yield from load_parquet(path)


def _flags(text: str) -> tuple[bool, bool, bool]:
    code = bool(_FENCE.search(text)) or sum(1 for line in text.splitlines() if _CODE_LINE.search(line)) >= 6
    math = len(_MATH.findall(text)) >= 3
    table = len(_TABLE_LINE.findall(text)) >= 3
    return code, math, table


def score_pair(doc_id: str, rows: Iterator[dict]) -> Iterator[dict]:
    a = b = None
    for row in rows:
        if "a_text" in row:
            a = row
        else:
            b = row
    if a is None or b is None:
        return  # B dropped it as short, or (impossible by construction) A missing
    wa, wb = set(_WORD.findall(a["a_text"].lower())), set(_WORD.findall(b["b_text"].lower()))
    union = len(wa | wb)
    jac = len(wa & wb) / union if union else 1.0
    code, math, table = _flags(a["a_text"] + "\n" + b["b_text"])
    keep = jac < KEEP_TEXT_BELOW
    yield {
        "doc_id": doc_id,
        "url": a["url"],
        "topic": a["topic"],
        "quality_bucket": a["quality_bucket"],
        "la": len(a["a_text"]),
        "lb": len(b["b_text"]),
        "jaccard": jac,
        "code": code,
        "math": math,
        "table": table,
        "a_text": a["a_text"][:TEXT_CAP] if keep else "",
        "b_text": b["b_text"][:TEXT_CAP] if keep else "",
    }


def _ctx(name: str, workers: int, ram: str) -> ZephyrContext:
    return ZephyrContext(name=name, max_workers=workers, resources=ResourceConfig(cpu=1, ram=ram, disk="8g"))


def run(root: str, a_tree: str, label_base: str, dataset: str, n_a_shards: int, seed: int) -> dict:
    out = f"{root}/diff_scan"
    with fsspec.open(f"{root}/labels/_meta.json", "r") as fh:
        label_names = json.load(fh)["label_names"]
    a_shards = sorted(fsspec_glob(f"{a_tree}/data-*.jsonl.gz"))
    picked = sorted(random.Random(seed).sample(a_shards, n_a_shards))
    logger.info("A sample: %d shards", len(picked))
    _ctx("swap-diff-a", n_a_shards, "8g").execute(
        Dataset.from_iterable(picked)
        .flat_map(functools.partial(a_sample_rows, label_base=label_base, dataset=dataset, label_names=label_names))
        .write_parquet(f"{out}/a_sample/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=A_SAMPLE_SCHEMA)
    )
    ids: set[str] = set()
    for p in fsspec_glob(f"{out}/a_sample/*.parquet"):
        ids.update(_read_parquet(p, ["doc_id"]).column("doc_id").to_pylist())
    logger.info("A sample: %d doc_ids; filtering B shards", len(ids))
    b_shards = sorted(fsspec_glob(f"{root}/deduped/data-*.jsonl.gz"))
    ctx = _ctx("swap-diff-b", 200, "4g")
    ctx.put("ids", ids)
    ctx.execute(
        Dataset.from_iterable(b_shards)
        .flat_map(b_sample_rows)
        .write_parquet(f"{out}/b_sample/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=B_SAMPLE_SCHEMA)
    )
    files = sorted(fsspec_glob(f"{out}/a_sample/*.parquet")) + sorted(fsspec_glob(f"{out}/b_sample/*.parquet"))
    logger.info("joining %d files by doc_id", len(files))
    _ctx("swap-diff-join", 200, "8g").execute(
        Dataset.from_iterable(files)
        .flat_map(_tagged_rows)
        .group_by(key=lambda r: r["doc_id"], reducer=score_pair, num_output_shards=256)
        .write_parquet(f"{out}/scored/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=SCORED_SCHEMA)
    )
    return select(out)


def select(out: str) -> dict:
    """Pick the most different pairs overall and per domain from the scored table; write selection.json."""
    cols = ["doc_id", "url", "topic", "quality_bucket", "la", "lb", "jaccard", "code", "math", "table"]
    tables = [_read_parquet(p, cols) for p in sorted(fsspec_glob(f"{out}/scored/*.parquet"))]
    meta = pa.concat_tables(tables).to_pylist()
    ok = [r for r in meta if r["la"] >= MIN_CHARS and r["lb"] >= MIN_CHARS]
    ok.sort(key=lambda r: r["jaccard"])
    groups: dict[str, list[dict]] = {"Most different overall": ok[:OVERALL]}
    groups["Code"] = [r for r in ok if r["code"] or r["topic"] == "Software Dev."][:PER_DOMAIN]
    groups["Math"] = [r for r in ok if r["math"]][:PER_DOMAIN]
    groups["Tables"] = [r for r in ok if r["table"]][:PER_DOMAIN]
    for d in DOMAINS:
        groups[d] = [r for r in ok if r["topic"] == d][:PER_DOMAIN]
    wanted = {r["doc_id"] for rs in groups.values() for r in rs}
    texts: dict[str, dict] = {}
    for p in sorted(fsspec_glob(f"{out}/scored/*.parquet")):
        t = _read_parquet(p, ["doc_id", "a_text", "b_text"])
        for row in t.to_pylist():
            if row["doc_id"] in wanted:
                texts[row["doc_id"]] = row
    jac = sorted(r["jaccard"] for r in meta)
    q = lambda f: jac[int(f * (len(jac) - 1))]  # noqa: E731
    summary = {
        "n_pairs": len(meta),
        "n_candidates_min_chars": len(ok),
        "jaccard_quantiles": {"p05": q(0.05), "p25": q(0.25), "p50": q(0.5), "p75": q(0.75), "p95": q(0.95)},
        "share_below_0p5": sum(1 for r in meta if r["jaccard"] < 0.5) / max(1, len(meta)),
        "per_topic_median_jaccard": {},
        "flag_counts": {k: sum(1 for r in meta if r[k]) for k in ("code", "math", "table")},
    }
    by_topic: dict[str, list[float]] = {}
    for r in meta:
        by_topic.setdefault(r["topic"], []).append(r["jaccard"])
    summary["per_topic_median_jaccard"] = {t: sorted(v)[len(v) // 2] for t, v in sorted(by_topic.items())}
    selection = {
        "summary": summary,
        "groups": {
            name: [{**r, "a_text": texts[r["doc_id"]]["a_text"], "b_text": texts[r["doc_id"]]["b_text"]} for r in rs]
            for name, rs in groups.items()
        },
    }
    with fsspec.open(f"{out}/selection.json", "w") as fh:
        json.dump(selection, fh)
    logger.info("selection: %s", json.dumps(summary))
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--a-tree", default=A_TREE)
    ap.add_argument("--label-base", default=LABEL_BASE)
    ap.add_argument("--source-dataset", default=SOURCE_DATASET)
    ap.add_argument("--n-a-shards", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-only", action="store_true", help="Skip the scan; rebuild selection.json from scored/.")
    args = ap.parse_args()
    if args.select_only:
        print(json.dumps(select(f"{args.root}/diff_scan"), indent=2))
        return
    print(
        json.dumps(
            run(args.root, args.a_tree, args.label_base, args.source_dataset, args.n_a_shards, args.seed), indent=2
        )
    )


if __name__ == "__main__":
    main()
