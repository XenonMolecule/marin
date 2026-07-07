#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inspect decontamination results: flag rate + per-eval-item attribution.

Re-scans a marked corpus against an **exact** n-gram -> eval-item index built
from the decon source (the same `extract_features`/`_bloom_hash` the mark step
used, so results match — but exact dict lookup means zero false positives, and
each hit carries the *specific* eval item it matched, which the merged bloom
cannot). Produces a compact `inspect_summary.json`:

    {
      "total_docs", "flagged_docs", "flag_pct",
      "per_task": {task: flagged_doc_count, ...},
      "samples": [{corpus_text, matched_ngram, eval_item_id, eval_task,
                   eval_text, num_eval_items_matched}, ...]
    }

The summary is small (stats + a capped sample per shard) and safe to pull
locally for the viz. READ-ONLY.

Usage (Iris, CPU, in the corpus region)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --region us-central1 --job-name decon-inspect-hq-core-v2 \\
        -- python experiments/baseline_collection/decon_inspect.py \\
           --corpus-path gs://marin-us-central1/documents/baseline_high_quality_deduped/10364warcs/deduped/ \\
           --decon-source gs://marin-us-central1/decontamination/dclm_core_v2/ \\
           --output-path gs://marin-us-central1/documents/baseline_high_quality_decon/10364warcs_core_v2/inspect/
"""

from __future__ import annotations

import argparse
import json
import logging

from fray import ResourceConfig
from marin.processing.classification.decon import NGramConfig, _bloom_hash, extract_features
from marin.processing.classification.deduplication.dedup_commons import DEFAULT_FILETYPES, _collect_input_files
from rigging.filesystem import filesystem as marin_filesystem
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

# Per-shard cap on attributed samples carried into the summary (full flag counts
# are always exact; only the example list is capped to keep the summary small).
SAMPLES_PER_SHARD = 8

# Corpus text kept per sample: full doc if short, else a window centred on the
# match so the highlighted n-gram is always visible (and the viz can scroll it).
_CORPUS_WINDOW = 16000

# Cap on distinct matched n-gram hashes carried per flagged doc (bounds output for
# pathological docs; a doc matching >200 distinct eval n-grams is plainly flagged).
_MAX_DOC_HASHES = 200

# Per-worker cache of the source index so each worker builds it once.
_INDEX_CACHE: dict = {}


def _read_jsonl(path: str) -> list[dict]:
    fs, p = url_to_fs(path)
    with fs.open(p, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_source_index(source_path: str, ngram_length: int) -> tuple[dict[int, int], list[tuple[str, str, str]]]:
    """ngram_hash -> item index, plus the list of (id, task, text) items.

    First item to claim a hash wins (cross-item n-gram collisions are rare and
    only affect which example we *display*, never the flag decision).
    """
    ng = NGramConfig(ngram_length=ngram_length, stride=0, overlap_threshold=0.0)
    index: dict[int, int] = {}
    items: list[tuple[str, str, str]] = []
    for path in _collect_input_files(input_paths=source_path, filetypes=DEFAULT_FILETYPES):
        for rec in _read_jsonl(path):
            idx = len(items)
            items.append((rec.get("id", ""), rec.get("task", ""), rec.get("text", "")))
            for feat in extract_features(rec.get("text", ""), ng):
                h = _bloom_hash(feat)
                index.setdefault(h, idx)
    logger.info("Source index: %d ngrams from %d eval items", len(index), len(items))
    return index, items


def _get_index(source_path: str, ngram_length: int):
    if not _INDEX_CACHE:
        index, items = _build_source_index(source_path, ngram_length)
        _INDEX_CACHE["index"] = index
        _INDEX_CACHE["items"] = items
    return _INDEX_CACHE["index"], _INDEX_CACHE["items"]


def run_inspect(
    corpus_path: str, source_path: str, output_path: str, ngram_length: int, text_field: str, df_threshold: int
) -> dict:
    corpus_files = _collect_input_files(input_paths=corpus_path, filetypes=DEFAULT_FILETYPES)
    ng = NGramConfig(ngram_length=ngram_length, stride=0, overlap_threshold=0.0)
    stats_dir = f"{output_path.rstrip('/')}/stats"

    def scan_shard(records, _):
        index, items = _get_index(source_path, ngram_length)
        total = 0
        df_local: dict[int, int] = {}  # eval-ngram hash -> corpus docs containing it (this shard)
        flagged_records: list[dict] = []  # per flagged doc: distinct matched hashes + tasks
        samples: list[dict] = []
        for rec in records:
            total += 1
            text = rec.get(text_field, "") or ""
            hit: dict[int, int] = {}  # ngram hash -> first eval-item index
            matched_ngram = matched_hash = matched_item = None
            for feat in extract_features(text, ng):
                h = _bloom_hash(feat)
                j = index.get(h)
                if j is not None:
                    if h not in hit:
                        hit[h] = j
                    if matched_ngram is None:
                        # Displayed n-gram and item stay consistent (n-gram ⊆ both texts).
                        matched_ngram, matched_hash, matched_item = feat, h, j
            if not hit:
                continue
            # Document frequency: count this doc once per distinct matched n-gram.
            for h in hit:
                df_local[h] = df_local.get(h, 0) + 1
            hashes = list(hit)[:_MAX_DOC_HASHES]
            tasks = sorted({items[j][1] for j in hit.values()})
            flagged_records.append({"h": hashes, "t": tasks})
            if len(samples) < SAMPLES_PER_SHARD:
                # Window the corpus text on the match so the highlight is always
                # visible even when the overlap is deep in a long document.
                off = text.find(matched_ngram)
                if off < 0 or len(text) <= _CORPUS_WINDOW:
                    shown, shown_off = text[:_CORPUS_WINDOW], off
                else:
                    start = max(0, off - _CORPUS_WINDOW // 2)
                    shown, shown_off = text[start : start + _CORPUS_WINDOW], off - start
                samples.append(
                    {
                        "corpus_text": shown,
                        "corpus_len": len(text),
                        "match_offset": shown_off,
                        "doc_match_offset": off,
                        "matched_ngram": matched_ngram,
                        "matched_hash": matched_hash,
                        "hashes": hashes,
                        "eval_item_id": items[matched_item][0],
                        "eval_task": items[matched_item][1],
                        "eval_text": items[matched_item][2][:4000],
                        "num_eval_items_matched": len(hit),
                    }
                )
        yield {"total": total, "df": df_local, "flagged_records": flagged_records, "samples": samples}

    ctx = ZephyrContext(name="decon-inspect", resources=ResourceConfig(cpu=2, ram="8g"))
    ctx.execute(
        Dataset.from_iterable(corpus_files)
        .flat_map(load_file)
        .map_shard(scan_shard)
        .write_jsonl(f"{stats_dir}/shard-{{shard:05d}}-of-{{total:05d}}.jsonl", skip_existing=True)
    )

    return _merge_stats(stats_dir, ngram_length, df_threshold)


def _merge_stats(stats_dir: str, ngram_length: int, df_threshold: int) -> dict:
    total = 0
    global_df: dict[int, int] = {}
    flagged_records: list[dict] = []
    samples: list[dict] = []
    for path in _collect_input_files(input_paths=stats_dir, filetypes=DEFAULT_FILETYPES):
        for rec in _read_jsonl(path):
            total += rec["total"]
            for h_str, count in rec["df"].items():  # JSON object keys are strings
                h = int(h_str)
                global_df[h] = global_df.get(h, 0) + count
            flagged_records.extend(rec["flagged_records"])
            samples.extend(rec["samples"])

    # Classify each flagged doc by its most-distinctive overlap (lowest-DF matched
    # n-gram). GPT-3's rule: an n-gram matching > df_threshold docs is ubiquitous
    # public text, not leakage; a doc is genuine contamination only if it has a
    # match with DF <= df_threshold.
    flagged = len(flagged_records)
    per_task: dict[str, int] = {}
    per_task_genuine: dict[str, int] = {}
    histogram: dict[int, int] = {}
    genuine = 0
    for fr in flagged_records:
        min_df = min((global_df[h] for h in fr["h"]), default=0)
        histogram[min_df] = histogram.get(min_df, 0) + 1
        is_genuine = min_df <= df_threshold
        genuine += is_genuine
        for task in fr["t"]:
            per_task[task] = per_task.get(task, 0) + 1
            if is_genuine:
                per_task_genuine[task] = per_task_genuine.get(task, 0) + 1

    out_samples = []
    for s in samples[:500]:
        hashes = s.pop("hashes", [])
        matched_hash = s.pop("matched_hash", None)
        s["matched_ngram_df"] = global_df.get(matched_hash, 0)
        s["doc_min_df"] = min((global_df[h] for h in hashes), default=0)
        out_samples.append(s)

    return {
        "ngram_length": ngram_length,
        "df_threshold": df_threshold,
        "total_docs": total,
        "flagged_docs": flagged,
        "flag_pct": (100.0 * flagged / total) if total else 0.0,
        "genuine_docs": genuine,
        "genuine_pct": (100.0 * genuine / total) if total else 0.0,
        "per_task": dict(sorted(per_task.items(), key=lambda kv: -kv[1])),
        "per_task_genuine": dict(sorted(per_task_genuine.items(), key=lambda kv: -kv[1])),
        # doc_min_df -> #flagged docs, so the viz slider can recompute the genuine
        # count live for any threshold.
        "min_df_histogram": {str(k): v for k, v in sorted(histogram.items())},
        "samples": out_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-path", required=True, help="Marked corpus dir (the docs themselves).")
    parser.add_argument("--decon-source", required=True, help="Eval-item source dir (the filter was built from).")
    parser.add_argument("--output-path", required=True, help="Where to write stats/ and inspect_summary.json.")
    parser.add_argument("--ngram-length", type=int, default=13)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--df-threshold",
        type=int,
        default=10,
        help="GPT-3 rule: an n-gram matching > this many corpus docs is ubiquitous public text, "
        "not leakage. A doc is 'genuine' contamination only if it has a match with DF <= threshold.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = run_inspect(
        args.corpus_path, args.decon_source, args.output_path, args.ngram_length, args.text_field, args.df_threshold
    )

    summary_path = f"{args.output_path.rstrip('/')}/inspect_summary.json"
    with marin_filesystem("gcs").open(summary_path, "w") as f:
        json.dump(summary, f)
    logger.info(
        "INSPECT done (n=%d): flagged %d (%.4f%%); genuine DF<=%d: %d (%.4f%%) of %d -> %s",
        summary["ngram_length"],
        summary["flagged_docs"],
        summary["flag_pct"],
        summary["df_threshold"],
        summary["genuine_docs"],
        summary["genuine_pct"],
        summary["total_docs"],
        summary_path,
    )


if __name__ == "__main__":
    main()
