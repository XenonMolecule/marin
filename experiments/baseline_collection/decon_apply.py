#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply CORE v2 decontamination removal — drop genuine-contamination docs.

Two passes over the (already deduped) corpus:
  1. Document frequency: count how many corpus docs each eval n-gram appears in.
  2. Drop any doc containing an eval n-gram with DF <= threshold (a *distinctive*
     eval passage — genuine leakage); keep everything else. GPT-3's rule: an
     n-gram in > threshold docs is ubiquitous public text (Bill of Rights,
     scripture, census boilerplate), NOT contamination, so it does not trigger
     removal.

Survivors = docs that passed BOTH dedup and decontam — the tokenizer input.
READ-ONLY w.r.t. the input; writes a fresh `data-*.jsonl.gz` tree.

Two corpus passes are required (the drop decision needs the global DF, known only
after pass 1). The DF pass is cached (`skip_existing`), so a retry resumes.

Usage (Iris, CPU, in the corpus region)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --region us-central1 --job-name decon-apply-high-quality \\
        -- python experiments/baseline_collection/decon_apply.py \\
           --input-path gs://marin-us-central1/documents/baseline_high_quality_deduped/10364warcs/deduped/ \\
           --decon-source gs://marin-us-central1/decontamination/dclm_core_v2/ \\
           --output-path gs://marin-us-central1/documents/baseline_high_quality_decon_deduped/10364warcs/deduped/ \\
           --ngram-length 15 --df-threshold 10
"""

from __future__ import annotations

import argparse
import logging

from fray import ResourceConfig
from marin.processing.classification.decon import NGramConfig, _bloom_hash, extract_features
from marin.processing.classification.deduplication.dedup_commons import DEFAULT_FILETYPES, _collect_input_files
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file

from experiments.baseline_collection.decon_inspect import _get_index, _read_jsonl

logger = logging.getLogger(__name__)


def run_apply(
    input_path: str, source_path: str, output_path: str, ngram_length: int, df_threshold: int, text_field: str
) -> dict:
    corpus_files = _collect_input_files(input_paths=input_path, filetypes=DEFAULT_FILETYPES)
    ng = NGramConfig(ngram_length=ngram_length, stride=0, overlap_threshold=0.0)
    df_dir = f"{output_path.rstrip('/')}_df"  # sibling of the corpus dir; not matched by the tokenizer glob

    # --- Pass 1: document frequency of each eval n-gram across the corpus ---
    def df_shard(records, _):
        index, _items = _get_index(source_path, ngram_length)
        df_local: dict[int, int] = {}
        for rec in records:
            text = rec.get(text_field, "") or ""
            seen: set[int] = set()
            for feat in extract_features(text, ng):
                h = _bloom_hash(feat)
                if h in index and h not in seen:  # count this doc once per distinct eval n-gram
                    seen.add(h)
                    df_local[h] = df_local.get(h, 0) + 1
        yield {"df": df_local}

    ZephyrContext(name="decon-apply-df", resources=ResourceConfig(cpu=2, ram="8g")).execute(
        Dataset.from_iterable(corpus_files)
        .flat_map(load_file)
        .map_shard(df_shard)
        .write_jsonl(f"{df_dir}/shard-{{shard:05d}}-of-{{total:05d}}.jsonl", skip_existing=True)
    )

    global_df: dict[int, int] = {}
    for path in _collect_input_files(input_paths=df_dir, filetypes=DEFAULT_FILETYPES):
        for rec in _read_jsonl(path):
            for h_str, count in rec["df"].items():
                h = int(h_str)
                global_df[h] = global_df.get(h, 0) + count
    # "Rare" eval n-grams = the contamination signal: present but in <= threshold docs.
    rare: frozenset[int] = frozenset(h for h, c in global_df.items() if c <= df_threshold)
    logger.info(
        "DF computed: %d eval n-grams appear in the corpus; %d are rare (DF<=%d) -> drop docs containing them",
        len(global_df),
        len(rare),
        df_threshold,
    )

    # --- Pass 2: drop docs containing any rare eval n-gram; keep the rest ---
    def filter_shard(records, _):
        for rec in records:
            text = rec.get(text_field, "") or ""
            if any(_bloom_hash(feat) in rare for feat in extract_features(text, ng)):
                continue  # genuine contamination -> drop
            yield rec

    results = (
        ZephyrContext(name="decon-apply-filter", resources=ResourceConfig(cpu=2, ram="8g"))
        .execute(
            Dataset.from_iterable(corpus_files)
            .flat_map(load_file)
            .map_shard(filter_shard)
            .write_jsonl(f"{output_path.rstrip('/')}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz", skip_existing=True)
        )
        .results
    )
    logger.info("Decontaminated corpus written -> %s (%d shards)", output_path, len(results) if results else 0)
    return {"eval_ngrams_in_corpus": len(global_df), "rare_ngrams_dropped_on": len(rare)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-path", required=True, help="Deduped corpus dir to decontaminate.")
    parser.add_argument("--decon-source", required=True, help="Eval-item source dir (the contamination reference).")
    parser.add_argument("--output-path", required=True, help="Where to write the survivor data-*.jsonl.gz tree.")
    parser.add_argument("--ngram-length", type=int, default=15)
    parser.add_argument("--df-threshold", type=int, default=10, help="Drop docs whose match has corpus DF <= this.")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_apply(
        args.input_path, args.decon_source, args.output_path, args.ngram_length, args.df_threshold, args.text_field
    )
    logger.info("Decon apply complete: %s", result)


if __name__ == "__main__":
    main()
