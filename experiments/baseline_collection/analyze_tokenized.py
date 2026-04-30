# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source stats on the tokenized pretraining caches.

Designed to diagnose *why* resiliparse beats filtered datasets on paloma loss.
Scans each of the five baseline-collection tokenized caches (plus llm_curated)
and emits a single ``tokenized_stats.json`` with per-source statistics:

- doc count, total tokens, tokens/doc quantiles
- % docs starting with BOS, % ending with EOS, special-token-per-doc stats
- top-30 most frequent token ids (as text)
- max repeat-run distribution (proxy for degenerate content)
- first-100-token hash collision rate (proxy for near-duplicate openings —
  useful for catching synthetic-rephraser duplication in ``nemotron_full``)
- docs-per-4096-token packed-sequence estimate (proxy for packing boundary
  rate; longer docs → fewer boundaries → smoother gradients)

Runs entirely on us-central2 (all caches live there). No cross-region I/O —
the cross-region guard in ``matched_viewer.py`` protects bad invocations.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py --cluster us-central2 --no_wait \\
        -- python experiments/baseline_collection/analyze_tokenized.py \\
               --threads 32 --sample-docs 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import numpy as np

from experiments.baseline_collection.matched_viewer import _assert_no_cross_region

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("analyze_tokenized")

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

SOURCES: dict[str, str] = {
    "resiliparse": "gs://marin-us-central2/tokenized/baseline_resiliparse-7278c1/train",
    "nemotron": "gs://marin-us-central2/tokenized/baseline_nemotron-c67de9/train",
    "nemotron_full": "gs://marin-us-central2/tokenized/baseline_nemotron_full-d4e3af/train",
    "dclm": "gs://marin-us-central2/tokenized/baseline_dclm-23e9be/train",
    "fineweb_edu": "gs://marin-us-central2/tokenized/baseline_fineweb_edu-7a3bc5/train",
    # Lives in us-central1 — only scan this source when running on us-central1 (e.g., via iris).
    "llm_curated": "gs://marin-us-central1/tokenized/baseline_llm_curated-3c07e4/train",
    # Paloma eval subsets (us-central2, /validation subdir)
    "paloma/4chan": "gs://marin-us-central2/tokenized/paloma/4chan-496ad5/validation",
    "paloma/c4_100_domains": "gs://marin-us-central2/tokenized/paloma/c4_100_domains-2b6db7/validation",
    "paloma/c4_en": "gs://marin-us-central2/tokenized/paloma/c4_en-cf1f79/validation",
    "paloma/dolma-v1_5": "gs://marin-us-central2/tokenized/paloma/dolma-v1_5-d3bed7/validation",
    "paloma/dolma_100_programing_languages": (
        "gs://marin-us-central2/tokenized/paloma/dolma_100_programing_languages-369132/validation"
    ),
    "paloma/dolma_100_subreddits": "gs://marin-us-central2/tokenized/paloma/dolma_100_subreddits-f25f70/validation",
    "paloma/falcon-refinedweb": "gs://marin-us-central2/tokenized/paloma/falcon-refinedweb-75d43b/validation",
    "paloma/gab": "gs://marin-us-central2/tokenized/paloma/gab-ccaced/validation",
    "paloma/m2d2_s2orc_unsplit": "gs://marin-us-central2/tokenized/paloma/m2d2_s2orc_unsplit-7dbcc1/validation",
    "paloma/m2d2_wikipedia_unsplit": "gs://marin-us-central2/tokenized/paloma/m2d2_wikipedia_unsplit-b33d23/validation",
    "paloma/manosphere_meta_sep": "gs://marin-us-central2/tokenized/paloma/manosphere_meta_sep-a07891/validation",
    "paloma/mc4": "gs://marin-us-central2/tokenized/paloma/mc4-ea36a2/validation",
    "paloma/ptb": "gs://marin-us-central2/tokenized/paloma/ptb-628036/validation",
    "paloma/redpajama": "gs://marin-us-central2/tokenized/paloma/redpajama-9d4ddd/validation",
    "paloma/twitterAAE_HELM_fixed": "gs://marin-us-central2/tokenized/paloma/twitterAAE_HELM_fixed-2e17c1/validation",
    "paloma/wikitext_103": "gs://marin-us-central2/tokenized/paloma/wikitext_103-1f5636/validation",
    # Uncheatable eval
    "uncheatable/wikipedia_english": (
        "gs://marin-us-central2/tokenized/uncheatable_eval/wikipedia_english-6330df/validation"
    ),
    "uncheatable/github_python": "gs://marin-us-central2/tokenized/uncheatable_eval/github_python-baab41/validation",
    "uncheatable/github_cpp": "gs://marin-us-central2/tokenized/uncheatable_eval/github_cpp-a9de07/validation",
    "uncheatable/bbc_news": "gs://marin-us-central2/tokenized/uncheatable_eval/bbc_news-4df59f/validation",
    "uncheatable/arxiv_physics": "gs://marin-us-central2/tokenized/uncheatable_eval/arxiv_physics-f4ad8c/validation",
    "uncheatable/arxiv_computer_science": (
        "gs://marin-us-central2/tokenized/uncheatable_eval/arxiv_computer_science-2b4f07/validation"
    ),
    "uncheatable/ao3_english": "gs://marin-us-central2/tokenized/uncheatable_eval/ao3_english-bb5666/validation",
}

OUTPUT_PATH = "gs://marin-us-central2/scratch/baseline_compare/tokenized_stats.json"
OUTPUT_PATH_LLM_CURATED = "gs://marin-us-central1/scratch/baseline_compare/tokenized_stats_llm_curated.json"
OUTPUT_PATH_EVALS = "gs://marin-us-central2/scratch/baseline_compare/tokenized_stats_evals.json"

LENGTH_BUCKETS = [0, 32, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 10**9]
REPEAT_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 10**9]


def _bucketize(v, edges):
    import bisect

    return max(0, min(len(edges) - 2, bisect.bisect_right(edges, v) - 1))


def _max_repeat_run(arr: np.ndarray) -> int:
    if arr.size == 0:
        return 0
    if arr.size == 1:
        return 1
    diffs = np.diff(arr)
    zero_mask = (diffs == 0).astype(np.int32)
    if not zero_mask.any():
        return 1
    best = 0
    cur = 0
    for z in zero_mask.tolist():
        cur = cur + 1 if z else 0
        if cur > best:
            best = cur
    return best + 1


def _process_doc(ids: np.ndarray, bos_id: int, eos_id: int, *, eot_id: int | None) -> dict:
    n = int(ids.shape[0])
    if n == 0:
        return None
    out = {
        "n": n,
        "first": int(ids[0]),
        "last": int(ids[-1]),
        "has_bos": bool(ids[0] == bos_id),
        "has_eos": bool(ids[-1] == eos_id),
        "n_bos": int((ids == bos_id).sum()),
        "n_eos": int((ids == eos_id).sum()),
        "max_repeat": _max_repeat_run(ids),
        "unique_frac": float(np.unique(ids).shape[0] / n),
        "first100_hash": hash(ids[:100].tobytes()) & 0xFFFFFFFFFFFFFFFF,
    }
    if eot_id is not None:
        out["n_eot"] = int((ids == eot_id).sum())
    return out


def _merge_token_counter(counters: list[Counter], top_n=100) -> dict:
    c: Counter = Counter()
    for ci in counters:
        c.update(ci)
    return dict(c.most_common(top_n))


def _scan_source(source: str, cache_dir: str, sample_docs: int, tokenizer) -> dict:
    """Pull a bounded sample of docs from a TreeCache and compute stats.

    We sample ``sample_docs`` random indices out of the cache and process each.
    That's cheap (TreeCache random access is O(1)) and bounded-memory.
    """
    from levanter.data.text.cache import load_lm_dataset_cache
    from levanter.data.text.formats import TextLmDatasetFormat

    _assert_no_cross_region(cache_dir)
    logger.info("[%s] opening cache", source)
    cache = load_lm_dataset_cache(cache_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    # get_batch_sync + slice is simplest; async_len via len(cache).
    total_docs = len(cache)
    logger.info("[%s] %d docs in cache", source, total_docs)

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.get_vocab().get("<|eot_id|>") if hasattr(tokenizer, "get_vocab") else None

    rng = np.random.default_rng(17 + hash(source) % 10**6)
    n = min(sample_docs, total_docs)
    indices = sorted(rng.choice(total_docs, size=n, replace=False).tolist())

    # Batch-fetch in chunks to avoid huge get_batch calls
    CHUNK = 1024
    length_hist = [0] * (len(LENGTH_BUCKETS) - 1)
    repeat_hist = [0] * (len(REPEAT_BUCKETS) - 1)
    token_counter: Counter = Counter()
    n_docs = 0
    n_with_bos = 0
    n_with_eos = 0
    total_tokens = 0
    total_bos = 0
    total_eos = 0
    total_eot = 0
    sum_unique_frac = 0.0
    first100_hashes: Counter = Counter()
    max_repeat_all = 0
    first_token_counter: Counter = Counter()

    for start in range(0, len(indices), CHUNK):
        batch_idx = indices[start : start + CHUNK]
        docs = cache.get_batch_sync(batch_idx)
        for d in docs:
            ids = np.asarray(d["input_ids"])
            res = _process_doc(ids, bos_id, eos_id, eot_id=eot_id)
            if res is None:
                continue
            n_docs += 1
            total_tokens += res["n"]
            n_with_bos += int(res["has_bos"])
            n_with_eos += int(res["has_eos"])
            total_bos += res["n_bos"]
            total_eos += res["n_eos"]
            total_eot += res.get("n_eot", 0)
            sum_unique_frac += res["unique_frac"]
            length_hist[_bucketize(res["n"], LENGTH_BUCKETS)] += 1
            repeat_hist[_bucketize(res["max_repeat"], REPEAT_BUCKETS)] += 1
            first100_hashes[res["first100_hash"]] += 1
            first_token_counter[res["first"]] += 1
            if res["max_repeat"] > max_repeat_all:
                max_repeat_all = res["max_repeat"]
            # token frequency: only count on first 512 tokens of each doc to
            # keep memory bounded; full-doc counters would blow up.
            for tok in ids[:512].tolist():
                token_counter[int(tok)] += 1
        if (start // CHUNK) % 16 == 0:
            logger.info("[%s] processed %d / %d", source, n_docs, n)

    # Estimate packing boundaries per 4096 tokens
    avg_doc_tokens = total_tokens / n_docs if n_docs else 0
    docs_per_4096 = 4096 / avg_doc_tokens if avg_doc_tokens > 0 else None

    # First-100-token hash collision stats
    unique_first100 = len(first100_hashes)
    most_common_first100 = first100_hashes.most_common(10)
    # Count of docs whose first-100 hash appears ≥2 times
    dup_doc_count = sum(cnt for h, cnt in first100_hashes.items() if cnt >= 2)

    # Decode top tokens for readability
    top_tokens_ids = token_counter.most_common(30)
    try:
        top_tokens = [
            {"id": int(tid), "count": int(cnt), "text": repr(tokenizer.decode([int(tid)], skip_special_tokens=False))}
            for tid, cnt in top_tokens_ids
        ]
    except Exception:
        top_tokens = [{"id": int(tid), "count": int(cnt)} for tid, cnt in top_tokens_ids]

    top_first_tokens_ids = first_token_counter.most_common(10)
    try:
        top_first_tokens = [
            {"id": int(tid), "count": int(cnt), "text": repr(tokenizer.decode([int(tid)], skip_special_tokens=False))}
            for tid, cnt in top_first_tokens_ids
        ]
    except Exception:
        top_first_tokens = [{"id": int(tid), "count": int(cnt)} for tid, cnt in top_first_tokens_ids]

    return {
        "total_docs": int(total_docs),
        "sample_docs": int(n_docs),
        "total_tokens_sampled": int(total_tokens),
        "avg_doc_tokens": float(avg_doc_tokens),
        "docs_per_4096_seq": docs_per_4096,
        "pct_docs_with_bos": 100 * n_with_bos / n_docs if n_docs else 0,
        "pct_docs_with_eos": 100 * n_with_eos / n_docs if n_docs else 0,
        "bos_per_1k_tokens": 1000 * total_bos / total_tokens if total_tokens else 0,
        "eos_per_1k_tokens": 1000 * total_eos / total_tokens if total_tokens else 0,
        "eot_per_1k_tokens": 1000 * total_eot / total_tokens if total_tokens else 0,
        "mean_unique_token_frac_per_doc": sum_unique_frac / n_docs if n_docs else 0,
        "length_hist_edges": LENGTH_BUCKETS,
        "length_hist": length_hist,
        "repeat_hist_edges": REPEAT_BUCKETS,
        "repeat_hist": repeat_hist,
        "max_repeat_observed": int(max_repeat_all),
        "first100_unique_hashes": int(unique_first100),
        "first100_duplicate_docs": int(dup_doc_count),
        "first100_dup_doc_fraction": float(dup_doc_count / n_docs) if n_docs else 0,
        "first100_top_collisions": [{"hash": int(h), "count": int(cnt)} for h, cnt in most_common_first100],
        "top_first_tokens": top_first_tokens,
        "top_tokens_first512": top_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-docs", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=5, help="parallel sources (one thread per source).")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Subset of sources to scan. Defaults to the central2 baseline sources only.",
    )
    parser.add_argument(
        "--eval-sources",
        action="store_true",
        help="Shortcut: scan all paloma/* + uncheatable/* eval sources.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Where to write the JSON. Defaults to central2; pass a central1 path when scanning llm_curated.",
    )
    args = parser.parse_args()

    from levanter.tokenizers import load_tokenizer

    tokenizer = load_tokenizer(TOKENIZER)

    # Default source set: central2 baseline collection only (excludes llm_curated on central1
    # and all paloma/uncheatable eval sources). Pass --sources or --eval-sources to override.
    if args.sources:
        selected = {s: SOURCES[s] for s in args.sources if s in SOURCES}
    elif args.eval_sources:
        selected = {s: p for s, p in SOURCES.items() if s.startswith(("paloma/", "uncheatable/"))}
    else:
        selected = {
            s: p for s, p in SOURCES.items() if s != "llm_curated" and not s.startswith(("paloma/", "uncheatable/"))
        }

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {ex.submit(_scan_source, s, p, args.sample_docs, tokenizer): s for s, p in selected.items()}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                results[src] = fut.result()
                logger.info("[%s] DONE: %s", src, {k: v for k, v in results[src].items() if not isinstance(v, list)})
            except Exception as e:
                logger.exception("[%s] FAILED: %s", src, e)
                results[src] = {"error": str(e)}

    if args.output_path:
        output_path = args.output_path
    elif set(selected) == {"llm_curated"}:
        output_path = OUTPUT_PATH_LLM_CURATED
    elif all(s.startswith(("paloma/", "uncheatable/")) for s in selected):
        output_path = OUTPUT_PATH_EVALS
    else:
        output_path = OUTPUT_PATH
    _assert_no_cross_region(output_path)
    with fsspec.open(output_path, "w") as f:
        f.write(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Wrote %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
