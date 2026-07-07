# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-core throughput benchmark for the stage-1 fastText useful-classifier on REAL WARC docs.

FastText scoring is CPU-bound and its cost is dominated by the actual document text (body_strip
regex + n-gram hashing in predict), so — unlike the data-independent ModernBERT forward pass — this
MUST be timed on real extracted HTML to be faithful. We score real ``raw_html`` documents straight
from the extracted parquet shards (in-region us-central2) through the EXACT production path reused
from ``cascade_survivor_filter.py``: ``to_fasttext_text`` (body_strip) -> ``model.f.predict``.

Reports the single-core rate (the per-core number, since fastText predict is single-threaded and the
real cascade fans out one process per shard), split into representation vs predict, plus a parallel
run that loads N worker processes each scoring its own shard to confirm near-linear core scaling.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import multiprocessing as mp
import time

import fasttext
import fsspec

# Reuse the EXACT production scoring path (faithfulness + no parallel reimplementation).
from experiments.baseline_collection.cascade_survivor_filter import (
    NOUSE_TMPL,
    USEFUL_TMPL,
    iter_html,
    predict_useful_prob,
    to_fasttext_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ft_bench")

DEFAULT_MODEL = "gs://marin-us-central2/classifiers/useful_fasttext/body_strip_scale_w80_strat_prep_mc500/model.bin"
DEFAULT_DATA_BASE = "gs://marin-us-central2/datasets/high_quality_3000_distill_random"
HTML_CAP = 1_000_000  # matches cascade_survivor_filter: cap pathological multi-MB markup


def fetch_model(gcs_path: str) -> str:
    """fasttext.load_model needs a local file — copy the model.bin down once."""
    local = "/tmp/ft_bench_model.bin"
    logger.info("downloading model %s ...", gcs_path)
    with fsspec.open(gcs_path, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    return local


def load_docs(data_base: str, shard: int, max_docs: int) -> list[str]:
    """Real extracted raw_html docs, interleaving useful + no_useful for a realistic length mix."""
    useful = list(_take(iter_html(USEFUL_TMPL.format(base=data_base, i=shard)), max_docs // 2))
    nouse = list(_take(iter_html(NOUSE_TMPL.format(base=data_base, i=shard)), max_docs - len(useful)))
    docs = useful + nouse
    logger.info(
        "loaded %d real docs from shard %05d (useful=%d, no_useful=%d)", len(docs), shard, len(useful), len(nouse)
    )
    return docs


def _take(it, n):
    for i, x in enumerate(it):
        if i >= n:
            return
        yield x


def bench_single_core(model, docs: list[str]) -> dict:
    """Time the two production stages separately on one core, then derive the end-to-end rate."""
    capped = [d[:HTML_CAP] for d in docs]
    n = len(capped)

    t0 = time.perf_counter()
    texts = [to_fasttext_text(d, "body_strip") for d in capped]
    rep_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for t in texts:
        predict_useful_prob(model, t)
    pred_s = time.perf_counter() - t0

    e2e_s = rep_s + pred_s
    mean_html_kb = sum(len(d) for d in capped) / n / 1024
    mean_ft_chars = sum(len(t) for t in texts) / n
    return {
        "n_docs": n,
        "mean_html_kb": round(mean_html_kb, 2),
        "mean_ft_text_chars": round(mean_ft_chars, 1),
        "representation_docs_per_sec": round(n / rep_s, 1),
        "predict_docs_per_sec": round(n / pred_s, 1),
        "end_to_end_docs_per_sec_per_core": round(n / e2e_s, 1),
        "ms_per_doc": round(e2e_s / n * 1e3, 3),
    }


def _worker_score_shard(args) -> tuple[int, float]:
    """One worker = one core: load model, read its OWN shard, time end-to-end compute (excl. GCS read)."""
    model_path, data_base, shard, max_docs = args
    model = fasttext.load_model(model_path)
    docs = load_docs(data_base, shard, max_docs)
    capped = [d[:HTML_CAP] for d in docs]
    t0 = time.perf_counter()
    for d in capped:
        predict_useful_prob(model, to_fasttext_text(d, "body_strip"))
    return len(capped), time.perf_counter() - t0


def bench_parallel(model_path: str, data_base: str, start_shard: int, workers: int, docs_per_worker: int) -> dict:
    """Fan out `workers` processes (one shard each), mirroring the real cascade Pool, to measure
    aggregate throughput and confirm per-core rate holds under parallelism."""
    tasks = [(model_path, data_base, start_shard + w, docs_per_worker) for w in range(workers)]
    ctx = mp.get_context("spawn")
    wall0 = time.perf_counter()
    with ctx.Pool(workers) as pool:
        results = pool.map(_worker_score_shard, tasks)
    wall = time.perf_counter() - wall0
    total_docs = sum(n for n, _ in results)
    per_worker_rates = [round(n / s, 1) for n, s in results]
    return {
        "workers": workers,
        "total_docs": total_docs,
        "wall_sec": round(wall, 2),
        "aggregate_docs_per_sec": round(total_docs / wall, 1),
        "per_core_docs_per_sec_under_load": round(sum(per_worker_rates) / len(per_worker_rates), 1),
        "per_worker_rates": per_worker_rates,
    }


def run(args):
    model_path = fetch_model(args.model)
    model = fasttext.load_model(model_path)

    docs = load_docs(args.data_base, args.shard, args.max_docs)
    # Warm up (first predicts pay one-time setup costs we don't want in the steady-state rate).
    for d in docs[: min(50, len(docs))]:
        predict_useful_prob(model, to_fasttext_text(d[:HTML_CAP], "body_strip"))

    single = bench_single_core(model, docs)
    logger.info("===== FASTTEXT_INFERENCE_BENCHMARK (single core) =====")
    for k, v in single.items():
        logger.info("  %-34s %s", k, v)

    summary = {"model": args.model, "data_base": args.data_base, "single_core": single}

    # Free the single-core working set (40k docs + texts) before fanning out workers that each
    # load their own model + shard — otherwise the parent's retained lists + workers OOM the container.
    del docs, model
    gc.collect()

    if args.workers > 1:
        logger.info("parallel run: %d workers, %d docs/worker (own shard each) ...", args.workers, args.docs_per_worker)
        par = bench_parallel(model_path, args.data_base, args.shard, args.workers, args.docs_per_worker)
        for k, v in par.items():
            logger.info("  %-34s %s", k, v)
        summary["parallel"] = par

    logger.info("FT_BENCH_JSON %s", json.dumps(summary))


def _parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="fastText model.bin (default: w80-stratified).")
    p.add_argument("--data-base", default=DEFAULT_DATA_BASE, help="extracted dataset base with data/ + data_no_useful/.")
    p.add_argument(
        "--shard", type=int, default=500, help="shard index for the single-core sample (>=300 = test-disjoint)."
    )
    p.add_argument("--max-docs", type=int, default=40000, help="real docs to score on one core.")
    p.add_argument("--workers", type=int, default=16, help="parallel processes (1 = skip the scaling run).")
    p.add_argument("--docs-per-worker", type=int, default=20000, help="docs each parallel worker scores from its shard.")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
