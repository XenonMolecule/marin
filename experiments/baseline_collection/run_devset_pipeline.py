# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Devset parity harness — run a pipeline over the 1934-doc gold devset and score.

This is the BENCHMARK entrypoint (distinct from run_extract_standalone.py, which
processes a WARC pool). It feeds the gold devset docs through the SAME offline
schedulers the pool uses, emits per-doc run records, and (for a single-shard run)
scores them in-marin against the vendored reference metrics — proving the offline
port reproduces the frozen cert numbers (two-stage ~93.7/90.8, topQ >=133;
one-call ~91.3/87.3).

Parallelism across TPUs: launch N jobs with --num-shards N and --shard-index
0..N-1, all pointing at the SAME --out-gcs run dir. Each worker processes its
disjoint slice and writes records into the shared dir (skip-existing makes a
preempted shard resumable). Because scoring needs every doc together, sharded
runs do NOT score inline — run `score_devset` on the shared run dir once all
shards finish.

The devset is staged to GCS as a tarball by stage_devset.py; this downloads and
extracts it (or reads a local --devset-dir), so a TPU job needs no sibling repo.

Usage (one TPU, full run + inline score)::

    python -m experiments.baseline_collection.run_devset_pipeline \
        --pipeline llm_pipeline_v1 \
        --devset-gcs gs://marin-us-central1/devset/marin_devset_1934.tar.gz

Usage (shard i of N, shared run dir, score separately after)::

    python -m experiments.baseline_collection.run_devset_pipeline \
        --pipeline llm_pipeline_v1 \
        --devset-gcs gs://marin-us-central1/devset/marin_devset_1934.tar.gz \
        --out-gcs gs://marin-us-central1/devset/runs/llm_pipeline_v1 \
        --num-shards 8 --shard-index 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time

from experiments.baseline_collection.devset.dataset import load_devset
from experiments.baseline_collection.devset.staging import fetch_devset_dir, gcs_done_record_ids, upload_record

logger = logging.getLogger(__name__)


def _run_pipeline(records, chat_fn, codec, formatter, pipeline):
    """Dispatch a group of records to the matching scheduler."""
    from experiments.baseline_collection.pipelines.one_call import run_one_call
    from experiments.baseline_collection.pipelines.pipeline_specs import OneCallPipeline
    from experiments.baseline_collection.pipelines.two_stage import run_two_stage

    if isinstance(pipeline, OneCallPipeline):
        return run_one_call(records, chat_fn, codec, formatter, pipeline)
    return run_two_stage(records, chat_fn, codec, formatter, pipeline)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="Pipeline id (llm_pipeline_v1 or llm_simple_v1)")
    parser.add_argument("--devset-gcs", default=None, help="GCS path to the staged devset tarball")
    parser.add_argument("--devset-dir", default=None, help="Local devset dir (alternative to --devset-gcs)")
    parser.add_argument("--subset", default=None, help="devset_subsets record-id file to restrict to (local path)")
    parser.add_argument("--out-gcs", default=None, help="Shared GCS run dir (required for multi-shard runs)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards (workers) over the devset")
    parser.add_argument("--shard-index", type=int, default=0, help="This worker's shard index [0, num-shards)")
    parser.add_argument("--model", default=None, help="Model path (default: v4 rephraser from local region)")
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size (auto-detect from JAX)")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--group-size", type=int, default=250, help="Docs per scheduler group")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.num_shards > 1 and not args.out_gcs:
        parser.error("multi-shard runs require --out-gcs (the shared run dir)")

    from experiments.baseline_collection.pipelines.pipeline_specs import get_pipeline, make_formatter

    pipeline = get_pipeline(args.pipeline)
    logger.info("Pipeline: %s (frozen cert %s)", pipeline.pipeline_id, pipeline.source_cert)

    work_dir = tempfile.mkdtemp(prefix="devset_")
    if args.devset_dir:
        devset_dir = args.devset_dir
    elif args.devset_gcs:
        devset_dir = fetch_devset_dir(args.devset_gcs, work_dir)
    else:
        raise ValueError("Provide --devset-gcs or --devset-dir")

    docs = load_devset(devset_dir)
    if args.subset:
        wanted = set(open(args.subset, encoding="utf-8").read().split())
        docs = [d for d in docs if d.record_id in wanted]
    # Deterministic static shard: disjoint slices across workers, order-stable.
    if args.num_shards > 1:
        docs = docs[args.shard_index :: args.num_shards]
    logger.info("Shard %d/%d: %d docs", args.shard_index, args.num_shards, len(docs))

    # Skip-existing against the shared GCS run dir (resume a preempted shard).
    done = gcs_done_record_ids(args.out_gcs) if args.out_gcs else set()
    todo = [d for d in docs if d.record_id not in done]
    if len(todo) < len(docs):
        logger.info("Skipping %d docs already in the shared run dir", len(docs) - len(todo))

    run_dir = os.path.join(work_dir, "run")
    records_dir = os.path.join(run_dir, "records")
    os.makedirs(records_dir, exist_ok=True)
    manifest = {
        "model": args.model or "(region-default v4 rephraser)",
        "pipeline_id": pipeline.pipeline_id,
        "source_cert": pipeline.source_cert,
        "group_size": args.group_size,
        "num_shards": args.num_shards,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    if not todo:
        logger.info("Nothing to do for this shard (all records already present).")
        _maybe_score(args, run_dir, docs)
        return

    # --- Engine setup (mirrors run_extract_standalone.main) ---
    from rigging.filesystem import marin_prefix

    model_name = args.model or f"{marin_prefix()}/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"
    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import jax

    devices = jax.devices()
    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    logger.info("Model: %s  tp=%d", model_name, tp)

    from vllm import LLM

    from experiments.baseline_collection.pipelines.offline_chat import make_vllm_chat_fn
    from experiments.baseline_collection.pipelines.token_budget import QwenTokenCodec

    t0 = time.monotonic()
    llm = LLM(model=model_name, tensor_parallel_size=tp, max_model_len=args.max_model_len, enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    logger.info("Engine loaded in %.1fs", time.monotonic() - t0)

    codec = QwenTokenCodec(tokenizer)
    formatter = make_formatter()
    chat_fn = make_vllm_chat_fn(llm)

    # --- Run the port over this shard in groups; write + upload one record per doc. ---
    t0 = time.monotonic()
    for start in range(0, len(todo), args.group_size):
        group = todo[start : start + args.group_size]
        recs = [{"html": d.html(), "record_id": d.record_id, "url": d.url} for d in group]
        results, _prof = _run_pipeline(recs, chat_fn, codec, formatter, pipeline)
        for d, res in zip(group, results, strict=True):
            out = {"record_id": d.record_id, "text": res.text, "error": res.error, "n_chunks": res.num_chunks}
            local = os.path.join(records_dir, f"{d.record_id}.json")
            with open(local, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False)
            if args.out_gcs:
                upload_record(args.out_gcs, d.record_id, local)
        logger.info(
            "Processed %d/%d docs (%.0fs)", min(start + args.group_size, len(todo)), len(todo), time.monotonic() - t0
        )

    if args.out_gcs:
        _upload_manifest(args.out_gcs, run_dir)
    _maybe_score(args, run_dir, docs)


def _maybe_score(args, run_dir: str, docs) -> None:
    """Score inline only for a single-shard full run (a shard can't see all docs).

    The scorer (metrics->token_f1->rapidfuzz) is imported lazily so sharded TPU
    workers, which never score, don't need the scoring stack installed.
    """
    if args.num_shards > 1:
        logger.info(
            "Shard %d/%d done. Score after ALL shards finish:\n"
            "  python -m experiments.baseline_collection.score_devset %s --devset-gcs %s",
            args.shard_index,
            args.num_shards,
            args.out_gcs or "<out-gcs>",
            args.devset_gcs or args.devset_dir,
        )
        return
    from experiments.baseline_collection.devset.metrics import run_summary
    from experiments.baseline_collection.devset.runs import load_run

    _print_headline(run_summary(load_run(run_dir), docs), args.pipeline)


def _upload_manifest(out_gcs: str, run_dir: str) -> None:
    import fsspec

    fs = fsspec.filesystem("gcs")
    dst = f"{out_gcs.rstrip('/')}/manifest.json".replace("gs://", "")
    with open(os.path.join(run_dir, "manifest.json"), "rb") as src, fs.open(dst, "wb") as out:
        out.write(src.read())


def _print_headline(summary: dict, pipeline_id: str) -> None:
    st = summary["keep_drop_strong"]["overall"]
    all_ = summary["keep_drop_all"]["overall"]
    tq = summary["top_quality"]
    g = summary["gold"]
    d = summary["decisions"]
    logger.info("=" * 60)
    logger.info("DEVSET PARITY — %s", pipeline_id)
    logger.info(
        "  decisions: kept=%d dropped=%d context=%d error=%d",
        d.get("keep", 0),
        d.get("drop", 0),
        d.get("context", 0),
        d.get("error", 0),
    )
    logger.info(
        "  strong: acc=%.1f  macro=%.1f   (all: acc=%.1f macro=%.1f)",
        100 * st["accuracy"],
        100 * st["macro_f1"],
        100 * all_["accuracy"],
        100 * all_["macro_f1"],
    )
    logger.info("  topQ: %d/%d kept", tq["kept"], tq["kept"] + tq["dropped"])
    logger.info(
        "  gold: kept-lev=%.3f  overall-lev=%.3f  (n_gold=%d)",
        g["kept_lev_sim"]["mean"],
        g["overall_lev_sim"]["mean"],
        g["n_gold"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
