# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the production llm_pipeline_v1_1 EXTRACTION stage over a WebMainBench page shard on one TPU worker.

Same engine and pipeline as ``fill_missing_batches.py`` (vLLM on TPU, ``run_two_stage`` with
``GUARDS_V1_1``, the Qwen3-8B rephraser SFT checkpoint from the worker's regional bucket), with one
difference: the filter stage is answered with a synthetic KEEP so every page is extracted. WebMainBench
rows are scored against gold with no page-level filtering, which is how the d5 rows on the other models
were produced.

Input: a ``.jsonl.gz`` of ``{"track_id", "url", "html", "meta"}`` rows on GCS (built by
``launch_wmb_tpu.py``). Output: one ``.jsonl.gz`` per group of pages next to ``--output``, in the html-dir
runner's record shape ``{"text", "metadata": {"track_id", "url", "meta", "pipeline": {...}}}`` so the
README scoring harness reads it unchanged; a group whose file already exists is skipped, which makes a
preempted job resumable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from typing import Any

import fsspec

from experiments.baseline_collection.pipelines.offline_chat import InferenceResult, make_vllm_chat_fn
from experiments.baseline_collection.pipelines.pipeline_specs import get_pipeline, make_formatter
from experiments.baseline_collection.pipelines.token_budget import QwenTokenCodec
from experiments.baseline_collection.pipelines.two_stage import run_two_stage

logger = logging.getLogger(__name__)
FILTER_BYPASS_VERDICT = "KEEP (filter bypassed: WebMainBench pages are scored unfiltered)"


def make_filter_bypassing_chat_fn(llm: Any, filter_spec: str):
    """ChatFn that answers filter calls locally with KEEP and sends everything else to vLLM."""
    inner = make_vllm_chat_fn(llm)

    def is_filter(messages: list[dict]) -> bool:
        return any(m["role"] == "user" and filter_spec in m["content"] for m in messages)

    def chat_fn(messages_list, *, enable_thinking, temperature, max_tokens):
        out: list[InferenceResult | None] = [None] * len(messages_list)
        todo = [i for i, m in enumerate(messages_list) if not is_filter(m)]
        for i in range(len(messages_list)):
            if i not in todo:
                out[i] = InferenceResult(
                    output_text=FILTER_BYPASS_VERDICT,
                    raw_output=FILTER_BYPASS_VERDICT,
                    reasoning_content=None,
                    completion_tokens=1,
                    finish_reason="stop",
                )
        if todo:
            results = inner(
                [messages_list[i] for i in todo],
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for i, r in zip(todo, results, strict=True):
                out[i] = r
        return out

    return chat_fn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="gs:// path of the page shard (.jsonl.gz)")
    parser.add_argument("--output", required=True, help="gs:// directory for this shard's group outputs")
    parser.add_argument("--pipeline", default="llm_pipeline_v1_1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--group-size", type=int, default=50, help="pages per run_two_stage call / output file")
    parser.add_argument(
        "--real-filter",
        action="store_true",
        help="run the pipeline's filter stage instead of answering it KEEP (keep/drop decisions, not WMB extraction)",
    )
    args = parser.parse_args()

    from rigging.filesystem import marin_prefix

    pipeline = get_pipeline(args.pipeline)
    model_name = args.model or f"{marin_prefix()}/checkpoints/qwen3-8b-rephraser-sft-v4-193d7b/hf/step-1318"

    marin_pfx = os.environ.get("MARIN_PREFIX")
    cache_dir = os.path.join(marin_pfx, "compilation-cache") if marin_pfx else "/tmp/marin-jax-compilation-cache"
    os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "1")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", cache_dir)
    os.environ.setdefault("VLLM_XLA_CACHE_PATH", cache_dir)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    fs = fsspec.filesystem("gcs")
    with fs.open(args.input.replace("gs://", ""), "rb") as f:
        rows = [json.loads(line) for line in gzip.open(f, "rt")]
    logger.info("Loaded %d pages from %s", len(rows), args.input)
    groups = [rows[i : i + args.group_size] for i in range(0, len(rows), args.group_size)]
    out_prefix = args.output.replace("gs://", "").rstrip("/")
    pending = [g for g in range(len(groups)) if not fs.exists(f"{out_prefix}/group_{g:04d}.jsonl.gz")]
    logger.info("%d groups, %d pending", len(groups), len(pending))
    if not pending:
        return

    import jax
    from vllm import LLM

    devices = jax.devices()
    tp = args.tp or len([d for d in devices if d.platform == "tpu"]) or len(devices)
    t0 = time.monotonic()
    llm: Any = LLM(
        model=model_name, tensor_parallel_size=tp, max_model_len=args.max_model_len, enable_prefix_caching=True
    )
    codec = QwenTokenCodec(llm.get_tokenizer())
    formatter = make_formatter()
    chat_fn = make_vllm_chat_fn(llm) if args.real_filter else make_filter_bypassing_chat_fn(llm, pipeline.filter_spec)
    logger.info("Engine loaded in %.1fs (tp=%d, model=%s)", time.monotonic() - t0, tp, model_name)

    for g in pending:
        records = [{"id": r["track_id"], "url": r.get("url", ""), "html": r["html"], "metadata": {}} for r in groups[g]]
        t0 = time.monotonic()
        results, profile = run_two_stage(records, chat_fn, codec, formatter, pipeline)
        elapsed = time.monotonic() - t0
        out_rows = []
        for r, res in zip(groups[g], results, strict=True):
            out_rows.append(
                {
                    "text": res.text,
                    "metadata": {
                        "track_id": r["track_id"],
                        "url": r.get("url", ""),
                        "meta": r.get("meta"),
                        "pipeline": {
                            "id": pipeline.pipeline_id,
                            "decision": res.decision,
                            "error": res.error,
                            "n_chunks": res.num_chunks,
                            "completion_tokens": res.completion_tokens,
                        },
                    },
                }
            )
        path = f"{out_prefix}/group_{g:04d}.jsonl.gz"
        with fs.open(path, "wb") as f, gzip.open(f, "wt") as z:
            for row in out_rows:
                z.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "group %d: %d pages in %.0fs (%.1f s/page) kept=%d errored=%d chunks=%d -> gs://%s",
            g,
            len(records),
            elapsed,
            elapsed / len(records),
            profile["kept"],
            profile["errored"],
            profile["n_chunks"],
            path,
        )


if __name__ == "__main__":
    main()
