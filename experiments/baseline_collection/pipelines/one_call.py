# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-call (llm_simple_v1) staged-batch scheduler for offline vLLM.

Merged judge+extract: a voting head (chunk 0, and optionally chunk 1 as a second
voter) decides keep/drop with the full spec (think-ON) and, on keep, its payload
is the doc's first extraction segment. Continuation chunks are pure transcription
(cont spec, think-OFF) with the loop-guard ladder. Voter/head calls get loop
truncation only — never a retry, since a retry could flip the verdict.

Expressed as cross-document staged batches: one batch per voter round, then one
continuation batch. Mirrors the reference `extract_one`.
"""

from __future__ import annotations

import time

from experiments.baseline_collection.pipelines.chunking import Chunk, chunk_html
from experiments.baseline_collection.pipelines.loop_guard import truncate_loops
from experiments.baseline_collection.pipelines.offline_chat import ChatFn
from experiments.baseline_collection.pipelines.pipeline_specs import DROP_MARKER, OneCallPipeline, is_drop
from experiments.baseline_collection.pipelines.preprocessing import preprocess_html_for_extraction
from experiments.baseline_collection.pipelines.prompt import PromptFormatter
from experiments.baseline_collection.pipelines.scheduler_base import (
    DECISION_DROP,
    DECISION_ERROR,
    DECISION_KEEP,
    ChunkState,
    DocResult,
    apply_loop_guard,
    classify_failure,
)
from experiments.baseline_collection.pipelines.token_budget import QwenTokenCodec

__all__ = ["run_one_call"]


def run_one_call(
    records: list[dict],
    chat_fn: ChatFn,
    codec: QwenTokenCodec,
    formatter: PromptFormatter,
    pipeline: OneCallPipeline,
) -> tuple[list[DocResult], dict]:
    """Run one group of WARC records through the one-call pipeline."""
    b = pipeline.budget
    prof: dict = {
        "pipeline_id": pipeline.pipeline_id,
        "n_docs": len(records),
        "kept": 0,
        "dropped": 0,
        "errored": 0,
        "n_chunks": 0,
        "stage_seconds": {},
    }
    results: list[DocResult] = [DocResult(decision=DECISION_ERROR, error="unprocessed") for _ in records]

    # --- Chunk every doc up front. ---
    t0 = time.monotonic()
    doc_chunks: dict[int, list[Chunk]] = {}
    for i, rec in enumerate(records):
        pre = preprocess_html_for_extraction(rec.get("html", ""), mode="aggressive")
        chunks = chunk_html(pre, b.chunk_tokens, codec.count, headroom=b.chunk_headroom)
        doc_chunks[i] = chunks
        prof["n_chunks"] += len(chunks)
    prof["stage_seconds"]["chunk"] = round(time.monotonic() - t0, 3)

    # --- Voting head: up to `filter_chunks` rounds, short-circuit on first keep. ---
    kept_at: dict[int, int] = {}
    head_payload: dict[int, str] = {}
    decided: set[int] = set()
    t0 = time.monotonic()
    for v in range(pipeline.filter_chunks):
        cand = [i for i in range(len(records)) if i not in decided and i not in kept_at and v < len(doc_chunks[i])]
        if not cand:
            continue
        msgs = [formatter.format(html=doc_chunks[i][v].text, spec=pipeline.main_spec) for i in cand]
        outs = chat_fn(msgs, enable_thinking=True, temperature=b.temperature, max_tokens=b.extract_max_tokens)
        for i, r in zip(cand, outs, strict=True):
            err = classify_failure(r)
            if err is not None:
                results[i] = DocResult(decision=DECISION_ERROR, error=err, completion_tokens=r.completion_tokens)
                decided.add(i)
                prof["errored"] += 1
                continue
            # Verdict-safe truncation on the head (never a retry).
            r.output_text = truncate_loops(r.output_text, doc_chunks[i][v].text, b.loop_threshold)
            if not is_drop(r.output_text):
                kept_at[i] = v
                head_payload[i] = r.output_text
                results[i] = DocResult(
                    decision=DECISION_KEEP, filter_reasoning=r.reasoning_content, completion_tokens=r.completion_tokens
                )
            else:
                # Drop the doc only after its last available voter has dropped.
                n_voters = min(pipeline.filter_chunks, len(doc_chunks[i]))
                if v == n_voters - 1:
                    results[i] = DocResult(
                        decision=DECISION_DROP,
                        text=DROP_MARKER,
                        drop_marker=DROP_MARKER,
                        filter_reasoning=r.reasoning_content,
                        completion_tokens=r.completion_tokens,
                    )
                    decided.add(i)
                    prof["dropped"] += 1
    prof["stage_seconds"]["voting_head"] = round(time.monotonic() - t0, 3)

    kept_idx = sorted(kept_at)
    if not kept_idx:
        return results, prof

    # --- Continuation chunks (after the keeping voter): think-OFF, loop-guarded. ---
    t0 = time.monotonic()
    states: list[ChunkState] = []
    for i in kept_idx:
        for c_idx in range(kept_at[i] + 1, len(doc_chunks[i])):
            states.append(
                ChunkState(
                    doc_idx=i,
                    chunk_idx=c_idx,
                    chunk_text=doc_chunks[i][c_idx].text,
                    messages=formatter.format(html=doc_chunks[i][c_idx].text, spec=pipeline.cont_spec),
                )
            )
    if states:
        outs = chat_fn(
            [st.messages for st in states],
            enable_thinking=False,
            temperature=b.temperature,
            max_tokens=b.extract_max_tokens,
        )
        for st, r in zip(states, outs, strict=True):
            st.result = r
        prof["loop_guard"] = apply_loop_guard(states, chat_fn, b)
    prof["stage_seconds"]["continuation_generate"] = round(time.monotonic() - t0, 3)

    # --- Join: keeping-chunk payload + subsequent non-drop continuation payloads. ---
    t0 = time.monotonic()
    by_doc: dict[int, list[ChunkState]] = {}
    for st in states:
        by_doc.setdefault(st.doc_idx, []).append(st)
    for i in kept_idx:
        cont = sorted(by_doc.get(i, []), key=lambda s: s.chunk_idx)
        chunk_err = next((s.result.error for s in cont if s.result.error), None)
        res = results[i]
        res.num_chunks = len(doc_chunks[i])
        res.completion_tokens += sum(s.result.completion_tokens for s in cont)
        if chunk_err is not None:
            res.decision = DECISION_ERROR
            res.error = chunk_err
            res.text = ""
            prof["errored"] += 1
            continue
        parts = [head_payload[i]] if not is_drop(head_payload[i]) else []
        parts += [s.result.output_text for s in cont if not is_drop(s.result.output_text)]
        res.text = "\n\n".join(parts)
        prof["kept"] += 1
    prof["stage_seconds"]["join"] = round(time.monotonic() - t0, 3)

    return results, prof
