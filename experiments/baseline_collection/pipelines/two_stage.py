# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-stage (llm_pipeline_v1) staged-batch scheduler for offline vLLM.

Re-expresses the reference per-document two-stage flow (filter -> chunked
extraction) as cross-document staged batches, where each stage is one
uniform-sampling-params `chat_fn` call and the verdict gates become partitions
between stages. At temperature 0 this reproduces the certified outputs up to the
engine's own batching nondeterminism (the loop-guard t=0.3 rerun is the only
nonzero-temp call).

The scheduler is engine-agnostic: it depends only on an injected `ChatFn`, so it
is unit-testable with a fake. WARC join-field assembly (warc_record_id,
snapshot, ...) is left to the caller; this returns index-aligned per-doc results
plus a timing/counts profile.
"""

from __future__ import annotations

import time

from experiments.baseline_collection.pipelines.bloat_filter import clean_output, is_bloat_chunk, strip_data_lines
from experiments.baseline_collection.pipelines.chunking import Chunk, chunk_html
from experiments.baseline_collection.pipelines.code_guard import code_underextracted
from experiments.baseline_collection.pipelines.coverage_guard import better_attempt, looks_early_stopped
from experiments.baseline_collection.pipelines.loop_guard import truncate_loops
from experiments.baseline_collection.pipelines.offline_chat import ChatFn
from experiments.baseline_collection.pipelines.pipeline_specs import (
    DROP_MARKER,
    ExtractionGuardConfig,
    PipelineBudget,
    TwoStagePipeline,
    is_drop,
)
from experiments.baseline_collection.pipelines.preprocessing import preprocess_html_for_extraction
from experiments.baseline_collection.pipelines.prompt import PromptFormatter
from experiments.baseline_collection.pipelines.scheduler_base import (
    DECISION_DROP,
    DECISION_ERROR,
    DECISION_KEEP,
    ChunkState,
    DocResult,
    FilterViewFn,
    apply_loop_guard,
    classify_failure,
    make_resiliparse_view,
)
from experiments.baseline_collection.pipelines.summarization_guard import better_extraction, looks_summarized
from experiments.baseline_collection.pipelines.tail_guard import looks_truncated
from experiments.baseline_collection.pipelines.token_budget import QwenTokenCodec, cap_view
from experiments.baseline_collection.pipelines.view_choice import scripts_carry_content

# Re-exported so callers/tests can import decision labels from this module.
__all__ = ["DECISION_DROP", "DECISION_ERROR", "DECISION_KEEP", "DocResult", "run_two_stage"]


# --- v1.1 extraction guards (all no-ops unless the pipeline's guard config
# enables them; llm_pipeline_v1 never calls these). ---


def _extraction_view(html: str, guards: ExtractionGuardConfig) -> str:
    """Preprocessing view for extraction: aggressive by default, switched to the
    safe (script-preserving) view only when the page's content lives in scripts."""
    aggressive = preprocess_html_for_extraction(html, mode="aggressive")
    if guards.view_choice:
        safe = preprocess_html_for_extraction(html, mode="safe")
        if scripts_carry_content(aggressive, safe):
            return safe
    return aggressive


def _apply_reroll_guards(
    states: list[ChunkState], chat_fn: ChatFn, budget: PipelineBudget, guards: ExtractionGuardConfig
) -> dict:
    """v1.1 per-chunk defense: coverage/code/tail guards drive a single t=0.3
    reroll (kept only if it's genuinely better), then loop-guard truncation.

    Replaces the v1 loop-guard rung ladder. Batches the reroll across all flagged
    chunks so the offline engine still runs one call, not one-per-chunk.
    """
    counts = {"flagged": 0, "rerolled": 0, "truncated": 0}
    todo = [
        st
        for st in states
        if st.result is not None
        and st.result.error is None
        and (
            looks_early_stopped(st.chunk_text, st.result.output_text)
            or code_underextracted(st.chunk_text, st.result.output_text)
            or looks_truncated(st.result.output_text)
        )
    ]
    counts["flagged"] = len(todo)
    if todo:
        outs = chat_fn(
            [st.messages for st in todo],
            enable_thinking=False,
            temperature=guards.reroll_temperature,
            max_tokens=budget.extract_max_tokens,
        )
        for st, r in zip(todo, outs, strict=True):
            if r.error is None and better_attempt(st.result.output_text, r.output_text, st.chunk_text):
                st.result = r
                counts["rerolled"] += 1
    for st in states:
        if st.result is None or st.result.error is not None:
            continue
        truncated = truncate_loops(st.result.output_text, st.chunk_text)
        if truncated != st.result.output_text:
            st.result.output_text = truncated
            counts["truncated"] += 1
    return counts


def _bloat_prefilter(states: list[ChunkState]) -> dict:
    """Per-chunk over-extraction defense (index walls dropped; script-data lines
    stripped). ``clean_output`` runs later at merge for the straddle-boundary
    filters."""
    counts = {"bloat_dropped": 0, "data_stripped": 0}
    for st in states:
        if st.result is None or st.result.error is not None:
            continue
        out = st.result.output_text
        if is_bloat_chunk(out):
            st.result.output_text = ""  # empty → excluded from the join
            counts["bloat_dropped"] += 1
            continue
        stripped = strip_data_lines(out)
        if stripped != out:
            st.result.output_text = stripped
            counts["data_stripped"] += 1
    return counts


def _summarization_retry(
    results: list[DocResult],
    kept_idx: list[int],
    doc_view: dict[int, str],
    chat_fn: ChatFn,
    codec: QwenTokenCodec,
    formatter: PromptFormatter,
    pipeline: TwoStagePipeline,
) -> int:
    """Re-extract summarized docs at half the chunk size; keep the retry only when
    ``better_extraction`` accepts it (≥2.5x longer, not more inventive)."""
    b = pipeline.budget
    g = pipeline.guards
    cand = [
        i
        for i in kept_idx
        if results[i].decision == DECISION_KEEP
        and results[i].text.strip()
        and looks_summarized(results[i].text, doc_view[i])
    ]
    if not cand:
        return 0
    half = max(1, b.chunk_tokens // g.half_chunk_divisor)
    states: list[ChunkState] = []
    for i in cand:
        for c_idx, ch in enumerate(chunk_html(doc_view[i], half, codec.count, headroom=b.chunk_headroom)):
            spec = pipeline.head_spec if c_idx == 0 else pipeline.cont_spec
            states.append(
                ChunkState(
                    doc_idx=i, chunk_idx=c_idx, chunk_text=ch.text, messages=formatter.format(html=ch.text, spec=spec)
                )
            )
    outs = chat_fn(
        [st.messages for st in states], enable_thinking=False, temperature=b.temperature, max_tokens=b.extract_max_tokens
    )
    for st, r in zip(states, outs, strict=True):
        st.result = r
        st.result.output_text = truncate_loops(r.output_text, st.chunk_text)
    by_doc: dict[int, list[ChunkState]] = {}
    for st in states:
        by_doc.setdefault(st.doc_idx, []).append(st)
    rescued = 0
    for i in cand:
        ds = sorted(by_doc.get(i, []), key=lambda s: s.chunk_idx)
        if any(s.result.error for s in ds):
            continue
        retry = "\n\n".join(s.result.output_text for s in ds if not is_drop(s.result.output_text))
        if g.merge_clean:
            retry = clean_output(retry)
        if retry.strip() and better_extraction(results[i].text, retry, doc_view[i]):
            results[i].text = retry
            results[i].completion_tokens += sum(s.result.completion_tokens for s in ds)
            rescued += 1
    return rescued


def run_two_stage(
    records: list[dict],
    chat_fn: ChatFn,
    codec: QwenTokenCodec,
    formatter: PromptFormatter,
    pipeline: TwoStagePipeline,
    *,
    filter_view_fn: FilterViewFn | None = None,
) -> tuple[list[DocResult], dict]:
    """Run one group of WARC records through the two-stage pipeline.

    Returns (results index-aligned to `records`, profile dict). ``filter_view_fn``
    defaults to the pipeline's configured view (main_content-pruned for
    llm_pipeline_v1, full-page for high_quality_v2); pass one only to inject a
    fake in tests.
    """
    b = pipeline.budget
    g = pipeline.guards
    if filter_view_fn is None:
        filter_view_fn = make_resiliparse_view(pipeline.filter_view_main_content)
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

    # --- Stage 1: filter (think-ON) over the whole-document text view. ---
    t0 = time.monotonic()
    views = [
        cap_view(filter_view_fn(r.get("html", ""), b.filter_view_min_chars), codec, b.filter_view_tokens)
        for r in records
    ]
    prof["stage_seconds"]["filter_view"] = round(time.monotonic() - t0, 3)

    t0 = time.monotonic()
    filter_msgs = [formatter.format(html=v, spec=pipeline.filter_spec) for v in views]
    filter_out = chat_fn(filter_msgs, enable_thinking=True, temperature=b.temperature, max_tokens=b.filter_max_tokens)
    prof["stage_seconds"]["filter_generate"] = round(time.monotonic() - t0, 3)

    kept_idx: list[int] = []
    for i, r in enumerate(filter_out):
        err = classify_failure(r)
        if err is not None:
            results[i] = DocResult(decision=DECISION_ERROR, error=err, completion_tokens=r.completion_tokens)
            prof["errored"] += 1
            continue
        if is_drop(r.output_text):
            results[i] = DocResult(
                decision=DECISION_DROP,
                text=DROP_MARKER,
                drop_marker=DROP_MARKER,
                filter_reasoning=r.reasoning_content,
                completion_tokens=r.completion_tokens,
            )
            prof["dropped"] += 1
            continue
        results[i] = DocResult(
            decision=DECISION_KEEP, filter_reasoning=r.reasoning_content, completion_tokens=r.completion_tokens
        )
        kept_idx.append(i)

    if not kept_idx:
        return results, prof

    # --- Stage 2: chunk kept docs, flatten, extract (think-OFF) in one batch. ---
    t0 = time.monotonic()
    doc_chunks: dict[int, list[Chunk]] = {}
    doc_view: dict[int, str] = {}
    states: list[ChunkState] = []
    for i in kept_idx:
        pre = _extraction_view(records[i].get("html", ""), g)
        doc_view[i] = pre
        chunks = chunk_html(pre, b.chunk_tokens, codec.count, headroom=b.chunk_headroom)
        doc_chunks[i] = chunks
        prof["n_chunks"] += len(chunks)
        for c_idx, ch in enumerate(chunks):
            spec = pipeline.head_spec if c_idx == 0 else pipeline.cont_spec
            states.append(
                ChunkState(
                    doc_idx=i,
                    chunk_idx=c_idx,
                    chunk_text=ch.text,
                    messages=formatter.format(html=ch.text, spec=spec),
                )
            )
    prof["stage_seconds"]["chunk"] = round(time.monotonic() - t0, 3)

    t0 = time.monotonic()
    extract_out = chat_fn(
        [st.messages for st in states], enable_thinking=False, temperature=b.temperature, max_tokens=b.extract_max_tokens
    )
    for st, r in zip(states, extract_out, strict=True):
        st.result = r
    prof["stage_seconds"]["extract_generate"] = round(time.monotonic() - t0, 3)

    # --- Degeneration defense over chunk outputs. ---
    t0 = time.monotonic()
    if g.reroll_guards:
        prof["loop_guard"] = _apply_reroll_guards(states, chat_fn, b, g)
    else:
        prof["loop_guard"] = apply_loop_guard(states, chat_fn, b)
    if g.bloat_prefilter:
        prof["bloat"] = _bloat_prefilter(states)
    prof["stage_seconds"]["loop_guard"] = round(time.monotonic() - t0, 3)

    # --- Join per doc: non-drop chunk outputs in chunk order. ---
    t0 = time.monotonic()
    by_doc: dict[int, list[ChunkState]] = {}
    for st in states:
        by_doc.setdefault(st.doc_idx, []).append(st)
    for i in kept_idx:
        doc_states = sorted(by_doc.get(i, []), key=lambda s: s.chunk_idx)
        chunk_err = next((s.result.error for s in doc_states if s.result.error), None)
        parts = [s.result.output_text for s in doc_states if not is_drop(s.result.output_text)]
        res = results[i]
        res.num_chunks = len(doc_chunks[i])
        res.completion_tokens += sum(s.result.completion_tokens for s in doc_states)
        if chunk_err is not None:
            res.decision = DECISION_ERROR
            res.error = chunk_err
            res.text = ""
            prof["errored"] += 1
        else:
            joined = "\n\n".join(parts)
            if g.merge_clean:
                joined = clean_output(joined)
            res.text = joined
            if not res.text.strip():
                # Filter said keep but every chunk transcribed to nothing:
                # reference scoring treats empty as a drop; persisting an empty
                # KEEP row would pollute training data. Classify as ERROR so it
                # is auditable/re-runnable rather than silently kept or dropped.
                res.decision = DECISION_ERROR
                res.error = "kept-but-empty: all chunk outputs empty/sentinel"
                prof["errored"] += 1
            else:
                prof["kept"] += 1
    prof["stage_seconds"]["join"] = round(time.monotonic() - t0, 3)

    if g.summarization_retry:
        t0 = time.monotonic()
        prof["summarization_rescued"] = _summarization_retry(
            results, kept_idx, doc_view, chat_fn, codec, formatter, pipeline
        )
        prof["stage_seconds"]["summarization_retry"] = round(time.monotonic() - t0, 3)

    return results, prof
