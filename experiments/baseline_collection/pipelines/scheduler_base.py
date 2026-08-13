# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared pieces for the staged-batch schedulers (two_stage, one_call).

Both pipelines partition documents across stages, extract chunks in cross-document
batches, and apply the same loop-guard ladder. The per-document decision logic
differs; everything else lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from experiments.baseline_collection.pipelines.loop_guard import truncate_loops, worst_line_amplification
from experiments.baseline_collection.pipelines.offline_chat import (
    ChatFn,
    InferenceResult,
    is_empty,
    is_premature_eos_filter,
    is_reasoning_runaway,
)
from experiments.baseline_collection.pipelines.pipeline_specs import PipelineBudget

# Decision labels for the durable record.
DECISION_KEEP = "keep"
DECISION_DROP = "drop"
DECISION_ERROR = "error"

FilterViewFn = Callable[[str, int], str]


@dataclass
class DocResult:
    """Index-aligned per-document outcome (caller adds WARC join fields)."""

    decision: str
    text: str = ""
    drop_marker: str | None = None
    filter_reasoning: str | None = None
    num_chunks: int = 0
    error: str | None = None
    completion_tokens: int = 0


@dataclass
class ChunkState:
    doc_idx: int
    chunk_idx: int
    chunk_text: str
    messages: list[dict]
    result: InferenceResult | None = None
    amp: int = 0


def make_resiliparse_view(main_content: bool) -> FilterViewFn:
    """Build a whole-document plain-text filter-view function.

    ``main_content=True`` prunes to the main content (llm_pipeline_v1); False
    renders the FULL page including chrome (high_quality_v2 — its extreme filter
    is rendering-aware and expects chrome present). Both keep the view-collapse
    guard: if the primary render is under ``min_chars``, re-render full-page
    (harmless when the primary was already full-page).
    """

    def _view(html: str, min_chars: int) -> str:
        from resiliparse.extract.html2text import extract_plain_text

        view = extract_plain_text(html, main_content=main_content, alt_texts=False)
        if len((view or "").strip()) < min_chars:
            view = extract_plain_text(html, main_content=False, alt_texts=False)
        return view or ""

    return _view


# Default filter view (main_content-pruned) — used by llm_pipeline_v1 and tests.
resiliparse_view = make_resiliparse_view(main_content=True)


def classify_failure(r: InferenceResult) -> str | None:
    """Return an error string if this result is a non-drop failure, else None.

    Offline the reference's server-oriented retries are no-ops at t=0, so we
    classify: reasoning-runaway / premature-EOS / empty are failures that must
    never be scored as a `[NO_USEFUL_CONTENT]` drop.
    """
    if r.error is not None:
        return r.error
    if is_reasoning_runaway(r):
        return "reasoning runaway (finish_reason=length, empty output)"
    if is_premature_eos_filter(r):
        return "premature EOS (reasoned, no verdict, empty output)"
    if is_empty(r):
        return "empty completion (0 tokens)"
    return None


def apply_loop_guard(states: list[ChunkState], chat_fn: ChatFn, budget: PipelineBudget) -> dict:
    """Batched loop-guard ladder over extraction chunk outputs (mutates in place).

    rung-1 think-on t=0 -> rung-2 think-off t=0.3, keeping whichever attempt
    amplifies least; survivors are truncated at loop onset (verdict-safe).
    Returns {amplified, rung1, rung2, truncated} counts for profiling.
    """
    thr = budget.loop_threshold
    counts = {"amplified": 0, "rung1": 0, "rung2": 0, "truncated": 0}
    for st in states:
        st.amp = worst_line_amplification(st.result.output_text, st.chunk_text)
    counts["amplified"] = sum(1 for st in states if st.amp >= thr)

    for rung, (think, temp) in enumerate(((True, budget.temperature), (False, budget.loop_retry_temperature))):
        todo = [st for st in states if st.amp >= thr]
        if not todo:
            break
        outs = chat_fn(
            [st.messages for st in todo], enable_thinking=think, temperature=temp, max_tokens=budget.extract_max_tokens
        )
        improved = 0
        for st, r in zip(todo, outs, strict=True):
            if r.error is not None:
                continue
            ramp = worst_line_amplification(r.output_text, st.chunk_text)
            if ramp < st.amp:
                st.result, st.amp = r, ramp
                improved += 1
        counts["rung1" if rung == 0 else "rung2"] = improved

    for st in states:
        if st.amp >= thr:
            st.result.output_text = truncate_loops(st.result.output_text, st.chunk_text, thr)
            counts["truncated"] += 1
    return counts
