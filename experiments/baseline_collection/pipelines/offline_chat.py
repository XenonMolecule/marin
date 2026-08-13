# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline vLLM chat adapter for the extraction pipelines.

The reference pipelines call an OpenAI-compatible server, which splits
`reasoning_content` from `content` and reports `finish_reason`. marin runs the
model **offline** (`vllm.LLM`), which returns raw generated text (including any
`<think>...</think>` block) plus a `finish_reason`. This module reconstructs the
same `(output_text, reasoning_content, finish_reason, completion_tokens)`
contract from offline `RequestOutput`s, and exposes a batched `chat_fn` that the
schedulers call once per stage.

Determinism note: at temperature 0 the offline engine is deterministic, so the
reference's server-oriented retry loop (rate-limit / network / transient-empty)
collapses to nothing useful — a t=0 retry reproduces the same output. We
therefore CLASSIFY outcomes instead of retrying: a call is `ok`, `context`
(context-window overflow), or a non-drop failure (`reasoning_runaway` /
`empty` / `premature_eos`) that must never be scored as a `[NO_USEFUL_CONTENT]`
decision. Only the loop-guard's think-on / t=0.3 reruns re-issue a call.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# --- Output marker parsing (ported verbatim from small-rephraser worker.py) ---

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_UNCLOSED_RE = re.compile(r"^<think>.*?(?=\[\[ ## text ## \]\])", re.DOTALL)
OUTPUT_MARKER_RE = re.compile(
    r"\[\[ ## text ## \]\]\s*(.*?)\s*\[\[ ## completed ## \]\]",
    re.DOTALL,
)
OUTPUT_MARKER_OPEN_RE = re.compile(r"^\s*\[\[ ## text ## \]\]\s*", re.DOTALL)
OUTPUT_MARKER_CLOSE_RE = re.compile(r"\s*\[\[ ## completed ## \]\]\s*$", re.DOTALL)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def parse_output_markers(raw: str) -> str:
    """Extract text between [[ ## text ## ]] and [[ ## completed ## ]].

    Strips Qwen3 <think>...</think> reasoning (closed or unclosed) first, then
    pulls the payload; falls back to the marker-stripped raw when the closing
    marker is absent (truncated completions).

    Known edge (accepted): a no-think transcription of a page whose visible
    text contains a literal unclosed "<think>" is treated as reasoning and the
    doc surfaces as ERROR rather than a bad keep — fails safe, vanishingly
    rare, auditable via the error sidecar.
    """
    raw = THINK_BLOCK_RE.sub("", raw)
    raw = THINK_UNCLOSED_RE.sub("", raw)
    match = OUTPUT_MARKER_RE.search(raw)
    if match:
        return match.group(1).strip()
    raw = OUTPUT_MARKER_OPEN_RE.sub("", raw)
    raw = OUTPUT_MARKER_CLOSE_RE.sub("", raw)
    return raw.strip()


def split_reasoning(raw: str) -> str | None:
    """Recover the reasoning channel from raw offline text.

    A server's reasoning parser sends thinking as a separate field; offline it is
    inline as `<think>...</think>`. Return the thinking text (without tags) when a
    think block is present, else None. Handles the unclosed case (runaway
    reasoning cut at max_tokens) by returning everything after `<think>`.
    """
    if _THINK_OPEN not in raw:
        return None
    after_open = raw.split(_THINK_OPEN, 1)[1]
    if _THINK_CLOSE in after_open:
        return after_open.split(_THINK_CLOSE, 1)[0].strip()
    return after_open.strip() or None


@dataclass
class InferenceResult:
    """Offline mirror of the reference InferenceResult contract."""

    output_text: str
    raw_output: str
    reasoning_content: str | None
    completion_tokens: int
    finish_reason: str | None
    error: str | None = None


def result_from_raw(raw: str, completion_tokens: int, finish_reason: str | None) -> InferenceResult:
    """Build an InferenceResult from one offline generation's raw text.

    Offline the reasoning is inline as `<think>...</think>` rather than a separate
    server field. An UNCLOSED think block (runaway reasoning cut at max_tokens)
    has no content channel at all, so its output_text is empty — otherwise
    `parse_output_markers` would return the think-dump as "output" and mask a
    reasoning-runaway failure as real text.
    """
    has_open = _THINK_OPEN in raw
    has_close = _THINK_CLOSE in raw
    output_text = "" if (has_open and not has_close) else parse_output_markers(raw)
    return InferenceResult(
        output_text=output_text,
        raw_output=raw,
        reasoning_content=split_reasoning(raw),
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
    )


# --- Outcome classification (offline: classify, do not retry no-ops) ---


def is_reasoning_runaway(r: InferenceResult) -> bool:
    """Runaway reasoning ate the whole budget before any text: finish_reason
    'length' with empty output. A truncation FAILURE, not a drop decision."""
    return r.error is None and r.finish_reason == "length" and not (r.output_text or "").strip()


def is_premature_eos_filter(r: InferenceResult) -> bool:
    """Model reasoned, never reached the output field, stopped cleanly, and the
    reasoning stated no verdict. Empty output would be mis-scored as a drop, so
    surface as a failure instead. Mirrors the reference premature-EOS guard."""
    reas = r.reasoning_content or ""
    return (
        r.error is None
        and r.finish_reason != "length"
        and not (r.output_text or "").strip()
        and bool(reas.strip())
        and "NO_USEFUL" not in reas
        and "KEEP" not in reas[-300:].upper()
    )


def is_empty(r: InferenceResult) -> bool:
    """Nothing generated at all (no output, no reasoning, 0 tokens)."""
    return (
        r.error is None
        and not (r.output_text or "").strip()
        and not (r.reasoning_content or "").strip()
        and r.completion_tokens == 0
    )


# --- Batched chat function type + vLLM implementation ---


class ChatFn(Protocol):
    """A batched chat primitive: one uniform-sampling-params stage per call."""

    def __call__(
        self,
        messages_list: list[list[dict]],
        *,
        enable_thinking: bool,
        temperature: float,
        max_tokens: int,
    ) -> list[InferenceResult]: ...


def make_vllm_chat_fn(llm: Any) -> ChatFn:
    """Wrap an offline `vllm.LLM` into a batched ChatFn.

    Each call maps to one `llm.chat(list_of_conversations, ...)` with a single
    `SamplingParams` and `chat_template_kwargs={"enable_thinking": ...}` so the
    whole batch shares one thinking mode (the constraint that makes staged
    batches the natural offline shape).
    """
    from vllm import SamplingParams

    def chat_fn(
        messages_list: list[list[dict]],
        *,
        enable_thinking: bool,
        temperature: float,
        max_tokens: int,
    ) -> list[InferenceResult]:
        if not messages_list:
            return []
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = llm.chat(
            messages_list,
            params,
            chat_template_kwargs={"enable_thinking": enable_thinking},
            use_tqdm=False,
        )
        results: list[InferenceResult] = []
        for out in outputs:
            gen = out.outputs[0]
            results.append(
                result_from_raw(
                    raw=gen.text,
                    completion_tokens=len(gen.token_ids),
                    finish_reason=gen.finish_reason,
                )
            )
        return results

    return chat_fn


# Callable that scores output-token thinking vs response split is reused from the
# runner (`_split_thinking_response`); the schedulers pass token counts through
# to the profiling sidecar rather than recomputing here.
StageProfiler = Callable[[str, float, int], None]
