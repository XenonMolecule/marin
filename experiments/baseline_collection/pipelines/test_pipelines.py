# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the extraction pipeline machinery (no vLLM / no TPU).

The scheduler is exercised with a fake ChatFn and an identity filter-view so the
data-dependent control flow (filter gate, chunk join, loop-guard, failure
classification) is validated without an engine.
"""

from __future__ import annotations

import dataclasses

import pytest

from experiments.baseline_collection.pipelines import (
    bloat_filter,
    code_guard,
    coverage_guard,
    summarization_guard,
    tail_guard,
    view_choice,
)
from experiments.baseline_collection.pipelines import pipeline_specs as ps
from experiments.baseline_collection.pipelines.loop_guard import (
    truncate_loops,
    worst_line_amplification,
)
from experiments.baseline_collection.pipelines.offline_chat import (
    is_premature_eos_filter,
    is_reasoning_runaway,
    parse_output_markers,
    result_from_raw,
    split_reasoning,
)
from experiments.baseline_collection.pipelines.two_stage import (
    DECISION_DROP,
    DECISION_ERROR,
    DECISION_KEEP,
    run_two_stage,
)


class FakeCodec:
    """Char-per-token codec: enough for chunk sizing in tests (no HF load)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))

    def decode(self, ids: list[int]) -> str:
        return "x" * len(ids)

    def count(self, text: str) -> int:
        return len(text)


def _filter_raw(payload: str) -> str:
    return f"<think>some reasoning</think>[[ ## text ## ]]\n{payload}\n[[ ## completed ## ]]"


def _extract_raw(payload: str) -> str:
    return f"[[ ## text ## ]]\n{payload}\n[[ ## completed ## ]]"


def make_fake_chat(*, extract_payload: str = "EXTRACTED"):
    """Filter (think-on): KEEP unless the html contains DROPME. Extract
    (think-off): echo a fixed payload, or a looping payload if html has LOOPME."""

    def chat_fn(messages_list, *, enable_thinking, temperature, max_tokens):
        out = []
        for msgs in messages_list:
            user = msgs[-1]["content"]
            if enable_thinking and temperature == 0.0 and max_tokens <= 4096:
                payload = "[NO_USEFUL_CONTENT]" if "DROPME" in user else "KEEP"
                raw = _filter_raw(payload)
            else:
                payload = extract_payload
                raw = _extract_raw(payload)
            out.append(result_from_raw(raw, completion_tokens=max(1, len(raw.split())), finish_reason="stop"))
        return out

    return chat_fn


IDENTITY_VIEW = lambda html, min_chars: html  # noqa: E731


@pytest.fixture
def pipeline():
    # Small chunk budget so multi-paragraph docs split into several chunks.
    budget = ps.PipelineBudget(chunk_tokens=12, filter_view_tokens=100000)
    return ps.build_two_stage_v1(budget=budget)


def test_parse_and_split():
    raw = "<think>reasoning here</think>[[ ## text ## ]]\nbody\n[[ ## completed ## ]]"
    assert parse_output_markers(raw) == "body"
    assert split_reasoning(raw) == "reasoning here"
    assert split_reasoning("no think here[[ ## text ## ]]\nx\n[[ ## completed ## ]]") is None


def test_is_drop():
    assert ps.is_drop("")
    assert ps.is_drop("   ")
    assert ps.is_drop("Language: English\n[NO_USEFUL_CONTENT]")  # substring, not leading
    assert not ps.is_drop("KEEP")
    assert not ps.is_drop("real extracted content")


def test_filter_drop(pipeline):
    records = [{"html": "this page is DROPME junk"}]
    results, prof = run_two_stage(
        records, make_fake_chat(), FakeCodec(), ps.make_formatter(), pipeline, filter_view_fn=IDENTITY_VIEW
    )
    assert results[0].decision == DECISION_DROP
    assert results[0].text == ps.DROP_MARKER
    assert results[0].filter_reasoning == "some reasoning"
    assert prof["dropped"] == 1 and prof["kept"] == 0


def test_keep_single_chunk(pipeline):
    records = [{"html": "body"}]  # under the tiny fixture chunk budget -> one chunk
    results, prof = run_two_stage(
        records, make_fake_chat(), FakeCodec(), ps.make_formatter(), pipeline, filter_view_fn=IDENTITY_VIEW
    )
    assert results[0].decision == DECISION_KEEP
    assert results[0].text == "EXTRACTED"
    assert results[0].num_chunks == 1
    assert prof["kept"] == 1


def test_keep_multi_chunk_join(pipeline):
    # </p> closers give tier-1 breakpoints; small budget forces several chunks.
    html = "AAAA</p>BBBB</p>CCCC</p>DDDD</p>EEEE"
    records = [{"html": html}]
    results, prof = run_two_stage(
        records, make_fake_chat(), FakeCodec(), ps.make_formatter(), pipeline, filter_view_fn=IDENTITY_VIEW
    )
    assert results[0].decision == DECISION_KEEP
    assert results[0].num_chunks >= 2
    # Each chunk echoes EXTRACTED; joined with blank lines.
    parts = results[0].text.split("\n\n")
    assert parts == ["EXTRACTED"] * results[0].num_chunks
    assert prof["n_chunks"] == results[0].num_chunks


def test_mixed_batch(pipeline):
    records = [
        {"html": "keep me article"},
        {"html": "DROPME gallery"},
        {"html": "another keeper"},
    ]
    results, prof = run_two_stage(
        records, make_fake_chat(), FakeCodec(), ps.make_formatter(), pipeline, filter_view_fn=IDENTITY_VIEW
    )
    assert [r.decision for r in results] == [DECISION_KEEP, DECISION_DROP, DECISION_KEEP]
    assert prof["kept"] == 2 and prof["dropped"] == 1


def test_reasoning_runaway_is_error_not_drop():
    r = result_from_raw("<think>looping forever with no close", completion_tokens=4096, finish_reason="length")
    assert is_reasoning_runaway(r)
    # A runaway on the filter must NOT become a [NO_USEFUL_CONTENT] drop.
    budget = ps.PipelineBudget()
    pipe = ps.build_two_stage_v1(budget=budget)

    def runaway_chat(messages_list, *, enable_thinking, temperature, max_tokens):
        return [
            result_from_raw("<think>runaway", completion_tokens=max_tokens, finish_reason="length")
            for _ in messages_list
        ]

    results, prof = run_two_stage(
        [{"html": "x"}], runaway_chat, FakeCodec(), ps.make_formatter(), pipe, filter_view_fn=IDENTITY_VIEW
    )
    assert results[0].decision == DECISION_ERROR
    assert prof["errored"] == 1 and prof["dropped"] == 0


def test_premature_eos_filter_detection():
    # Reasoned, no verdict word, empty output field, clean stop -> failure.
    r = result_from_raw("<think>hmm this is tricky</think>", completion_tokens=10, finish_reason="stop")
    assert is_premature_eos_filter(r)
    # But a verdict living in the reasoning is a real decision, not a failure.
    r2 = result_from_raw("<think>this is NO_USEFUL clearly</think>", completion_tokens=10, finish_reason="stop")
    assert not is_premature_eos_filter(r2)


def test_loop_guard_truncates_runaway():
    looped = "\n".join(["repeat line"] * 200)
    assert worst_line_amplification(looped, "repeat line") >= 20
    cut = truncate_loops(looped, "repeat line", threshold=20)
    assert cut.count("repeat line") <= 3


# --- one-call (llm_simple_v1) --------------------------------------------------

from experiments.baseline_collection.pipelines.one_call import run_one_call  # noqa: E402


def make_onecall_chat(cont_payload: str = "CONT"):
    """Voting head (think-on): drop if the chunk contains 'DROP', else keep with
    payload KEEPHEAD. Continuation (think-off): echo cont_payload."""

    def chat_fn(messages_list, *, enable_thinking, temperature, max_tokens):
        out = []
        for msgs in messages_list:
            user = msgs[-1]["content"]
            if enable_thinking:
                payload = "[NO_USEFUL_CONTENT]" if "DROP" in user else "KEEPHEAD"
                raw = _filter_raw(payload)
            else:
                raw = _extract_raw(cont_payload)
            out.append(result_from_raw(raw, completion_tokens=5, finish_reason="stop"))
        return out

    return chat_fn


@pytest.fixture
def onecall():
    # budget=int(11*0.98)=10: two <=10-char blocks split into exactly two chunks.
    budget = ps.PipelineBudget(chunk_tokens=11, filter_view_tokens=100000)
    return ps.build_one_call_v1(budget=budget)


def test_onecall_keep_single_chunk(onecall):
    results, prof = run_one_call([{"html": "good"}], make_onecall_chat(), FakeCodec(), ps.make_formatter(), onecall)
    assert results[0].decision == DECISION_KEEP
    assert results[0].text == "KEEPHEAD"
    assert prof["kept"] == 1


def test_onecall_drop_single_chunk(onecall):
    results, prof = run_one_call([{"html": "DROP this"}], make_onecall_chat(), FakeCodec(), ps.make_formatter(), onecall)
    assert results[0].decision == DECISION_DROP
    assert results[0].text == ps.DROP_MARKER
    assert prof["dropped"] == 1


def test_onecall_second_voter_rescue(onecall):
    # chunk0 (DROPme</p>) drops; chunk1 (keepME) rescues via the second voter.
    results, _ = run_one_call(
        [{"html": "DROPme</p>keepME"}], make_onecall_chat(), FakeCodec(), ps.make_formatter(), onecall
    )
    assert results[0].num_chunks == 2
    assert results[0].decision == DECISION_KEEP
    assert results[0].text == "KEEPHEAD"


def test_onecall_both_voters_drop(onecall):
    results, prof = run_one_call(
        [{"html": "DROPa</p>DROPb"}], make_onecall_chat(), FakeCodec(), ps.make_formatter(), onecall
    )
    # Both voters ran (2 chunks) and both dropped -> doc drops.
    assert results[0].decision == DECISION_DROP
    assert prof["dropped"] == 1


# --------------------------------------------------------------------------
# llm_pipeline_v1.1 extraction-guard tests: (a) each guard's logic is a
# faithful port, (b) the guards fire for v1.1 and are OFF for v1 (gating).
# --------------------------------------------------------------------------


def _pipe(chunk_tokens: int, guards_on: bool) -> ps.TwoStagePipeline:
    b = ps.PipelineBudget(chunk_tokens=chunk_tokens, filter_view_tokens=100000)
    base = ps.build_two_stage_v1(budget=b)
    return dataclasses.replace(base, guards=ps.GUARDS_V1_1 if guards_on else ps.ExtractionGuardConfig())


def _keep_chat(payload: str):
    """Filter always KEEP; extraction (think-off) echoes a fixed payload."""

    def chat(messages_list, *, enable_thinking, temperature, max_tokens):
        out = []
        for _ in messages_list:
            raw = _filter_raw("KEEP") if (enable_thinking and max_tokens <= 4096) else _extract_raw(payload)
            out.append(result_from_raw(raw, completion_tokens=5, finish_reason="stop"))
        return out

    return chat


# ---- guard-logic unit tests (faithful port) ----


def test_loop_guard_ignores_added_markdown():
    md = "\n".join(["**Re: topic**"] * 24)  # 24 bold lines, absent from plain input
    assert worst_line_amplification(md, "Re: topic " * 24) < ps.PipelineBudget().loop_threshold
    assert worst_line_amplification("\n".join(["```"] * 30), "") == 0  # structural md ignored


def test_view_choice_novelty_threshold():
    agg = "<p>the body repeats the same short phrase over and over again here</p>"
    safe = agg + "<script>one extra sentence that is reasonably long indeed.</script>"
    assert not view_choice.scripts_carry_content(agg, safe)  # far under 200 new sentences


def test_bloat_index_wall_and_data_lines():
    wall = "\n".join(f"Some Topic Number {i} ({i})" for i in range(40))
    assert bloat_filter.is_bloat_chunk(wall) == "counted_index"
    assert bloat_filter.is_bloat_chunk("A real sentence. Another real sentence here now.") is None
    data = '"5002863",' * 40
    stripped = bloat_filter.strip_data_lines(f"real content line\n{data}\nmore real content")
    assert "real content line" in stripped and "more real content" in stripped and "5002863" not in stripped


def test_code_guard_underextraction():
    chunk = "<pre>" + "x = compute(i);\n" * 200 + "</pre>"
    assert code_guard.code_underextracted(chunk, "just prose, no code kept at all")
    assert not code_guard.code_underextracted("<p>prose only</p>", "prose only")


def test_coverage_early_stop():
    chunk = "<p>" + "word " * 600 + "</p>"  # ~3000 chars of content
    assert coverage_guard.looks_early_stopped(chunk, "tiny")
    assert not coverage_guard.looks_early_stopped(chunk, "word " * 400)  # ~2000, above 0.45
    assert coverage_guard.better_attempt("short", "a much longer and non-degenerate retry output", chunk)


def test_tail_guard_truncation_signals():
    assert tail_guard.looks_truncated("a " * 300 + "ends with no terminal punctuation")
    assert not tail_guard.looks_truncated("a " * 300 + "ends properly.")
    assert tail_guard.looks_truncated("```python\n" + "code line\n" * 60)  # unbalanced fence


def test_summarization_guard():
    src = "<p>" + "the quick brown fox jumps over the lazy dog again and yet again. " * 90 + "</p>"
    summary = "In brief the article describes an animal that leaps repeatedly somewhere."
    assert summarization_guard.looks_summarized(summary, src)
    faithful = "the quick brown fox jumps over the lazy dog again and yet again. " * 30
    assert summarization_guard.better_extraction(summary, faithful, src)


# ---- end-to-end wiring + gating (v1 keeps; v1.1 guards fire) ----


def test_bloat_gating_v1_vs_v11():
    wall = "\n".join(f"Some Topic Number {i} ({i})" for i in range(40))
    chat = _keep_chat(wall)
    r1, _ = run_two_stage(
        [{"html": "x"}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, False), filter_view_fn=IDENTITY_VIEW
    )
    r2, _ = run_two_stage(
        [{"html": "x"}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, True), filter_view_fn=IDENTITY_VIEW
    )
    assert r1[0].decision == DECISION_KEEP and "(0)" in r1[0].text  # v1 keeps the index wall
    assert r2[0].decision == DECISION_ERROR  # v1.1 drops it -> kept-but-empty


def test_data_line_gating_v1_vs_v11():
    payload = "real extracted content here\n" + '"5002863",' * 40 + "\nmore real content"
    chat = _keep_chat(payload)
    r1, _ = run_two_stage(
        [{"html": "x"}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, False), filter_view_fn=IDENTITY_VIEW
    )
    r2, _ = run_two_stage(
        [{"html": "x"}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, True), filter_view_fn=IDENTITY_VIEW
    )
    assert "5002863" in r1[0].text  # v1 keeps the data line
    assert r2[0].decision == DECISION_KEEP and "5002863" not in r2[0].text  # v1.1 strips it
    assert "real extracted content" in r2[0].text


def test_reroll_gating_v1_vs_v11():
    big = "<p>" + "word " * 700 + "</p>"  # early-stop-eligible chunk content

    def chat(messages_list, *, enable_thinking, temperature, max_tokens):
        out = []
        for _ in messages_list:
            if enable_thinking and max_tokens <= 4096:
                raw = _filter_raw("KEEP")
            elif temperature == 0.0:
                raw = _extract_raw("tiny")  # first pass: early-stopped
            else:
                raw = _extract_raw("recovered " * 80)  # t=0.3 reroll: long, faithful-ish
            out.append(result_from_raw(raw, completion_tokens=5, finish_reason="stop"))
        return out

    r1, _ = run_two_stage(
        [{"html": big}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, False), filter_view_fn=IDENTITY_VIEW
    )
    r2, p2 = run_two_stage(
        [{"html": big}], chat, FakeCodec(), ps.make_formatter(), _pipe(100000, True), filter_view_fn=IDENTITY_VIEW
    )
    assert r1[0].text == "tiny"  # v1: no reroll (t=0 only), keeps the short output
    assert p2["loop_guard"]["flagged"] >= 1 and p2["loop_guard"]["rerolled"] >= 1
    assert "recovered" in r2[0].text  # v1.1: reroll kept
