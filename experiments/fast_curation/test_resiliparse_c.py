# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase C extraction-engine dispatch and resiliparse-rs crash isolation.

No TPU and no network: the Rust extractor is not installed here. Instead the pool's extraction
function is swapped for one that *really kills its process* on a sentinel page, which is exactly
what the Rust engine does on ``<frameset>`` pages (0.108% of pages measured) — a SIGSEGV, not a
Python exception. ``test_sentinel_is_a_real_process_death`` proves the stand-in dies the same way.
"""

from __future__ import annotations

import functools
import json
import multiprocessing as mp
import os
import resource
import signal

import pytest

from experiments.fast_curation import batch_format, cpu_phase_a, cpu_phase_c, preprocess
from experiments.fast_curation.spec import Extractor, get_spec

# A page shaped like the ones that really crash the Rust engine.
CRASH_SENTINEL = "<html><frameset cols='20%,80%'><frame src='a.html'></frameset></html>"
# SIGKILL, not SIGSEGV. Both are uncatchable hard process deaths and the pool cannot tell them
# apart — it sees a dead child and raises BrokenProcessPool either way — but macOS routes a real
# SIGSEGV through ReportCrash, so running the suite spammed the developer's crash reporter with
# dozens of "Python crashed" reports. SIGKILL is also the *literal* signal an OOM-kill delivers,
# which is the other way this pool dies in production.
CRASH_SIGNAL = signal.SIGKILL
NO_CAP = 10**9


def _segfault_extract(args: tuple[list[str], int]) -> list[str]:
    """Stand-in for ``preprocess.resiliparse_rs_batch`` that hard-kills its process on :data:`CRASH_SENTINEL`.

    Top-level (not a closure/lambda) so a ``spawn`` pool child can unpickle it. The kill is a real
    signal death — no exception is raised and no ``except`` can intercept it — so the pool child is
    gone and the executor really is left broken, just as with the Rust extractor. Core dumps are
    disabled first so the test cannot litter the tree with a core file.
    """
    htmls, _max_html_chars = args
    for html in htmls:
        if html == CRASH_SENTINEL:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.kill(os.getpid(), CRASH_SIGNAL)
    return [f"text::{h}" for h in htmls]


def _raising_extract(args: tuple[list[str], int]) -> list[str]:
    """Every page fails with an ordinary exception (the child survives) — the broken-install case."""
    raise ValueError("extractor is misconfigured")


class _RecordingJustextPool:
    """Stands in for ``JustextPool``: records ``run(args, timeout)`` calls."""

    def __init__(self):
        self.calls: list[tuple] = []

    def run(self, args: list[tuple], timeout):
        self.calls.append((args, timeout))
        return [f"jt::{a[0]}" for a in args]


class _RecordingRsPool:
    """Stands in for ``ResiliparseRsPool``: records ``run(htmls, doc_ids, max_html_chars)`` calls."""

    def __init__(self, crash_ids: frozenset[str] = frozenset()):
        self.calls: list[tuple] = []
        self._crash_ids = crash_ids

    def run(self, htmls: list[str], doc_ids: list[str], max_html_chars: int):
        self.calls.append((htmls, doc_ids, max_html_chars))
        texts = ["" if d in self._crash_ids else f"rs::{h}" for h, d in zip(htmls, doc_ids, strict=True)]
        return texts, [d for d in doc_ids if d in self._crash_ids]


@pytest.mark.parametrize(
    ("spec_id", "engine"),
    [
        ("fastpipe_v1", Extractor.JUSTEXT),
        ("fastpipe_v2", Extractor.JUSTEXT),
        ("fastpipe_v3", Extractor.JUSTEXT),
        ("lpv11_fastpipe_v1", Extractor.RESILIPARSE_RS),
    ],
)
def test_extraction_engine_per_spec(spec_id: str, engine: Extractor):
    assert get_spec(spec_id).extraction_engine is engine


def test_extract_chunk_dispatches_justext_with_spec_knobs():
    """The live v1-v3 path is unchanged: same 4-tuple args, same per-doc timeout, never any crashes."""
    spec = get_spec("fastpipe_v3")
    pool = _RecordingJustextPool()
    html = "<html><body><p>hello</p></body></html>"

    texts, crashed = cpu_phase_c._extract_chunk(spec, pool, [html], ["d0"])

    assert texts == [f"jt::{html}"]
    assert crashed == []
    args, timeout = pool.calls[0]
    assert args == [(html, spec.justext_lang, spec.justext_max_html_chars, spec.justext_paragraph_sep)]
    assert timeout == spec.justext_timeout


def test_extract_chunk_dispatches_resiliparse_rs():
    spec = get_spec("lpv11_fastpipe_v1")
    pool = _RecordingRsPool(crash_ids=frozenset({"d1"}))

    texts, crashed = cpu_phase_c._extract_chunk(spec, pool, ["<p>a</p>", CRASH_SENTINEL], ["d0", "d1"])

    assert texts == ["rs::<p>a</p>", ""]
    assert crashed == ["d1"]
    assert pool.calls == [(["<p>a</p>", CRASH_SENTINEL], ["d0", "d1"], spec.justext_max_html_chars)]


def test_extract_chunk_justext_serial_without_pool():
    """``--justext-procs 1`` (pool=None) still extracts, on the jusText line only."""
    spec = get_spec("fastpipe_v3")
    html = (
        "<html><body><p>This is the main article body with several informative sentences. "
        "It should survive boilerplate removal because it is the real content here.</p></body></html>"
    )

    texts, crashed = cpu_phase_c._extract_chunk(spec, None, [html], ["d0"])

    assert crashed == []
    assert "informative sentences" in texts[0]


def test_sentinel_is_a_real_process_death():
    """Evidence for the isolation test below: the sentinel really kills its own process.

    If it merely raised, the exit code would be 1 (traceback) and the pool would never break, so the
    crash-isolation path would not be exercised at all.
    """
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_segfault_extract, args=(([CRASH_SENTINEL], NO_CAP),))
    proc.start()
    proc.join(120)

    assert proc.exitcode == -CRASH_SIGNAL  # negative => killed by that signal, not a raised exception
    # ...and non-sentinel pages return normally in-process.
    assert _segfault_extract((["<p>x</p>"], NO_CAP)) == ["text::<p>x</p>"]


def test_pool_survives_hard_crash_reports_id_and_stays_usable():
    """A page that kills its worker costs that page only: empty text, recorded id, WARC continues."""
    pool = cpu_phase_c.ResiliparseRsPool(2, "/nonexistent-pkg-dir", task_chunk=4, extract_fn=_segfault_extract)
    htmls = [f"<p>doc {i}</p>" for i in range(6)]
    htmls[3] = CRASH_SENTINEL
    doc_ids = [f"d{i}" for i in range(6)]
    try:
        texts, crashed = pool.run(htmls, doc_ids, NO_CAP)
        # The pool was poisoned by the crash; the next batch must still work (rebuilt on demand).
        after_texts, after_crashed = pool.run(["<p>next</p>"], ["d9"], NO_CAP)
    finally:
        pool.close()

    assert crashed == ["d3"]
    assert texts[3] == ""  # the pipeline's "dropped" convention
    assert [t for i, t in enumerate(texts) if i != 3] == [f"text::{h}" for i, h in enumerate(htmls) if i != 3]
    assert (after_texts, after_crashed) == (["text::<p>next</p>"], [])


def test_pool_raises_when_every_page_fails():
    """A broken install (every page failing) must fail loudly, not write a corpus of empty docs."""
    n = cpu_phase_c.RS_MIN_CRASH_SAMPLE
    pool = cpu_phase_c.ResiliparseRsPool(1, "/nonexistent-pkg-dir", task_chunk=4, extract_fn=_raising_extract)
    try:
        with pytest.raises(RuntimeError, match="install is broken"):
            pool.run([f"<p>{i}</p>" for i in range(n)], [f"d{i}" for i in range(n)], NO_CAP)
    finally:
        pool.close()


def _seed_warc(root: str, spec, warc_hash: str, docs: list[tuple[str, str, float]]) -> None:
    """Write one WARC's Phase-A pre-survivors + Phase-B keeplist. ``docs`` = (doc_id, html, prob)."""
    rows = [
        {
            "doc_id": doc_id,
            "url": f"http://example.com/{doc_id}",
            "warc_hash": warc_hash,
            "snapshot": "2024-10",
            "fasttext_score": 0.5,
            "html": html,
            "input_ids": [1, 2, 3],
            "n_tokens": 3,
        }
        for doc_id, html, _ in docs
    ]
    batch_format.write_presurvivors(f"{spec.presurvivors_prefix(root)}/data-{warc_hash}.parquet", rows)
    batch_format.write_keeplist(
        f"{spec.keeplist_prefix(root)}/data-{warc_hash}.parquet",
        [d for d, _, _ in docs],
        [p for _, _, p in docs],
        # lpv11 keeplists carry pooled_prob; a pooled-dropped doc has a NaN modernbert_prob.
        pooled_probs=[0.0 if p != p else 0.9 for _, _, p in docs],
    )


def test_process_one_completes_warc_despite_a_crashing_page(tmp_path):
    """End to end: one killer page does not stop the WARC, and its loss is reported, not silent."""
    spec = get_spec("lpv11_fastpipe_v1")
    root = str(tmp_path)
    warc_hash = "deadbeef"
    keep_prob = spec.modernbert_threshold + 0.1
    docs = [
        ("good0", "<p>doc 0</p>", keep_prob),
        ("crash", CRASH_SENTINEL, keep_prob),
        ("good1", "<p>doc 1</p>", keep_prob),
        ("mb_drop", "<p>doc 2</p>", spec.modernbert_threshold - 0.1),
        ("pooled_drop", "<p>doc 3</p>", float("nan")),  # never scored by ModernBERT
        ("good2", "<p>doc 4</p>", keep_prob),
    ]
    _seed_warc(root, spec, warc_hash, docs)

    pool = cpu_phase_c.ResiliparseRsPool(2, "/nonexistent-pkg-dir", task_chunk=4, extract_fn=_segfault_extract)
    try:
        stats = cpu_phase_c._process_one(
            "s3://commoncrawl/fake.warc.gz", warc_hash, lambda: None, spec=spec, pool=pool, bucket=root
        )
    finally:
        pool.close()

    kept = batch_format.read_table(f"{spec.kept_prefix(root)}/data-{warc_hash}.parquet")
    # lpv11 has a pooled stage, so the corpus carries pooled_prob alongside the other scores.
    assert kept.schema == batch_format.kept_schema_for(spec)
    assert kept.column("doc_id").to_pylist() == ["good0", "good1", "good2"]
    assert kept.column("text").to_pylist() == ["text::<p>doc 0</p>", "text::<p>doc 1</p>", "text::<p>doc 4</p>"]
    assert stats["docs_in"] == 4  # 3 good + the crasher passed ModernBERT
    assert stats["docs_out"] == 3
    assert set(stats["compute_seconds"]) == {"resiliparse_rs"}

    with open(f"{spec.namespace(root)}/timing_c/data-{warc_hash}.json") as f:
        timing = json.load(f)
    assert timing["extractor"] == "resiliparse_rs"
    assert timing["n_extract_crashed"] == 1
    assert timing["crashed_doc_ids"] == ["crash"]
    assert timing["n_modernbert_keep"] == 4
    assert timing["n_kept"] == 3


def test_process_one_text_consumes_the_fused_pool_contract(tmp_path, monkeypatch):
    """END TO END: TEXT-line Phase A must route each fused-pool ``(text, in_population)`` result to
    its fate and land the survivors in the presurvivor parquet with correct funnel counters.

    Regression: the unit tests each covered one seam, but nothing ran ``_process_one_text`` against
    the fused pool's tuple results — a silently-unapplied edit left the loop treating them as
    strings, and every worker of a 32-worker fleet died on its first chunk
    (``AttributeError: 'tuple' object has no attribute 'encode'``)."""
    spec = get_spec("lpv11_fastpipe_v2")
    records = [
        {"doc_id": d, "url": f"http://{d}", "warc_hash": "h01", "snapshot": "CC-MAIN-2020-05", "html": f"<p>{d}</p>"}
        for d in ("keep", "prefiltered", "empty_extract", "below_gate")
    ]
    monkeypatch.setattr(cpu_phase_a, "_decode_one_warc", lambda wp, with_text_body=True: list(records))
    monkeypatch.setattr(cpu_phase_a, "_gcs_exists", lambda p: False)  # keep chunk probes off GCS

    class _FusedPool:
        """Returns the fused shape for each fate the loop must route."""

        def run(self, htmls, doc_ids, cap):
            table = {
                "keep": ("Real article text worth keeping.", True),
                "prefiltered": ("", False),  # empty text_body -> out of population
                "empty_extract": ("", True),  # in population, extractor found nothing
                "below_gate": ("junk", True),  # extracted but fastText rejects it
            }
            return [table[d] for d in doc_ids], []

    class _Model:
        class f:
            @staticmethod
            def predict(text, k, threshold, mode):
                return [("__label__useful", 0.9 if "real article" in text else 1e-6)]

    tokenizer = preprocess.load_tokenizer("answerdotai/ModernBERT-base")
    stats = cpu_phase_a._process_one_text(
        "warc://x",
        "h01",
        lambda: None,
        spec=spec,
        model=_Model(),
        tokenize_batch=functools.partial(preprocess.tokenize_trunc_batch_arrow, tokenizer, max_length=128),
        tokenizer_impl="hf",
        pool=_FusedPool(),
        bucket=str(tmp_path),
        cleanup_chunks=False,
    )

    t = batch_format.read_table(f"{spec.presurvivors_prefix(str(tmp_path))}/data-h01.parquet")
    assert t.column("doc_id").to_pylist() == ["keep"], "exactly the doc that passed every gate survives"
    assert t.column("text").to_pylist() == ["Real article text worth keeping."]
    assert t.column("input_ids").to_pylist()[0][0] == 50281  # [CLS]-framed, really tokenized
    assert stats == {
        "docs_in": 4,
        "docs_out": 1,
        "wall_seconds": stats["wall_seconds"],
        "compute_seconds": stats["compute_seconds"],
    }
    with open(f"{spec.namespace(str(tmp_path))}/timing_a/data-h01.json") as f:
        timing = json.load(f)
    assert (timing["n_prefilter_dropped"], timing["n_extract_empty"]) == (1, 1)
    assert timing["tokenizer_impl"] == "hf"


def test_frameset_screen_refuses_before_the_engine_is_touched():
    """``<frameset>`` pages hard-SEGFAULT the Rust engine; the child-side screen must return ""
    WITHOUT importing/calling it (this test runs with no artifact installed — reaching the engine
    would raise ImportError, not return). Both the lowercase and uppercase spellings the training
    corpora screened must be refused; screening is what keeps the pool alive and parallel."""
    from experiments.fast_curation import preprocess

    assert preprocess.resiliparse_rs_text(CRASH_SENTINEL, NO_CAP) == ""
    assert preprocess.resiliparse_rs_text("<html><FRAMESET rows='*'></FRAMESET></html>", NO_CAP) == ""


def test_screen_batch_population_filter_skips_extraction():
    """The fused TEXT-line task must mark empty-``text_body`` pages out-of-population WITHOUT
    extracting them (same reason as above: no artifact here, so touching the engine would raise),
    and screened framesets stay IN-population with an empty extraction — matching what the old
    decode-filter + crash-isolation pipeline produced for them."""
    from experiments.fast_curation import preprocess

    no_body = "<html><head><title>t</title></head><body>   </body></html>"
    frameset_with_body = f"<html><body>real body text here</body>{CRASH_SENTINEL}</html>"
    out = preprocess.resiliparse_rs_screen_batch(([no_body, frameset_with_body], NO_CAP))
    assert out[0] == ("", False), "no body text -> out of population, never extracted"
    assert out[1] == ("", True), "frameset with body text -> in population, screened to empty"


def test_pool_crash_result_matches_the_fused_task_shape(tmp_path):
    """When a fused-task page crashes anyway (mixed-case frameset, unknown crasher), the isolation
    path must emit the caller's crash_result so the results stay shape-homogeneous with the fused
    ``(text, in_population)`` tuples — a bare "" would crash the strict zip in Phase A."""
    pool = cpu_phase_c.ResiliparseRsPool(2, str(tmp_path), extract_fn=_segfault_screen_extract, crash_result=("", True))
    try:
        results, crashed = pool.run(["<p>fine</p>", CRASH_SENTINEL, "<p>also fine</p>"], ["a", "b", "c"], NO_CAP)
    finally:
        pool.close()
    assert crashed == ["b"]
    assert results[1] == ("", True), "the crashed page carries the fused crash_result"
    assert results[0] == ("ok:<p>fine</p>", True) and results[2] == ("ok:<p>also fine</p>", True)


def _segfault_screen_extract(args: tuple[list[str], int]) -> list[tuple[str, bool]]:
    """Fused-shape stand-in that hard-kills the process on :data:`CRASH_SENTINEL`."""
    htmls, _cap = args
    out = []
    for h in htmls:
        if h == CRASH_SENTINEL:
            os.kill(os.getpid(), CRASH_SIGNAL)
        out.append((f"ok:{h}", True))
    return out


def test_size_aware_chunks_bound_bytes_without_dropping_docs():
    """A pool task is pickled into a child, so bound it by BYTES as well as doc count. An OOM-killed
    child looks exactly like a segfault to the isolation path, which would then record the page as a
    crash and emit empty text — silently losing a long document."""
    from experiments.fast_curation.cpu_phase_c import RS_TASK_MAX_BYTES, _size_aware_chunks

    small = ["x" * 1000] * 100
    chunks = _size_aware_chunks(small, task_chunk=25)
    assert [len(c) for c in chunks] == [25, 25, 25, 25], "small docs still pack to the doc-count cap"
    assert sum(len(c) for c in chunks) == 100, "no doc lost"

    big = ["y" * (10 * 1024 * 1024)] * 8  # 80 MB total, well over the per-task cap
    chunks = _size_aware_chunks(big, task_chunk=25)
    assert all(sum(len(h) for h in c) <= RS_TASK_MAX_BYTES for c in chunks), "every task within the byte cap"
    assert sum(len(c) for c in chunks) == 8, "no doc lost"

    huge = ["z" * (RS_TASK_MAX_BYTES + 1)]  # one doc larger than the whole cap
    assert _size_aware_chunks(huge, task_chunk=25) == [huge], "an oversized page gets its own task, not dropped"

    mixed = ["a" * 100, "b" * (RS_TASK_MAX_BYTES // 2), "c" * (RS_TASK_MAX_BYTES // 2), "d" * 100]
    chunks = _size_aware_chunks(mixed, task_chunk=25)
    assert [h for c in chunks for h in c] == mixed, "order and content preserved exactly"
