# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure per-document CPU stages of the fast-curation cascade.

This module is deliberately **torch/JAX-free** so it imports cleanly on a CPU Zephyr
worker (the JAX/levanter machinery lives only in ``tpu_phase.py``). The shared HTML
helpers (``body_strip`` / ``fasttext_text``) are imported from
``decode_warcs_clean`` — which already duplicates them away from the torch_xla-tainted
``cascade_chat_filter`` for exactly this reason.

Each heavy model (fastText, the HF tokenizer) is loaded once and cached as a process-level
singleton keyed by its identifier, so a Zephyr ``flat_map`` over many WARCs amortizes the
load.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading

import fsspec

# Reuse the canonical, torch-free preprocessing from the clean decoder. ``fasttext_text``
# == ``to_fasttext_text(html, "body_strip")``: body_strip -> whitespace-collapse -> lower.
from experiments.baseline_collection.decode_warcs_clean import body_strip, fasttext_text
from experiments.baseline_collection.extractors import justext_en

logger = logging.getLogger(__name__)

USEFUL_LABEL = "__label__useful"

# jusText DOM-parses the full page; a pathological multi-MB page can hang a worker for minutes
# (the try/except guards failures, not slowness). Skip extraction above this size -> empty -> the
# doc is dropped as no-content. Matches score_justext_redecode.MAX_JUSTEXT_HTML_CHARS.
MAX_JUSTEXT_HTML_CHARS = 3_000_000

# Process-level singletons (Zephyr runs one worker per process; flat_map reuses them).
_FASTTEXT_CACHE: dict[str, object] = {}
_TOKENIZER_CACHE: dict[str, object] = {}
_LOCK = threading.Lock()


def ft_text(html: str) -> str:
    """Raw HTML -> fastText/ModernBERT input text (``body_strip`` + ws-collapse + lower).

    Identical to ``decode_warcs_clean``'s ``text_body`` and to
    ``fasttext_useful_classifier.to_fasttext_text(html, "body_strip")``.
    """
    return fasttext_text(body_strip(html))


def load_fasttext(model_url: str):
    """Load (and cache) a fastText model from a gs:// path. Copies to a local temp file
    once per process because ``fasttext.load_model`` needs a real filesystem path."""
    with _LOCK:
        model = _FASTTEXT_CACHE.get(model_url)
        if model is not None:
            return model
        import fasttext

        fd, local = tempfile.mkstemp(prefix="ft_model_", suffix=".bin")
        os.close(fd)
        logger.info("downloading fastText model %s -> %s", model_url, local)
        with fsspec.open(model_url, "rb") as src, open(local, "wb") as dst:
            dst.write(src.read())
        model = fasttext.load_model(local)
        _FASTTEXT_CACHE[model_url] = model
        return model


def fasttext_useful_prob(model, text: str) -> float:
    """``P(__label__useful)`` from a binary fastText model; ``0.0`` for empty text.

    Uses the low-level ``model.f.predict`` (list of ``(prob, label)``) to dodge the
    NumPy-2.0 ``np.array(..., copy=False)`` failure in fasttext's high-level ``predict``
    (mirrors ``score_fasttext_useful._useful_prob``).
    """
    if not text:
        return 0.0
    # fastText's ``predict`` (C++/"strict") raises on some pathological inputs — notably text with an
    # embedded newline (it expects a single line), which happens when body_strip fails to clean
    # malformed HTML and leaves raw markup in ``text_body``. Guarding here is essential: without it a
    # single bad document kills the whole WARC and leaves it a permanent, cascade-blocking straggler.
    # Such a doc is treated as not-useful (dropped) — the right call for un-parseable content.
    try:
        preds = model.f.predict(text, 2, 0.0, "strict")
    except Exception as e:
        logger.warning("fastText predict failed on a %d-char doc (%s); dropping as not-useful", len(text), e)
        return 0.0
    for a, b in preds:
        prob, lbl = (a, b) if isinstance(b, str) else (b, a)
        if lbl == USEFUL_LABEL:
            return float(prob)
    return 0.0


def justext_text(
    html: str,
    lang: str = "English",
    max_html_chars: int = MAX_JUSTEXT_HTML_CHARS,
    paragraph_sep: str = "\n\n",
) -> str:
    """Run the custom JustText fork on RAW (un-preprocessed) HTML -> extracted plain text.

    This is the training ``text``. ``""`` on any failure (``justext_en`` already guards) or when
    the page exceeds ``max_html_chars`` (skip to avoid a multi-minute DOM-parse hang). The cap and
    ``paragraph_sep`` (the string joining kept paragraphs) are spec-governed
    (``PipelineSpec.justext_max_html_chars`` / ``justext_paragraph_sep``) so both are versioned;
    the module constants / ``"\\n\\n"`` are only the defaults.
    """
    if len(html) > max_html_chars:
        logger.warning("skipping jusText on %d-char page (> %d cap)", len(html), max_html_chars)
        return ""
    if lang == "English":
        return justext_en(html, paragraph_sep=paragraph_sep)
    # Non-English stoplists: call jusText directly with the requested language.
    try:
        import justext

        paragraphs = justext.justext(html, justext.get_stoplist(lang))
        return paragraph_sep.join(p.text for p in paragraphs if not p.is_boilerplate)
    except Exception:
        return ""


def resiliparse_rs_text(html: str, max_html_chars: int = MAX_JUSTEXT_HTML_CHARS) -> str:
    """Raw HTML -> markdown main content via the XenonMolecule fork's **Rust** engine.

    This is ``resiliparse._extract_rs`` (published as a prebuilt artifact by
    ``build_resiliparse_rs.py``), NOT ``resiliparse.extract.html2text`` — that is marin's core
    ``resiliparse`` dep, a *different* fork claiming the same import name. The two collide, so the
    artifact directory must already sit FIRST on ``sys.path``
    (``score_resiliparse_rs.install_extractor`` does exactly that). The import is function-local
    because the package is downloaded at runtime rather than installed.

    ~31x faster than :func:`justext_text` (291.8 vs 9.43 docs/s/core). Two crash classes are handled
    HERE, in the child, so one bad page costs one page instead of poisoning the pool:

    * ``<frameset>`` pages hard-SEGFAULT the Rust engine (0.1-0.25% of the web). The substring
      screen refuses them -> "" — the same screen (and the same answer: these are parking/redirect
      shells with no main content) the TEXT classifiers' training corpora were built with
      (``extract_prep_text_rs``/``extract_text_shards``), so screening is the parity-faithful
      behavior. Without it, ~every 5k-doc batch contained a crasher, the pool died, and the
      whole-batch isolation fallback silently ran extraction SINGLE-process — the canary's 14x gap.
    * The fork can PANIC on a doc (pyo3 ``PanicException`` inherits BaseException, so a plain
      ``except Exception`` misses it and it kills the worker on the way out) -> "".

    A mixed-case ``<FrameSet``, or any unknown crasher, still segfaults — callers keep
    ``ResiliparseRsPool``'s process isolation as the backstop.
    """
    if len(html) > max_html_chars:
        logger.warning("skipping resiliparse-rs on %d-char page (> %d cap)", len(html), max_html_chars)
        return ""
    if "<frameset" in html or "<FRAMESET" in html:
        return ""
    from resiliparse._extract_rs import extract_plain_text

    try:
        return extract_plain_text(html, main_content=True, preserve_formatting="markdown")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:  # pyo3 PanicException is a BaseException
        return ""


def resiliparse_rs_batch(args: tuple[list[str], int]) -> list[str]:
    """Picklable top-level wrapper mapping :func:`resiliparse_rs_text` over a chunk of pages.

    ``args`` is ``(htmls, max_html_chars)``. Batching per task amortizes task overhead; the chunk is
    kept small by the caller so one segfaulting page costs only its chunk's worth of re-isolation.
    """
    htmls, max_html_chars = args
    return [resiliparse_rs_text(h, max_html_chars) for h in htmls]


def resiliparse_rs_screen_batch(args: tuple[list[str], int]) -> list[tuple[str, bool]]:
    """TEXT-line fused pool task: ``(extracted_text, in_population)`` per page.

    ``in_population`` is the decode-time filter (``fasttext_text(body_strip(html))`` non-empty)
    moved into the pool so its ~2 min/WARC of regex work is parallelized instead of serial in the
    parent; a page outside the population skips extraction entirely. The population it defines is
    EXACTLY the one the cascade thresholds were calibrated on (the 100k sample passed this filter).
    """
    htmls, max_html_chars = args
    out: list[tuple[str, bool]] = []
    for h in htmls:
        if not fasttext_text(body_strip(h)):
            out.append(("", False))
            continue
        out.append((resiliparse_rs_text(h, max_html_chars), True))
    return out


def _justext_one(args: tuple[str, str, int, str]) -> str:
    """Picklable top-level wrapper for ProcessPool fan-out of JustText (the dominant CPU cost).

    JustText (the XenonMolecule learned/sklearn-tier fork) runs per fastText-survivor and is far
    too slow single-threaded at ~tens of thousands of survivors per WARC. ``cpu_phase`` maps this
    over a process pool to use all of the worker's cores. ``args`` is
    ``(html, lang, max_html_chars, paragraph_sep)``.
    """
    html, lang, max_html_chars, paragraph_sep = args
    return justext_text(html, lang, max_html_chars, paragraph_sep)


def load_tokenizer(tokenizer_ref: str):
    """Load (and cache) an HF tokenizer for ModernBERT pre-tokenization."""
    with _LOCK:
        tok = _TOKENIZER_CACHE.get(tokenizer_ref)
        if tok is not None:
            return tok
        from transformers import AutoTokenizer

        logger.info("loading tokenizer %s", tokenizer_ref)
        tok = AutoTokenizer.from_pretrained(tokenizer_ref)
        _TOKENIZER_CACHE[tokenizer_ref] = tok
        return tok


def tokenize_trunc(tokenizer, text: str, max_length: int) -> list[int]:
    """Tokenize ``text`` to token ids, truncated to ``max_length`` (single window).

    Matches the ModernBERT training/scoring path: ``tokenizer(text, truncation=True,
    max_length=max_length)["input_ids"]`` with default special tokens (the [CLS] at
    position 0 is what the deployed cls-pooling head reads).

    Char-cap the input at ``max_length * 8`` before tokenizing: the tokenizer otherwise
    processes the full (often 100k+ char) doc before truncating to ``max_length`` tokens, which
    is the second-largest CPU cost. At >=8 chars/token the first ``max_length`` tokens come
    entirely from this prefix, so the truncated ids are IDENTICAL — just computed far faster.

    Prefer :func:`tokenize_trunc_batch` on the hot path: a one-doc-at-a-time call cannot use the
    Rust tokenizer's cross-document Rayon parallelism and leaves the worker's cores idle.
    """
    return tokenizer(text[: max_length * 8], truncation=True, max_length=max_length)["input_ids"]


# Sub-batch size for the batched tokenizer call. Large enough to keep every core fed by the Rust
# Rayon pool, small enough that only this many char-capped substrings are materialized at once
# (bounds peak RSS on WARCs with tens of thousands of survivors, and composes with Phase A's
# streaming chunk-checkpointing — each chunk's survivors tokenize in one or a few of these).
_TOKENIZE_BATCH_SIZE = 1024


def tokenize_trunc_batch(
    tokenizer, texts: list[str], max_length: int, *, batch_size: int = _TOKENIZE_BATCH_SIZE
) -> list[list[int]]:
    """Batched equivalent of :func:`tokenize_trunc`: byte-identical ``input_ids`` per doc, far faster.

    Two stacked wins over the one-doc-at-a-time :func:`tokenize_trunc`:

    1. **Cross-document Rayon parallelism** — the Rust backend tokenizes the whole sub-batch in
       parallel across the worker's cores, instead of pinning one core per doc. This scales
       near-linearly with cores (measured ~3.2x on 4 threads). It is a no-op when
       ``TOKENIZERS_PARALLELISM=false`` (Iris/Fray inject that by default), so the Phase A entrypoint
       overrides it to ``"true"`` — safe there because Phase A does not fork.
    2. **Skip discarded metadata** — ``encode_batch_fast`` builds only token ids, not the offsets /
       word-ids / special-tokens mask that the ``transformers`` ``__call__`` wrapper computes and we
       throw away (~1.3x on top of the batching).

    ``encode_batch_fast`` reads truncation from the backend rather than a per-call arg, so we
    (re)assert it to this call's ``max_length`` before every batch. That is robust as long as the
    length is known and calls are serialized — always true here (each Phase A / chunk call tokenizes
    at one known ``max_length``, single-threaded at the Python level). Do NOT interleave the per-doc
    wrapper with a *different* truncation on the same tokenizer in the same process.

    Parity: each text is char-capped at ``max_length * 8`` and truncated to ``max_length`` (right
    side, ``longest_first``) exactly as :func:`tokenize_trunc`, so ids match the per-doc path
    element-for-element — asserted in ``test_preprocess.py`` over adversarial inputs including a
    head/tail-distinct long doc that would catch a wrong truncation direction.
    """
    if not texts:
        return []
    backend = tokenizer.backend_tokenizer
    # NOTE: revisit when adding chunking. encode_batch_fast reads truncation from the backend, so we
    # set it here to this call's max_length. When docs are chunked to fit the context and truncation
    # is dropped, remove this line (chunks are already <= max_length, so it's a no-op / unwanted).
    backend.enable_truncation(max_length, stride=0, strategy="longest_first", direction="right")
    out: list[list[int]] = []
    for i in range(0, len(texts), batch_size):
        chunk = [t[: max_length * 8] for t in texts[i : i + batch_size]]
        out.extend(enc.ids for enc in backend.encode_batch_fast(chunk))
    return out


# Texts chosen so any semantic drift between tokenizer implementations surfaces: truncation
# direction (head/tail-distinct long doc), the char cap, special-tokens-only empties, and UTF-8.
_PARITY_TEXTS = (
    "",
    " ",
    "short doc",
    "word " * 20000,
    " ".join(f"tok{i}word{i * 7 % 13}" for i in range(60000)),
    "some utf-8: café π 漢字 emoji \U0001f680 mixed in. " * 500,
    "a" * (8192 * 8 + 50),
)


def load_gigatoken(tokenizer_ref: str):
    """The raw ``gigatoken.Tokenizer`` built from the HF tokenizer (cached like the HF loader).

    Requires the ``gigatoken`` extra. It shares the SAME vocab/merges as the HF tokenizer, so its
    ids are drop-in — but never trust that silently: call :func:`assert_gigatoken_parity` once at
    worker startup before using it on real data.
    """
    key = f"gigatoken::{tokenizer_ref}"
    # Resolve the HF tokenizer BEFORE taking _LOCK: load_tokenizer takes the same non-reentrant
    # lock, so calling it while holding _LOCK self-deadlocks.
    hf_tok = load_tokenizer(tokenizer_ref)
    with _LOCK:
        tok = _TOKENIZER_CACHE.get(key)
        if tok is not None:
            return tok
        import gigatoken

        logger.info("wrapping %s in gigatoken %s", tokenizer_ref, getattr(gigatoken, "__version__", "?"))
        tok = gigatoken.Tokenizer(hf_tok)
        _TOKENIZER_CACHE[key] = tok
        return tok


def tokenize_trunc_batch_arrow(tokenizer, texts: list[str], max_length: int):
    """HF tokenize -> ``(pyarrow list<int32> column, int32 lengths)`` — the columnar contract.

    Same ids as :func:`tokenize_trunc_batch`; the arrow conversion is what the presurvivor write
    consumed anyway, moved into the tokenize seam so both implementations share one output type
    (and one timing boundary in ``timing_a``).
    """
    import numpy as np
    import pyarrow as pa

    ids = tokenize_trunc_batch(tokenizer, texts, max_length)
    column = pa.array(ids, type=pa.list_(pa.int32()))
    return column, np.array([len(x) for x in ids], dtype=np.int32)


def tokenize_trunc_batch_gigatoken(gt_tok, texts: list[str], max_length: int, *, cls_id: int, sep_id: int):
    """Gigatoken tokenize -> ``(pyarrow list<int32> column, int32 lengths)``, never touching Python.

    ~23x the HF path at real WARC scale: gigatoken's native ``encode_batch`` returns a ragged
    awkward array backed by offsets+values buffers; truncation to ``max_length - 2`` body tokens
    and the ``[CLS]``/``[SEP]`` assembly are vectorized array ops; the result converts to the
    parquet schema's ``list<int32>`` without materializing per-doc Python lists (which is where the
    naive integration spent ~90% of its time). Ids are byte-identical to
    :func:`tokenize_trunc_batch_arrow` — verified per worker by :func:`assert_gigatoken_parity`.
    """
    import awkward as ak
    import numpy as np
    import pyarrow as pa

    if not texts:
        return pa.array([], type=pa.list_(pa.int32())), np.array([], dtype=np.int32)
    capped = [t[: max_length * 8] for t in texts]
    ids = ak.values_astype(gt_tok.encode_batch(capped)[:, : max_length - 2], np.int32)
    n = len(ids)
    one = np.ones(n, np.int64)
    cls_col = ak.unflatten(np.full(n, cls_id, np.int32), one)
    sep_col = ak.unflatten(np.full(n, sep_id, np.int32), one)
    full = ak.concatenate([cls_col, ids, sep_col], axis=1)
    column = ak.to_arrow(full, list_to32=True).cast(pa.list_(pa.int32()))
    return column, np.asarray(ak.num(full)).astype(np.int32)


def assert_gigatoken_parity(tokenizer, gt_tok, max_length: int, *, cls_id: int, sep_id: int) -> None:
    """Fail fast unless gigatoken reproduces the HF tokenization EXACTLY on adversarial inputs.

    The token ids are the classifier input the cascade thresholds were calibrated on, so a
    tokenizer that is merely "close" silently shifts every score. Runs once at worker startup;
    costs a few hundred ms.
    """
    ref, ref_n = tokenize_trunc_batch_arrow(tokenizer, list(_PARITY_TEXTS), max_length)
    cand, cand_n = tokenize_trunc_batch_gigatoken(gt_tok, list(_PARITY_TEXTS), max_length, cls_id=cls_id, sep_id=sep_id)
    for i, (a, b, an, bn) in enumerate(zip(ref.to_pylist(), cand.to_pylist(), ref_n, cand_n, strict=True)):
        if a != b or an != bn:
            raise RuntimeError(
                f"gigatoken tokenization diverges from HF on parity text {i} "
                f"(len {len(a)}/{an} vs {len(b)}/{bn}); refusing to tokenize the corpus with it."
            )
    logger.info("gigatoken parity check passed (%d texts, max_length=%d)", len(ref), max_length)


def assert_justext_version(expected: str) -> None:
    """Fail fast if the installed JustText fork differs from the spec's pinned version,
    so a silent dependency drift can never poison the corpus."""
    import justext

    installed = getattr(justext, "__version__", None)
    # Spec stores e.g. "xenon-v4.2.0"; the installed package reports "4.2.0".
    want = expected.split("-v")[-1] if "-v" in expected else expected.lstrip("v")
    if installed is not None and installed != want:
        raise RuntimeError(
            f"JustText version mismatch: spec expects {expected!r} (=={want!r}) but installed "
            f"justext.__version__={installed!r}. Pin the right fork before extracting."
        )
    if installed is None:
        logger.warning("justext.__version__ is unavailable; cannot verify against spec %s", expected)
