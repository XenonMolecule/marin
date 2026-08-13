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

    ~31x faster than :func:`justext_text` (291.8 vs 9.43 docs/s/core). It also **segfaults** on a
    small fraction of pages (0.108% measured, all containing ``<frameset>``): a hard process death,
    not a Python exception, so a caller MUST run it in a separate process and treat the loss of that
    process as a dropped document — see ``cpu_phase_c.ResiliparseRsPool``.
    """
    if len(html) > max_html_chars:
        logger.warning("skipping resiliparse-rs on %d-char page (> %d cap)", len(html), max_html_chars)
        return ""
    from resiliparse._extract_rs import extract_plain_text

    return extract_plain_text(html, main_content=True, preserve_formatting="markdown")


def resiliparse_rs_batch(args: tuple[list[str], int]) -> list[str]:
    """Picklable top-level wrapper mapping :func:`resiliparse_rs_text` over a chunk of pages.

    ``args`` is ``(htmls, max_html_chars)``. Batching per task amortizes task overhead; the chunk is
    kept small by the caller so one segfaulting page costs only its chunk's worth of re-isolation.
    """
    htmls, max_html_chars = args
    return [resiliparse_rs_text(h, max_html_chars) for h in htmls]


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
