# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply the full DCLM-Baseline filtering pipeline to plain-text web records.

Implements the pipeline defined in ``dclm_full.yaml``: RefinedWeb heuristic
filters, language detection, Gopher repetition filters, line-level modifiers,
and a FastText quality classifier.  All filter logic is ported from the DCLM
codebase (https://github.com/mlfoundations/dclm) so that we can run it inside
marin's Zephyr pipeline infrastructure without pulling in DCLM's Ray code.

Usage as an ExecutorStep::

    filter_step = ExecutorStep(
        name="filtered/dclm_full_pipeline",
        fn=dclm_filter,
        config=DclmFilterConfig(
            input_path=extract_text / "*.jsonl.gz",
            output_path=this_output_path(),
            lid_model_path=download_lid_model,
            quality_model_path=download_quality_model,
            banlists_path=download_banlists,
        ),
        resources=ResourceConfig.with_cpu(cpu=8, ram="32g"),
        pip_dependency_groups=["cpu", "dclm"],
    )
"""

import json
import logging
import os
import re
import string
import tempfile
from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from urllib.parse import urlparse

from zephyr import Dataset, ZephyrContext, load_file

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DclmFilterConfig:
    input_path: str
    """Glob pattern for input JSONL files containing text records."""

    output_path: str
    """Where to write filtered output JSONL files."""

    lid_model_path: str
    """GCS path to directory containing ``lid.176.bin``."""

    quality_model_path: str
    """GCS path to directory containing ``fasttext_oh_eli5.bin``."""

    banlists_path: str
    """GCS path to directory containing ban-list text files and ``iana_tlds.txt``."""

    text_column: str = "text"
    """Column name containing the text to filter."""

    url_column: str = "url"
    """Column name containing the page URL."""


# ---------------------------------------------------------------------------
# Utility functions (ported from DCLM core_utils.py)
# ---------------------------------------------------------------------------


def _split_paragraphs(text: str, paragraph_end: str = "\n", remove_empty: bool = True) -> list[str]:
    """Split text into paragraphs/lines."""
    paragraphs = re.split(paragraph_end, text)
    if remove_empty:
        paragraphs = [p for par in paragraphs if (p := par.strip())]
    return paragraphs


def _split_words(text: str, model: str = "fasttext", ignore_punctuation: bool = False) -> list[str]:
    """Split text into words using the specified tokenizer model.

    Supported models: ``fasttext``, ``uniseg``, ``split``.
    """
    if model == "uniseg":
        from uniseg.wordbreak import words

        tokens = words(text)
    elif model == "fasttext":
        import fasttext

        tokens = fasttext.FastText.tokenize(text)
    elif model == "split":
        tokens = text.split()
    else:
        raise ValueError(f"Unknown word tokenizer: {model}")

    if ignore_punctuation:
        return [w for w in tokens if w[0].isalnum() or w[0].isspace()]
    return [w for w in tokens if w.strip()]


def _is_space_or_punct(s: str) -> bool:
    """Return True if the string is empty or contains only spaces/punctuation."""
    punct = set(string.punctuation)
    for char in s:
        if char not in punct and char != " ":
            return False
    return True


# ---------------------------------------------------------------------------
# URL filters (ported from DCLM metadata_filters.py)
# ---------------------------------------------------------------------------


def _make_url_substring_filter(
    banlist_path: str,
    exact_domain_match: bool = False,
    ignore_chars: list[str] | None = None,
    num_banned_substrs: int = 1,
    match_substrings: bool = True,
    case_sensitive: bool = False,
) -> Callable[[dict], bool]:
    """Build a URL filter from a ban list file.

    Returns a function that returns True if the page should be *kept*.
    """
    from retrie.retrie import Blacklist

    with open(banlist_path) as f:
        banlist = f.read().splitlines()

    banlist = banlist if case_sensitive else [b.lower() for b in banlist]
    ignore_chars = ignore_chars or []

    if exact_domain_match:
        banset = set(banlist)

        def url_filter(page: dict) -> bool:
            url = urlparse(page.get("url", "")).netloc
            url = url if case_sensitive else url.lower()
            for char in ignore_chars:
                url = url.replace(char, "")
            return url not in banset

    else:
        re_flags = re.IGNORECASE if not case_sensitive else None
        pattern = re.compile(Blacklist(banlist, match_substrings=match_substrings, re_flags=re_flags).compiled)

        def url_filter(page: dict) -> bool:
            url = page.get("url", "")
            url = url if case_sensitive else url.lower()
            for char in ignore_chars:
                url = url.replace(char, "")
            return len(set(pattern.findall(url))) < num_banned_substrs

    return url_filter


# ---------------------------------------------------------------------------
# URL removal modifier (ported from DCLM modifiers.py)
# ---------------------------------------------------------------------------


def _make_url_removal_modifier(tlds_filepath: str) -> Callable[[dict], dict]:
    """Build a modifier that strips URLs from page text."""
    from retrie.retrie import Blacklist

    with open(tlds_filepath) as f:
        tlds_list = [re.escape(tld) for tld in f.read().splitlines()]

    tlds_regex = Blacklist(tlds_list, match_substrings=True).compiled

    url_regex = re.compile(
        rf"\s{{0,10}}(?:((https?|ftp)://))?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}"
        rf"\.({tlds_regex.pattern})\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    )
    ipv4_regex = re.compile(
        r"\s{0,10}\b((https?|ftp)://)?(?:[0-2]?[0-9]{1,2}\.){3}[0-2]?[0-9]{1,2}" r"[-a-zA-Z0-9()@:%_\+.~#?&//=]*"
    )

    def modify(page: dict) -> dict:
        text = page["text"]
        if tlds_regex.match(text):
            text = url_regex.sub("", text)
        text = ipv4_regex.sub("", text)
        page["text"] = text
        return page

    return modify


# ---------------------------------------------------------------------------
# Newline normalization modifier
# ---------------------------------------------------------------------------

_NEWLINE_COLLAPSE_RE = re.compile(r"\n{3,}")


def _newline_removal_modifier(page: dict) -> dict:
    """Collapse 3+ consecutive newlines to 2."""
    page["text"] = _NEWLINE_COLLAPSE_RE.sub("\n\n", page["text"])
    return page


# ---------------------------------------------------------------------------
# Language detection (ported from DCLM language_id_enrichers.py)
# ---------------------------------------------------------------------------


def _fasttext_predict(model, text: str, k: int = 1):
    """Call fasttext predict, working around NumPy 2.x incompatibility.

    The ``fasttext`` package uses ``np.array(probs, copy=False)`` internally,
    which raises ``ValueError`` under NumPy 2.x.  We call the C++ layer
    directly — ``model.f.predict`` returns a list of ``(prob, label)`` tuples —
    and repack the results ourselves using ``np.asarray``.
    """
    import numpy as np

    predictions = model.f.predict(text, k, 0.0, "strict")
    if predictions:
        probs, labels = zip(*predictions)
    else:
        probs, labels = ([], [])
    return list(labels), np.asarray(probs)


def _make_language_enricher(lid_model_path: str) -> Callable[[dict], dict]:
    """Build an enricher that adds ``language_id_whole_page_fasttext`` to each page."""
    import fasttext

    model = fasttext.load_model(lid_model_path)

    def enrich(page: dict) -> dict:
        text = page["text"]
        if _is_space_or_punct(text):
            page["language_id_whole_page_fasttext"] = {}
        else:
            labels, probs = _fasttext_predict(model, text.replace("\n", ""))
            lang = labels[0].replace("__label__", "")
            prob = probs[0]
            page["language_id_whole_page_fasttext"] = {lang: prob}
        return page

    return enrich


def _language_filter(page: dict) -> bool:
    """Keep only English pages with probability > 0.65."""
    lang_info = page.get("language_id_whole_page_fasttext", {})
    return lang_info.get("en", 0.0) > 0.65


# ---------------------------------------------------------------------------
# Content quality filters (ported from DCLM content_filters.py)
# ---------------------------------------------------------------------------


def _page_length_filter(page: dict) -> bool:
    """Keep pages with 50-100,000 words (ignoring punctuation)."""
    word_count = len(_split_words(page["text"], ignore_punctuation=True))
    return 50 <= word_count <= 100_000


def _word_length_filter(page: dict) -> bool:
    """Keep pages with average word length between 3 and 10."""
    words = page["text"].split()
    if not words:
        return False
    avg = sum(len(w) for w in words) / len(words)
    return 3 <= avg <= 10


def _symbol_ratio_filter(page: dict) -> bool:
    """Filter pages where symbol-to-word ratio exceeds 0.1."""
    symbols = ["#", "...", ". . .", "\u2026"]
    num_symbols = sum(page["text"].count(s) for s in symbols)
    num_words = len(page["text"].split())
    if num_words == 0:
        return False
    return num_symbols / num_words <= 0.1


def _bullet_count_filter(page: dict) -> bool:
    """Filter pages where > 90% of lines start with a bullet."""
    lines = _split_paragraphs(page["text"], paragraph_end="\n")
    if not lines:
        return False
    bullet_count = sum(any(line.startswith(b) for b in ["\u25cf", "\u2022", "*", "-"]) for line in lines)
    return bullet_count <= 0.9 * len(lines)


def _ellipsis_count_filter(page: dict) -> bool:
    """Filter pages where > 30% of lines end with an ellipsis."""
    lines = _split_paragraphs(page["text"], paragraph_end="\n")
    if not lines:
        return False
    ellipsis_count = sum(any(line.endswith(e) for e in ["...", ". . .", "\u2026"]) for line in lines)
    return ellipsis_count <= 0.3 * len(lines)


def _alphabetic_word_ratio_filter(page: dict) -> bool:
    """Filter pages where > 20% of words contain no alphabetic character."""
    words = page["text"].split()
    if not words:
        return False
    non_alpha = sum(1 for w in words if not any(c.isalpha() for c in w))
    return non_alpha / len(words) <= 0.2


_STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}


def _stop_word_filter(page: dict) -> bool:
    """Keep pages with >= 2 stop word occurrences (not unique)."""
    count = 0
    for word in page["text"].split():
        if word.lower() in _STOP_WORDS:
            count += 1
            if count >= 2:
                return True
    return False


# ---------------------------------------------------------------------------
# Gopher repetition filters (ported from DCLM content_filters.py)
# ---------------------------------------------------------------------------


def _repetition_filter(
    text: str,
    granularity: str | int,
    max_fraction: float,
    count_characters: bool = True,
    cache: dict | None = None,
) -> bool:
    """Return True if the page passes the repetition check."""
    from nltk import ngrams

    if not text:
        return False

    if cache is None:
        cache = {}

    if isinstance(granularity, str):
        sep = "\n\n" if granularity == "paragraph" else "\n"

        if granularity not in cache:
            cache[granularity] = _split_paragraphs(text, paragraph_end=sep, remove_empty=True)
        segments = cache[granularity]

        if len(segments) <= 1:
            return len(segments) == 1

        if granularity + "/count" not in cache:
            cache[granularity + "/chars"] = sum(len(s) for s in segments)
            cache[granularity + "/count"] = Counter(segments)
        total_chars = cache[granularity + "/chars"]
        segment_counts = cache[granularity + "/count"]

        if count_characters:
            repeated_fraction = sum(len(seg) * cnt for seg, cnt in segment_counts.items() if cnt > 1) / total_chars
        else:
            repeated_fraction = sum(cnt for cnt in segment_counts.values() if cnt > 1) / len(segments)

        return repeated_fraction <= max_fraction

    elif isinstance(granularity, int):
        if "words" not in cache:
            cache["words"] = _split_words(text, ignore_punctuation=True, model="uniseg")
            cache["words/chars"] = sum(len(w) for w in cache["words"])
        words = cache["words"]
        total_chars = cache["words/chars"]

        n_grams = list(ngrams(words, granularity))
        if not n_grams:
            return True

        ngram_counts = Counter(n_grams)
        ordered = ngram_counts.most_common()
        most_common_ngram, most_common_count = ordered[0]
        if most_common_count == 1:
            return True

        if granularity in {2, 3, 4}:
            # Check fraction taken by the most common n-gram
            most_common_length = sum(len(w) for w in most_common_ngram)
            for ng, cnt in ordered:
                if cnt != most_common_count:
                    break
                most_common_length = max(most_common_length, sum(len(w) for w in ng))
            repeated_fraction = (most_common_length * most_common_count) / total_chars
        else:
            # Fraction of characters in any repeated n-gram
            repeated_word_indices: set[int] = set()
            for idx, ng in enumerate(n_grams):
                if ngram_counts[ng] > 1:
                    repeated_word_indices.update(range(idx, idx + granularity))
            repeated_fraction = sum(len(words[i]) for i in repeated_word_indices) / total_chars

        return repeated_fraction <= max_fraction

    raise ValueError(f"granularity must be 'line', 'paragraph', or an int, got {granularity}")


def _massive_web_repetition_filters(page: dict) -> bool:
    """Gopher-style repetition checks across many granularities."""
    text = page["text"]
    cache: dict = {}
    checks = [
        ("line", 0.3, False),
        ("paragraph", 0.3, False),
        ("line", 0.2, True),
        ("paragraph", 0.2, True),
    ]
    for granularity, threshold, count_chars in checks:
        if not _repetition_filter(text, granularity, threshold, count_characters=count_chars, cache=cache):
            return False

    for n, threshold in [
        (2, 0.2),
        (3, 0.18),
        (4, 0.16),
        (5, 0.15),
        (6, 0.14),
        (7, 0.13),
        (8, 0.12),
        (9, 0.11),
        (10, 0.10),
    ]:
        if not _repetition_filter(text, n, threshold, cache=cache):
            return False

    return True


# ---------------------------------------------------------------------------
# Line-level modifiers (ported from DCLM modifiers.py)
# ---------------------------------------------------------------------------


def _word_counter_enricher(page: dict) -> dict:
    """Record word count before line-level modifiers are applied."""
    page["previous_word_count"] = len(_split_words(page["text"], ignore_punctuation=True))
    return page


def _uppercase_ratio_line_modifier(page: dict) -> dict:
    """Remove lines where uppercase characters exceed 50% of line length."""
    lines = page["text"].split("\n")
    kept = []
    for line in lines:
        if not line or sum(c.isupper() for c in line) / len(line) <= 0.5:
            kept.append(line)
    page["text"] = "\n".join(kept).strip()
    return page


def _numeric_ratio_line_modifier(page: dict) -> dict:
    """Remove lines where numeric characters exceed 99.9999% of line length."""
    lines = page["text"].split("\n")
    kept = []
    for line in lines:
        if not line or sum(c.isdigit() for c in line) / len(line) <= 0.999999:
            kept.append(line)
    page["text"] = "\n".join(kept).strip()
    return page


_COUNTER_RE = re.compile(
    r"^\W*\d(?:,|\.|\d)*(?:K|k|M|m|B|b)?\s+"
    r"(?:likes|shares|comments|retweets|reposts|quotes|bookmarks|upvotes|downvotes|downloads|views|followers)\W*$"
)


def _counter_line_modifier(page: dict) -> dict:
    """Remove lines that look like social media counters (e.g. '3 likes')."""
    lines = page["text"].split("\n")
    kept = [line for line in lines if not _COUNTER_RE.search(line.lower())]
    page["text"] = "\n".join(kept).strip()
    return page


def _line_length_modifier(page: dict) -> dict:
    """Remove lines with fewer than 2 words."""
    lines = page["text"].split("\n")
    kept = [line for line in lines if len(line.split()) >= 2 or not line]
    page["text"] = "\n".join(kept)
    return page


def _make_substring_line_modifier(
    banlist: str | list[str],
    location: str = "any",
    max_length: int | None = None,
    remove_substring_only: bool = False,
    case_sensitive: bool = False,
) -> Callable[[dict], dict]:
    """Build a line modifier that removes lines containing banned substrings."""
    if isinstance(banlist, str):
        banlist = [banlist]
    banlist = banlist if case_sensitive else [b.lower() for b in banlist]

    pat = f"(?:{'|'.join(re.escape(b) for b in banlist)})"
    if location == "prefix":
        pat = rf"^{pat}\s?"
    elif location == "suffix":
        pat = rf"\s?{pat}$"
    else:
        pat = rf"\s?{pat}"

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(pat, flags)

    def modify(page: dict) -> dict:
        lines = page["text"].split("\n")
        kept = []
        for line in lines:
            if max_length is not None and len(line.split()) > max_length:
                kept.append(line)
                continue
            if remove_substring_only:
                modified = pattern.sub("", line)
                if line and (not modified or modified.isspace()):
                    continue
                kept.append(modified)
            else:
                if not pattern.search(line):
                    kept.append(line)
        new_doc = "\n".join(kept).strip()
        page["text"] = new_doc
        return page

    return modify


def _word_removal_ratio_filter(page: dict) -> bool:
    """Filter out pages where > 5% of words were removed by line modifiers."""
    prev_count = page.get("previous_word_count", 0)
    if prev_count == 0:
        return False
    new_count = len(_split_words(page["text"], ignore_punctuation=True))
    ratio_removed = (prev_count - new_count) / prev_count
    return ratio_removed <= 0.05


# ---------------------------------------------------------------------------
# FastText quality classifier (ported from DCLM quality_prediction_enrichers)
# ---------------------------------------------------------------------------


def _make_quality_enricher(model_path: str) -> Callable[[dict], dict]:
    """Build an enricher that adds ``fasttext_oh_eli5_vs_rw_v2_prob`` score."""
    import fasttext

    model = fasttext.load_model(model_path)

    def enrich(page: dict) -> dict:
        text = " ".join(page["text"].strip().splitlines())
        labels, probs = _fasttext_predict(model, text)
        pred_label = labels[0]
        hq_prob = probs[0]
        if pred_label == "__label__cc":
            hq_prob = 1 - hq_prob
        page["fasttext_oh_eli5_vs_rw_v2_prob"] = float(hq_prob)
        return page

    return enrich


def _quality_filter(page: dict) -> bool:
    """Keep pages with quality score >= 0.018112."""
    return page.get("fasttext_oh_eli5_vs_rw_v2_prob", 0.0) >= 0.018112


# ---------------------------------------------------------------------------
# Pipeline assembly — one worker-init, one per-record function
# ---------------------------------------------------------------------------


def _download_from_gcs(gcs_path: str, local_dir: str, filename: str) -> str:
    """Download a file from GCS to a local directory. Returns local path."""
    import fsspec

    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        return local_path

    remote_path = os.path.join(gcs_path, filename)
    fs, _, _ = fsspec.get_fs_token_paths(remote_path)

    if not fs.exists(remote_path):
        raise FileNotFoundError(f"Remote file not found: {remote_path}")

    logger.info("Downloading %s -> %s", remote_path, local_path)
    fs.get(remote_path, local_path)
    return local_path


def _init_pipeline(config: DclmFilterConfig, local_dir: str) -> Callable[[dict], list[dict]]:
    """Initialize all models and compiled regexes, returning the combined filter function.

    Called once per worker. The returned function takes a page dict and returns
    ``[page]`` to keep or ``[]`` to drop.
    """
    # Download models and resources from GCS to local temp
    lid_path = _download_from_gcs(config.lid_model_path, local_dir, "lid.176.bin")
    quality_path = _download_from_gcs(config.quality_model_path, local_dir, "fasttext_oh_eli5.bin")

    banlists_dir = os.path.join(local_dir, "banlists")
    os.makedirs(banlists_dir, exist_ok=True)
    banlist_files = [
        "refinedweb_banned_domains_curated.txt",
        "refinedweb_banned_words_strict_reverse_engineered.txt",
        "refinedweb_banned_words_hard_reverse_engineered.txt",
        "refinedweb_banned_words_soft_reverse_engineered.txt",
    ]
    for fname in banlist_files:
        _download_from_gcs(config.banlists_path, banlists_dir, fname)
    tlds_path = _download_from_gcs(config.banlists_path, banlists_dir, "iana_tlds.txt")

    # Build URL filters
    url_filter_curated = _make_url_substring_filter(
        os.path.join(banlists_dir, "refinedweb_banned_domains_curated.txt"),
        exact_domain_match=True,
        ignore_chars=["www"],
    )
    url_filter_strict = _make_url_substring_filter(
        os.path.join(banlists_dir, "refinedweb_banned_words_strict_reverse_engineered.txt"),
        ignore_chars=["-", "."],
    )
    url_filter_hard = _make_url_substring_filter(
        os.path.join(banlists_dir, "refinedweb_banned_words_hard_reverse_engineered.txt"),
        match_substrings=False,
    )
    url_filter_soft = _make_url_substring_filter(
        os.path.join(banlists_dir, "refinedweb_banned_words_soft_reverse_engineered.txt"),
        num_banned_substrs=2,
        match_substrings=False,
    )

    # Build URL removal modifier
    url_removal = _make_url_removal_modifier(tlds_path)

    # Build language enricher
    language_enricher = _make_language_enricher(lid_path)

    # Build substring line modifiers
    items_in_cart_modifier = _make_substring_line_modifier(
        "items in cart",
        max_length=10,
        remove_substring_only=True,
    )
    read_more_modifier = _make_substring_line_modifier(
        "Read more...",
        location="suffix",
        max_length=10,
        remove_substring_only=True,
    )
    sign_in_modifier = _make_substring_line_modifier(
        "Sign-in",
        location="prefix",
        max_length=10,
        remove_substring_only=True,
    )

    # Build quality enricher
    quality_enricher = _make_quality_enricher(quality_path)

    def apply_pipeline(page: dict) -> list[dict]:
        """Apply the full DCLM pipeline to a single page. Returns [page] or []."""
        # --- URL filtering ---
        if not url_filter_curated(page):
            return []
        if not url_filter_strict(page):
            return []
        if not url_filter_hard(page):
            return []
        if not url_filter_soft(page):
            return []

        # --- URL removal from text ---
        page = url_removal(page)
        if not page["text"]:
            return []

        # --- Newline normalization ---
        page = _newline_removal_modifier(page)

        # --- Language detection and filtering ---
        page = language_enricher(page)
        if not _language_filter(page):
            return []

        # --- Heuristic content filters ---
        if not _page_length_filter(page):
            return []
        if not _word_length_filter(page):
            return []
        if not _symbol_ratio_filter(page):
            return []
        if not _bullet_count_filter(page):
            return []
        if not _ellipsis_count_filter(page):
            return []
        if not _alphabetic_word_ratio_filter(page):
            return []
        if not _stop_word_filter(page):
            return []
        if not _massive_web_repetition_filters(page):
            return []

        # --- Line-level modifiers ---
        page = _word_counter_enricher(page)
        page = _uppercase_ratio_line_modifier(page)
        page = _numeric_ratio_line_modifier(page)
        page = _counter_line_modifier(page)
        page = _line_length_modifier(page)
        page = items_in_cart_modifier(page)
        page = read_more_modifier(page)
        page = sign_in_modifier(page)

        if not page["text"]:
            return []
        if not _word_removal_ratio_filter(page):
            return []

        # --- FastText quality classifier ---
        page = quality_enricher(page)
        if not _quality_filter(page):
            return []

        # Strip internal metadata before writing output
        for key in ["language_id_whole_page_fasttext", "previous_word_count", "fasttext_oh_eli5_vs_rw_v2_prob"]:
            page.pop(key, None)

        return [page]

    return apply_pipeline


# Module-level cache for lazy per-worker initialization. Each worker process
# initializes models/regexes once on first record, then reuses for all
# subsequent records. This avoids passing unpicklable objects (fasttext C++
# models, compiled regex closures) through Ray serialization.
_worker_pipeline_cache: dict[str, Callable[[dict], list[dict]]] = {}


def _get_or_init_pipeline(config: DclmFilterConfig) -> Callable[[dict], list[dict]]:
    """Return the cached pipeline, initializing on first call per worker process."""
    cache_key = f"{config.lid_model_path}|{config.quality_model_path}|{config.banlists_path}"
    if cache_key not in _worker_pipeline_cache:
        local_dir = tempfile.mkdtemp(prefix="dclm_resources_")
        logger.info("Initializing DCLM pipeline on worker (downloading models to %s)", local_dir)
        _worker_pipeline_cache[cache_key] = _init_pipeline(config, local_dir)
        logger.info("DCLM pipeline initialization complete")
    return _worker_pipeline_cache[cache_key]


def _apply_dclm_pipeline(record: dict) -> list[dict]:
    """Per-record function called by Zephyr. Lazy-inits the pipeline on first call."""
    from zephyr import zephyr_worker_ctx

    ctx = zephyr_worker_ctx()
    config: DclmFilterConfig = ctx.get_shared("dclm_config")
    pipeline_fn = _get_or_init_pipeline(config)
    return pipeline_fn(record)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def dclm_filter(config: DclmFilterConfig) -> None:
    """Apply the full DCLM-Baseline filtering pipeline.

    Reads plain-text JSONL records, applies all DCLM filters (URL filtering,
    language detection, heuristic content quality filters, line-level modifiers,
    FastText quality classification), and writes surviving records as JSONL.

    Also writes ``dclm_filter_stats.json`` with input/output document counts.
    """
    import fsspec

    pipeline = (
        Dataset.from_files(config.input_path)
        .flat_map(load_file)
        .flat_map(_apply_dclm_pipeline)
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    # Pass only the serializable config dataclass (all string fields) to workers.
    # Each worker lazily initializes models/regexes on first record via
    # _get_or_init_pipeline(), avoiding pickle of unpicklable C++ objects.
    with ZephyrContext(name="dclm-filter") as ctx:
        ctx.put("dclm_config", config)
        output_files = ctx.execute(pipeline)

    # Count output documents
    output_count = 0
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    for fpath in fs.glob(f"{config.output_path}/data-*.jsonl.gz"):
        with fs.open(fpath, "rb") as f:
            import gzip

            with gzip.open(f, "rt") as gz:
                for _line in gz:
                    output_count += 1

    stats = {
        "output_files": len(output_files),
        "output_documents": output_count,
    }

    stats_path = f"{config.output_path}/dclm_filter_stats.json"
    with fs.open(stats_path, "w") as f:
        f.write(json.dumps(stats, indent=2))

    logger.info(
        "DCLM filtering complete: %d output files, %d documents written to %s",
        len(output_files),
        output_count,
        config.output_path,
    )
