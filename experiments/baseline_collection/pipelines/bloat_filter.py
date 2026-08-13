# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mechanical over-extraction filters applied to per-chunk extraction output.

The bloat front (index walls, tag clouds, headline feeds) survived 22+ spec
wordings and a full distillation round: the model cannot reliably tell a wall
of tag names from list content when a chunk contains nothing else. These
filters catch the shapes structurally, after generation, where the signature is
unambiguous.

Measured on the 420-doc devset gold set (2026-07-26): the counted-index
signature fires on exactly 2 documents — the two sullivanlaw archives that
over-extract 32x and 13.6x — and on zero others, so the false-positive risk is
nil. Filters are per-chunk: a document keeps every chunk whose output is real
content.
"""

from __future__ import annotations

import re

_COUNTED = re.compile(r"\(\d+\)\s*$")
_DATE_ONLY = re.compile(
    r"^\s*(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}?,?\s*\d{4}\s*$",
    re.IGNORECASE,
)
_SENTENCE_END = (".", "?", "!", '."', '?"', '!"', ".*", "*")


def _lines(text: str) -> list[str]:
    return [l.strip() for l in (text or "").splitlines() if len(l.strip()) > 2]


def counted_index_fraction(text: str) -> float:
    """Share of lines shaped like index entries: 'Some Topic Name (3)'."""
    lines = _lines(text)
    if len(lines) < 10:
        return 0.0
    return sum(1 for l in lines if _COUNTED.search(l)) / len(lines)


def headline_feed_fraction(text: str) -> float:
    """Share of lines that are a bare date or a short non-sentence headline.

    A "latest posts"/archive feed alternates headline and date lines, neither
    ending in sentence punctuation. Real prose fails this test immediately.
    """
    lines = _lines(text)
    if len(lines) < 12:
        return 0.0
    hits = 0
    for l in lines:
        if _DATE_ONLY.match(l):
            hits += 1
        elif len(l) < 90 and not l.endswith(_SENTENCE_END) and not l.startswith(("|", "```", "#", "$")):
            hits += 1
    return hits / len(lines)


def is_bloat_chunk(text: str, counted_thresh: float = 0.7, feed_thresh: float | None = None) -> str | None:
    """Return the filter name if this chunk's output is an index wall.

    Only `counted_index` is enabled by default. The headline_feed heuristic was
    measured across the 420-doc gold set and REJECTED: it strips real content
    (booksie 1,863 -> 441 chars against 9,947 gold; 9.5k of a bls release), and
    causing under-extraction to prevent over-extraction is the wrong trade —
    over-extraction is the tolerated failure, never the reverse. Pass
    feed_thresh explicitly to experiment with it.
    """
    if len(text or "") < 500:
        return None
    if counted_index_fraction(text) >= counted_thresh:
        return "counted_index"
    if feed_thresh is not None and headline_feed_fraction(text) >= feed_thresh:
        return "headline_feed"
    return None


# A `<script>` payload copied verbatim into the output: quoted ID arrays and
# similar machine data. One Forbes article carried ~100k chars of
# `"5002863","5002865",...` this way, dwarfing its 4k of real text.
# Validated against all 420 gold devset extractions: it removes zero gold lines.
def is_data_line(line: str, min_len: int = 200, max_letter: float = 0.15) -> bool:
    s = (line or "").strip()
    if len(s) < min_len:
        return False
    if sum(c.isalpha() for c in s) / len(s) > max_letter:
        return False
    struct = sum(c in '",;:[]{}|0123456789 \t.-_' for c in s)
    return struct / len(s) >= 0.9


def strip_data_lines(text: str) -> str:
    """Drop machine-data lines from an extraction, keeping everything else."""
    if not text:
        return text
    lines = text.splitlines()
    kept = [l for l in lines if not is_data_line(l)]
    return "\n".join(kept) if len(kept) != len(lines) else text


# A tag cloud or A-Z index rendered as hundreds of bare short lines. One
# felinedocs tag page carried 467 such lines (`abdomen` ... `younger`) around
# 1.3k of real article. Alphabetical ordering across a long run of short,
# unpunctuated lines is the signature: prose is never sorted, and speaker lines
# end in `:` so forum threads are untouched.
# Validated against all 420 gold devset extractions: it removes zero gold lines.
def _bare_index_line(line: str, max_len: int = 50) -> bool:
    s = (line or "").strip()
    return (
        0 < len(s) <= max_len
        and not s.endswith((".", "!", "?", ":", ";", ","))
        and not s.startswith(("#", ">", "|", "```", "- [", "*", "1."))
    )


def index_runs(text: str, min_run: int = 25, sorted_frac: float = 0.8) -> list[tuple[int, int]]:
    lines = (text or "").splitlines()
    runs, i = [], 0
    while i < len(lines):
        if not _bare_index_line(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and (_bare_index_line(lines[j]) or not lines[j].strip()):
            j += 1
        block = [l.strip().lower() for l in lines[i:j] if l.strip()]
        if len(block) >= min_run:
            asc = sum(1 for a, b in zip(block, block[1:]) if a <= b)
            if asc / max(1, len(block) - 1) >= sorted_frac:
                runs.append((i, j))
        i = j
    return runs


def strip_index_runs(text: str) -> str:
    runs = index_runs(text)
    if not runs:
        return text
    drop = {i for a, b in runs for i in range(a, b)}
    return "\n".join(l for k, l in enumerate(text.splitlines()) if k not in drop)


# The model narrating the spec instead of obeying it: "According to the
# extraction spec, we should extract the main content... Thus we output
# nothing." — written *instead of* outputting nothing. Rare (2 of 419 docs) but
# it is the model talking to itself inside the product.
# Validated against all 420 gold extractions: it removes zero gold paragraphs.
_META = re.compile(
    r"\b(?:the spec (?:also |further )?(?:says|states|requires)|per the spec"
    r"|according to the spec|the instructions? (?:say|state)"
    r"|we (?:should|must|cannot|can't|will|need to) (?:extract|output|produce|include|omit)"
    r"|thus we output|so we output|we output nothing"
    r"|there is no (?:main )?content to extract"
    r"|this (?:section|chunk) (?:is|holds|contains) only)\b",
    re.I,
)


def strip_meta_paragraphs(text: str, max_words: int = 160) -> str:
    """Drop paragraphs that talk about the extraction rather than doing it.

    The word cap keeps the filter off any paragraph long enough to also be
    carrying real page content.
    """
    if not text or not _META.search(text):
        return text
    kept = [para for para in text.split("\n\n") if not (_META.search(para) and len(para.split()) < max_words)]
    return "\n\n".join(kept)


def clean_output(text: str) -> str:
    """All mechanical output filters, applied to a merged extraction."""
    return strip_trailing_feed(strip_meta_paragraphs(strip_index_runs(strip_data_lines(strip_entity_runs(text)))))


# A feed of other articles appended after the piece ends, written as bare
# headline/date pairs with no heading to introduce it (gaylaxymag: a 496-char
# poem followed by 1.9k of other headlines). The spec's "where to stop" rule
# cannot reach this one -- a spec variant that keys on an introducing heading
# fixed techtarget's furniture (CLEAN 0.367 -> 0.977) and left this untouched,
# because there is no heading to key on. The date pairing is the only signal.
# Validated against all 420 gold extractions: it removes zero gold lines.
_FEED_DATE = re.compile(
    r"^\**\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\s*\**$"
    r"|^\**\s*\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*\**$",
    re.I,
)


def trailing_feed_start(text: str, min_pairs: int = 4) -> int | None:
    """First line of a trailing headline/date feed, or None."""
    lines = (text or "").splitlines()
    idx = [i for i, l in enumerate(lines) if _FEED_DATE.match(l.strip())]
    if len(idx) < min_pairs:
        return None
    best, run = None, [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if 1 <= b - a <= 3:
            run.append(b)
        else:
            if len(run) >= min_pairs and (best is None or len(run) > len(best)):
                best = run
            run = [b]
    if len(run) >= min_pairs and (best is None or len(run) > len(best)):
        best = run
    if not best:
        return None
    start = best[0]
    while start > 0 and lines[start - 1].strip():  # the headline above the first date
        start -= 1
    # only strip when the feed runs to the end: a dated list mid-article is content
    if len([l for l in lines[best[-1] + 1 :] if l.strip()]) > 3:
        return None
    return start


def strip_trailing_feed(text: str) -> str:
    s = trailing_feed_start(text)
    return "\n".join((text or "").splitlines()[:s]).rstrip() if s is not None else text


# Runs of HTML entities copied verbatim out of a page's markup. On a 3dmdb
# product page this was 19,540 of the extraction's 30,005 chars -- 65% of the
# output was `&#39;&#39;&#39;...`. Rare (1 doc in WMB en-dev, 1 in the devset)
# but it destroys the document it lands on, and it is unambiguous: neither gold
# corpus contains a single run of this shape (0/545 WMB, 0/420 devset).
_ENTITY_RUN = re.compile(r"(?:&#?\w{1,8};){8,}")


def strip_entity_runs(text: str) -> str:
    return _ENTITY_RUN.sub(" ", text) if text else text
