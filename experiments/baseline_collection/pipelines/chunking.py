# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split long HTML docs into token-budgeted chunks at natural breakpoints.

Used when a doc exceeds the extractor's context window: instead of emitting
[CONTEXT_LENGTH_EXCEEDED], split the (preprocessed) body into chunks of at most
`max_tokens`, extract each chunk, and merge the outputs.

The chunk budget is a TUNABLE: context_window ≈ chunk_tokens (input HTML)
+ spec tokens + prompt overhead + completion budget (reasoning + output).
Dedicating more budget to reasoning/output means smaller chunks, and vice versa.

Split-point policy (never inside <pre>/<code>/protected spans):
  tier 1 — after block-level closers (</p>, </div>, </section>, </table>, ...),
           <hr>, or a blank line: real structural boundaries.
  tier 2 — after row/item closers (</tr>, </li>, </dd>, ...) or <br>: used when
           a single block (e.g. a giant table) exceeds the budget by itself.
  fallback — newline / whitespace / hard character cut, in that order, only
           when a protected or unbroken span alone exceeds the budget.

Chunks are contiguous slices of the input: "".join(c.text for c in chunks)
reconstructs the document exactly.

Efficiency: one regex pass for breakpoints + protected spans, one tokenizer
pass over the doc (each inter-breakpoint segment encoded once, prefix-summed),
then greedy chunk assembly aiming at balanced chunk sizes under the cap.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

# Spans we never split inside. (<script>/<style> usually pre-stripped; kept for safety.)
_PROTECT_TAGS = "pre|code|textarea|script|style|svg|math"
_PROTECT_RE = re.compile(rf"<({_PROTECT_TAGS})\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)

_TIER1_BLOCK_TAGS = (
    "p|div|section|article|aside|main|header|footer|nav|form|fieldset|"
    "table|ul|ol|dl|blockquote|figure|details|h[1-6]|pre"
)
_TIER1_RE = re.compile(rf"(?:</(?:{_TIER1_BLOCK_TAGS})\s*>|<hr\b[^>]*/?>|\n\s*\n)", re.IGNORECASE)
_TIER2_RE = re.compile(r"(?:</(?:tr|li|dt|dd|option|thead|tbody)\s*>|<br\b[^>]*/?>)", re.IGNORECASE)

TokenCounter = Callable[[str], int]


@dataclass
class Chunk:
    text: str
    start: int  # char offset in the input
    end: int
    est_tokens: int
    split_kind: str  # boundary kind that ENDS this chunk: tier1|tier2|fallback|eof


def _protected_intervals(html: str) -> list[tuple[int, int]]:
    """Merged (start, end) spans of protected elements."""
    spans = [(m.start(), m.end()) for m in _PROTECT_RE.finditer(html)]
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _in_protected(pos: int, intervals: list[tuple[int, int]]) -> bool:
    i = bisect.bisect_right(intervals, (pos, float("inf"))) - 1
    return i >= 0 and intervals[i][0] < pos < intervals[i][1]


def _breakpoints(html: str) -> list[tuple[int, int]]:
    """Sorted (pos, tier) split positions, outside protected spans.

    pos is a char offset one may cut AT (end of the matched boundary).
    """
    protected = _protected_intervals(html)
    points = {}
    for tier, regex in ((1, _TIER1_RE), (2, _TIER2_RE)):
        for m in regex.finditer(html):
            pos = m.end()
            if _in_protected(pos, protected):
                continue
            # keep the strongest tier at a given position
            if pos not in points or tier < points[pos]:
                points[pos] = tier
    return sorted(points.items())


def _char_cuts(text: str, budget_chars: int) -> list[int]:
    """Relative cut positions ~budget_chars apart: prefer newline, then
    whitespace, else hard cut."""
    cuts = []
    pos = 0
    n = len(text)
    while n - pos > budget_chars:
        window_end = pos + budget_chars
        nl = text.rfind("\n", pos + budget_chars // 2, window_end)
        ws = -1 if nl != -1 else _last_ws(text, pos + budget_chars // 2, window_end)
        cut = nl + 1 if nl != -1 else (ws + 1 if ws != -1 else window_end)
        cuts.append(cut)
        pos = cut
    return cuts


def _fallback_split(text: str, offset: int, budget: int, count_tokens: TokenCounter) -> list[int]:
    """Absolute cut positions guaranteeing every piece ≤ budget tokens (verified
    with the real tokenizer, not a char estimate — token density varies)."""
    cuts: list[int] = []
    pending: list[tuple[int, int]] = [(0, len(text))]
    while pending:
        lo, hi = pending.pop()
        tokens = count_tokens(text[lo:hi])
        if tokens <= budget:
            continue
        chars_per_tok = max(1.0, (hi - lo) / tokens)
        budget_chars = max(1, int(budget * chars_per_tok * 0.9))
        rel = _char_cuts(text[lo:hi], budget_chars)
        if not rel:  # estimate says fits but count says no: force a midpoint cut
            rel = [(hi - lo) // 2]
        cuts.extend(offset + lo + c for c in rel)
        # verify each resulting piece; over-budget ones get re-split
        edges = [lo] + [lo + c for c in rel] + [hi]
        pending.extend(pairwise(edges))
    return sorted(set(cuts))


def _last_ws(text: str, lo: int, hi: int) -> int:
    for i in range(hi - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def chunk_html(
    html: str,
    max_tokens: int,
    count_tokens: TokenCounter,
    headroom: float = 0.98,
) -> list[Chunk]:
    """Split html into chunks of ≈≤ max_tokens (est.) at natural breakpoints.

    Per-segment token counts are estimated independently then summed, so chunk
    estimates can drift a few tokens from an exact re-encode; `headroom` shrinks
    the working budget to absorb that. A doc within budget returns one chunk.
    """
    if not html:
        return []
    budget = max(1, int(max_tokens * headroom))
    total = count_tokens(html)
    if total <= budget:
        return [Chunk(html, 0, len(html), total, "eof")]

    points = _breakpoints(html)
    # Segment boundaries: 0, breakpoints..., len(html)
    bounds = [0] + [p for p, _ in points] + [len(html)]
    tiers = {p: t for p, t in points}
    seg_tokens = [count_tokens(html[bounds[i] : bounds[i + 1]]) for i in range(len(bounds) - 1)]

    # Oversized segments get internal fallback cuts (protected/unbroken spans).
    aug_bounds = [0]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if seg_tokens[i] > budget:
            for cut in _fallback_split(html[s:e], s, budget, count_tokens):
                aug_bounds.append(cut)
                tiers[cut] = 3
        aug_bounds.append(e)
    bounds = sorted(set(aug_bounds))
    seg_tokens = [count_tokens(html[bounds[i] : bounds[i + 1]]) for i in range(len(bounds) - 1)]

    # prefix[i] = tokens in html[:bounds[i]]
    prefix = [0]
    for t in seg_tokens:
        prefix.append(prefix[-1] + t)

    # Balanced target: even chunks under the cap beat maximal chunks + tiny tail.
    n_chunks = -(-prefix[-1] // budget)  # ceil
    target = prefix[-1] / n_chunks

    chunks: list[Chunk] = []
    start_i = 0  # index into bounds
    while start_i < len(bounds) - 1:
        start_tok = prefix[start_i]
        # candidates: bound indices j>start_i with prefix[j]-start_tok <= budget
        hi = bisect.bisect_right(prefix, start_tok + budget) - 1
        hi = max(hi, start_i + 1)  # always advance, even if a segment overflows
        if bounds[hi] >= len(html):
            j = len(bounds) - 1
        else:
            # among candidates, prefer tier1 nearest the balanced target, then tier2, then any
            want = start_tok + target
            js = range(start_i + 1, hi + 1)
            j = None
            for tier_pref in (1, 2, None):
                cand = [k for k in js if tier_pref is None or tiers.get(bounds[k]) == tier_pref]
                if cand:
                    j = min(cand, key=lambda k: abs(prefix[k] - want))
                    break
        s_char, e_char = bounds[start_i], bounds[j]
        kind = (
            "eof" if e_char >= len(html) else {1: "tier1", 2: "tier2", 3: "fallback"}.get(tiers.get(e_char), "fallback")
        )
        chunks.append(Chunk(html[s_char:e_char], s_char, e_char, prefix[j] - start_tok, kind))
        start_i = j
    return chunks
