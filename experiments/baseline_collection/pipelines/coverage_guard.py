# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Coverage guard v2 — expectation-based early-stop detection.

v1 compared output length against the chunk's raw text (tags stripped) with a
15% floor. That misses the common case: mathisfunforum pid=31619 emitted 2,155
chars where the page holds ~10k of real content, but 2,155 was just above 15%
of the raw text, so no reroll fired — and the doc scored 0.179 in one run
versus 0.907 in another. The early-stop lottery is the single largest source of
run-to-run variance in the campaign.

v2 estimates how much CONTENT the chunk actually holds (resiliparse main-text
rendering, falling back to stripped tags) and fires when the extraction returns
much less than that. Comparing against content rather than markup makes the
threshold meaningful across page types.

Selection is also fixed: v1 kept whichever attempt was longer, which let a
degenerate reroll win. v2 keeps the longer attempt only when it is not
loop-degenerate.
"""

from __future__ import annotations

import re

from experiments.baseline_collection.pipelines.loop_guard import THRESHOLD, worst_line_amplification

_TAG_RE = re.compile(r"<[^>]+>")


def expected_content_chars(html_chunk: str) -> int:
    """Rough size of the readable content in this chunk."""
    try:
        from resiliparse.extract.html2text import extract_plain_text

        text = extract_plain_text(html_chunk, main_content=False, alt_texts=False)
    except Exception:
        text = _TAG_RE.sub(" ", html_chunk or "")
    return len(re.sub(r"\s+", " ", text).strip())


def looks_early_stopped(html_chunk: str, output: str, min_expected: int = 2000, ratio: float = 0.45) -> bool:
    """True if the output is far smaller than the chunk's readable content.

    Deliberately more sensitive than v1 (0.45 of content vs 0.15 of markup):
    extraction legitimately drops boilerplate, but dropping more than half of a
    content-rich chunk is the early-stop signature. Costs one extra call on the
    pages it flags.
    """
    expected = expected_content_chars(html_chunk)
    if expected < min_expected:
        return False
    return len((output or "").strip()) < ratio * expected


def better_attempt(original: str, retry: str, html_chunk: str) -> bool:
    """Should the retry replace the original?

    Longer wins, but only if the retry is not a degeneration: a looping reroll
    is longer and worse, and v1's naive length rule would have kept it.
    """
    o, r = (original or "").strip(), (retry or "").strip()
    if len(r) <= len(o) * 1.15:
        return False
    return worst_line_amplification(r, html_chunk) < THRESHOLD
