# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect an extraction that turned into a summary, and re-extract it smaller.

A seventh silent-stop mechanism, distinct from the six the other guards cover.
Handed a large window, the model stops transcribing and starts *writing about*
the page: it consumes the whole input, reaches the document's real ending, and
compresses everything between. lectlaw is the type specimen — 10,830 chars
against 40,923 of gold, output ending on the page's true last line
("THE END. / Brought to you by ..."), 2,691 completion tokens against a 12,288
budget. Nothing overflowed, nothing was truncated, nothing looped.

It is invisible to the other guards by construction: the output ends cleanly
(loop guard sees nothing), the budget is untouched (length bisect sees nothing),
and the coverage guard, which does fire, prescribes a t=0.3 reroll — which comes
back equally short, because summarizing is what the model does with that window,
not a sampling accident.

Two signals identify it, and neither needs gold:

  COVERAGE   output much shorter than the chunk's measured content
  INVENTION  a high rate of n-grams absent from the input itself

The second is what makes the trigger precise. Summarizing is paraphrase, so it
*writes new text*: lectlaw's shipped output carries 156 invented spans (rate
0.607) against a corpus mean of 0.065. Coverage alone cannot separate a
summarized page from a page that is genuinely mostly boilerplate; measured over
the 46 flagged docs, re-chunking on coverage alone is a wash (dFIDELITY -0.0008:
+0.098 KEPT bought with -0.131 CLEAN, because a smaller window has less context
for judging what is furniture). Adding the invention test picks out the real
cases: 10 docs, mean dFIDELITY +0.179.

On lectlaw the re-extraction fixes content and invention together:

    1 chunk    10,830 chars   KEPT 0.040   CLEAN 0.149   156 invented spans
    6k chunks  39,092 chars   KEPT 0.881   CLEAN 0.924    76 invented spans

The threshold is fit on 46 documents, so it is deliberately taken off the
argmax (0.10) and set at a flatter point; the curve runs +0.0034 to +0.0043
across 0.08-0.16, so the exact cut is not load-bearing.
"""
from __future__ import annotations

import re

from experiments.baseline_collection.pipelines.coverage_guard import expected_content_chars

_TAG = re.compile(r"</?[a-zA-Z!][^<>]{0,2000}>", re.S)
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

INVENTION_THRESHOLD = 0.12
COVERAGE_RATIO = 0.45


def _norm(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", (text or ""))).strip().lower()


def invention_rate(output: str, source_html: str, n: int = 6) -> float:
    """Share of the output made of n-grams that are not in its own input.

    Measured against the input the model was given, not the raw page: the model
    can only copy from what it saw, so anything else it produced, it wrote.
    Checked line by line, because scanning the flattened output reports every
    place the extraction's field order differs from the source's -- those are
    reorderings, not inventions.
    """
    src = f" {_norm(_TAG.sub(' ', source_html))} "
    words = _norm(output).split()
    if len(words) < n:
        return 0.0
    novel = 0
    for line in (output or "").splitlines():
        w = _norm(line).split()
        if len(w) < n:
            continue
        i = 0
        while i <= len(w) - n:
            if f" {' '.join(w[i:i + n])} " not in src:
                novel += 1
                i += n
            else:
                i += 1
    return novel * n / len(words)


def looks_summarized(
    output: str,
    source_html: str,
    invention_threshold: float = INVENTION_THRESHOLD,
    coverage_ratio: float = COVERAGE_RATIO,
    min_expected: int = 2000,
) -> bool:
    """True when the extraction reads like a summary of its input."""
    text = (output or "").strip()
    if not text:
        return False  # empty is the blank-doc guards' problem
    expected = expected_content_chars(source_html)
    if expected < min_expected or len(text) >= coverage_ratio * expected:
        return False
    return invention_rate(text, source_html) >= invention_threshold


MIN_GROWTH = 2.5


def better_extraction(original: str, retry: str, source_html: str, min_growth: float = MIN_GROWTH) -> bool:
    """Keep the re-extraction only if it recovers a document's worth of content.

    Accepting any longer retry was measured wrong: over the six docs the guard
    fired on, it took two large wins (+0.653, +0.634) and four losses, netting
    +0.523 where the wins alone were +1.287. The losses share a shape — the
    original was already at or past gold length, so re-chunking added furniture
    rather than recovering text.

    Growth separates them cleanly. A genuinely truncated extraction roughly
    triples (2.99x, 3.35x); every false positive grew 2.11x or less. If a
    smaller window barely changes the length, the original was not truncated.

    An invention test does NOT work here and was tried: it accepts the wsj
    regression (-0.236, invention fell 46%) and rejects gonzalo123 (+0.634,
    invention rose 30%), because a paraphrased document and a padded one both
    move invention in the same direction.

    Fitted on six documents, so the threshold is soft; the gap between 2.99x
    and 2.11x is what it rests on.
    """
    o, r = (original or "").strip(), (retry or "").strip()
    if not r or not o:
        return False
    if len(r) < min_growth * len(o):
        return False
    return invention_rate(r, source_html) <= invention_rate(o, source_html) + 0.06
