# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose the preprocessing view: does this page keep its content in scripts?

`--preprocess_mode aggressive` strips every <script> tag. That is right for the
overwhelming majority of pages, where scripts hold tracking, ads and hydration
payloads that merely duplicate the rendered body. It is catastrophic for the
few that render their content from those payloads: patheos' comment thread
(4% of gold survives aggressive), khanacademy transcripts (10%), searxiv
search results (34-52%).

Measured over the 40 devset docs where aggressive discards >100k chars of text,
only **5** actually lose gold. The other 35 keep >=0.93 of it, and switching
them to the safe view is expensive -- blanket switching measured -0.126 mean
fidelity, wrecking pages like keralapscblog (CLEAN 0.900 -> 0.038) whose
scripts are pure noise.

So the whole problem is telling the 5 from the 35, and the signals that look
plausible do not work:

  * view size ratio   -- forbes/sap carries a 397k JSON id array, the largest
                         script payload in the set, and needs nothing from it.
  * prose-likeness of the added text -- quizlet's flashcard JSON reads exactly
                         like prose and scores highest of any document, but its
                         cards are already in the body.
  * any test on the OUTPUT -- cannot work in principle. The extraction
                         faithfully covers the view it was handed, so a starved
                         doc looks healthy downstream. Detection has to happen
                         at the view.

What separates them is **novelty, not volume**: on the 35, the scripts repeat
what the body already says; on the 5, they carry writing that exists nowhere
else. Counting sentences in the safe view that are absent from the aggressive
view rescues 5/5 with 1/35 false positives, where every measure above scored
near chance.
"""
from __future__ import annotations

import re

_TAG = re.compile(r"<[^>]+>")

# 5/5 recall with a single false positive across the 40-doc audit. The counts
# are far apart either side of it (the true cases run 284-3,631 new sentences;
# all but one of the 35 negatives sit under 140), so the exact cut is not
# load-bearing.
NEW_SENTENCE_THRESHOLD = 200
MIN_WORDS = 8


def _sentences(text: str, min_words: int = MIN_WORDS) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " \n ", text or "")
    parts = re.split(r"(?<=[.!?])\s+|\n", flat)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def _norm(text: str) -> str:
    t = re.sub(r"[^\w\s]", " ", _TAG.sub(" ", text or ""))
    return re.sub(r"\s+", " ", t).strip().lower()


def _shingles(text: str, n: int = 5) -> set:
    w = text.split()
    if len(w) < n:
        return {tuple(w)} if w else set()
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def new_sentence_count(aggressive_html: str, safe_html: str) -> int:
    """Sentences the safe view carries that the aggressive view does not.

    Sentences are split before normalizing: `_norm` removes the very
    punctuation the split depends on, so normalizing first collapses the whole
    document into a single "sentence".
    """
    base = _shingles(_norm(aggressive_html))
    if not base:
        return 0
    new = 0
    for s in _sentences(_TAG.sub(" ", safe_html or "")):
        sh = _shingles(_norm(s))
        if sh and len(sh & base) / len(sh) < 0.5:
            new += 1
    return new


def scripts_carry_content(aggressive_html: str, safe_html: str, threshold: int = NEW_SENTENCE_THRESHOLD) -> bool:
    """True when the page's writing lives in its scripts, not its body."""
    return new_sentence_count(aggressive_html, safe_html) >= threshold
