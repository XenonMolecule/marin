# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Degeneration defense, shared by both runners (single source of truth).

The class (MR-caught 2026-07-20): greedy no-think transcription of repetitive
input collapses into a repeated line, invisible to mean-lev and absent from
gold. Zoo-measured resolution (141 docs): no-retry 33 amplified / max x5074;
think-retry 4 / x1030; ladder (think -> t0.3) 1 / x22 — with mean gold-lev
IMPROVING as loops die (.717 -> .739 -> .743). Truncation converts any
survivor into a shortened doc: catastrophic loops structurally impossible.
"""
from __future__ import annotations

import collections
import re

THRESHOLD = 20

# Structural markdown lines are legitimately repeated in the output while
# absent from the HTML input (a 10-code-block page emits 20+ identical ```
# lines; pipe-table separators repeat per table). Counting them as loops
# truncated healthy markdown extractions (holzmann-cfd: complete 18k-char
# output cut to 2.2k at the first fences). They cannot be degeneration
# carriers: they hold no content.
_MD_STRUCTURAL_RE = re.compile(r"^\s*(```[\w+-]*|\$\$|---+|\*\*\*+|___+|\|[\s\-:|]+\|?)\s*$")


# Inline markdown is something WE add: a thread title emitted as
# `**Re: Science laws and theorems**` on all 24 posts appears 24 times in the
# output and ZERO times in the source HTML, which reads as amplification 24
# against a threshold of 20. That truncated a complete 14,162-char forum
# extraction to 1,902 chars (mathisfunforum, MR-caught 2026-07-27). Compare
# de-marked text so our own formatting cannot look like a loop.
_MD_MARKS_RE = re.compile(r"(\*\*|__|\*|_|`|~~|^#{1,6}\s+|^>\s?|^[-*+]\s+)", re.M)


def demark(line: str) -> str:
    return _MD_MARKS_RE.sub("", line or "").strip()


def _content_lines(text: str) -> list[str]:
    return [l for l in (text or "").split("\n") if len(l.strip()) >= 2 and not _MD_STRUCTURAL_RE.match(l)]


def worst_line_amplification(out_text: str, in_text: str) -> int:
    """Max count of any output line repeated beyond its input count.

    Structural markdown lines (fences, $$, hr, table separators) are ignored —
    they repeat legitimately and carry no content."""
    lines = _content_lines(out_text)
    if not lines:
        return 0
    line, n = collections.Counter(lines).most_common(1)[0]
    return n - (in_text or "").count(demark(line)[:60])


def truncate_loops(text: str, chunk_text: str, threshold: int = THRESHOLD) -> str:
    """Verdict-safe last resort: cut output at loop onset (verdicts live at the
    start of the output, so truncating a later loop can never flip one)."""
    if worst_line_amplification(text, chunk_text) < threshold:
        return text
    lines = (text or "").split("\n")
    counted = collections.Counter(_content_lines(text))
    loop_line = counted.most_common(1)[0][0]
    seen = 0
    for idx, l in enumerate(lines):
        if l == loop_line:
            seen += 1
            if seen > max((chunk_text or "").count(demark(loop_line)[:60]), 3):
                return "\n".join(lines[:idx]).rstrip()
    return text
