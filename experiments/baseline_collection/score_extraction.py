# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Objectively score an extraction against the source page's reading order.

When building a gold extraction by sampling an LLM K times, we need to pick the BEST sample rather
than trust one. The dominant failure mode observed is silent paragraph reordering (an LLM relocating
a paragraph), which is a correctness bug, not a formatting choice. Since the raw HTML gives the true
reading order, we can score each sample directly:

  * order fidelity — how many paragraph pairs are out of source order (0 = perfect),
  * coverage      — how many of the sample's paragraphs actually trace back to the source
                    (guards against over-cropping / hallucination),
  * cleanliness   — undecoded entities / mojibake / replacement chars remaining.

`select_best` ranks samples by (fewest inversions, most covered paragraphs, cleanest).
"""

from __future__ import annotations

import html as _html
import re

_TAGS = re.compile(r"(?s)<[^>]+>")
_SCRIPT = re.compile(r"(?is)<(script|style).*?</\1>")
_WS = re.compile(r"\s+")


def source_linear(raw_html: str) -> str:
    """Reconstruct the page's linear reading order: strip tags, decode entities, normalize whitespace."""
    s = _SCRIPT.sub(" ", raw_html or "")
    s = _TAGS.sub(" ", s)
    s = _html.unescape(s)
    return _WS.sub(" ", s).strip()


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if len(p.strip()) > 40]


def _positions(text: str, src: str) -> list[int]:
    """Source-offset of each paragraph's first ~6 words; -1 if not found."""
    out = []
    for p in _paragraphs(text):
        anchor = " ".join(_WS.sub(" ", p).split()[:6])
        out.append(src.find(anchor))
    return out


def order_inversions(text: str, src: str) -> tuple[int, int]:
    """(#out-of-order paragraph pairs, #paragraphs matched in source)."""
    pos = [p for p in _positions(text, src) if p != -1]
    inv = sum(1 for a in range(len(pos)) for b in range(a + 1, len(pos)) if pos[b] < pos[a])
    return inv, len(pos)


def cleanliness_issues(text: str) -> int:
    """Count residual encoding problems (want 0): undecoded entities, mojibake, replacement chars."""
    n = 0
    n += len(re.findall(r"&(?:lt|gt|amp|nbsp|#\d+|#x[0-9a-fA-F]+);", text or ""))
    n += (text or "").count("�")
    n += len(re.findall(r"Ã|â€|Â ", text or ""))
    return n


def score_sample(text: str, src: str) -> dict:
    inv, matched = order_inversions(text, src)
    return {
        "inversions": inv,
        "matched_paras": matched,
        "cleanliness_issues": cleanliness_issues(text),
        "chars": len(text or ""),
    }


def select_best(samples: list[str], raw_html: str) -> tuple[int, list[dict]]:
    """Return (best_index, per-sample scores).

    Completeness dominates: a drastically shorter sample is under-extracted and must never win on
    order-fidelity alone (a 1-paragraph sample trivially has 0 inversions). So first drop samples far
    shorter than the longest, THEN rank the full-length survivors by fewest inversions / most coverage.
    """
    src = source_linear(raw_html)
    scores = [score_sample(s, src) for s in samples]
    maxlen = max((sc["chars"] for sc in scores), default=1)
    pool = [i for i in range(len(samples)) if scores[i]["chars"] >= 0.6 * maxlen] or list(range(len(samples)))
    order = sorted(
        pool,
        key=lambda i: (scores[i]["inversions"], -scores[i]["matched_paras"], scores[i]["cleanliness_issues"]),
    )
    return order[0], scores
