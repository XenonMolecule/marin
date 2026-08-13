# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Code-preservation guard: detect chunks whose code listings were skipped.

A distinct failure from early-stop and budget truncation: the model reaches the
end of the document, emits the surrounding prose, and silently skips or
abbreviates long <pre> listings in between (tips4java: 39 of 133 gold code
lines gone, output complete to 100% of the document, no error, budget unused).
Neither the tail guard (ending is clean) nor the coverage guard (overall ratio
is fine) can see it.

The signal is volumetric: compare characters inside <pre>/<code> in the chunk
input against characters inside ``` fences (plus indented code lines) in the
output. Code the model chose to render as plain text still counts, so a spec
that keeps code unfenced is not punished.
"""

from __future__ import annotations

import re

_PRE_RE = re.compile(r"<(pre|code)\b[^>]*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_ENTITIES = (("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " "))


def _clean(s: str) -> str:
    s = _TAG_RE.sub("", s)
    for a, b in _ENTITIES:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def input_code_chars(html_chunk: str) -> int:
    """Characters of code-ish text inside <pre>/<code> in the chunk input."""
    return sum(len(_clean(m.group(2))) for m in _PRE_RE.finditer(html_chunk or ""))


def output_code_chars(text: str) -> int:
    """Characters inside ``` fences, plus indented/monospace-looking lines."""
    total, in_fence = 0, False
    for line in (text or "").splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            total += len(line.strip())
        elif line.startswith(("    ", "\t")) and len(line.strip()) > 4:
            total += len(line.strip())
    return total


def code_underextracted(html_chunk: str, output: str, min_input_chars: int = 500, ratio: float = 0.6) -> bool:
    """True if the chunk carried substantial code and the output kept little.

    Conservative: only fires when the input has real code volume, so prose
    pages and pages whose code is a couple of inline snippets never trip it.
    """
    in_chars = input_code_chars(html_chunk)
    if in_chars < min_input_chars:
        return False
    return output_code_chars(output) < ratio * in_chars
