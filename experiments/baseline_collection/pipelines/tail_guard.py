# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect suspicious (likely premature-EOS) extraction tails.

vLLM at t=0 is not run-to-run deterministic under concurrent batching; a small
fraction of long extractions stop early (observed: mid-$$ block on WebMainBench
d8bfa68a at 1855/12288 tokens; forum threads cut after the first posts on the
marin devset — identical requests complete fine on retry). This module gives
runners a cheap, conservative signal to retry once.

Deliberately conservative: false positives cost one extra call; false negatives
just keep today's behavior.
"""

from __future__ import annotations

# Endings that plausibly close a document (sentence-final punctuation across
# scripts, closing quotes/brackets/fences, sentinel-style closers).
_OK_TAIL_CHARS = set(".!?。！？…\"'”’)]}>*`|$")


def unbalanced_fences(text: str) -> bool:
    """Odd number of ``` fences or $$ display-math delimiters."""
    return text.count("```") % 2 == 1 or text.count("$$") % 2 == 1


def ends_mid_sentence(text: str) -> bool:
    tail = text.rstrip()
    if not tail:
        return False
    return tail[-1] not in _OK_TAIL_CHARS


def looks_truncated(output: str | None, min_chars: int = 400) -> bool:
    """True if the output smells like a premature stop.

    Only flags outputs longer than min_chars: very short outputs are usually a
    legitimate near-empty page (or a sentinel), and retrying them is wasteful —
    the empty-output retry in the runners already covers those.
    """
    if not output:
        return False
    text = output.strip()
    if len(text) < min_chars:
        return False
    return unbalanced_fences(text) or ends_mid_sentence(text)
