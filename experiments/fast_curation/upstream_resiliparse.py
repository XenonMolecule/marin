# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The upstream (Cython) resiliparse extraction call, isolated so pool children import nothing heavy.

This is the call behind the ``resiliparse_10k`` baseline
(``marin.transform.extract_text_from_html._extract_text``): main-content mode, no alt texts, no
``<noscript>`` bodies, stripped. Keep the arguments identical to that function; arm B of the
extractor ablation is only comparable to arm C because the two share this exact call.
"""

from __future__ import annotations

from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree

# ``ExtractTextConfig.min_text_length``: the baseline drops extractions shorter than this.
MIN_TEXT_CHARS = 50


def upstream_resiliparse_text(html: str) -> str:
    """Upstream resiliparse main-content text of ``html``, stripped; ``""`` when the parser fails."""
    try:
        tree = HTMLTree.parse(html)
        text = extract_plain_text(tree, main_content=True, alt_texts=False, noscript=False)
    except Exception:
        return ""
    return text.strip()


def extract_batch(htmls: list[str]) -> list[str]:
    """Pool unit of work: extract a batch of pages in one child call."""
    return [upstream_resiliparse_text(h) for h in htmls]
