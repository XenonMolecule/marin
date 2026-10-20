# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The upstream (Cython) resiliparse extraction call, isolated so pool children import nothing heavy.

This is the call behind the ``resiliparse_10k`` baseline
(``marin.transform.extract_text_from_html._extract_text``): main-content mode, no alt texts, no
``<noscript>`` bodies, stripped. Keep the arguments identical to that function; arm B of the
extractor ablation is only comparable to arm C because the two share this exact call.
"""

from __future__ import annotations

# ``ExtractTextConfig.min_text_length``: the baseline drops extractions shorter than this.
MIN_TEXT_CHARS = 50


def upstream_resiliparse_text(html: str) -> str:
    """Upstream resiliparse main-content text of ``html``, stripped; ``""`` when the parser fails."""
    # Imported INSIDE the function on purpose, and it must stay that way. marin's core
    # ``resiliparse`` (upstream, Cython) and the XenonMolecule Rust fork claim the SAME import
    # name and cannot coexist: the fork has no ``extract.html2text`` submodule. A module-scope
    # import here therefore explodes in any process that puts the fork first on ``sys.path`` --
    # which is exactly what ``cpu_phase_c._init_rs_worker`` does in every pool child. Hoisting it
    # killed every child of the lpv11 fork worker (2026-10-02), and the pool reported the import
    # failure as a smoke-test "SIGSEGV".
    from resiliparse.extract.html2text import extract_plain_text  # noqa: PLC0415
    from resiliparse.parse.html import HTMLTree  # noqa: PLC0415

    try:
        tree = HTMLTree.parse(html)
        text = extract_plain_text(tree, main_content=True, alt_texts=False, noscript=False)
    except Exception:
        return ""
    return text.strip()


def extract_batch(htmls: list[str]) -> list[str]:
    """Pool unit of work: extract a batch of pages in one child call."""
    return [upstream_resiliparse_text(h) for h in htmls]
