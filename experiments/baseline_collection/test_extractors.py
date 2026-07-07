# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the extractor registry: the failure-guard contract (never raise,
always return str), the candidate/reference partition, and that the always-present
core extractors recover the article body on a realistic page."""

from __future__ import annotations

from experiments.baseline_collection.extractors import (
    ALL_EXTRACTORS,
    EXTRACTORS,
    REFERENCE_EXTRACTORS,
    bs4_get_text,
    extract_all,
    resiliparse_main,
)

# A realistic page: nav + footer boilerplate around a long article body. The body
# is long enough that stopword-density removers (justext) keep it.
_ARTICLE = (
    "<html><head><title>Site</title></head><body>"
    "<nav>Home About Contact Subscribe</nav>"
    "<article><h1>The Headline</h1>"
    "<p>BODYMARKER The annual migration patterns of arctic terns span nearly the "
    "entire globe, from their breeding grounds in the far north to wintering areas "
    "near Antarctica, a round trip covering tens of thousands of kilometers.</p>"
    "<p>Researchers tracked individual birds across multiple seasons to understand "
    "how they navigate such vast distances using a combination of cues.</p>"
    "</article><footer>FOOTERMARKER copyright 2026 all rights reserved</footer>"
    "</body></html>"
)


def test_registry_partition_is_clean():
    assert ALL_EXTRACTORS == {**EXTRACTORS, **REFERENCE_EXTRACTORS}
    assert not (EXTRACTORS.keys() & REFERENCE_EXTRACTORS.keys())  # disjoint roles


def test_guard_contract_never_raises_returns_str():
    # Garbage / empty input must degrade to "" per extractor, never raise.
    for html in ("", "<not really html", "<html><body></body></html>", "\x00\x01"):
        out = extract_all(html, url=None)
        assert out.keys() == ALL_EXTRACTORS.keys()
        assert all(isinstance(v, str) for v in out.values())


def test_core_extractors_recover_body():
    # resiliparse (core dep) is a boilerplate remover: keeps the body, drops the
    # footer. bs4 is the raw floor: keeps everything including the footer.
    main = resiliparse_main(_ARTICLE)
    assert "BODYMARKER" in main
    assert "FOOTERMARKER" not in main

    floor = bs4_get_text(_ARTICLE)
    assert "BODYMARKER" in floor and "FOOTERMARKER" in floor
