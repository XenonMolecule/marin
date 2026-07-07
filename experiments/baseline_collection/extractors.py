# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open-source HTML->text extractors for the extraction-router bake-off.

Each adapter takes ``(html, url)`` and returns plain text, or ``""`` on any
failure -- one library choking on a weird page must never kill a doc (mirrors
``fasttext_useful_classifier.resiliparse_text``). Library imports are inside the
adapters so a missing optional dep degrades to ``""`` instead of an ImportError.

Two roles (see ``EXTRACTORS`` vs ``REFERENCE_EXTRACTORS``):
  * candidates -- boilerplate removers that are real router targets;
  * reference  -- "take-everything" floors that anchor the recall ceiling and
    show how much the removers discard (not router targets).

Install with the ``extraction-bakeoff`` extra (see lib/marin/pyproject.toml).
The exact APIs of the less-common libs are validated by the smoke probe in
``__main__`` / the bake-off; the guard keeps any drift from raising.
"""

from collections.abc import Callable

# An extractor: raw HTML + optional source URL -> plain text ("" on failure).
Extractor = Callable[[str, str | None], str]


def resiliparse_main(html: str, url: str | None = None) -> str:
    """resiliparse main-content extraction (the DCLM / marin baseline)."""
    try:
        from resiliparse.extract.html2text import extract_plain_text
        from resiliparse.parse.html import HTMLTree

        return extract_plain_text(HTMLTree.parse(html), main_content=True, alt_texts=False, noscript=False)
    except Exception:
        return ""


def resiliparse_full(html: str, url: str | None = None) -> str:
    """resiliparse with main_content=False: ~all visible text (recall floor)."""
    try:
        from resiliparse.extract.html2text import extract_plain_text
        from resiliparse.parse.html import HTMLTree

        return extract_plain_text(HTMLTree.parse(html), main_content=False, alt_texts=False, noscript=False)
    except Exception:
        return ""


def trafilatura_default(html: str, url: str | None = None) -> str:
    try:
        import trafilatura

        return trafilatura.extract(html, url=url) or ""
    except Exception:
        return ""


def trafilatura_recall(html: str, url: str | None = None) -> str:
    """trafilatura tuned toward recall (keeps more borderline content)."""
    try:
        import trafilatura

        return trafilatura.extract(html, url=url, favor_recall=True) or ""
    except Exception:
        return ""


def justext_en(html: str, url: str | None = None, paragraph_sep: str = "\n\n") -> str:
    """jusText: stopword-density boilerplate removal; keep non-boilerplate paras.

    ``paragraph_sep`` joins the kept paragraphs. Default ``"\\n\\n"`` matches the gold/benchmark
    (a single ``"\\n"`` mashes forum posts and paragraphs together); callers that version this
    choice (e.g. the fast_curation cascade) pass it explicitly.
    """
    try:
        import justext

        paragraphs = justext.justext(html, justext.get_stoplist("English"))
        return paragraph_sep.join(p.text for p in paragraphs if not p.is_boilerplate)
    except Exception:
        return ""


def python_readability(html: str, url: str | None = None) -> str:
    """Arc90 readability (readability-lxml): summary HTML -> text via bs4."""
    try:
        from bs4 import BeautifulSoup
        from readability import Document

        summary_html = Document(html).summary()
        return BeautifulSoup(summary_html, "html.parser").get_text(" ", strip=True)
    except Exception:
        return ""


def goose3_extract(html: str, url: str | None = None) -> str:
    try:
        from goose3 import Goose

        with Goose() as g:
            return g.extract(raw_html=html).cleaned_text or ""
    except Exception:
        return ""


def boilerpy3_article(html: str, url: str | None = None) -> str:
    try:
        from boilerpy3 import extractors

        return extractors.ArticleExtractor().get_content(html) or ""
    except Exception:
        return ""


def bs4_get_text(html: str, url: str | None = None) -> str:
    """BeautifulSoup all-visible-text (raw floor; no boilerplate removal)."""
    try:
        from bs4 import BeautifulSoup

        return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        return ""


def inscriptis_text(html: str, url: str | None = None) -> str:
    """Layout-preserving HTML->text (strong on tables); recall-leaning floor."""
    try:
        from inscriptis import get_text

        return get_text(html)
    except Exception:
        return ""


# Boilerplate-removal candidates -- real router targets, ranked into the bake-off.
EXTRACTORS: dict[str, Extractor] = {
    "resiliparse_main": resiliparse_main,
    "trafilatura": trafilatura_default,
    "trafilatura_recall": trafilatura_recall,
    "justext": justext_en,
    "python_readability": python_readability,
    "goose3": goose3_extract,
    "boilerpy3": boilerpy3_article,
}

# "Take-everything" reference lines -- anchor the recall ceiling, not router targets.
REFERENCE_EXTRACTORS: dict[str, Extractor] = {
    "resiliparse_full": resiliparse_full,
    "bs4_get_text": bs4_get_text,
    "inscriptis": inscriptis_text,
}

ALL_EXTRACTORS: dict[str, Extractor] = {**EXTRACTORS, **REFERENCE_EXTRACTORS}


def extract_all(html: str, url: str | None = None) -> dict[str, str]:
    """Run every extractor on one page. Each value is text or ``""`` on failure."""
    return {name: fn(html, url) for name, fn in ALL_EXTRACTORS.items()}


if __name__ == "__main__":
    # Smoke probe: confirm each adapter's API against the installed libs. Prints
    # the output length per extractor (0 => missing dep or API drift to fix).
    sample = (
        "<html><head><title>T</title></head><body>"
        "<nav>home about contact</nav>"
        "<article><h1>Real Heading</h1>"
        "<p>This is the main article body with several informative sentences. "
        "It should survive boilerplate removal because it is the real content.</p>"
        "<p>A second substantive paragraph continues the article discussion here.</p>"
        "</article><footer>copyright 2026</footer></body></html>"
    )
    for name, text in extract_all(sample, url="https://example.com/article").items():
        preview = text[:60].replace("\n", " ")
        print(f"{name:20s} len={len(text):5d}  {preview!r}")
