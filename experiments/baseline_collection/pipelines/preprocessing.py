# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HTML preprocessing for the extraction pipeline.

The whole point of preprocessing is to make the HTML smaller without removing
anything a reader (or the gold) would want. "Safe" is an EMPIRICAL property, not
a syntactic one — validate every rule with scripts/audit_preprocessing.py, which
measures gold-token recall against the RAW HTML STRING (extractor-independent).

Audit over all 393 gold devset docs (docs losing >2% of gold tokens):

    rule                                    damaged   size
    script_strip (naive)                     15/393    82%   <- unsafe
    body_slice   (even first..last </body>)  25/393    84%   <- unsafe
    CURRENT (script+body, shipped ≤ iter-21) 35/393    73%   <- unsafe
    style / comment / svg / attr scrub        0/393  96-82%   safe
    strip_scripts_keep_prose                  0/393    93%    safe
    preprocess_html_for_extraction (now)      0/393    66%   <- safe AND smaller

Why the old rules were unsafe:
- <script> can BE the content. Khan Academy renders articles/exercises from a
  `perseusContent` JSON blob; the gold for record_01726 contains text that exists
  only inside a <script>. Keeping scripts that carry a long natural-language
  string (and dropping pure-behaviour ones) is safe and still cuts ~7%.
- <head> can BE the content. Scholarly pages (biorxiv/searxiv) put title,
  authors, and abstract ONLY in <meta> tags — the gold for record_01522 *starts*
  with `<meta citation_title>`. Slicing to <body> deleted it (recall .725 -> .076).
- Nested <body> is real: ad iframes open a second <body>, so a non-greedy
  <body>(.*?)</body> match ends at an AD's close tag. record_01119 (558KB) became
  233 chars of ad boilerplate — a guaranteed false drop on a top-quality doc.
  `body_slice()` below is kept (first <body> to LAST </body>) for callers that
  want it, but it is NOT used by default: it still costs 5 gold docs and buys
  only 6% size.
"""

import re

SCRIPT_TAG_RE = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
STYLE_TAG_RE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
SVG_TAG_RE = re.compile(r"<svg\b[^>]*>.*?</svg\s*>", re.IGNORECASE | re.DOTALL)
COMMENT_RE = re.compile(r"<!--(?!\[if).*?-->", re.DOTALL)
BODY_OPEN_RE = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
BODY_CLOSE_RE = re.compile(r"</body\s*>", re.IGNORECASE)
DATA_SCRIPT_RE = re.compile(
    r'<script\b[^>]*type\s*=\s*["\']?(?:application/(?:ld\+)?json|application/json)["\']?[^>]*>',
    re.IGNORECASE,
)
# Attributes that never carry reader-facing text. alt/title/aria-label are KEPT.
NOISY_ATTR_RE = re.compile(
    r"\s+(?:class|style|id|onclick|onload|target|rel|width|height|srcset|sizes|"
    r'data-[\w-]+)\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s>]+)',
    re.IGNORECASE,
)
# A quoted string this long inside a <script> means the block carries prose.
_PROSE_STRING_RE = re.compile(r'"((?:[^"\\]|\\.){40,})"')
MIN_SCRIPT_PROSE = 200


def strip_scripts_keep_prose(html: str, min_prose: int = MIN_SCRIPT_PROSE) -> str:
    """Drop <script> blocks that are pure behaviour; keep ones carrying content.

    A block is kept if it declares a JSON/JSON-LD type or contains a quoted
    string of at least `min_prose` chars (state blobs, perseusContent, etc.).
    """
    out, pos = [], 0
    for match in SCRIPT_TAG_RE.finditer(html):
        block = match.group(0)
        out.append(html[pos : match.start()])
        head = block[: block.find(">") + 1]
        longest = max((len(s) for s in _PROSE_STRING_RE.findall(block)), default=0)
        if longest >= min_prose or DATA_SCRIPT_RE.match(head):
            out.append(block)
        pos = match.end()
    out.append(html[pos:])
    return "".join(out)


def body_slice(html: str) -> str:
    """First <body> to LAST </body>. NOT used by default — see module docstring
    (costs 5 gold docs; <head> meta tags are content on scholarly pages)."""
    open_match = BODY_OPEN_RE.search(html)
    if not open_match:
        return html
    last_close = None
    for match in BODY_CLOSE_RE.finditer(html):
        last_close = match
    if last_close is None or last_close.start() <= open_match.end():
        return html[open_match.end() :]
    return html[open_match.end() : last_close.start()]


_PROSE_STRING_ONLY_RE = re.compile(r'"((?:[^"\\]|\\.){40,})"')


def scripts_to_prose(html: str, min_prose: int = MIN_SCRIPT_PROSE) -> str:
    """Replace each <script> with the prose it carries, unescaped, as plain text.

    Keeping the raw block preserves content but hands the extractor JSON syntax,
    which it then reproduces as if it were page text. Pulling the long strings
    out and unescaping them keeps the content without the syntax.
    """
    import codecs

    def repl(match):
        strings = [s for s in _PROSE_STRING_ONLY_RE.findall(match.group(0)) if len(s) >= min_prose]
        if not strings:
            return ""
        out = []
        for s in strings:
            try:
                out.append(codecs.decode(s, "unicode_escape"))
            except Exception:
                out.append(s)
        return "\n<div>" + "\n".join(out) + "</div>\n"

    return SCRIPT_TAG_RE.sub(repl, html)


NBSP_ENTITY_RE = re.compile(r"&(?:nbsp|#160|#xa0);", re.IGNORECASE)


def _common_strips(html: str) -> str:
    cleaned = STYLE_TAG_RE.sub("", html)
    cleaned = SVG_TAG_RE.sub("", cleaned)
    cleaned = COMMENT_RE.sub("", cleaned)
    # &nbsp;-soup (ASCII-art formulas, layout padding) halts the extractor's
    # transcription mid-document; decoding ONLY the non-breaking-space entities
    # keeps the HTML valid (never &lt;/&gt;) and reads straight through.
    # Measured: programmingpraxis lev .065 -> .864 from this line alone.
    cleaned = NBSP_ENTITY_RE.sub(" ", cleaned)
    return NOISY_ATTR_RE.sub("", cleaned)


_TAG_RE = re.compile(r"<[^>]+>")


def _visible_text_len(html: str) -> int:
    """Cheap proxy for 'how much reader-facing text is in here'."""
    return len(_TAG_RE.sub(" ", html).split())


def preprocess_adaptive(html: str, keep_ratio: float = 0.5) -> str:
    """Aggressive by default; prose-mode ONLY for pages aggressive would gut.

    Aggressive (strip scripts + slice to <body>) wins on every end metric because
    for ~91% of pages the head/script material is noise the extractor reproduces.
    But on ~9% it deletes the actual article (Khan renders from a script blob;
    biorxiv puts title/authors in <head> meta). Those pages are identifiable
    without an LLM: aggressive leaves them with far less visible text than prose
    mode does. Cost is two regex passes per doc.
    """
    agg = body_slice(SCRIPT_TAG_RE.sub("", html))
    agg_len = _visible_text_len(agg)
    pro = _common_strips(scripts_to_prose(html))
    if agg_len < keep_ratio * _visible_text_len(pro):
        return pro
    return _common_strips(agg)


def preprocess_html_for_extraction(html: str, mode: str = "safe") -> str:
    """Shrink HTML before extraction. `mode` is an attributable experiment knob.

    Audited over 393 gold docs (docs losing >2% of gold tokens / output size):
      safe       0/393 @ 66%  — keep content-bearing scripts verbatim + whole head
      prose      6/393 @ ~60% — scripts reduced to their prose (no JSON syntax)
      aggressive 35/393 @ 73% — strip all scripts + slice to <body> (the old default)
      none       0/393 @ 100%
    """
    if mode == "none":
        return html
    if mode == "adaptive":
        return preprocess_adaptive(html)
    if mode == "aggressive":
        return _common_strips(body_slice(SCRIPT_TAG_RE.sub("", html)))
    if mode == "prose":
        return _common_strips(scripts_to_prose(html))
    if mode == "safe":
        return _common_strips(strip_scripts_keep_prose(html))
    raise ValueError(f"unknown preprocess mode: {mode!r}")
