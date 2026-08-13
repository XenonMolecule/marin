# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gemini (flash) client for the edge-case search: query expansion + reranking.

Thin wrapper over the generateContent REST API (mirrors
``run_gemini_crosscheck.gemini_backoff``): raw urllib, SSL-cert-aware, with
429/5xx exponential backoff, plus structured-JSON output via ``responseSchema``.
Reads ``GEMINI_API_KEY`` from the environment.

Two calls power the search:

- :func:`expand_query` — turn a natural-language query ("ocaml tutorials") into
  BM25 query strings + keywords the lexical index can actually match.
- :func:`rerank` — given the NL query and candidate snippets, return the indices
  of the most relevant candidates (an LLM reranker over lexical recall).
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


def _ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)


def generate_json(
    prompt: str,
    *,
    schema: dict,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    tries: int = 5,
) -> dict:
    """Call Gemini with a forced JSON schema; return the parsed object.

    Raises RuntimeError on repeated failure so the caller can surface it.
    """
    text = _generate_text(
        prompt, schema=schema, model=model, temperature=temperature, max_tokens=max_tokens, tries=tries
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        salvaged = _salvage_json(text)
        if salvaged is not None:
            return salvaged
        # Deterministic parse failure — don't retry (retrying wastes the whole budget).
        raise RuntimeError(f"gemini returned non-JSON ({len(text)} chars): {text[:200]!r}") from None


def _salvage_json(text: str) -> dict | None:
    """Best-effort recovery of a JSON object from a fenced/prefixed/truncated response."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{") :] if "{" in t else t
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(t[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _generate_text(prompt, *, schema, model, temperature, max_tokens, tries) -> str:
    """POST to generateContent with retries on transient (429/5xx/network) errors; return text."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set in the environment")
    url = _ENDPOINT.format(model=model, key=key)
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            # Disable "thinking": expand/rerank are direct judgment tasks. Thinking
            # tokens would otherwise consume the output budget (truncating the JSON)
            # and add several seconds of latency.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    ctx = _ssl_context()
    last = ""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
                d = json.loads(r.read())
            cands = d.get("candidates") or []
            if not cands:
                last = f"no candidate: {json.dumps(d)[:200]}"
                raise ValueError(last)
            return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts") or [])
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:150].decode(errors='replace')}"
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(min(20, 2 * 2**attempt))
                continue
            break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            if attempt < tries - 1:
                time.sleep(2 * 2**attempt)
                continue
            break
    raise RuntimeError(f"gemini generate failed: {last}")


_EXPAND_SCHEMA = {
    "type": "object",
    "properties": {
        "example_queries": {"type": "array", "items": {"type": "string"}},
        "bm25_queries": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["example_queries", "bm25_queries", "keywords", "rationale"],
}

_EXPAND_PROMPT = """You help search a large corpus of web-page text with a BM25 (lexical, bag-of-words) index.
The user is a data-curation researcher probing how different extraction pipelines handle a topic or edge case.

User query: {query}

A corpus over-represents pages that DISCUSS a topic (essays, reviews, encyclopedia entries, tutorials) and
under-represents pages that ARE AN ACTUAL EXAMPLE of it (the primary source / raw artifact itself). Produce BOTH
kinds of lexical queries so retrieval surfaces real examples, not only commentary:

- "example_queries": 3-5 queries that match the VERBATIM INTERIOR TEXT of an actual instance — strings that appear
  in the BODY of a primary source, so BM25 lands on the real thing and NOT a page describing it. Prefer an actual
  opening sentence, a distinctive line of prose or dialogue, a characteristic interior passage, real code tokens, a
  formula, or jargon that only appears inside the artifact. Do NOT use bare titles or author names alone — those
  match reviews, catalogs, and encyclopedia entries just as well as the work itself — and avoid analytical words
  ("analysis", "overview", "review", "guide"). e.g. for "18th century English literature":
  "I was born in the year 1632 in the city of York" or "It is a truth universally acknowledged that a single man";
  for "ocaml tutorials": "let rec" and "match with".
- "bm25_queries": 3-5 short queries (2-6 words) for pages ABOUT the topic (the discussion/analysis facet).
  Concrete nouns, jargon, phrasings an author writing about it would use. Vary across facets.
- "keywords": 8-15 individual high-signal terms/phrases relevant pages tend to contain.
- "rationale": one sentence on your strategy.

No boolean operators or quotes in any query. Return JSON only."""


def expand_query(query: str, *, model: str = DEFAULT_MODEL) -> dict:
    """Expand a natural-language query into example-finding + about-topic BM25 queries + keywords."""
    return generate_json(_EXPAND_PROMPT.format(query=query), schema=_EXPAND_SCHEMA, model=model, max_tokens=2048)


_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score"],
            },
        }
    },
    "required": ["ranked"],
}

_RERANK_PROMPT = """You are reranking candidate web-page snippets by relevance to a researcher's query. Rank the
genuinely relevant, substantive pages highest — and KEEP every good on-topic result — but filter out clear junk.

Query: {query}
{example_note}
Candidates (id: snippet):
{candidates}

EXCLUDE (do not return) only candidates that are clearly bad:
- NOT written in the same language as the query — e.g. drop Russian/German/French pages for an English query —
  unless the query is explicitly about that language or about translations;
- obvious low-quality junk: navigation/menu or link/tag/category index dumps, SEO spam, bare product/catalog/
  bookstore listings (mostly publisher, ISBN, price), login/cookie walls, or garbled/machine-translated text.

Otherwise KEEP good on-topic content — do NOT drop a solid result just to be strict. Return up to {top_n} best
first (score ≥ 0.4); returning fewer is fine when some are excluded, but do not pad with off-topic filler.
JSON: a "ranked" array of {{"id": <candidate id>, "score": <0-1>, "reason": <≤8 words>}}. Judge by the page's
actual CONTENT, not incidental keyword matches."""

_EXAMPLE_NOTE = """
Prefer pages that ARE AN ACTUAL EXAMPLE of the query — the primary source / raw artifact whose BODY is the real
content (the literary prose itself, the actual code, the real document) — over pages that merely DISCUSS,
DESCRIBE, SUMMARIZE, REVIEW, or CATALOG it. Critically: a page whose text reads like an encyclopedia or blurb
("X is a novel by Y, first published in 1719...", plot summaries, "themes include..."), a review, a study guide,
or a bookstore/catalog listing (publisher, ISBN, price) is ABOUT the work, NOT the work — drop it or rank it far
below any page that contains the actual primary text. Rank real examples highest and say so in the reason.
"""


def rerank(
    query: str,
    candidates: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    snippet_chars: int = 260,
    top_n: int = 15,
    prefer_examples: bool = True,
) -> list[dict]:
    """Rerank candidates ``[{id, snippet}]``; return the top ``top_n`` as ``[{id, score, reason}]``.

    Only the top matches are requested (not a full ranking) to keep the generated
    output — and thus latency — small. ``prefer_examples`` ranks primary-source
    instances above commentary about the topic.
    """
    lines = [f"{c['id']}: {(c.get('snippet') or '').replace(chr(10), ' ')[:snippet_chars]}" for c in candidates]
    prompt = _RERANK_PROMPT.format(
        query=query, candidates="\n".join(lines), top_n=top_n, example_note=_EXAMPLE_NOTE if prefer_examples else ""
    )
    out = generate_json(prompt, schema=_RERANK_SCHEMA, model=model, max_tokens=4096)
    ranked = out.get("ranked", [])
    ranked.sort(key=lambda r: -r.get("score", 0))
    return ranked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "ocaml tutorials"
    print(json.dumps(expand_query(q), indent=2))


if __name__ == "__main__":
    main()
