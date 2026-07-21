# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Phase-1 match-quality preview: render Phase-C doc<->eval-example matches for human review.

Before seeding the dev set from the Phase A/B/C search, the user should see whether the matches
are actually good: is a doc that "matched" the water-cycle eval item really about the water cycle,
or a spurious keyword hit? This renders sampled matched docs as HTML, each showing the matched eval
SUBJECT(s), the overlap fraction, the verifier tier, and the snippet with the eval keywords
HIGHLIGHTED — so match quality is eyeballable.

Reads a sample parquet with columns {url, domain, category, subjects, best_frac, verifier_score,
snippet} + keyword_examples.json (subject -> kws) locally; writes scratch/match_preview.html.
"""

from __future__ import annotations

import argparse
import collections
import html
import json
import math
import re
import sys

import pyarrow.parquet as pq

_TOKEN = re.compile(r"[a-z0-9]+")


def _reverify(full_text: str, subj_kws: list[str], min_frac: float = 0.6) -> bool:
    """Does the FULL doc text actually contain >=min_frac of this eval example's keywords?
    Reproduces the scan rule on the full text — a recorded match that fails here is a bug/spurious."""
    toks = set(_TOKEN.findall(full_text.lower()))
    hit = sum(1 for k in subj_kws if all(w in toks for w in _TOKEN.findall(k.lower())))
    return hit >= max(3, math.ceil(len(subj_kws) * min_frac))


def _highlight(snippet: str, kws: list[str]) -> str:
    """Escape snippet, then bold whole-word occurrences of any matched keyword/phrase."""
    esc = html.escape(snippet)
    for kw in sorted(kws, key=len, reverse=True):
        if not kw.strip():
            continue
        pat = re.compile(r"(?<![a-zA-Z0-9])(" + re.escape(html.escape(kw)) + r")(?![a-zA-Z0-9])", re.IGNORECASE)
        esc = pat.sub(r"<mark>\1</mark>", esc)
    return esc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/tmp/rsample.parquet")
    ap.add_argument("--examples", default="/tmp/keyword_examples.json")
    ap.add_argument("--out", default="scratch/match_preview.html")
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--hqworse", default="/tmp/hqworse.parquet")
    ap.add_argument("--broad", default="scratch/eval_attribution/keywords_broad.json")
    args = ap.parse_args()

    subj2kws = {e["subject"]: e["kws"] for e in json.load(open(args.examples))["examples"]}
    # subject -> the actual eval prompt(s): records give (task, idx); hq_worse gives the prompt text.
    hqw = pq.read_table(args.hqworse, columns=["task", "idx", "text"]).to_pylist()
    task_idx_text = {(r["task"], str(r["idx"])): r["text"] for r in hqw}
    subj2prompt: dict[str, list[str]] = collections.defaultdict(list)
    for rec in json.load(open(args.broad))["records"]:
        txt = task_idx_text.get((rec.get("task"), str(rec.get("idx"))))
        if txt and txt not in subj2prompt[rec["subject"]]:
            subj2prompt[rec["subject"]].append(txt)
    rows = pq.read_table(args.sample).to_pylist()
    # Group by category; within a category show a spread of best_frac (weak -> strong) so the user
    # can calibrate where matches get spurious.
    bycat: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        bycat[r.get("category", "?")].append(r)
    verify_tally: dict = collections.defaultdict(collections.Counter)  # cat -> {True: n_valid, False: n_spurious}

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'></head><body>",
        "<style>body{font:14px/1.5 -apple-system,sans-serif;max-width:1000px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}"
        "h2{border-bottom:2px solid #ddd;padding-top:1.5rem}.doc{border:1px solid #e0e0e0;border-radius:6px;padding:.7rem 1rem;margin:.6rem 0}"
        ".meta{font-size:12px;color:#666}.subj{color:#0a6;font-weight:600}mark{background:#fde68a;padding:0 1px}"
        ".frac{font-weight:700}.hi{color:#2a9d4a}.lo{color:#c0392b}.snip{margin-top:.4rem;color:#333}"
        ".evalq{margin:.3rem 0;padding:.3rem .5rem;background:#eef4ff;border-left:3px solid #7aa2f7;font-size:13px;color:#334}"
        "summary.cat{cursor:pointer;font-size:17px;font-weight:700;padding:.5rem 0;border-bottom:2px solid #ddd;margin-top:.6rem}"
        "details{margin:.3rem 0}button{margin:.2rem .4rem .2rem 0;padding:.3rem .7rem;cursor:pointer}"
        ".ok{color:#2a9d4a;font-weight:700}.bad{color:#c0392b;font-weight:700}table{border-collapse:collapse;font-size:13px}</style>",
        "<div><button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=true)\">expand all</button>"
        "<button onclick=\"document.querySelectorAll('details').forEach(d=>d.open=false)\">collapse all</button></div>",
        "<h1>Phase-C match preview — do the doc↔eval matches look real?</h1>",
        "<p>Each doc shows the eval <span class='subj'>subject(s)</span> it matched, the overlap "
        "<span class='frac'>fraction</span> (≥0.60 by construction), the verifier tier, and the snippet "
        "with matched eval keywords <mark>highlighted</mark>. Skim for spurious matches.</p>",
    ]
    for cat in sorted(bycat):
        docs = sorted(bycat[cat], key=lambda r: r.get("best_frac", 0))
        # even spread across the best_frac range
        step = max(1, len(docs) // args.per_category)
        shown = docs[::step][: args.per_category]
        parts.append(
            f"<details><summary class='cat'>{html.escape(cat)} "
            f"<span class='meta'>({len(bycat[cat])} matched docs; showing {len(shown)})</span></summary>"
        )
        for r in shown:
            subjects = [s for s in (r.get("subjects") or "").split("|") if s]
            kws = [k for s in subjects for k in subj2kws.get(s, [])]
            frac = r.get("best_frac", 0) or 0
            cls = "hi" if frac >= 0.8 else "lo"
            vs = r.get("verifier_score", "?")
            full = r.get("full_text") or ""
            doc_toks = set(_TOKEN.findall(full.lower()))
            matched_kws = [k for k in dict.fromkeys(kws) if k and all(w in doc_toks for w in _TOKEN.findall(k.lower()))]
            evalq = []
            for s in subjects[:3]:
                skws = subj2kws.get(s, [])
                badge = ""
                if full and skws:
                    ok = _reverify(full, skws)
                    badge = "<span class='ok'>✓</span> " if ok else "<span class='bad'>⚠ spurious?</span> "
                    verify_tally[cat][ok] += 1
                for p in subj2prompt.get(s, [])[:1]:
                    evalq.append(f"{badge}<b>{html.escape(s)}</b> → <i>{html.escape(p)}</i>")
            evalq_html = ("<div class='evalq'>EVAL ITEM(S): " + "<br>".join(evalq) + "</div>") if evalq else ""
            mk = (
                "<div class='mk'>matched keywords in doc: "
                + ", ".join(f"<mark>{html.escape(k)}</mark>" for k in matched_kws)
                + "</div>"
                if matched_kws
                else "<div class='mk bad'>⚠ NO keywords actually found in the fetched doc text (real spurious / truncation)</div>"
            )
            body_full = full or r.get("snippet") or ""
            preview = _highlight(body_full[:1500], kws)
            rest = body_full[1500:]
            more = (
                f"<details><summary>show full doc ({len(body_full):,} chars)</summary>{_highlight(rest, kws)}</details>"
                if rest
                else ""
            )
            parts.append(
                f"<div class='doc'><div class='meta'>[{html.escape(r.get('domain','')[:40])}] "
                f"verifier={vs}/4 · <span class='frac {cls}'>frac={frac:.2f}</span></div>"
                f"{evalq_html}{mk}"
                f"<div class='snip'>DOC: {preview}{more}</div></div>"
            )
        parts.append("</details>")
    if any(verify_tally.values()):
        parts.append(
            "<h2>Full-text re-verification audit</h2><p>Fraction of recorded matches that re-clear ≥60% "
            "on the FULL doc text (low % = spurious matches / a Phase-C linking problem).</p>"
            "<table border='1' cellpadding='4'><tr><th>category</th><th>✓ valid</th><th>⚠ spurious</th><th>% valid</th></tr>"
        )
        for cat in sorted(verify_tally):
            t = verify_tally[cat]
            v, s = t[True], t[False]
            if v + s:
                parts.append(
                    f"<tr><td>{html.escape(cat)}</td><td>{v}</td><td>{s}</td><td>{100 * v / (v + s):.0f}%</td></tr>"
                )
                print(f"[audit] {cat:22} valid={v} spurious={s} ({100 * v / (v + s):.0f}% valid)")
        parts.append("</table>")
    parts.append("</body></html>")
    with open(args.out, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out} ({sum(len(v) for v in bycat.values())} docs across {len(bycat)} categories)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
