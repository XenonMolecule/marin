# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manager gate battery for gold extractions — flags anything that doesn't clear the quality bar.

Per doc, on the Sonnet extraction:
  1. tool-XML contamination        — leaked </content>, </invoke>, </…> tool framing (hard fail).
  2. meta-commentary / agent-text  — preamble/postamble/markdown-fence leakage (start & end).
  3. order fidelity                — out-of-source-order paragraph pairs (0 = clean; review if >0).
  4. cleanliness                   — undecoded entities / mojibake / U+FFFD left behind.
  5. length coverage vs resiliparse— gold must be >= the mechanical baseline (weak: resiliparse capped).
  6. GEMINI_DROPPED                — substantive lines the independent gemini extraction kept but Sonnet
                                     omitted. Matching is normalized (accents/markdown/pipes/dashes
                                     stripped) so pure FORMATTING differences don't false-flag — only a
                                     genuinely dropped paragraph/section fires.

`--sanitize <files...>` rewrites extraction files with tool-XML framing stripped (deterministic fix).
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata

from experiments.baseline_collection.score_extraction import (
    cleanliness_issues,
    order_inversions,
    source_linear,
)

_META_START = re.compile(
    r"(?i)^\s*(here (is|are)|here's|below (is|are)|the following|i (have )?(extracted|produced|removed)|"
    r"extraction notes?|note:|okay|sure|this (page|is|forum)|the (main )?(content|article|extraction|page))\b"
)
_META_END = re.compile(
    r"(?i)(let me know|i hope (this|that)|the extraction (is|was) complete|"
    r"no (real |substantive )?(comments|content)|end of extraction)\s*$"
)
_TOOLTAG = re.compile(r"</?(?:content|invoke|antml:[\w-]+|function_calls|parameter|thinking)\b[^>]*>", re.I)
_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    """Fold to comparable form: strip accents, lowercase, drop all non-alphanumerics (markdown markers,
    table pipes/dashes, punctuation). Formatting differences vanish; real word content remains."""
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return _WS.sub(" ", re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()


def sanitize(text: str) -> str:
    """Strip leaked tool-call XML framing (deterministic — never trust the model to never leak it)."""
    return _TOOLTAG.sub("", text or "").strip() + "\n"


def check(sonnet: str, raw_html: str, resiliparse: str | None, gemini: str | None) -> dict:
    issues: list[str] = []
    s = sonnet or ""
    head, tail = s[:220].strip(), s[-220:].strip()

    if _TOOLTAG.search(s):
        issues.append(f"TOOL_XML_CONTAMINATION :: {_TOOLTAG.findall(s)[:4]} [hard fail — sanitize]")
    if _META_START.search(head):
        issues.append(f"META_START :: {head[:70]!r}")
    if _META_END.search(tail):
        issues.append(f"META_END :: {tail[-70:]!r}")
    if s.lstrip().startswith("```") or s.rstrip().endswith("```"):
        issues.append("MARKDOWN_FENCE")

    inv, matched = order_inversions(s, source_linear(raw_html))
    if inv > 0:
        issues.append(f"ORDER_INV={inv} (matched {matched} paras) [review — often a metric false-positive]")

    cl = cleanliness_issues(s)
    if cl > 0:
        issues.append(f"CLEANLINESS={cl} (entities/mojibake/U+FFFD)")

    resi_ratio = None
    if resiliparse:
        resi_ratio = len(s) / max(1, len(resiliparse))
        if resi_ratio < 0.9:
            issues.append(f"UNDER_VS_RESI ratio={resi_ratio:.2f} (sonnet {len(s)} < resiliparse {len(resiliparse)})")

    gem_ratio = None
    gem_only: list[str] = []
    if gemini and not gemini.startswith("["):
        gem_ratio = len(s) / max(1, len(gemini))
        son_norm = _norm(s)
        for ln in gemini.splitlines():
            t = ln.strip()
            if len(t) > 40 and not t.startswith(">") and "```" not in t:
                key = _norm(t)[:45]
                if key and key not in son_norm:
                    gem_only.append(t)
        if len(gem_only) >= 3:
            issues.append(f"GEMINI_DROPPED={len(gem_only)} substantive lines gemini kept but Sonnet omitted [REAL — review]")

    return {
        "chars": len(s),
        "inversions": inv,
        "matched_paras": matched,
        "cleanliness": cl,
        "resi_ratio": round(resi_ratio, 2) if resi_ratio is not None else None,
        "gem_ratio": round(gem_ratio, 2) if gem_ratio is not None else None,
        "gemini_only": gem_only[:12],
        "gemini_only_n": len(gem_only),
        "issues": issues,
        "verdict": "PASS" if not issues else "FLAG",
    }


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--sanitize":
        n = 0
        for path in sys.argv[2:]:
            txt = open(path).read()
            clean = sanitize(txt)
            if clean != txt:
                open(path, "w").write(clean)
                n += 1
        print(f"sanitized {n}/{len(sys.argv) - 2} files (stripped tool-XML framing)")
        return 0

    ROOT = "scratch/gold_extraction"
    manifest = json.load(open(sys.argv[1]))
    resi = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    out_dir = sys.argv[3] if len(sys.argv) > 3 else f"{ROOT}/extract_out"

    rows = []
    for d in manifest:
        hid = d["html"].split("/")[-1].replace(".html", "")
        try:
            sonnet = open(f"{out_dir}/{hid}.txt").read()
        except FileNotFoundError:
            rows.append({"url": d["url"], "register": d["register"], "verdict": "MISSING", "issues": ["no Sonnet output"]})
            continue
        raw = open(d["html"]).read()
        try:
            gem = open(f"{out_dir}/gemini_{hid}.txt").read()
        except FileNotFoundError:
            gem = None
        r = check(sonnet, raw, resi.get(d["url"]), gem)
        r.update({"url": d["url"], "register": d["register"]})
        rows.append(r)

    npass = sum(1 for r in rows if r["verdict"] == "PASS")
    print(f"{'verdict':<8}{'register':<14}{'chars':>8}{'inv':>5}{'resi':>6}{'gem':>6}  issues / url")
    for r in rows:
        rr, gg = r.get("resi_ratio"), r.get("gem_ratio")
        print(f"{r['verdict']:<8}{r.get('register', ''):<14}{r.get('chars', 0):>8}{r.get('inversions', 0):>5}"
              f"{(rr if rr is not None else 0):>6}{(gg if gg is not None else 0):>6}  {r['url'][:48]}")
        for iss in r.get("issues", []):
            print(f"          - {iss}")
    print(f"\nPASS {npass}/{len(rows)}   FLAG {len(rows) - npass}")
    json.dump(rows, open(f"{out_dir}/../extraction_checks.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
