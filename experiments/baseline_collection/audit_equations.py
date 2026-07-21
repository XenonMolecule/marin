# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit the gold extractions for DROPPED EQUATIONS — a serious violation.

Signal: a line in the independent gemini extraction that looks like a formula (a math relation plus
math markup or a short formula body) whose content is ABSENT from the Sonnet extraction. Catches the
E=mc^2 class of drop (equation rendered as an image / special markup that Sonnet skipped). Normalized
matching (accents/markdown stripped) so pure formatting differences don't false-flag.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata

_REL = re.compile(r"[=≈≤≥≠∝→←↔]")
_MATHY = re.compile(r"\^|_|\\frac|\\sqrt|\\sum|\\int|[∫∑√∞±×÷·θπΔΩλμσαβγφψ²³½¼]")
_WS = re.compile(r"\s+")


def _norm(t: str) -> str:
    t = "".join(c for c in unicodedata.normalize("NFKD", t) if not unicodedata.combining(c))
    return _WS.sub(" ", re.sub(r"[^a-z0-9=^]+", " ", t.lower())).strip()


def _is_equation(line: str) -> bool:
    """A formula line: has a relation AND (math markup OR is short/dense enough to be a formula)."""
    t = line.strip()
    if not _REL.search(t):
        return False
    if _MATHY.search(t):
        return True
    # short line dominated by a relation between symbol/number operands (e.g. 'E = mc2', 'a = b + c')
    return len(t) < 90 and bool(re.search(r"[A-Za-z0-9)\]]\s*[=≈≤≥≠]\s*[A-Za-z0-9(\-]", t))


def dropped_equations(sonnet: str, gemini: str) -> list[str]:
    son = _norm(sonnet)
    out = []
    for ln in (gemini or "").splitlines():
        t = ln.strip()
        if len(t) < 3 or t.startswith(">") or "```" in t:
            continue
        if _is_equation(t) and _norm(t)[:40] and _norm(t)[:40] not in son:
            out.append(t)
    return out


def main() -> int:
    manifest = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scratch/gold_extraction/extract_manifest.json"))
    out_dir = "scratch/gold_extraction/extract_out"
    hits = []
    for d in manifest:
        hid = os.path.basename(d["html"]).replace(".html", "")
        sp, gp = f"{out_dir}/{hid}.txt", f"{out_dir}/gemini_{hid}.txt"
        if not (os.path.exists(sp) and os.path.exists(gp)):
            continue
        gem = open(gp).read()
        if gem.startswith("["):
            continue
        dropped = dropped_equations(open(sp).read(), gem)
        if dropped:
            hits.append({"url": d["url"], "register": d["register"], "hid": hid, "n": len(dropped), "eqs": dropped[:6]})
    hits.sort(key=lambda x: -x["n"])
    print(f"docs where Sonnet dropped equation-like lines gemini kept: {len(hits)}\n")
    for h in hits:
        print(f"  [{h['n']}] {h['register']:<14} {h['url'][:60]}")
        for e in h["eqs"]:
            print(f"        DROPPED: {e[:100]}")
    json.dump(hits, open("scratch/gold_extraction/equation_audit.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
