# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precompute hq->gold inline diffs for the extraction viewer (cached; the O(n^2) work is slow).

Runs offline so it never blocks the viewer. Per doc: if hq dropped the page, the diff is trivially
"all gold added"; for small docs a granular word-level diff; for huge docs a line-level diff (so it
stays feasible). Writes {url: diff_html} and rebuilds the viewer with the cache.
"""

from __future__ import annotations

import difflib
import html as H
import json
import sys

from experiments.baseline_collection.build_extraction_diff import _word_diff


def _line_diff(a: str, b: str) -> str:
    la, lb = a.splitlines(keepends=True), b.splitlines(keepends=True)
    sm = difflib.SequenceMatcher(a=la, b=lb, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(H.escape("".join(la[i1:i2])))
        if tag in ("delete", "replace"):
            parts.append("<del>" + H.escape("".join(la[i1:i2])) + "</del>")
        if tag in ("insert", "replace"):
            parts.append("<ins>" + H.escape("".join(lb[j1:j2])) + "</ins>")
    return "".join(parts)


def _diff_for(hq: str, gold: str) -> str:
    hq, gold = hq or "", gold or ""
    if hq.strip() == "[NO_USEFUL_CONTENT]":
        return "<del>[NO_USEFUL_CONTENT]</del>\n<ins>" + H.escape(gold) + "</ins>"
    if len(hq) + len(gold) < 90_000:
        return _word_diff(hq, gold)  # granular word-level (under the builder's cap)
    return _line_diff(hq, gold)  # feasible for huge docs


def main() -> int:
    results = json.load(open(sys.argv[1]))
    cache_path = sys.argv[2]
    cache = {}
    for r in results:
        gold = r.get("gold_text")
        cache[r["url"]] = {
            "hq": _diff_for(r.get("hq_text"), gold),
            "resi": _diff_for(r.get("resiliparse_text"), gold),
        }
        print(f"  diffed [{r['register']}] -> hq {len(cache[r['url']]['hq'])}c, resi {len(cache[r['url']]['resi'])}c")
    json.dump(cache, open(cache_path, "w"))
    print(f"wrote {len(cache)} diffs to {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
