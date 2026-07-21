# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Word-level grounding gate: verify every span of an extraction is literally present in the raw HTML.

The hallucination gate for the gold pipeline — especially for JS-blob edge cases where content comes
from inside `<script>` tags (JSON-escaped) and a model could invent or grab non-source text. Decodes
the raw (JSON `\\uXXXX`/`\\n`/`\\"` escapes + HTML entities) so script-embedded content matches, then
slides a word window over the extraction and checks each window is a substring of the normalized raw.
Reports grounded fraction + sample ungrounded spans per doc, for Sonnet and gemini.

  python -m experiments.baseline_collection.ground_extraction <manifest.json> [window=12] [step=6]
"""

from __future__ import annotations

import html
import json
import os
import re
import sys

ROOT = "scratch/gold_extraction"
_UESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_WS = re.compile(r"\s+")
_NONALNUM = re.compile(r"[^a-z0-9 ]")


def _decode(s: str) -> str:
    s = _UESC.sub(lambda m: chr(int(m.group(1), 16)), s)
    s = s.replace("\\n", " ").replace("\\t", " ").replace("\\r", " ").replace('\\"', '"').replace("\\/", "/")
    return html.unescape(s)


def _norm(s: str) -> str:
    return _WS.sub(" ", _NONALNUM.sub(" ", _decode(s).lower())).strip()


def ground(extraction: str, raw_norm: str, window: int, step: int) -> tuple[float, list[str]]:
    """Per-LINE 5-gram coverage: a line is grounded if >=60% of its 5-word shingles are in the raw.

    Robust to reformatting (joined headers, `term — definition` pairs whose fields are separate in the
    source JSON) — those still share nearly all their internal 5-grams with the raw. A fabricated line
    shares almost none, so it surfaces. `window`/`step` are kept for the CLI but the gram size is 5."""
    bad: list[str] = []
    grounded = total = 0
    for raw_line in extraction.splitlines():
        words = _norm(raw_line).split()
        if not words:
            continue
        total += 1
        if len(words) < 5:
            ok = " ".join(words) in raw_norm
        else:
            grams = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
            ok = sum(g in raw_norm for g in grams) / len(grams) >= 0.6
        if ok:
            grounded += 1
        elif len(bad) < 12:
            bad.append(raw_line.strip()[:140])
    return (grounded / total if total else 1.0), bad


def main() -> int:
    manifest = json.load(open(sys.argv[1]))
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    print(f"{'model':<8}{'register':<14}{'words':>7}{'grounded':>9}   url")
    report = []
    for d in manifest:
        hid = d["html"].split("/")[-1].replace(".html", "")
        raw_norm = _norm(open(d["html"]).read())
        for model, path in (
            ("sonnet", f"{ROOT}/extract_out/{hid}.txt"),
            ("gemini", f"{ROOT}/extract_out/gemini_{hid}.txt"),
            ("deepseek", f"{ROOT}/extract_out/deepseek_{hid}.txt"),
            ("MERGED", f"{ROOT}/extract_out/{hid}_merged.txt"),
        ):
            if not os.path.exists(path):
                continue
            txt = open(path).read()
            frac, miss = ground(txt, raw_norm, window, step)
            nwords = len(_norm(txt).split())
            flag = "" if frac >= 0.98 else "  <-- CHECK"
            print(f"{model:<8}{d.get('register', ''):<14}{nwords:>7}{frac * 100:>8.1f}%{flag}   {d['url'][:46]}")
            for m in miss[:4]:
                print(f"           ungrounded: {m}")
            report.append(
                {
                    "hid": hid,
                    "url": d["url"],
                    "model": model,
                    "grounded": round(frac, 4),
                    "words": nwords,
                    "ungrounded_samples": miss,
                }
            )
    json.dump(report, open(f"{ROOT}/edge_grounding.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
