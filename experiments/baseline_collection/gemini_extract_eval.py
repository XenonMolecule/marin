# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a Gemini extraction prompt against the agent-produced gold extractions.

The Claude-agent extractions are the target; this measures how close a scalable Gemini prompt gets,
so we can iterate the prompt until Gemini is good enough to run at scale. Reads GEMINI_API_KEY from the
environment (never stored in this file). Compares Gemini output to agent gold with a token-level
edit-distance similarity (difflib ratio ~= 1 - normalized Levenshtein).
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = "scratch/gold_extraction"


def gemini(prompt: str, html: str, model: str, retries: int = 2) -> str:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": prompt + "\n\n=== RAW PAGE HTML ===\n" + html}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65536},
    }
    d = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            return f"[HTTP {e.code}: {e.read()[:300].decode(errors='replace')}]"
        except Exception as e:  # socket.timeout, connection resets, etc.
            if attempt < retries:
                time.sleep(4 * (attempt + 1))
                continue
            return f"[error after {retries + 1} tries: {type(e).__name__}: {e}]"
    cands = (d or {}).get("candidates") or []
    if not cands:
        return f"[no candidate: {json.dumps(d)[:300]}]"
    parts = cands[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts) or f"[empty/blocked: {json.dumps(cands[0])[:200]}]"


def similarity(a: str, b: str) -> float:
    """Token-level similarity in [0,1] (~1 - normalized edit distance)."""
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


def main() -> int:
    import urllib.error  # noqa: F401  (referenced in gemini())

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=f"{ROOT}/prompt_v1.md")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--out-tag", default="v1")
    args = ap.parse_args()

    os.makedirs(f"{ROOT}/gemini_out", exist_ok=True)
    prompt = open(args.prompt).read()
    results = json.load(open(f"{ROOT}/extraction_results_noted.json"))
    staged = json.load(open(f"{ROOT}/noted_staged.json"))
    url2i = {s["url"]: s["_i"] for s in staged}

    rows = []
    for r in results:
        i = url2i.get(r["url"])
        gold = r.get("gold_text") or ""
        if i is None or not gold:
            continue
        html = open(f"{ROOT}/html_noted/doc_{i}.html").read()
        out = gemini(prompt, html, args.model)
        sim = similarity(out, gold)
        rows.append({"i": i, "register": r["register"], "sim": round(sim, 3), "gem_len": len(out), "gold_len": len(gold)})
        json.dump({"text": out}, open(f"{ROOT}/gemini_out/{args.out_tag}_doc_{i}.json", "w"))
        print(f"  doc_{i:<2} [{r['register']:<14}] sim={sim:.3f}  gemini={len(out):>6}c  gold={len(gold):>6}c")

    avg = sum(x["sim"] for x in rows) / len(rows) if rows else 0.0
    print(f"\n[{args.model}] AVG token-similarity to agent gold: {avg:.3f} over {len(rows)} docs")
    json.dump({"model": args.model, "prompt": args.prompt, "avg": avg, "rows": rows},
              open(f"{ROOT}/gemini_eval_{args.out_tag}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
