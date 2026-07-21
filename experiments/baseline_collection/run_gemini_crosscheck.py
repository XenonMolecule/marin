# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent gemini-2.5-flash cross-check for the gold extraction set (the second verifier).

Runs the SAME prompt over each doc's HTML with gemini, so the manager gate can diff Sonnet vs an
independent model and catch dropped content. Concurrent with rate-limit-aware exponential backoff
(Google 429s a naive fan-out), and skip-existing so it resumes cheaply. Reads GEMINI_API_KEY from env.
"""

from __future__ import annotations

import collections
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "scratch/gold_extraction"


def gemini_backoff(prompt: str, html: str, model: str, tries: int = 6) -> str:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": prompt + "\n\n=== RAW PAGE HTML ===\n" + html}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 65536},
    }
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300, context=ctx) as r:
                d = json.loads(r.read())
            cands = d.get("candidates") or []
            if not cands:
                return f"[no candidate: {json.dumps(d)[:200]}]"
            return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts") or [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < tries - 1:
                time.sleep(min(90, 4 * 2**attempt))
                continue
            return f"[HTTP {e.code}: {e.read()[:150].decode(errors='replace')}]"
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(4 * 2**attempt)
                continue
            return f"[error: {type(e).__name__}: {e}]"


def main() -> int:
    manifest = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/extract_manifest.json"))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    prompt = open(f"{ROOT}/prompt_v3.md").read()

    def run(d: dict) -> str:
        hid = os.path.basename(d["html"]).replace(".html", "")
        out = f"{ROOT}/extract_out/gemini_{hid}.txt"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return "skip"
        txt = gemini_backoff(prompt, open(d["html"]).read(), "gemini-2.5-flash")
        open(out, "w").write(txt)
        return "err" if txt.startswith("[") else "ok"

    counts: collections.Counter = collections.Counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(run, manifest)):
            counts[r] += 1
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(manifest)}  {dict(counts)}", flush=True)
    print(f"gemini cross-check done: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
