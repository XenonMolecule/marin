# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concurrent extraction eval against any OpenAI-compatible endpoint (e.g. gpt-oss-120b via a provider).

Mirrors gemini_extract_eval but (a) speaks the OpenAI /chat/completions schema so it works for any
provider (OpenRouter/Groq/Together/Fireworks/Cerebras/vLLM), and (b) fans the per-doc calls out over a
thread pool so a slow model doesn't serialize the whole run. Scores token-similarity to the agent gold
and writes per-doc outputs into the same gemini_out/ folder so results merge with the Gemini tables.

Reads the API key from the env var named by --api-key-env (never stored). base_url + model are args.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from experiments.baseline_collection.gemini_extract_eval import ROOT, similarity
from experiments.baseline_collection.gemini_model_compare import load_docs

_RETRYABLE = {408, 409, 429, 500, 502, 503, 504}


def chat(base_url: str, model: str, key: str, prompt: str, html: str, max_tokens: int, retries: int = 3) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt + "\n\n=== RAW PAGE HTML ===\n" + html}],
    }
    ctx = ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or None)
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    # some providers front the API with Cloudflare, which 403s (err 1010) the default
                    # python-urllib User-Agent; send a normal one.
                    "User-Agent": "Mozilla/5.0 (marin-extract-eval)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=600, context=ctx) as r:
                d = json.loads(r.read())
            choices = d.get("choices") or []
            if not choices:
                return f"[no choice: {json.dumps(d)[:300]}]"
            return choices[0].get("message", {}).get("content") or f"[empty: {json.dumps(choices[0])[:200]}]"
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode(errors="replace")
            if e.code in _RETRYABLE and attempt < retries:
                time.sleep(4 * (attempt + 1))
                continue
            return f"[HTTP {e.code}: {msg}]"
        except Exception as e:  # timeout / reset
            if attempt < retries:
                time.sleep(4 * (attempt + 1))
                continue
            return f"[error after {retries + 1} tries: {type(e).__name__}: {e}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible root, e.g. https://openrouter.ai/api/v1")
    ap.add_argument("--model", required=True, help="e.g. openai/gpt-oss-120b")
    ap.add_argument("--api-key-env", default="OAI_API_KEY")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--out-tag", default="oai")
    args = ap.parse_args()

    key = os.environ[args.api_key_env]
    prompt = open(f"{ROOT}/prompt_v1.md").read()
    docs = load_docs()
    os.makedirs(f"{ROOT}/gemini_out", exist_ok=True)
    label = args.model.replace("/", "_")

    def run(d: dict) -> dict:
        out = chat(args.base_url, args.model, key, prompt, d["html"], args.max_tokens)
        json.dump({"text": out}, open(f"{ROOT}/gemini_out/{args.out_tag}_{label}_doc_{d['i']}.json", "w"))
        sim = similarity(out, d["gold"])
        return {"i": d["i"], "register": d["register"], "sim": round(sim, 3),
                "out_len": len(out), "gold_len": len(d["gold"])}

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = sorted(ex.map(run, docs), key=lambda r: r["i"])

    for r in rows:
        print(f"  doc_{r['i']:<2} [{r['register']:<14}] sim={r['sim']:.3f}  out={r['out_len']:>6}c  gold={r['gold_len']:>6}c")
    avg = sum(r["sim"] for r in rows) / len(rows) if rows else 0.0
    print(f"\n[{args.model}] AVG token-similarity to agent gold: {avg:.3f} over {len(rows)} docs (concurrency={args.concurrency})")
    json.dump({"model": args.model, "base_url": args.base_url, "avg": avg, "rows": rows},
              open(f"{ROOT}/oai_eval_{args.out_tag}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
