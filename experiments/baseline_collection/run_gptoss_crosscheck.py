# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""3rd-vote extraction: gpt-oss-120b (Together) over the gold set — the weakest vote for 3-way merge.

Independent model family (OpenAI open weights) so a 3-way {Sonnet, gemini, gpt-oss} disagreement is
strong signal for the merge-doctor pass. Concurrent + backoff; skip-existing; writes gptoss_<hid>.txt.
Docs whose HTML exceeds gpt-oss's 128k context error out (it simply abstains as a checker there).
Reads OAI_API_KEY (Together) from env.
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.baseline_collection.oai_extract_eval import chat

ROOT = "scratch/gold_extraction"
BASE_URL = "https://api.together.xyz/v1"
MODEL = "openai/gpt-oss-120b"


def main() -> int:
    manifest = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/extract_manifest.json"))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    key = os.environ["OAI_API_KEY"]
    prompt = open(f"{ROOT}/prompt_v3.md").read()

    def run(d: dict) -> str:
        hid = os.path.basename(d["html"]).replace(".html", "")
        out = f"{ROOT}/extract_out/gptoss_{hid}.txt"
        if os.path.exists(out) and os.path.getsize(out) > 0 and not open(out).read().startswith("["):
            return "skip"
        txt = chat(BASE_URL, MODEL, key, prompt, open(d["html"]).read(), max_tokens=32000)
        open(out, "w").write(txt)
        return "err" if txt.startswith("[") else "ok"

    counts: collections.Counter = collections.Counter()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(run, manifest)):
            counts[r] += 1
            if (i + 1) % 40 == 0:
                print(f"  {i + 1}/{len(manifest)}  {dict(counts)}  ({int(time.time()-t0)}s)", flush=True)
    print(f"gpt-oss cross-check done: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
