# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""3rd-vote extraction: DeepSeek-V4-Pro (Together) over the gold set — the independent long-context vote.

Replaces gpt-oss for the 3-way {Sonnet, gemini, DeepSeek} merge: its 512k context clears the >128k
JS-blob edge docs that gpt-oss abstained on. Uses prompt_v4 (the completeness rubric that mandates
digging `<script>` data blobs) so it's a real coverage vote on these pages, not just a shell reader.
Concurrent; skip-existing; writes deepseek_<hid>.txt. Reads OAI_API_KEY (Together) from env.
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
MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def main() -> int:
    manifest = json.load(open(sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/extract_manifest.json"))
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    key = os.environ["OAI_API_KEY"]
    prompt = open(f"{ROOT}/prompt_v4.md").read()

    def run(d: dict) -> str:
        hid = os.path.basename(d["html"]).replace(".html", "")
        out = f"{ROOT}/extract_out/deepseek_{hid}.txt"
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
                print(f"  {i + 1}/{len(manifest)}  {dict(counts)}  ({int(time.time() - t0)}s)", flush=True)
    print(f"deepseek cross-check done: {dict(counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
