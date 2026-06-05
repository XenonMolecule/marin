# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the random-3k curation caches start each doc with BOS.

Same check as verify_10k_bos.py, for the independent random-3000-WARC draw
caches (dclm_random_3000 / nemotron_full_random_3000 / resiliparse_random_dedup_3000).
The nemotron cache lacks the ``_bos_fixed`` marker, and the upstream merge
reverted the Levanter BOS patch, so a recently-tokenized cache could be
BOS-broken — verify before launching the fixed-model sweep on these methods.

Run IN-REGION (us-central2, where all three caches live):

    uv run iris --cluster marin job run --region us-central2 \
        --cpu 2 --memory 8GB --disk 16GB --enable-extra-resources \
        --priority batch --no-wait --extra cpu \
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \
        -- python experiments/baseline_collection/verify_random3k_bos.py
"""

from __future__ import annotations

import sys

import numpy as np
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

CACHES: dict[str, str] = {
    "dclm_random_3000": "gs://marin-us-central2/tokenized/baseline_dclm-cf177e/train",
    "nemotron_full_random_3000": "gs://marin-us-central2/tokenized/baseline_nemotron_full-75f981/train",
    "resiliparse_random_dedup_3000": "gs://marin-us-central2/tokenized/resiliparse_random_dedup_3000warcs-7de1e2/train",
}


def check(name: str, cache_dir: str, n: int, tokenizer) -> dict:
    bos = tokenizer.bos_token_id
    print(f"\n=== {name} ===\n  dir: {cache_dir}", flush=True)
    ds = load_lm_dataset_cache(cache_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    total = len(ds)
    rng = np.random.default_rng(0)
    idx = sorted({0, 1, 2, total // 2, total - 1, *rng.choice(total, size=max(0, n - 5), replace=False).tolist()})[:n]
    docs = ds.get_batch_sync(idx)
    n_bos = sum(1 for d in docs if np.asarray(d["input_ids"]).size and int(np.asarray(d["input_ids"])[0]) == bos)
    frac = n_bos / len(idx)
    verdict = "BOS-PRESENT" if frac >= 0.99 else ("BOS-MISSING" if frac <= 0.01 else "PARTIAL")
    print(f"  total_docs={total}  first-token==BOS({bos}): {n_bos}/{len(idx)} ({frac:.0%})  -> {verdict}")
    print(f"  example doc[0] head: {np.asarray(docs[0]['input_ids'])[:6].tolist()}")
    return {"name": name, "verdict": verdict, "frac": frac}


def main() -> int:
    tok = load_tokenizer(TOKENIZER)
    print(f"bos_id={tok.bos_token_id} eos_id={tok.eos_token_id}", flush=True)
    results = [check(n, p, 40, tok) for n, p in CACHES.items()]
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['verdict']:<12} {r['name']} ({r['frac']:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
