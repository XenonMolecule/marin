# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off BOS sniff for the three Apr-7 cache hashes flagged by issue #5149.

Loads each cache, checks whether the first token of docs 0/1/2/mid/-1 is BOS.
Based on verify_bos_fix_tokens.py's loader pattern.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "SSL_CERT_FILE",
    str(Path(__file__).resolve().parents[2] / ".venv/lib/python3.11/site-packages/certifi/cacert.pem"),
)

from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

CACHES = {
    "baseline_raw_html-7efa1b": "gs://marin-us-central2/tokenized/baseline_raw_html-7efa1b/train",
    "baseline_dclm-7ae3fe": "gs://marin-us-central2/tokenized/baseline_dclm-7ae3fe/train",
    "baseline_dclm-b9ccd2": "gs://marin-us-central2/tokenized/baseline_dclm-b9ccd2/train",
}


def sniff(name: str, path: str, tokenizer) -> None:
    bos = tokenizer.bos_token_id
    print(f"\n=== {name} ===")
    print(f"  path: {path}")
    try:
        ds = load_lm_dataset_cache(path, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    except Exception as e:
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        return
    n = len(ds)
    print(f"  total_docs: {n}")
    if n == 0:
        print("  empty cache")
        return
    indices = sorted({0, 1, 2, min(100, n - 1), n // 2, n - 1})
    try:
        docs = ds.get_batch_sync(indices)
    except Exception as e:
        print(f"  GET_BATCH FAILED: {type(e).__name__}: {e}")
        return
    hits = 0
    for idx, doc in zip(indices, docs, strict=True):
        ids = np.asarray(doc["input_ids"])
        has_bos = ids.size and int(ids[0]) == bos
        if has_bos:
            hits += 1
        print(f"  doc[{idx}]: len={ids.size} first5={ids[:5].tolist()} BOS_at_start={has_bos}")
    print(f"  VERDICT: {hits}/{len(indices)} docs start with BOS  →  {'PROPER' if hits == len(indices) else 'BUGGED'}")


def main() -> int:
    tokenizer = load_tokenizer(TOKENIZER)
    print(f"bos_id={tokenizer.bos_token_id}  eos_id={tokenizer.eos_token_id}")
    for name, path in CACHES.items():
        sniff(name, path, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
