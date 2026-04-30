# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-for-token verification that the BOS-fixed caches only differ from the
originals by a prepended BOS token.

For each (old, new) cache pair:
  1. Assert total_rows match (same number of docs).
  2. Sample N random doc indices.
  3. Fetch input_ids from both.
  4. Assert ``new[i] == [BOS] + old[i]`` byte-for-byte.

If any assertion fails, print a diff-style summary.

Usage:
    uv run python experiments/baseline_collection/verify_bos_fix_tokens.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Set SSL certs for gcsfs on macOS.
os.environ.setdefault(
    "SSL_CERT_FILE",
    str(Path(__file__).resolve().parents[2] / ".venv/lib/python3.11/site-packages/certifi/cacert.pem"),
)

from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

CACHE_PAIRS = {
    "nemotron_full": (
        "gs://marin-us-central1/tokenized/baseline_nemotron_full-d4e3af/train",
        "gs://marin-us-central1/tokenized/baseline_nemotron_full_bos_fixed-4b1ce7/train",
    ),
    "llm_curated": (
        "gs://marin-us-central1/tokenized/baseline_llm_curated-3c07e4/train",
        "gs://marin-us-central1/tokenized/baseline_llm_curated_bos_fixed-d04ef8/train",
    ),
}


def verify_pair(name: str, old_dir: str, new_dir: str, n_samples: int, tokenizer) -> dict:
    bos_id = tokenizer.bos_token_id
    print(f"\n=== {name} ===", flush=True)
    print(f"  old: {old_dir}")
    print(f"  new: {new_dir}")

    old = load_lm_dataset_cache(old_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    new = load_lm_dataset_cache(new_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)

    old_n = len(old)
    new_n = len(new)
    print(f"  total_rows: old={old_n}  new={new_n}")
    if old_n != new_n:
        return {
            "name": name,
            "ok": False,
            "reason": f"doc-count mismatch: old={old_n} new={new_n}",
        }

    rng = np.random.default_rng(0)
    # Include doc 0 explicitly plus random samples for coverage.
    indices = sorted(
        {0, 1, 2, old_n // 4, old_n // 2, old_n - 1, *rng.choice(old_n, size=n_samples - 7, replace=False).tolist()}
    )[:n_samples]

    old_docs = old.get_batch_sync(indices)
    new_docs = new.get_batch_sync(indices)

    n_bos_prefix_match = 0
    n_old_already_has_bos = 0
    n_new_missing_bos = 0
    n_tail_mismatch = 0
    mismatches = []
    for idx, od, nd in zip(indices, old_docs, new_docs, strict=True):
        o = np.asarray(od["input_ids"])
        n = np.asarray(nd["input_ids"])
        old_has_bos = int(o[0]) == bos_id if o.size else False
        new_has_bos = int(n[0]) == bos_id if n.size else False
        if old_has_bos:
            n_old_already_has_bos += 1
        if not new_has_bos:
            n_new_missing_bos += 1
            mismatches.append((idx, f"new[0]={int(n[0])} != BOS={bos_id}"))
            continue
        # Expected: new == [BOS] + old  (if old didn't already start with BOS)
        if not old_has_bos:
            tail_ok = n.shape[0] == o.shape[0] + 1 and np.array_equal(n[1:], o)
        else:
            # old already had BOS (shouldn't happen for the broken caches, but handle it)
            tail_ok = np.array_equal(n, o)
        if tail_ok:
            n_bos_prefix_match += 1
        else:
            n_tail_mismatch += 1
            if len(mismatches) < 5:
                mismatches.append(
                    (
                        idx,
                        f"new.len={n.shape[0]} old.len={o.shape[0]}  "
                        f"new[:5]={n[:5].tolist()}  new[1:][:5]={n[1:6].tolist()}  "
                        f"old[:5]={o[:5].tolist()}",
                    )
                )

    n = len(indices)
    print(
        f"  sampled={n}  [BOS] + old == new: {n_bos_prefix_match}  "
        f"new missing BOS: {n_new_missing_bos}  tail mismatch: {n_tail_mismatch}  "
        f"old already had BOS: {n_old_already_has_bos}"
    )
    if mismatches:
        print("  first mismatches:")
        for idx, msg in mismatches[:5]:
            print(f"    idx={idx}: {msg}")
    ok = n_bos_prefix_match == n
    return {
        "name": name,
        "ok": ok,
        "sampled": n,
        "matched": n_bos_prefix_match,
        "new_missing_bos": n_new_missing_bos,
        "tail_mismatch": n_tail_mismatch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50, help="random doc indices to verify per cache")
    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER)
    print(f"bos_id={tokenizer.bos_token_id}  eos_id={tokenizer.eos_token_id}")

    results = []
    for name, (old, new) in CACHE_PAIRS.items():
        results.append(verify_pair(name, old, new, args.samples, tokenizer))

    print("\n=== SUMMARY ===")
    all_ok = True
    for r in results:
        status = "✅ OK" if r.get("ok") else "❌ FAIL"
        print(f"  {status}  {r['name']}: {r}")
        all_ok = all_ok and r.get("ok", False)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
