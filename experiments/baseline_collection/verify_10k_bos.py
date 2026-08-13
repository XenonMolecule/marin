# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify whether the ExpC 10k tokenized caches start each doc with a BOS token.

Unlike `verify_bos_fix_tokens.py` (which diffs an old/new cache pair), the 10k
caches have no pre-fix counterpart — they were tokenized once. So we directly
check the absolute property: does every doc's first token equal the tokenizer
BOS id? A BOS-correct cache should be ~100%; a cache built under the
2026-04-10 Levanter regression (add_special_tokens=False) will be ~0%.

Run IN-REGION (us-central2) so the cache read is not cross-region egress:

    uv run iris --cluster marin job run --region us-central2 \
        --cpu 2 --memory 8GB --disk 16GB --priority batch --no-wait \
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> --extra cpu \
        -- python experiments/baseline_collection/verify_10k_bos.py
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

# ExpC 10k caches (us-central2). dclm_10k / nemotron_10k feed the sliced,
# simulated-epoching expC_T33T sweep. Override with --caches name=gs://... to
# check other caches (e.g. the fineweb 10k pair).
DEFAULT_CACHES: dict[str, str] = {
    "dclm_10k": "gs://marin-us-central2/tokenized/dclm_400m_1x_10k_dclm-3df0ba/train",
    "nemotron_10k": "gs://marin-us-central2/tokenized/dclm_400m_1x_10k_nemotron_full-3dcb75/train",
}


def check_cache(name: str, cache_dir: str, n_samples: int, tokenizer) -> dict:
    bos_id = tokenizer.bos_token_id
    print(f"\n=== {name} ===", flush=True)
    print(f"  dir: {cache_dir}")

    ds = load_lm_dataset_cache(cache_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    total = len(ds)
    print(f"  total_docs: {total}")

    rng = np.random.default_rng(0)
    anchors = {0, 1, 2, total // 4, total // 2, 3 * total // 4, total - 1}
    extra = rng.choice(total, size=max(0, n_samples - len(anchors)), replace=False).tolist()
    indices = sorted(set(anchors) | set(extra))[:n_samples]

    docs = ds.get_batch_sync(indices)
    n_bos = 0
    examples = []
    for idx, d in zip(indices, docs, strict=True):
        ids = np.asarray(d["input_ids"])
        first = int(ids[0]) if ids.size else -1
        if first == bos_id:
            n_bos += 1
        if len(examples) < 6:
            examples.append((idx, ids[:6].tolist()))

    n = len(indices)
    frac = n_bos / n if n else 0.0
    print(f"  sampled={n}  first-token==BOS({bos_id}): {n_bos}/{n}  ({frac:.0%})")
    print("  example first-6 tokens per doc:")
    for idx, head in examples:
        print(f"    idx={idx}: {head}")

    verdict = "BOS-PRESENT" if frac >= 0.99 else ("BOS-MISSING" if frac <= 0.01 else "PARTIAL/UNCLEAR")
    print(f"  VERDICT: {verdict}")
    return {"name": name, "sampled": n, "n_bos": n_bos, "frac": frac, "verdict": verdict}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=40, help="docs to sample per cache")
    parser.add_argument(
        "--caches",
        nargs="+",
        metavar="NAME=GS_PATH",
        help="caches to check as name=gs://.../train pairs (default: the ExpC 10k pair)",
    )
    args = parser.parse_args()

    caches = dict(spec.split("=", 1) for spec in args.caches) if args.caches else DEFAULT_CACHES

    tokenizer = load_tokenizer(TOKENIZER)
    print(f"bos_id={tokenizer.bos_token_id}  eos_id={tokenizer.eos_token_id}", flush=True)

    results = [check_cache(name, d, args.samples, tokenizer) for name, d in caches.items()]

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['verdict']:<16} {r['name']}: {r['n_bos']}/{r['sampled']} docs start with BOS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
