# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decode sampled documents from tokenized caches so the raw text can be inspected.

Companion to `verify_10k_bos.py`: that script checks token-level structure (BOS);
this one detokenizes whole documents so extraction/encoding artifacts (mojibake,
stripped newlines, boilerplate, truncation) are visible to a human. Writes one
text report per cache to --output-prefix.

Run IN-REGION (us-central2) so the cache read is not cross-region egress.
"""

from __future__ import annotations

import argparse
import sys

import fsspec
import numpy as np
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"


def dump_cache(name: str, cache_dir: str, n_docs: int, max_chars: int, out_prefix: str, tokenizer) -> None:
    print(f"=== {name}: {cache_dir}", flush=True)
    ds = load_lm_dataset_cache(cache_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    total = len(ds)

    rng = np.random.default_rng(0)
    indices = sorted(rng.choice(total, size=n_docs, replace=False).tolist())
    docs = ds.get_batch_sync(indices)

    lines = [f"cache: {cache_dir}\ntotal_docs: {total}\n"]
    for idx, d in zip(indices, docs, strict=True):
        ids = np.asarray(d["input_ids"])
        text = tokenizer.decode(ids, skip_special_tokens=False)
        n_replacement = text.count("�")
        lines.append(
            f"\n{'=' * 80}\ndoc idx={idx}  n_tokens={ids.size}  n_chars={len(text)}  "
            f"n_newlines={text.count(chr(10))}  n_replacement_chars={n_replacement}\n{'-' * 80}\n"
            f"{text[:max_chars]}\n"
        )

    out_path = f"{out_prefix}/{name}.txt"
    with fsspec.open(out_path, "w") as f:
        f.write("".join(lines))
    print(f"  wrote {out_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", nargs="+", required=True, metavar="NAME=GS_PATH")
    parser.add_argument("--output-prefix", required=True, help="gs:// prefix for per-cache text reports")
    parser.add_argument("--docs", type=int, default=12)
    parser.add_argument("--max-chars", type=int, default=3000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER)
    for spec in args.caches:
        name, cache_dir = spec.split("=", 1)
        dump_cache(name, cache_dir, args.docs, args.max_chars, args.output_prefix, tokenizer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
