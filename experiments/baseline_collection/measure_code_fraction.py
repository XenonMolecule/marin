# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate how much code-like text each tokenized cache contains.

Samples N random docs per cache, decodes them, and applies line-level heuristics
to classify code-ish lines (statement terminators/braces at line end, indented
call-like lines, common programming keywords). Reports per-cache the fraction of
docs containing a code block (>=3 code-ish lines) and the fraction of characters
on code-ish lines. Heuristics are crude but applied identically to every cache,
so between-cache ratios are meaningful even if absolute levels are not.

Run IN-REGION (us-central2) so the cache reads are not cross-region egress.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import fsspec
import numpy as np
from levanter.data.text.cache import load_lm_dataset_cache
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import load_tokenizer

TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"

CODE_LINE_END = re.compile(r"[;{}]\s*$|^\s*[})\]];?\s*$")
CODE_KEYWORD = re.compile(
    r"\b(def |return |import |from \w+ import|#include|public static|function \w+\(|"
    r"void \w+\(|int \w+\(|console\.log|printf\(|System\.out|=> |var \w+ =|const \w+ =|"
    r"if \(|for \(|while \()"
)
INDENTED_CALL = re.compile(r"^\s{4,}\S+.*[()=]")


def is_code_line(line: str) -> bool:
    if len(line.strip()) < 3:
        return False
    return bool(CODE_LINE_END.search(line) or CODE_KEYWORD.search(line) or INDENTED_CALL.match(line))


def measure_cache(name: str, cache_dir: str, n_docs: int, tokenizer) -> dict:
    print(f"=== {name}: {cache_dir}", flush=True)
    ds = load_lm_dataset_cache(cache_dir, TextLmDatasetFormat(), tokenizer, enforce_eos=True)
    total = len(ds)
    rng = np.random.default_rng(0)
    indices = sorted(rng.choice(total, size=min(n_docs, total), replace=False).tolist())
    docs = ds.get_batch_sync(indices)

    docs_with_block = 0
    code_chars = 0
    total_chars = 0
    examples = []
    for idx, d in zip(indices, docs, strict=True):
        text = tokenizer.decode(np.asarray(d["input_ids"]), skip_special_tokens=True)
        lines = text.split("\n")
        code_lines = [ln for ln in lines if is_code_line(ln)]
        total_chars += len(text)
        code_chars += sum(len(ln) for ln in code_lines)
        if len(code_lines) >= 3:
            docs_with_block += 1
            if len(examples) < 3:
                examples.append({"idx": idx, "sample_code_lines": code_lines[:5]})

    result = {
        "name": name,
        "sampled_docs": len(indices),
        "frac_docs_with_code_block": docs_with_block / len(indices),
        "frac_chars_on_code_lines": code_chars / max(1, total_chars),
        "examples": examples,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", nargs="+", required=True, metavar="NAME=GS_PATH")
    parser.add_argument("--output", required=True, help="gs:// path for the JSON report")
    parser.add_argument("--docs", type=int, default=3000)
    args = parser.parse_args()

    tokenizer = load_tokenizer(TOKENIZER)
    results = []
    for spec in args.caches:
        name, cache_dir = spec.split("=", 1)
        results.append(measure_cache(name, cache_dir, args.docs, tokenizer))

    with fsspec.open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
