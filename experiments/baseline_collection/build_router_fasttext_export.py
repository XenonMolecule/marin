# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the jusText router dataset into a fastText training set.

The router parts (justext_router_labels/{split}/part-*.jsonl.gz) carry per-doc:
  text     = body_strip HTML (the doc)
  label    = 1 if jusText's extraction reached >= 0.85 normalized Levenshtein sim to the gold
             LLM extraction, else 0  (1 = handle with jusText; 0 = route to the 1.7B model)

This emits fastText lines `__label__<extractable|needs_llm> <to_fasttext_text(text)>` (body text
collapsed + lowercased — same prep as the useful classifier), one gzipped file per split.

Output: {router_root}/fasttext/{train,dev,test}.txt.gz  (gunzip before `fasttext supervised`).
"""

import argparse
import gzip
import json
import os
import re
import time

import fsspec

_WS_RE = re.compile(r"\s+")
SPLITS = ("dev", "test", "train")  # dev/test first (done sooner); skip a split with no parts yet
LABELS = {1: "extractable", 0: "needs_llm"}


def to_fasttext_text(body_html: str) -> str:
    return _WS_RE.sub(" ", body_html).strip().lower()


def iter_parts(router_root: str, split: str):
    fs, root = fsspec.core.url_to_fs(f"{router_root}/{split}")
    parts = sorted(fs.glob(f"{root}/part-*.jsonl.gz"))
    for p in parts:
        path = p if p.startswith("gs://") else f"gs://{p}"
        with fsspec.open(path, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
            for line in g:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--router-root", required=True, help="gs://.../justext_router_labels")
    ap.add_argument("--out-root", required=True, help="gs://.../justext_router_labels/fasttext")
    ap.add_argument("--splits", default="dev,test,train")
    args = ap.parse_args()

    for split in args.splits.split(","):
        fs, root = fsspec.core.url_to_fs(f"{args.router_root}/{split}")
        if not fs.exists(root) or not fs.glob(f"{root}/part-*.jsonl.gz"):
            print(f"{split}: no parts yet, skipping", flush=True)
            continue
        out_path = f"{args.out_root}/{split}.txt.gz"
        tmp = f"/app/_router_ft_{split}.txt.gz"
        n = pos = empty = 0
        t0 = time.time()
        with gzip.open(tmp, "wt", encoding="utf-8") as out:
            for d in iter_parts(args.router_root, split):
                ft = to_fasttext_text(d.get("text", ""))
                if not ft:
                    empty += 1
                    continue
                lab = LABELS[int(d["label"])]
                out.write(f"__label__{lab} {ft}\n")
                n += 1
                pos += int(d["label"])
                if n % 5000 == 0:
                    print(
                        f"{split}: {n} written, pos={pos} ({100*pos/n:.1f}%), {n/(time.time()-t0):.0f} rows/s",
                        flush=True,
                    )
        with open(tmp, "rb") as src, fsspec.open(out_path, "wb") as dst:
            dst.write(src.read())
        os.remove(tmp)
        print(
            f"{split}: DONE {n} lines, extractable={pos} ({100*pos/max(n,1):.1f}%), "
            f"needs_llm={n-pos}, skipped_empty={empty} -> {out_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
