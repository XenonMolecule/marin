# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the winning fastText model on the *exact same* frozen test docs the
ModernBERT eval used, so the two classifiers' per-doc scores can be aligned
index-by-index for a true cascade (stacked-filter) analysis.

Reproduces modernbert_tpu_smoke.read_test verbatim (sorted shards, random.Random(0),
even per-shard sampling) so doc i here == doc i in the BERT result JSON's `preds`.
Outputs {ft_probs, labels} — labels double as an alignment check against BERT's preds.
Runs in-region (CPU); fastText is cheap, only the ~7k probs leave the job.
"""

from __future__ import annotations

import argparse
import json
import random

import fasttext
import fsspec

LABEL_USEFUL = "__label__useful"


def _valid_lines(path: str):
    """Yield (text, label) per valid line — same filter as modernbert read_fasttext
    (skip blank lines and lines with empty text). Streaming: O(1) memory."""
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, _, text = line.partition(" ")
            if not text:
                continue
            yield text, (1 if label == LABEL_USEFUL else 0)


def read_test(path: str, max_rows: int) -> tuple[list[str], list[int]]:
    """Reproduces modernbert_tpu_smoke.read_test (seed 0) for exact alignment, but
    memory-safe: two streaming passes per shard (count, then keep the sampled indices)
    instead of loading the whole shard. Same rng.shuffle sequence -> identical sample
    and identical output order, so doc i here == doc i in BERT's preds."""
    fs = fsspec.filesystem("gcs")
    shards = sorted("gs://" + p for p in fs.glob(path))
    per = max(1, max_rows // len(shards))
    rng = random.Random(0)
    texts, labels = [], []
    for shard in shards:
        count = sum(1 for _ in _valid_lines(shard))  # pass 1: count valid lines
        order = list(range(count))
        rng.shuffle(order)  # consumes rng exactly as the original (one shuffle per shard)
        sel = order[:per]
        want = set(sel)
        content: dict[int, tuple[str, int]] = {}
        for i, tl in enumerate(_valid_lines(shard)):  # pass 2: keep only sampled
            if i in want:
                content[i] = tl
        for j in sel:  # emit in shuffled order, matching the original
            t, lab = content[j]
            texts.append(t)
            labels.append(lab)
    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True, help="Test glob (must match the BERT eval's --test).")
    ap.add_argument("--model", required=True, help="gs:// path to fastText model.bin.")
    ap.add_argument("--rows", type=int, default=7000)
    ap.add_argument("--out", required=True, help="gs:// path for the {ft_probs, labels} JSON.")
    args = ap.parse_args()

    texts, labels = read_test(args.test, args.rows)
    local = "/tmp/ft_model.bin"
    with fsspec.open(args.model, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    model = fasttext.load_model(local)

    probs = []
    for t in texts:
        t = t.replace("\n", " ")  # fastText.predict rejects embedded newlines
        lbls, prs = model.predict(t, k=2)
        d = dict(zip(lbls, [float(p) for p in prs]))
        probs.append(d.get(LABEL_USEFUL, 0.0))

    with fsspec.open(args.out, "wt", encoding="utf-8") as f:
        json.dump({"ft_probs": probs, "labels": labels}, f)
    print(f"scored {len(probs)} docs -> {args.out}; useful={sum(labels)} ({sum(labels)/len(labels):.1%})")


if __name__ == "__main__":
    main()
