# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a TEXT-trained fastText model on the frozen 7k, aligned doc-for-doc with the HTML runs.

``score_fasttext_on_bert_test.py`` samples the frozen test by *counting valid lines* and shuffling
with ``random.Random(0)``. A line is "valid" only if it has non-empty text — and main-content
extraction legitimately returns "" for a nontrivial slice of junk pages. Running that sampler
directly on the TEXT file would therefore (a) silently DROP those pages from the evaluation, which
biases the comparison, and (b) change ``count``, which changes the shuffle, which changes the doc
ORDER and breaks index-for-index alignment with ``ft_w640_on_frozen7k.json`` and the
``pooled_*_on_frozen7k.json`` files that ``cascade_analysis.py`` consumes.

So the selection is computed from the **HTML** file (identical to every published frozen-7k number)
and then applied POSITIONALLY to the row-aligned TEXT file produced by
``extract_prep_text_rs.py --in-file/--out-file``. Docs whose extracted text is empty stay in the
evaluation and are scored at whatever fastText returns for an empty input (the class prior), which
is exactly what would happen in deployment.

Output schema matches ``score_fasttext_on_bert_test.py`` ({ft_probs, labels}) so the existing
``cascade_analysis.py`` tooling works unchanged.

Run (CPU, in the region holding the two .txt.gz files)::

    python experiments/baseline_collection/score_fasttext_text_frozen7k.py \\
      --html-test gs://.../full_prep_body_strip_test7k/test_sample_7k.txt.gz \\
      --text-test gs://.../full_prep_resiliparse_rs_test7k/test_sample_7k.txt.gz \\
      --model gs://.../resiliparse_scale_w640_sub0p22_strat_prep_mc500_TEXT/model.bin \\
      --out gs://.../eval/ft_w640_TEXT_on_frozen7k.json
"""

from __future__ import annotations

import argparse
import json
import random

import fasttext
import fsspec

LABEL_USEFUL = "__label__useful"
# The published frozen-7k doc set (arch sweep, 2026-08-10). Asserted so a silent change in the
# sample can never be mistaken for a modelling result.
EXPECTED_DOCS = 6477
EXPECTED_USEFUL = 1513


def _useful_prob(model, text: str) -> float:
    """P(__label__useful). Uses the raw C++ binding: fasttext-wheel's Python wrapper calls
    ``np.array(..., copy=False)``, which numpy>=2 turns into a hard error."""
    preds = model.f.predict(text, 2, 0.0, "strict")
    return next((float(p) for p, lbl in preds if lbl == LABEL_USEFUL), 0.0)


def _lines(path: str) -> list[str]:
    with fsspec.open(path, "rt", compression="gzip", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def select_frozen_indices(html_lines: list[str], max_rows: int) -> list[int]:
    """The absolute line numbers of the frozen sample, in the published order.

    Reproduces ``score_fasttext_on_bert_test.read_test`` for the single-shard case: enumerate the
    valid lines (non-blank, non-empty text), shuffle their ordinals with seed 0, keep the first
    ``max_rows``, and emit in that shuffled order.
    """
    valid = [i for i, line in enumerate(html_lines) if line and line.partition(" ")[2]]
    order = list(range(len(valid)))
    random.Random(0).shuffle(order)
    return [valid[j] for j in order[:max_rows]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html-test", required=True, help="The HTML frozen-7k file the published numbers used.")
    ap.add_argument("--text-test", required=True, help="Its row-aligned TEXT twin.")
    ap.add_argument("--model", required=True, help="gs:// path to the TEXT model.bin.")
    ap.add_argument("--rows", type=int, default=7000)
    ap.add_argument("--out", required=True, help="gs:// path for the {ft_probs, labels} JSON.")
    ap.add_argument("--skip-assert", action="store_true", help="Bypass the frozen doc-count assertion.")
    args = ap.parse_args()

    html_lines = _lines(args.html_test)
    text_lines = _lines(args.text_test)
    if len(html_lines) != len(text_lines):
        raise RuntimeError(f"row misalignment: html has {len(html_lines)} lines, text has {len(text_lines)}")

    picked = select_frozen_indices(html_lines, args.rows)
    labels = [1 if html_lines[i].partition(" ")[0] == LABEL_USEFUL else 0 for i in picked]
    for i in picked:  # labels must agree line-for-line, else the files are not twins
        if html_lines[i].partition(" ")[0] != text_lines[i].partition(" ")[0]:
            raise RuntimeError(f"label mismatch at line {i}")
    if not args.skip_assert and (len(picked) != EXPECTED_DOCS or sum(labels) != EXPECTED_USEFUL):
        raise RuntimeError(
            f"frozen sample drifted: got {len(picked)} docs / {sum(labels)} useful, "
            f"expected {EXPECTED_DOCS}/{EXPECTED_USEFUL}"
        )

    local = "/root/ft_text_model.bin"
    with fsspec.open(args.model, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    model = fasttext.load_model(local)

    n_empty = 0
    probs = []
    for i in picked:
        text = text_lines[i].partition(" ")[2].replace("\n", " ")
        n_empty += not text
        probs.append(_useful_prob(model, text))

    with fsspec.open(args.out, "wt", encoding="utf-8") as f:
        json.dump({"ft_probs": probs, "labels": labels, "n_empty_text": n_empty}, f)
    print(
        f"scored {len(probs)} docs -> {args.out}; useful={sum(labels)} "
        f"({sum(labels)/len(labels):.1%}); empty extractions kept: {n_empty}"
    )


if __name__ == "__main__":
    main()
