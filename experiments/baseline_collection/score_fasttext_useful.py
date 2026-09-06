# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a fastText useful-classifier over the 100k comparison sample -> scalar P(useful) column.

Mirrors score_modernbert_useful (same ``comparison_sample`` reader, raw scalar so the threshold is
tunable). ``--input html`` feeds the body_strip HTML the ``*_prep_mc500`` models trained on;
``--input text`` feeds the resiliparse-rs main-content text of that HTML (the ``*_TEXT`` models).
fastText is CPU + fast, so this is one un-sharded job. Writes to the SAME ``model_scores/<col>/``
layout so score_modernbert_useful's ``join`` merges it as another column.

Run (CPU, us-east5 where the sample lives; model.bin pulled from us-central2 once)::

    uv run iris --cluster marin job run --region us-east5 --cpu 8 --memory 32GB --disk 20GB \\
      --enable-extra-resources --extra cpu --extra dclm --priority interactive --no-wait \\
      --job-name ft-score -- python -m experiments.baseline_collection.score_fasttext_useful
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import fsspec

from experiments.baseline_collection.comparison_sample import SCORES_DIR, TIMING_DIR, ClassifierInput, read_sample

logger = logging.getLogger(__name__)

# Default = the w80 model (us-central2). Width sweep (w160/w320) lives in us-east5; pass --model-url/--col.
MODEL = "gs://marin-us-central2/classifiers/useful_fasttext/body_strip_scale_w80_strat_prep_mc500/model.bin"
COL = "fasttext_useful_prob"
USEFUL_LABEL = "__label__useful"


def _useful_prob(model, text: str) -> float:
    """P(__label__useful) from a binary fastText model; 0.0 for empty text.

    Uses the low-level ``model.f.predict`` (list of (prob, label)) to dodge the
    NumPy-2.0 ``np.array(..., copy=False)`` failure in fasttext's ``predict``.
    """
    if not text:
        return 0.0
    for a, b in model.f.predict(text, 2, 0.0, "strict"):
        prob, lbl = (a, b) if isinstance(b, str) else (b, a)
        if lbl == USEFUL_LABEL:
            return float(prob)
    return 0.0


def run(model_url: str, col: str, kind: ClassifierInput) -> None:
    import fasttext
    import pyarrow as pa
    import pyarrow.parquet as pq

    local = "/tmp/ft_model.bin"
    logger.info("downloading model %s", model_url)
    with fsspec.open(model_url, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    model = fasttext.load_model(local)

    # fastText TEXT prep shards carry a bare line for empty extractions, i.e. "" (default empty_text).
    ids, texts = read_sample("marin-us-east5", kind)
    scores: list[float] = []
    t_score = 0.0  # pure inference time (excludes parquet I/O), for a clean per-core docs/s
    for start in range(0, len(texts), 20000):
        chunk = texts[start : start + 20000]
        t0 = time.monotonic()
        scores.extend(_useful_prob(model, text) for text in chunk)
        t_score += time.monotonic() - t0
        logger.info("scored %d docs", len(scores))

    logger.info(
        "TIMING col=%s: scored %d docs in %.1fs = %.1f docs/s/core (single CPU; excludes parquet I/O)",
        col,
        len(ids),
        t_score,
        len(ids) / t_score if t_score else 0.0,
    )

    # Persist the timing next to the scores: it is the number stage_registry needs (docs/s/core),
    # and a log line is unreadable whenever finelog is down.
    with fsspec.open(f"{TIMING_DIR}/{col}.json", "w") as fh:
        json.dump(
            {
                "col": col,
                "model_url": model_url,
                "input": str(kind),
                "docs": len(ids),
                "seconds": round(t_score, 3),
                "docs_per_sec_per_core": round(len(ids) / t_score, 1) if t_score else 0.0,
                "note": "single CPU core, excludes parquet I/O",
            },
            fh,
            indent=2,
        )

    out_path = f"{SCORES_DIR}/{col}/part-000-of-001.parquet"
    table = pa.table({"warc_record_id": ids, col: scores})
    with fsspec.open(out_path, "wb") as fh:
        pq.write_table(table, fh)
    mean = sum(scores) / max(len(scores), 1)
    logger.info("wrote %d scores -> %s (mean P=%.4f)", len(ids), out_path, mean)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-url", default=MODEL, help="fastText model.bin (gs://). Default = w80.")
    p.add_argument("--col", default=COL, help="Output score column name.")
    p.add_argument(
        "--input",
        type=ClassifierInput,
        choices=list(ClassifierInput),
        default=ClassifierInput.HTML,
        help="html = body_strip HTML (what the *_prep_mc500 models trained on); text = resiliparse-rs "
        "main-content text of that HTML (the *_TEXT models). See comparison_sample.",
    )
    args = p.parse_args()
    run(args.model_url, args.col, args.input)


if __name__ == "__main__":
    main()
