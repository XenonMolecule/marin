# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a fastText useful-classifier over the 100k comparison sample -> scalar P(useful) column.

Mirrors score_modernbert_useful (same body_strip preprocessing, raw scalar so the threshold is
tunable). fastText is CPU + fast, so this is one un-sharded job. Writes to the SAME
``bert_scores/<col>/`` layout so score_modernbert_useful's ``join`` merges it as another column.

Run (CPU, us-east5 where the sample lives; model.bin pulled from us-central2 once)::

    uv run iris --cluster marin job run --region us-east5 --cpu 8 --memory 32GB --disk 20GB \\
      --enable-extra-resources --extra cpu --extra dclm --priority interactive --no-wait \\
      --job-name ft-score -- python -m experiments.baseline_collection.score_fasttext_useful
"""

from __future__ import annotations

import argparse
import logging
import re
import time

import fsspec
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SCORES_DIR = f"{OUT_ROOT}/model_scores"
SAMPLE_DIR = f"{OUT_ROOT}/sample_100k"

# Default = the w80 model (us-central2). Width sweep (w160/w320) lives in us-east5; pass --model-url/--col.
MODEL = "gs://marin-us-central2/classifiers/useful_fasttext/body_strip_scale_w80_strat_prep_mc500/model.bin"
COL = "fasttext_useful_prob"
USEFUL_LABEL = "__label__useful"
MAX_TEXT_CHARS = 1_000_000

_WS_RE = re.compile(r"\s+")


def _preprocess(stripped_html: str | None) -> str:
    """body_strip HTML -> classifier input (whitespace-collapse + lowercase), matching to_fasttext_text."""
    return _WS_RE.sub(" ", (stripped_html or "")[:MAX_TEXT_CHARS]).strip().lower()


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


def run(model_url: str, col: str) -> None:
    import fasttext
    import pyarrow as pa
    import pyarrow.parquet as pq

    local = "/tmp/ft_model.bin"
    logger.info("downloading model %s", model_url)
    with fsspec.open(model_url, "rb") as src, open(local, "wb") as dst:
        dst.write(src.read())
    model = fasttext.load_model(local)

    ids: list[str] = []
    scores: list[float] = []
    t_score = 0.0  # pure inference time (excludes parquet I/O), for a clean per-core docs/s
    for path in sorted(fsspec_glob(f"{SAMPLE_DIR}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "stripped_html"])
        rids = t.column("warc_record_id").to_pylist()
        htmls = t.column("stripped_html").to_pylist()
        t0 = time.monotonic()
        for rid, html in zip(rids, htmls, strict=True):
            ids.append(rid)
            scores.append(_useful_prob(model, _preprocess(html)))
        t_score += time.monotonic() - t0
        if len(ids) % 20000 < len(rids):
            logger.info("scored %d docs", len(ids))

    logger.info(
        "TIMING col=%s: scored %d docs in %.1fs = %.1f docs/s/core (single CPU; excludes parquet I/O)",
        col,
        len(ids),
        t_score,
        len(ids) / t_score if t_score else 0.0,
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
    args = p.parse_args()
    run(args.model_url, args.col)


if __name__ == "__main__":
    main()
