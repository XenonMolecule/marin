# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the ONE-EXTRACTION classifier input and measure how far it is from the trained-on input.

Production will run resiliparse-rs exactly once, on the raw HTML: its output is the training text AND,
after ``normalize_text`` (whitespace-collapse, strip, lower), the TEXT classifiers' input. The TEXT
models, however, were trained on text extracted from the *body_strip* HTML (already lowercased and
whitespace-collapsed BEFORE extraction) — ``clf_text_resiliparse_rs``. This job writes the deployment
variant, row-aligned:

    clf_text_resiliparse_rs_fromraw = normalize_text(text_resiliparse_rs)      # text_resiliparse_rs = extract(raw_html)

and a comparison JSON (exact-match rate, char similarity, length ratio, empty-mismatch counts) so the
"do we need to retrain?" question rests on measurement. Score agreement is the decisive test and comes
from scoring a TEXT model on both inputs (``score_modernbert_useful`` / ``score_fasttext_useful``).

Run (CPU, us-east5; rapidfuzz needs --extra extraction-bakeoff)::

    uv run iris --cluster marin job run --region us-east5 --cpu 4 --memory 32GB \\
      --enable-extra-resources --extra cpu --extra extraction-bakeoff --priority interactive --no-wait \\
      --job-name clf-text-fromraw -- python -m experiments.baseline_collection.build_clf_text_from_raw
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz.distance import Levenshtein

from experiments.baseline_collection.comparison_sample import (
    CLF_TEXT_COLUMN,
    CLF_TEXT_FROM_RAW_COLUMN,
    OUT_ROOT,
    clf_text_dir,
    clf_text_from_raw_dir,
    normalize_text,
)
from experiments.fsspec_paths import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_TEXT_DIR = f"{OUT_ROOT}/text_resiliparse_rs"  # score_resiliparse_rs output = extract(raw_html)
REPORT = f"{OUT_ROOT}/clf_text_source_comparison.json"
LEV_CAP = 20_000
_SCHEMA = pa.schema([("warc_record_id", pa.string()), (CLF_TEXT_FROM_RAW_COLUMN, pa.string())])


def run(input_bucket: str) -> None:
    raw_files = sorted(fsspec_glob(f"{RAW_TEXT_DIR}/*.parquet"))
    clf_files = sorted(fsspec_glob(f"{clf_text_dir(input_bucket)}/*.parquet"))
    if len(raw_files) != len(clf_files):
        raise RuntimeError(f"{len(raw_files)} raw-text shards vs {len(clf_files)} clf-text shards")
    out_dir = clf_text_from_raw_dir(input_bucket)

    n = exact = both_empty = only_a_empty = only_b_empty = 0
    sims: list[float] = []
    len_ratio: list[float] = []
    for i, (rf, cf) in enumerate(zip(raw_files, clf_files, strict=True)):
        with fsspec.open(rf, "rb") as fh:
            rt = pq.ParquetFile(fh).read()
        with fsspec.open(cf, "rb") as fh:
            ct = pq.ParquetFile(fh).read()
        ids = rt.column("warc_record_id").to_pylist()
        if ids != ct.column("warc_record_id").to_pylist():
            raise RuntimeError(f"shard {i}: raw text and clf text are not row-aligned")
        b = [normalize_text(t) for t in rt.column("text_resiliparse_rs").to_pylist()]  # extract(raw) -> norm
        a = [t or "" for t in ct.column(CLF_TEXT_COLUMN).to_pylist()]  # as trained
        for x, y in zip(a, b, strict=True):
            n += 1
            if not x and not y:
                both_empty += 1
                continue
            if not x:
                only_a_empty += 1
            elif not y:
                only_b_empty += 1
            if x == y:
                exact += 1
            sims.append(Levenshtein.normalized_similarity(x[:LEV_CAP], y[:LEV_CAP]))
            len_ratio.append(len(y) / max(len(x), 1))
        with fsspec.open(f"{out_dir}/{cf.rsplit('/', 1)[-1]}", "wb") as fh:
            pq.write_table(
                pa.Table.from_pydict({"warc_record_id": ids, CLF_TEXT_FROM_RAW_COLUMN: b}, schema=_SCHEMA),
                fh,
                compression="zstd",
            )
        if i % 25 == 0:
            logger.info(
                "shard %d/%d: n=%d exact=%.3f sim=%.4f",
                i,
                len(raw_files),
                n,
                exact / max(n, 1),
                float(np.mean(sims)) if sims else 0,
            )

    s = np.array(sims)
    r = np.array(len_ratio)
    report = {
        "n_docs": n,
        "a": "clf_text_resiliparse_rs = norm(extract(body_strip_lower_ws(html)))  [as trained]",
        "b": "clf_text_resiliparse_rs_fromraw = norm(extract(raw_html))  [one-extraction deployment]",
        "exact_match_rate": exact / n,
        "both_empty": both_empty,
        "only_trained_input_empty": only_a_empty,
        "only_deployment_input_empty": only_b_empty,
        "char_similarity_mean": float(s.mean()),
        "char_similarity_p10": float(np.percentile(s, 10)),
        "char_similarity_p50": float(np.percentile(s, 50)),
        "frac_similarity_below_0.9": float((s < 0.9).mean()),
        "frac_similarity_below_0.5": float((s < 0.5).mean()),
        "len_ratio_b_over_a_mean": float(r.mean()),
        "len_ratio_p10": float(np.percentile(r, 10)),
        "len_ratio_p90": float(np.percentile(r, 90)),
    }
    with fsspec.open(REPORT, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("REPORT %s", json.dumps(report))
    logger.info("DONE -> %s", out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-bucket", default="marin-us-east5")
    run(p.parse_args().input_bucket)


if __name__ == "__main__":
    main()
