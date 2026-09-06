# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared I/O for the 100k extractor-comparison sample the cascade planner is built on.

One place for the sample's paths, the classifier-input preprocessing, and row-aligned readers for
BOTH classifier input representations, so every scorer (fastText, ModernBERT-family, pooled/eqx)
sees doc *i* as doc *i* and identical text:

* ``ClassifierInput.HTML`` — ``preprocess_html(stripped_html)``: body_strip HTML, whitespace-collapsed
  and lowercased (``to_fasttext_text``). What the HTML-trained lpv11 classifiers consume.
* ``ClassifierInput.TEXT`` — the ``clf_text_resiliparse_rs`` side artifact
  (``build_clf_text_resiliparse_rs``): the XenonMolecule Rust fork's main-content extraction OF that
  same HTML, normalized the same way. What the TEXT-trained lpv11 classifiers consume. Its row order
  is asserted against the sample's ids on every read, because a silent misalignment would poison
  every text score.

Deliberately light (fsspec + pyarrow only) so CPU jobs can import it without pulling in JAX.
"""

from __future__ import annotations

import logging
import re
from enum import StrEnum

import fsspec
import pyarrow.parquet as pq

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SCORES_DIR = f"{OUT_ROOT}/model_scores"  # {col}/part-i-of-N.parquet, one dir per classifier column
SCORED_OUT = f"{OUT_ROOT}/sample_100k_scored"  # the join output every downstream reader consumes
TIMING_DIR = f"{OUT_ROOT}/model_timing"  # <col>.json — measured docs/s/core for stage_registry
SAMPLE_NS = "documents/extractor_compare/high_quality_200warc/sample_100k"
CLF_TEXT_COLUMN = "clf_text_resiliparse_rs"
CLF_TEXT_NS = f"documents/extractor_compare/high_quality_200warc/{CLF_TEXT_COLUMN}"
# The single-extraction production variant: lower(collapse(extract(raw_html))) — i.e. the extractor-
# comparison text `text_resiliparse_rs` normalized AFTER extraction (build_clf_text_from_raw). Used to
# test whether TEXT models trained on CLF_TEXT_COLUMN tolerate the one-extraction deployment.
CLF_TEXT_FROM_RAW_COLUMN = "clf_text_resiliparse_rs_fromraw"
CLF_TEXT_FROM_RAW_NS = f"documents/extractor_compare/high_quality_200warc/{CLF_TEXT_FROM_RAW_COLUMN}"

MAX_TEXT_CHARS = 1_000_000  # cap pathological multi-MB markup (matches cascade_survivor_filter)
_WS_RE = re.compile(r"\s+")


class ClassifierInput(StrEnum):
    HTML = "html"  # body_strip HTML (lowercased, ws-collapsed)
    TEXT = "text"  # resiliparse-rs main-content text of that HTML, same normalization (as trained)
    TEXT_FROM_RAW = "text_from_raw"  # lower(collapse(extract(raw_html))): the one-extraction deployment


def sample_dir(input_bucket: str) -> str:
    return f"gs://{input_bucket}/{SAMPLE_NS}"


def clf_text_dir(input_bucket: str) -> str:
    return f"gs://{input_bucket}/{CLF_TEXT_NS}"


def clf_text_from_raw_dir(input_bucket: str) -> str:
    return f"gs://{input_bucket}/{CLF_TEXT_FROM_RAW_NS}"


def normalize_text(text: str | None) -> str:
    """The post-extraction normalization every TEXT classifier expects: whitespace-collapse, strip, lower."""
    return _WS_RE.sub(" ", text or "").strip().lower()


def preprocess_html(stripped_html: str | None) -> str:
    """body_strip HTML -> classifier input (whitespace-collapse + lowercase), matching to_fasttext_text."""
    return _WS_RE.sub(" ", (stripped_html or "")[:MAX_TEXT_CHARS]).strip().lower()


def _read_column(directory: str, column: str) -> tuple[list[str], list[str | None]]:
    ids: list[str] = []
    vals: list[str | None] = []
    files = sorted(fsspec_glob(f"{directory}/*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet under {directory}")
    for path in files:
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", column])
        ids.extend(t.column("warc_record_id").to_pylist())
        vals.extend(t.column(column).to_pylist())
    return ids, vals


def read_sample(input_bucket: str, kind: ClassifierInput, empty_text: str = "") -> tuple[list[str], list[str]]:
    """Return (warc_record_ids, classifier_input_texts) for the 100k sample, in shard+row order.

    ``empty_text`` is what an empty TEXT extraction becomes: "" for fastText (as in its prep shards),
    ``extract_text_shards.EMPTY_PLACEHOLDER`` for the neural TEXT models. Ignored for HTML.
    """
    ids, htmls = _read_column(sample_dir(input_bucket), "stripped_html")
    if kind is ClassifierInput.HTML:
        texts = [preprocess_html(h) for h in htmls]
    else:
        directory, column = (
            (clf_text_dir(input_bucket), CLF_TEXT_COLUMN)
            if kind is ClassifierInput.TEXT
            else (clf_text_from_raw_dir(input_bucket), CLF_TEXT_FROM_RAW_COLUMN)
        )
        text_ids, raw = _read_column(directory, column)
        if text_ids != ids:
            raise RuntimeError(
                f"{directory} is not row-aligned with {sample_dir(input_bucket)} "
                f"({len(text_ids)} vs {len(ids)} rows) — rebuild it (build_clf_text_*)"
            )
        texts = [t if t else empty_text for t in raw]
    logger.info("sample: %d docs (%s input)", len(ids), kind)
    return ids, texts
