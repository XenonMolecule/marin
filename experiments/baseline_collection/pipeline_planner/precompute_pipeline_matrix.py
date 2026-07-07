# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Precompute the static artifacts the cascade-planning dashboard runs on.

The dashboard evaluates any pipeline client-side over a cached per-doc matrix — no model re-runs. This
job builds that matrix from ``sample_100k_scored`` plus two derived per-doc column families:

  * **Levenshtein similarity vs 8B** for each non-gold extractor (1.7B / 0.6B / jusText), via
    ``rapidfuzz.distance.Levenshtein.normalized_similarity`` — matches the aggregate definition in
    ``build_extractor_comparison_dataset.py`` (union convention: missing extraction => "").
  * **Response/output token count** per extractor via the **llama3** tokenizer
    (``meta-llama/Meta-Llama-3.1-8B``, the curation-pipeline standard) — what training consumes. (LLM
    ``<think>`` tokens are NOT counted here; reasoning is a compute cost carried by the throughput, not output.)

Outputs (written to ``OUT_PREFIX`` in-region; download into ``web/data/`` for the static/Vercel deploy):
  * ``pipeline_matrix_{10k,100k}.json`` — columnar numeric matrix (the browser engine's input).
  * ``docs/shard-XXXX.json.gz`` — full doc text (url + stripped_html + raw_html[capped] + 4 extractor texts)
    for the 10k subset, sharded so the frontend lazy-fetches one shard per borderline cluster.
  * ``sample_docs.json`` — ~20 docs bundled for an offline borderline fallback.

Run (CPU, us-east5 where the sample lives; HF_TOKEN for the gated llama tokenizer)::

    uv run iris --cluster marin job run --region us-east5 --cpu 16 --memory 64GB --disk 40GB \
      --enable-extra-resources --extra cpu --extra extraction-bakeoff --priority interactive --no-wait \
      --job-name pipeline-precompute -e HF_TOKEN <token> -- \
      python -m experiments.baseline_collection.pipeline_planner.precompute_pipeline_matrix
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import fsspec
import numpy as np
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

from experiments.baseline_collection.pipeline_planner import stage_registry as reg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCORED_DIR = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc/sample_100k_scored"
OUT_PREFIX = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc/pipeline_planner"

LLAMA_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
SUBSAMPLE_SEED = 42
SUBSAMPLE_N = 10_000
SHARD_SIZE = 100  # docs per gzipped doc-text shard (10k => 100 shards)
RAW_HTML_CAP = 256 * 1024  # cap per-doc raw_html to tame multi-MB outliers
N_SAMPLE_DOCS = 20
LEV_CAP = 20_000  # cap chars for Levenshtein (O(n*m)); long-doc outliers would otherwise stall workers
MAX_WORKERS = 16  # HARD cap: os.cpu_count() can be 200+ on these boxes; one llama tokenizer/worker => OOM

# Non-gold extractors get a per-doc similarity + token column; 8B is the gold reference (sim = 1.0).
_EXTRACTORS = [("1p7b", "text_1p7b"), ("0p6b", "text_0p6b"), ("justext", "text_justext")]
_ALL_TEXT = [("8b", "text_8b"), *_EXTRACTORS]
_DOC_TEXT_COLS = ["url", "stripped_html", "raw_html", "text_8b", "text_1p7b", "text_0p6b", "text_justext"]

_TOK = None  # per-worker lazy tokenizer


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(LLAMA_TOKENIZER)
    return _TOK


def _perdoc(rows: list[dict]) -> list[dict]:
    """Levenshtein-vs-8B + llama response-token counts for a chunk of docs (runs in a worker process)."""
    from rapidfuzz.distance import Levenshtein

    tok = _tokenizer()
    out = []
    for r in rows:
        gold = r.get("text_8b") or ""
        rec: dict = {"tok_8b": len(tok.encode(gold)) if gold else 0}
        for key, col in _EXTRACTORS:
            t = r.get(col) or ""
            rec[f"tok_{key}"] = len(tok.encode(t)) if t else 0
            rec[f"lev_{key}"] = float(Levenshtein.normalized_similarity(t[:LEV_CAP], gold[:LEV_CAP]))
            rec[f"both_{key}"] = bool(t) and bool(gold)
        out.append(rec)
    return out


def _read_scored() -> dict[str, list]:
    """Load every column the matrix + doc shards need, in shard+row order."""
    needed = (
        ["warc_record_id", "label_8b", "label_1p7b", "label_0p6b", *_DOC_TEXT_COLS]
        + reg.REQUIRED_SCORE_COLS
    )
    cols: dict[str, list] = {c: [] for c in dict.fromkeys(needed)}
    files = sorted(fsspec_glob(f"{SCORED_DIR}/*.parquet"))
    if not files:
        raise RuntimeError(f"no scored parquet under {SCORED_DIR}")
    for path in files:
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=list(cols))
        for c in cols:
            cols[c].extend(t.column(c).to_pylist())
    n = len(cols["warc_record_id"])
    logger.info("loaded %d docs, %d columns", n, len(cols))
    return cols


def _build_matrix(cols: dict[str, list], perdoc: list[dict]) -> dict:
    """Columnar numeric matrix for the browser engine (scores NaN-filled; labels/booleans as 0/1)."""

    def f32(vals):
        return [None if v is None else float(v) for v in vals]  # JSON null -> NaN in JS

    matrix = {c: f32(cols[c]) for c in reg.REQUIRED_SCORE_COLS}
    matrix["gold_useful"] = [int(v == "useful") for v in cols["label_8b"]]
    matrix["label_1p7b_useful"] = [int(v == "useful") for v in cols["label_1p7b"]]
    matrix["label_0p6b_useful"] = [int(v == "useful") for v in cols["label_0p6b"]]
    for key, _ in _EXTRACTORS:
        matrix[f"lev_{key}"] = [round(r[f"lev_{key}"], 5) for r in perdoc]
        matrix[f"both_{key}"] = [int(r[f"both_{key}"]) for r in perdoc]
    for key, _ in _ALL_TEXT:
        matrix[f"tok_{key}"] = [r[f"tok_{key}"] for r in perdoc]
    matrix["warc_record_id"] = cols["warc_record_id"]
    return matrix


def _subsample_idx(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.default_rng(SUBSAMPLE_SEED).choice(n, size=k, replace=False))


def _slice_matrix(matrix: dict, idx: np.ndarray) -> dict:
    idxl = idx.tolist()
    return {col: [vals[i] for i in idxl] for col, vals in matrix.items()}


def _write_json(obj, path: str) -> None:
    with fsspec.open(path, "w") as fh:
        json.dump(obj, fh)
    logger.info("wrote %s", path)


def _write_doc_shards(cols: dict[str, list], idx: np.ndarray) -> None:
    """Sharded gzipped full-doc text for the subsample; frontend maps doc position -> shard (pos // SHARD_SIZE)."""
    ids = cols["warc_record_id"]
    for shard_i, start in enumerate(range(0, len(idx), SHARD_SIZE)):
        chunk = idx[start : start + SHARD_SIZE].tolist()
        shard = {}
        for pos in chunk:
            rec = {}
            for c in _DOC_TEXT_COLS:
                v = cols[c][pos]
                if c == "raw_html" and v and len(v) > RAW_HTML_CAP:
                    v = v[:RAW_HTML_CAP]
                rec[c] = v
            shard[ids[pos]] = rec
        with fsspec.open(f"{OUT_PREFIX}/docs/shard-{shard_i:04d}.json.gz", "wb") as fh:
            fh.write(gzip.compress(json.dumps(shard).encode("utf-8")))
    logger.info("wrote %d doc shards (%d docs/shard)", (len(idx) + SHARD_SIZE - 1) // SHARD_SIZE, SHARD_SIZE)


def main() -> None:
    global OUT_PREFIX
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-prefix", default=OUT_PREFIX)
    args = p.parse_args()
    OUT_PREFIX = args.out_prefix

    cols = _read_scored()
    # Validate every registry-referenced column is present before doing expensive work.
    missing = [c for c in reg.REQUIRED_SCORE_COLS + reg.REQUIRED_TEXT_COLS if c not in cols]
    if missing:
        raise SystemExit(f"artifact is missing registry columns: {missing}")
    n = len(cols["warc_record_id"])

    # Per-doc Levenshtein + token counts, fanned across cores.
    rows = [{k: cols[k][i] for k in ("text_8b", "text_1p7b", "text_0p6b", "text_justext")} for i in range(n)]
    n_workers = min(MAX_WORKERS, max(1, (os.cpu_count() or 2) - 1))
    chunks = [rows[i : i + 1000] for i in range(0, n, 1000)]
    logger.info("per-doc compute: %d docs over %d workers, %d chunks", n, n_workers, len(chunks))
    perdoc: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for part in ex.map(_perdoc, chunks):
            perdoc.extend(part)
            done += len(part)
            if done % 20000 < 1000:
                logger.info("per-doc compute: %d/%d done", done, n)

    matrix = _build_matrix(cols, perdoc)
    idx10k = _subsample_idx(n, SUBSAMPLE_N)
    meta = {"n_full": n, "n_subsample": int(len(idx10k)), "tokenizer": LLAMA_TOKENIZER, "registry": json.loads(reg.registry_json())}

    _write_json({"meta": {**meta, "n": n}, "columns": matrix}, f"{OUT_PREFIX}/pipeline_matrix_100k.json")
    _write_json({"meta": {**meta, "n": int(len(idx10k))}, "columns": _slice_matrix(matrix, idx10k)},
                f"{OUT_PREFIX}/pipeline_matrix_10k.json")
    _write_doc_shards(cols, idx10k)

    sample = {cols["warc_record_id"][i]: {c: (cols[c][i][:RAW_HTML_CAP] if c == "raw_html" and cols[c][i] else cols[c][i])
                                          for c in _DOC_TEXT_COLS} for i in idx10k[:N_SAMPLE_DOCS].tolist()}
    _write_json(sample, f"{OUT_PREFIX}/sample_docs.json")
    logger.info("DONE -> %s (download pipeline_matrix_*.json + docs/ + sample_docs.json into web/data/)", OUT_PREFIX)


if __name__ == "__main__":
    main()
