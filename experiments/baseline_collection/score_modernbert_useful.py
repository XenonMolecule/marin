# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the survivor ModernBERT useful-classifiers over the 100k comparison sample (JAX/Levanter).

Uses the NEW Levanter ModernBERT path (PR #6518: `load_hf_sequence_classifier` round-trips the HF
checkpoint) — NOT the old torch_xla `modernbert_warc_filter.py`. For one checkpoint per run, loads
its fine-tuned HF weights, scores every sample doc, and writes the RAW scalar P(useful) (sigmoid/
softmax prob, not a thresholded label) as one column per context-length model so the threshold can
be tuned downstream.

Model input = the SAME preprocessing the classifiers were trained on: ``to_fasttext_text`` =
``body_strip(html)`` → whitespace-collapse → lowercase. The sample already stores ``stripped_html``
= ``body_strip(html)``, so the input is just ``re.sub(r"\\s+"," ", stripped_html).strip().lower()``.

Data-parallel over the chips (batch sharded on a 1-D "data" mesh; model replicated — base fits on one
chip), splash attention at 8192 (vanilla OOMs there). Runs as a standalone TPU Iris job::

    uv run iris --cluster marin job run --region us-east5 \\
      --tpu v6e-4 --enable-extra-resources --extra tpu --memory 64GB \\
      --priority interactive --no-wait --job-name mb-score-1M-ctx8192 \\
      -e HF_TOKEN hf_... -- \\
      python -m experiments.baseline_collection.score_modernbert_useful score --model bert_useful_prob_1M_ctx8192

Then merge the per-model columns onto the sample::

    ... --job-name mb-score-join -- \\
      python -m experiments.baseline_collection.score_modernbert_useful join
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import time

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from levanter.layers.attention import AttentionMask
from levanter.models.modernbert import ModernBertConfig, load_hf_sequence_classifier
from marin.utils import fsspec_glob
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# --- Inputs / outputs ------------------------------------------------------

# Output always lands in us-east5 (where the sample + join run); INPUTS (checkpoints + sample) are
# read from --input-bucket so v5litepod spillover in another region reads a local mirror.
OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SCORES_DIR = f"{OUT_ROOT}/model_scores"  # {col}/part-i-of-N.parquet (BERT + fastText + future classifiers)
SCORED_OUT = f"{OUT_ROOT}/sample_100k_scored"
SAMPLE_NS = "documents/extractor_compare/high_quality_200warc/sample_100k"
CKPT_NS = "checkpoints/modernbert-useful"

TOKENIZER_REF = "answerdotai/ModernBERT-base"  # identical tokenizer for every survivor checkpoint
PAD_TOKEN_ID = 50283
MAX_TEXT_CHARS = 1_000_000  # cap pathological multi-MB markup (matches cascade_survivor_filter)

# col -> (checkpoint run-id, training/eval context length). 1M@8192 FIRST (run it first).
# base/large entries derive their architecture from the checkpoint's own config.json (see run_score),
# so a `large` checkpoint loads correctly despite the base default dims.
MODELS: dict[str, tuple[str, int]] = {
    "bert_useful_prob_1M_ctx8192": ("mb-clf-1M-surv-f", 8192),
    "bert_useful_prob_200k_ctx8192": ("mb-clf-200k-surv-f", 8192),
    "bert_useful_prob_200k_ctx4096": ("mb-clf-200k-ctx4096-surv-f", 4096),
    "bert_useful_prob_200k_ctx2048": ("mb-clf-200k-ctx2048-surv-f", 2048),
    "bert_useful_prob_200k_ctx1024": ("mb-clf-200k-ctx1024-surv-f", 1024),
    "bert_useful_prob_base_1M_rand": ("mb-clf-base-1M-rand-e5", 8192),
    "bert_useful_prob_large_1M_surv": ("mb-clf-large-1M-surv-e5", 8192),
    "bert_useful_prob_base_10M": ("mb-clf-base-10M-c8192", 8192),
    "bert_useful_prob_large_10M": ("mb-clf-large-10M-c8192", 8192),
}


def _sample_dir(input_bucket: str) -> str:
    return f"gs://{input_bucket}/{SAMPLE_NS}"


def _hf_ref(input_bucket: str, run_id: str) -> str:
    return f"gs://{input_bucket}/{CKPT_NS}/{run_id}/hf"


_WS_RE = re.compile(r"\s+")


def _preprocess(stripped_html: str | None) -> str:
    """body_strip HTML -> classifier input (whitespace-collapse + lowercase), matching to_fasttext_text."""
    return _WS_RE.sub(" ", (stripped_html or "")[:MAX_TEXT_CHARS]).strip().lower()


def _read_sample(input_bucket: str) -> tuple[list[str], list[str]]:
    """Return (warc_record_ids, preprocessed_texts) for the 100k sample, in shard+row order."""
    import pyarrow.parquet as pq

    ids: list[str] = []
    texts: list[str] = []
    for path in sorted(fsspec_glob(f"{_sample_dir(input_bucket)}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "stripped_html"])
        ids.extend(t.column("warc_record_id").to_pylist())
        texts.extend(_preprocess(h) for h in t.column("stripped_html").to_pylist())
    logger.info("sample: %d docs", len(ids))
    return ids, texts


# --- Scoring (data-parallel, splash@8192) ----------------------------------


def _score(model, texts: list[str], tokenizer, Pos: Axis, batch_size: int) -> np.ndarray:
    """P(useful) per text. Batch sharded over the 'data' mesh; pad each chunk to batch_size for static shapes."""
    from levanter.utils.tree_utils import inference_mode

    model = inference_mode(model, True)
    Batch = Axis("batch", batch_size)

    def _probs_impl(m, tokens, mask):
        logits = m(tokens, mask).astype(jnp.float32)
        return hax.nn.softmax(logits, axis="label")["label", 1]

    _probs = hax.named_jit(_probs_impl, axis_resources={"batch": "data"})

    out = np.zeros((len(texts),), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        n = len(chunk)
        ids = np.full((batch_size, Pos.size), PAD_TOKEN_ID, dtype=np.int32)
        seg = np.full((batch_size, Pos.size), -1, dtype=np.int32)
        for r, text in enumerate(chunk):
            enc = tokenizer(text, truncation=True, max_length=Pos.size)["input_ids"]
            ids[r, : len(enc)] = enc
            seg[r, : len(enc)] = 0
        tokens = hax.named(ids, (Batch, Pos))
        seg_named = hax.named(seg, (Batch, Pos))
        mask = AttentionMask(is_causal=False).with_segment_ids(seg_named, seg_named)
        out[start : start + n] = np.asarray(_probs(model, tokens, mask).array)[:n]
        if (start // batch_size) % 50 == 0:
            logger.info("scored %d/%d", start + n, len(texts))
    return out


def run_score(col: str, batch_size: int, limit: int | None, num_shards: int, shard_idx: int, input_bucket: str) -> None:
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq
    from haliax.partitioning import ResourceAxis, set_mesh
    from jax.sharding import Mesh

    if col not in MODELS:
        raise ValueError(f"unknown --model {col!r}; choices: {list(MODELS)}")
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"--shard-idx {shard_idx} out of range for --num-shards {num_shards}")
    run_id, max_seq_len = MODELS[col]
    hf_ref = _hf_ref(input_bucket, run_id)
    logger.info("scoring col=%s ref=%s ctx=%d on devices=%s", col, hf_ref, max_seq_len, jax.devices())

    ids, texts = _read_sample(input_bucket)
    if limit is not None:
        ids, texts = ids[:limit], texts[:limit]
        logger.info("LIMIT: scoring only first %d docs (smoke)", len(ids))
    if num_shards > 1:
        # Contiguous disjoint slice per shard so K TPUs split the 100k (join is by record_id anyway).
        chunk = math.ceil(len(ids) / num_shards)
        s, e = shard_idx * chunk, min((shard_idx + 1) * chunk, len(ids))
        ids, texts = ids[s:e], texts[s:e]
        logger.info("shard %d/%d: docs [%d:%d] = %d", shard_idx, num_shards, s, e, len(ids))

    backend = "splash" if max_seq_len >= 8192 else "vanilla"  # vanilla OOMs at 8192; equivalent (M4-validated)
    # Derive architecture (hidden/layers/heads) from the checkpoint's own config.json so base AND large
    # checkpoints load correctly; override only eval context, attn backend, labels, pad.
    converter = ModernBertConfig().hf_checkpoint_converter(ref_checkpoint=hf_ref)
    hf_config = converter.hf_config_from_hf_checkpoint(hf_ref)
    config = dataclasses.replace(
        ModernBertConfig.from_hf_config(hf_config),
        max_seq_len=max_seq_len,
        attn_backend=backend,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
    )
    logger.info("arch: hidden=%d layers=%d heads=%d (from checkpoint)", config.hidden_dim, config.num_layers, config.num_heads)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)

    # Data-parallel mesh with haliax's resource axes; set_mesh (not jax `with mesh:`) so the
    # safetensors loader's best_effort_sharding sees the "data" axis (matches use_test_mesh).
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with set_mesh(mesh):
        model = load_hf_sequence_classifier(config, hf_ref, axis_mapping=None, dtype=jnp.bfloat16)
        t_score = time.monotonic()
        probs = _score(model, texts, tokenizer, config.max_Pos, batch_size)
        score_secs = time.monotonic() - t_score
    logger.info(
        "TIMING col=%s: scored %d docs in %.1fs = %.2f docs/s = %.3f docs/chip/s (%d chips, ctx=%d)",
        col,
        len(texts),
        score_secs,
        len(texts) / score_secs if score_secs else 0.0,
        len(texts) / score_secs / n_dev if score_secs else 0.0,
        n_dev,
        max_seq_len,
    )

    if limit is not None:
        out_path = f"{SCORES_DIR}/{col}_smoke.parquet"  # smoke stays flat, out of the real per-col dir
    else:
        out_path = f"{SCORES_DIR}/{col}/part-{shard_idx:03d}-of-{num_shards:03d}.parquet"
    table = pa.table({"warc_record_id": ids, col: probs.tolist()})
    with fsspec.open(out_path, "wb") as fh:
        pq.write_table(table, fh)
    logger.info("wrote %d scores -> %s (mean P=%.4f)", len(ids), out_path, float(probs.mean()))


# --- Join scores back onto the sample --------------------------------------


def run_join() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    score_files = sorted(fsspec_glob(f"{SCORES_DIR}/*/part-*.parquet"))  # {col}/part-i-of-N (all shards, all cols)
    if not score_files:
        raise RuntimeError(f"no score parquets under {SCORES_DIR}/*/; run `score` first")
    by_id: dict[str, dict[str, float]] = {}
    cols: list[str] = []
    for f in score_files:
        with fsspec.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read()
        col = next(c for c in t.column_names if c != "warc_record_id")
        if col not in cols:  # one column per model, not one per shard-part
            cols.append(col)
        rid = t.column("warc_record_id").to_pylist()
        val = t.column(col).to_pylist()
        for r, v in zip(rid, val, strict=True):
            by_id.setdefault(r, {})[col] = v
    logger.info("joining %d score columns: %s", len(cols), cols)

    # This join REBUILDS sample_100k_scored from the original sample, so re-apply the jusText re-decode
    # (clean raw_html + text_justext) when staged — otherwise a BERT join run after the jusText join would
    # silently revert raw_html and drop text_justext (they live only in the scored file). Order-independent.
    from experiments.baseline_collection.score_justext_redecode import load_staging  # heavy deps (zephyr) — load lazily

    staged = load_staging()
    if staged:
        logger.info("re-applying jusText staging: %d clean raw_html + text_justext", len(staged))

    sample_dir = _sample_dir("marin-us-east5")  # join always reads the canonical us-east5 sample
    n_shards = len(fsspec_glob(f"{sample_dir}/*.parquet"))
    for i, path in enumerate(sorted(fsspec_glob(f"{sample_dir}/*.parquet"))):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read()
        rids = t.column("warc_record_id").to_pylist()
        for col in cols:
            t = t.append_column(col, pa.array([by_id.get(r, {}).get(col) for r in rids], type=pa.float32()))
        if staged:
            old_raw = t.column("raw_html").to_pylist()
            new_raw = [staged[r][0] if staged.get(r, (None,))[0] is not None else old_raw[k] for k, r in enumerate(rids)]
            idx = t.column_names.index("raw_html")
            t = t.set_column(idx, "raw_html", pa.array(new_raw, type=pa.string()))
            t = t.append_column("text_justext", pa.array([staged.get(r, (None, None))[1] for r in rids], type=pa.string()))
        out_path = f"{SCORED_OUT}/sample-{i:05d}-of-{n_shards:05d}.parquet"
        with fsspec.open(out_path, "wb") as fh:
            pq.write_table(t, fh)
    logger.info("joined sample with %d score columns%s -> %s", len(cols), " + jusText" if staged else "", SCORED_OUT)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_score = sub.add_parser("score", help="Score ONE model over the sample -> per-model parquet.")
    p_score.add_argument("--model", required=True, choices=list(MODELS), help="Which checkpoint/column to score.")
    p_score.add_argument("--batch-size", type=int, default=32, help="Must be a multiple of the chip count.")
    p_score.add_argument("--limit", type=int, default=None, help="Smoke: score only the first N docs.")
    p_score.add_argument("--num-shards", type=int, default=1, help="Split the 100k across this many TPUs.")
    p_score.add_argument("--shard-idx", type=int, default=0, help="Which contiguous slice this job scores.")
    p_score.add_argument(
        "--input-bucket",
        default="marin-us-east5",
        help="Bucket holding the checkpoints + sample (use a regional mirror for v5litepod spillover).",
    )
    sub.add_parser("join", help="Merge all per-model score columns onto the sample -> sample_100k_scored.")
    args = parser.parse_args()

    if args.stage == "score":
        run_score(args.model, args.batch_size, args.limit, args.num_shards, args.shard_idx, args.input_bucket)
    elif args.stage == "join":
        run_join()


if __name__ == "__main__":
    main()
