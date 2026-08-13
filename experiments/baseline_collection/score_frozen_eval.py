# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-score a finished ModernBERT useful-classifier checkpoint on the FROZEN 7k eval set and dump
per-document (P(useful), label) pairs — the raw material for an operating curve (recall vs % useless
removed), which training only summarized to scalar best-F1/threshold and discarded.

Loads the fine-tuned HF checkpoint exactly as ``score_modernbert_useful.run_score`` does
(``load_hf_sequence_classifier`` under a data-parallel mesh; architecture derived from the
checkpoint's own config.json so base AND large load), reads the SAME deterministic frozen eval as
training (``read_frozen_eval`` seed=0, 7000 rows), scores, and writes a tiny JSON next to the
checkpoint:

    {"run_id", "ctx", "n", "best_f1", "best_threshold", "preds": [[p, y], ...]}

Run IN-REGION (checkpoint + test set must be local to the run region — never cross-region). The
output JSON is a few hundred KB, safe to fetch to a laptop afterward.

  uv run iris --cluster marin job run --region us-east5 --tpu v6e-8 --memory 128GB \\
      --extra marin:tpu --enable-extra-resources -e WANDB_API_KEY ... -e HF_TOKEN ... -- \\
      python -m experiments.baseline_collection.score_frozen_eval \\
          --run-id mb-clf-large-1M-surv-e5 --ctx 8192 --bucket gs://marin-us-east5
"""

import argparse
import dataclasses
import json
import logging
import os

import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from haliax.partitioning import ResourceAxis, set_mesh
from jax.sharding import Mesh
from levanter.layers.attention import AttentionMask
from levanter.main.train_classifier import f1_sweep, read_frozen_eval
from levanter.models.modernbert import ModernBertConfig, load_hf_sequence_classifier
from levanter.utils.tree_utils import inference_mode
from transformers import AutoTokenizer

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

TOKENIZER_REF = "answerdotai/ModernBERT-base"  # shared tokenizer for every survivor checkpoint
PAD_TOKEN_ID = 50283
USEFUL_LABEL = "__label__useful"
EVAL_ROWS = 7000  # matches ClassificationDataConfig.eval_rows (the frozen-7k test)


def score(model, texts: list[str], tokenizer, Pos: Axis, batch_size: int) -> np.ndarray:
    """P(useful) per text; batch sharded on the 'data' mesh, each chunk padded to batch_size."""
    model = inference_mode(model, True)
    Batch = Axis("batch", batch_size)
    _probs = hax.named_jit(
        lambda m, tokens, mask: hax.nn.softmax(m(tokens, mask).astype(jnp.float32), axis="label")["label", 1],
        axis_resources={"batch": "data"},
    )
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


def score_one(
    run_id: str,
    ctx: int,
    bucket: str,
    texts: list[str],
    labels: list[int],
    tokenizer,
    mesh,
    batch_size: int,
    backend: str,
) -> None:
    hf_ref = f"{bucket}/checkpoints/modernbert-useful/{run_id}/hf"
    out_path = f"{bucket}/checkpoints/modernbert-useful/{run_id}/frozen_eval_preds.json"
    # Idempotent: a finished preds JSON means this checkpoint is done. Skip it so a job that gets
    # preempted mid-run (no checkpoint — scoring restarts from scratch) makes forward progress across
    # restarts instead of re-scoring completed checkpoints on a contended (tonyhlee) v5p node.
    if fsspec_glob(out_path):
        logger.info("skip run_id=%s — preds already exist at %s", run_id, out_path)
        return
    # Resumable: scoring has no checkpoint, so on a contended (tonyhlee) v5p node a 7000-doc pass gets
    # preempted before finishing and restarts from scratch (preempt-loop). Persist probs to a partial
    # file every PARTIAL_CHUNK docs and resume from it, so progress accumulates across preemptions.
    partial_path = f"{bucket}/checkpoints/modernbert-useful/{run_id}/frozen_eval_partial.json"
    PARTIAL_CHUNK = 800  # multiple of batch_size; checkpoint cadence (small → bank progress fast under preempt)
    probs: list[float] = []
    if fsspec_glob(partial_path):
        with fsspec.open(partial_path, "r") as fh:
            probs = json.load(fh)["probs"]
        logger.info("resuming run_id=%s from partial: %d/%d already scored", run_id, len(probs), len(texts))
    logger.info("scoring run_id=%s ctx=%d ref=%s (from doc %d)", run_id, ctx, hf_ref, len(probs))

    if backend == "auto":
        backend = "splash" if ctx >= 8192 else "vanilla"  # splash needed at 8192 on small-HBM (v6e 32GB)
    converter = ModernBertConfig().hf_checkpoint_converter(ref_checkpoint=hf_ref)
    hf_config = converter.hf_config_from_hf_checkpoint(hf_ref)
    config = dataclasses.replace(
        ModernBertConfig.from_hf_config(hf_config),
        max_seq_len=ctx,
        attn_backend=backend,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
    )
    logger.info(
        "arch: hidden=%d layers=%d heads=%d (from checkpoint)", config.hidden_dim, config.num_layers, config.num_heads
    )
    with set_mesh(mesh):
        model = load_hf_sequence_classifier(config, hf_ref, axis_mapping=None, dtype=jnp.bfloat16)
        for s in range(len(probs), len(texts), PARTIAL_CHUNK):
            chunk_probs = score(model, texts[s : s + PARTIAL_CHUNK], tokenizer, config.max_Pos, batch_size)
            probs.extend(float(p) for p in chunk_probs)
            with fsspec.open(partial_path, "w") as fh:  # checkpoint after each chunk
                json.dump({"probs": probs}, fh)
            logger.info("run_id=%s partial-checkpointed %d/%d", run_id, len(probs), len(texts))

    probs = np.asarray(probs, dtype=np.float32)
    best_f1, best_t = f1_sweep(probs, np.asarray(labels))
    logger.info("re-scored best_f1=%.4f @ t=%.2f (sanity vs recorded wandb value)", best_f1, best_t)
    payload = {
        "run_id": run_id,
        "ctx": ctx,
        "n": len(labels),
        "best_f1": best_f1,
        "best_threshold": best_t,
        "preds": [[round(float(pr), 5), int(y)] for pr, y in zip(probs.tolist(), labels, strict=True)],
    }
    with fsspec.open(out_path, "w") as fh:
        json.dump(payload, fh)
    logger.info("wrote %d preds -> %s (best_f1=%.4f)", len(labels), out_path, best_f1)
    for pp in fsspec_glob(partial_path):  # drop the resume checkpoint now that final preds exist
        fsspec.open(pp).fs.rm(pp)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # Persistent XLA compile cache: on a tonyhlee-contended v5p node the splash@8192 compile (~4min) gets
    # preempted before any docs are scored, so every restart recompiles → zero progress. A GCS-backed
    # cache turns post-first-compile restarts into ~second cache hits, leaving the window for scoring.
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        logger.info("persistent JAX compile cache: %s", cache_dir)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-ids", required=True, help="Comma-separated run-ids to score sequentially on one node.")
    p.add_argument("--ctx", type=int, required=True, help="Eval context length (shared; the checkpoints' training ctx).")
    p.add_argument(
        "--bucket", required=True, help="Region bucket holding the checkpoints + test set, e.g. gs://marin-us-east5"
    )
    p.add_argument("--batch-size", type=int, default=32, help="Must be a multiple of the chip count.")
    p.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "vanilla", "splash"],
        help="Attention backend. 'vanilla' compiles far faster (no Pallas) and fits at 8192 on "
        "big-HBM v5p (95GB) — preferred when a contended node preempts during the slow splash compile.",
    )
    args = p.parse_args()

    test_glob = f"{args.bucket}/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz"
    test_paths = sorted(fsspec_glob(test_glob))
    if not test_paths:
        raise ValueError(f"no test shards at {test_glob}")
    texts, labels = read_frozen_eval(test_paths, USEFUL_LABEL, rows=EVAL_ROWS)
    logger.info(
        "frozen eval: %d docs (%d useful) from %d shards; devices=%s",
        len(texts),
        sum(labels),
        len(test_paths),
        jax.devices(),
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))

    for run_id in [r.strip() for r in args.run_ids.split(",") if r.strip()]:
        score_one(run_id, args.ctx, args.bucket, texts, labels, tokenizer, mesh, args.batch_size, args.backend)


if __name__ == "__main__":
    main()
