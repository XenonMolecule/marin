# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone CHUNKED re-scorer: load a finished chunked classifier's HF checkpoint and run the
full chunked eval (score every chunk of each frozen-7k doc, aggregate) — decoupled from the training
job, so it can't be knocked out by a mid-eval preemption. Mirrors ``score_frozen_eval`` but uses
``score_texts_chunked`` + ``aggregate_sweep`` instead of truncated scoring.

Run IN-REGION on a NON-preemptible TPU (so a ~30-min eval completes in one shot):

  uv run iris --cluster marin job run --region us-east5 --tpu v6e-4 --memory 128GB \\
      --extra marin:tpu --enable-extra-resources -e WANDB_API_KEY ... -e HF_TOKEN ... -- \\
      python -m experiments.baseline_collection.score_chunked_eval \\
          --run-id mb-clf-large-200k-c8192-chunk-ov --ctx 8192 --overlap 0.5 --backend splash \\
          --bucket gs://marin-us-east5
"""

import argparse
import dataclasses
import json
import logging

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import ResourceAxis, set_mesh
from jax.sharding import Mesh
from levanter.main.train_classifier import aggregate_sweep, read_frozen_eval, score_texts_chunked
from levanter.models.modernbert import ModernBertConfig, load_hf_sequence_classifier
from transformers import AutoTokenizer

from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

TOKENIZER_REF = "answerdotai/ModernBERT-base"
PAD_TOKEN_ID = 50283
USEFUL_LABEL = "__label__useful"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--ctx", type=int, required=True)
    p.add_argument("--overlap", type=float, default=0.0, help="0.0 non-overlap (stride=ctx); 0.5 → stride=ctx/2.")
    p.add_argument("--backend", default="splash", choices=["vanilla", "splash"])
    p.add_argument("--bucket", default="gs://marin-us-east5")
    p.add_argument("--eval-rows", type=int, default=7000)
    p.add_argument("--max-doc-tokens", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=16, help="Must divide the chip count on the padded batch.")
    args = p.parse_args()

    hf_ref = f"{args.bucket}/checkpoints/modernbert-useful/{args.run_id}/hf"
    out_path = f"{args.bucket}/checkpoints/modernbert-useful/{args.run_id}/chunked_eval_{args.eval_rows}.json"
    if fsspec_glob(out_path):
        logger.info("skip %s — chunked eval already exists at %s", args.run_id, out_path)
        return

    test_glob = f"{args.bucket}/classifiers/useful_fasttext/full_prep_body_strip/test/*.txt.gz"
    test_paths = sorted(fsspec_glob(test_glob))
    if not test_paths:
        raise ValueError(f"no test shards at {test_glob}")
    texts, labels = read_frozen_eval(test_paths, USEFUL_LABEL, rows=args.eval_rows)
    logger.info("frozen eval: %d docs (%d useful); devices=%s", len(texts), sum(labels), jax.devices())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    stride = args.ctx if args.overlap == 0.0 else max(1, round(args.ctx * (1.0 - args.overlap)))

    converter = ModernBertConfig().hf_checkpoint_converter(ref_checkpoint=hf_ref)
    hf_config = converter.hf_config_from_hf_checkpoint(hf_ref)
    config = dataclasses.replace(
        ModernBertConfig.from_hf_config(hf_config),
        max_seq_len=args.ctx,
        attn_backend=args.backend,
        num_labels=2,
        pad_token_id=PAD_TOKEN_ID,
    )
    logger.info(
        "arch hidden=%d layers=%d; ctx=%d stride=%d backend=%s",
        config.hidden_dim,
        config.num_layers,
        args.ctx,
        stride,
        args.backend,
    )

    # Resumable: score docs in blocks and checkpoint per-doc chunk probs to GCS after each block, so a
    # preemption on the (preemptible-only) v6e pool loses at most one block instead of restarting the eval.
    partial_path = (
        f"{args.bucket}/checkpoints/modernbert-useful/{args.run_id}/chunked_eval_partial_{args.eval_rows}.json"
    )
    per_doc: list = []
    if fsspec_glob(partial_path):
        with fsspec.open(partial_path, "r") as fh:
            per_doc = [np.asarray(d, dtype=np.float32) for d in json.load(fh)["per_doc"]]
        logger.info("resuming from partial: %d/%d docs already scored", len(per_doc), len(texts))
    DOC_BLOCK = 500
    with set_mesh(mesh):
        model = load_hf_sequence_classifier(config, hf_ref, axis_mapping=None, dtype=jnp.bfloat16)
        for s in range(len(per_doc), len(texts), DOC_BLOCK):
            block = score_texts_chunked(
                model,
                texts[s : s + DOC_BLOCK],
                tokenizer,
                config.max_Pos,
                PAD_TOKEN_ID,
                stride=stride,
                max_doc_tokens=args.max_doc_tokens,
                batch_size=args.batch_size,
            )
            per_doc.extend(block)
            with fsspec.open(partial_path, "w") as fh:
                json.dump({"per_doc": [[round(float(p), 5) for p in d] for d in per_doc]}, fh)
            logger.info("scored %d/%d docs", len(per_doc), len(texts))
    results, best_agg = aggregate_sweep(per_doc, labels)
    best_f1, best_t = results[best_agg]
    logger.info(
        "run=%s CHUNKED best_f1=%.4f agg=%s @ t=%.2f  | per-agg: %s",
        args.run_id,
        best_f1,
        best_agg,
        best_t,
        {k: round(v[0], 4) for k, v in results.items()},
    )
    payload = {
        "run_id": args.run_id,
        "ctx": args.ctx,
        "overlap": args.overlap,
        "n": len(labels),
        "best_f1": best_f1,
        "best_agg": best_agg,
        "best_threshold": best_t,
        "per_agg": {k: {"f1": v[0], "threshold": v[1]} for k, v in results.items()},
    }
    with fsspec.open(out_path, "w") as fh:
        json.dump(payload, fh)
    logger.info("wrote chunked eval -> %s", out_path)
    for pp in fsspec_glob(partial_path):  # drop the resume checkpoint now that the final result exists
        fsspec.open(pp).fs.rm(pp)


if __name__ == "__main__":
    main()
