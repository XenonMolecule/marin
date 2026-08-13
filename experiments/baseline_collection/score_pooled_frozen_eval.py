# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a trained ``pooled_transformer`` useful-classifier over the frozen lpv11 7k eval set.

Writes per-doc ``P(useful)`` aligned index-for-index with the fastText w640 scores produced by
``score_fasttext_on_bert_test.py``, so the two can be combined into a cascade (stacked-filter)
analysis by ``cascade_analysis.py``.

Alignment: the doc order is ``levanter.main.train_classifier.read_frozen_eval`` (per shard: read all
valid ``__label__x text`` lines in file order, shuffle the indices with ``random.Random(seed)``, keep
the first ``rows // n_shards``). With the single-file frozen eval this is the same sequence the
fastText script emits, so doc *i* here is doc *i* there; the emitted ``labels`` are the check.

The checkpoint is the generic equinox format written by ``save_eqx_classifier`` (``model.eqx`` +
``config.json``), NOT an HF checkpoint — the architecture is read back out of ``config.json`` so any
pooled run id works.

Run as a standalone TPU Iris job (the model is ~26M params; the whole 7k takes seconds)::

    uv run iris --cluster marin job run --region us-east5 \\
      --tpu v6e-4 --enable-extra-resources --extra tpu --memory 64GB \\
      --priority interactive --no-wait --job-name pooled-score-frozen7k -- \\
      python -m experiments.baseline_collection.score_pooled_frozen_eval \\
        --ckpt gs://marin-us-east5/checkpoints/modernbert-useful/mb-clf-lpv11-pooled-1M-r5/hf \\
        --out gs://marin-us-east5/classifiers/useful_fasttext_lpv11/eval/pooled_1M_on_frozen7k.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time

import fsspec
import jax
import numpy as np
from haliax.partitioning import ResourceAxis, set_mesh
from jax.sharding import Mesh
from levanter.main.train_classifier import read_frozen_eval, score_texts
from levanter.models.pooled_transformer import PooledTransformerConfig, load_pooled_transformer_classifier
from marin.utils import fsspec_glob
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

TEST_GLOB = "gs://marin-us-east5/classifiers/useful_fasttext_lpv11/full_prep_body_strip_test7k/test_sample_7k.txt.gz"
TOKENIZER_REF = "answerdotai/ModernBERT-base"
USEFUL_LABEL = "__label__useful"
MODERNBERT_VOCAB_SIZE = 50368  # training rounds Axis("vocab", len(tokenizer)) for partitioning


def load_config(ckpt: str, ctx: int | None) -> PooledTransformerConfig:
    """Rebuild the training-time ``PooledTransformerConfig`` from the checkpoint's ``config.json``."""
    with fsspec.open(f"{ckpt}/config.json", "rt", encoding="utf-8") as f:
        raw = json.load(f)
    config_class = raw.pop("config_class", None)
    if config_class != "PooledTransformerConfig":
        raise ValueError(f"{ckpt}/config.json is a {config_class!r}, not a PooledTransformerConfig")
    config = PooledTransformerConfig(**raw)
    return config if ctx is None else dataclasses.replace(config, max_seq_len=ctx)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="gs:// dir holding model.eqx + config.json.")
    ap.add_argument("--test", default=TEST_GLOB, help="Frozen eval glob (must match the fastText run's).")
    ap.add_argument("--rows", type=int, default=7000, help="read_frozen_eval row budget (per-shard share).")
    ap.add_argument("--seed", type=int, default=0, help="read_frozen_eval shuffle seed.")
    ap.add_argument("--ctx", type=int, default=None, help="Eval context length; default = the training ctx.")
    ap.add_argument("--batch-size", type=int, default=64, help="Padded to a fixed size for static shapes.")
    ap.add_argument("--vocab-size", type=int, default=MODERNBERT_VOCAB_SIZE, help="Embedding rows in the checkpoint.")
    ap.add_argument("--run-id", default=None, help="Label for the output JSON; defaults to the ckpt's parent dir.")
    ap.add_argument("--out", required=True, help="gs:// path for the {probs, labels, run_id, ctx} JSON.")
    args = ap.parse_args()

    config = load_config(args.ckpt, args.ctx)
    run_id = args.run_id or args.ckpt.rstrip("/").rsplit("/", 2)[-2]
    logger.info("ckpt=%s run_id=%s ctx=%d devices=%s", args.ckpt, run_id, config.max_seq_len, jax.devices())

    paths = sorted(fsspec_glob(args.test))
    if not paths:
        raise FileNotFoundError(f"no frozen-eval shards match {args.test}")
    texts, labels = read_frozen_eval(paths, USEFUL_LABEL, rows=args.rows, seed=args.seed)
    logger.info("frozen eval: %d docs, %d useful (%.1f%%)", len(texts), sum(labels), 100 * sum(labels) / len(labels))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)

    # set_mesh (not `with mesh:`) so haliax-internal init paths see the resource axes.
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with set_mesh(mesh):
        model = load_pooled_transformer_classifier(config, args.ckpt, vocab_size=args.vocab_size)
        t0 = time.monotonic()
        probs = score_texts(model, texts, tokenizer, config.max_Pos, config.pad_token_id, batch_size=args.batch_size)
        elapsed = time.monotonic() - t0
    logger.info(
        "scored %d docs in %.1fs = %.1f docs/s = %.1f docs/chip/s",
        len(texts),
        elapsed,
        len(texts) / elapsed,
        len(texts) / elapsed / n_dev,
    )

    payload = {"probs": [float(p) for p in probs], "labels": labels, "run_id": run_id, "ctx": config.max_seq_len}
    with fsspec.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info("wrote %d scores -> %s (mean P=%.4f)", len(probs), args.out, float(np.mean(probs)))


if __name__ == "__main__":
    main()
