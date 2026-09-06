# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score an equinox-checkpointed useful-classifier (pooled_transformer, funnelbert) over the 100k sample.

The cascade planner needs one scalar ``P(useful)`` column per candidate stage, over the SAME 100k
docs in the SAME order as every other classifier. This is the eqx counterpart to
``score_modernbert_useful`` (HF/safetensors) and ``score_fasttext_useful`` (CPU) — all three read the
sample through ``comparison_sample``, so doc *i* is doc *i* across every family and both input
representations (body_strip HTML, or the resiliparse-rs TEXT of that HTML for the ``*-text-*`` runs).

Checkpoints are the generic equinox format written by ``save_eqx_classifier`` (``model.eqx`` +
``config.json``), NOT HF — ``config.json`` names its ``config_class``, which selects the loader
(``_EQX_ARCHS``), so any pooled or funnelbert run id works. Output lands in the shared
``model_scores/<col>/`` layout that ``score_modernbert_useful join`` merges onto the sample.

Sibling script ``score_pooled_frozen_eval`` scores the frozen lpv11 7k eval set instead; use that for
threshold/cascade curves against the fastText w640 run, and this one for the planner.

Run as a standalone TPU Iris job (these models are ~26M params; the 100k takes minutes)::

    uv run iris --cluster marin job run --region us-east5 \\
      --tpu v6e-4 --enable-extra-resources --extra tpu --memory 64GB \\
      --priority interactive --no-wait --job-name pooled-lpv11-1M -- \\
      python -m experiments.baseline_collection.score_pooled_useful --model pooled_lpv11_prob_1M_r5

Then merge every score column onto the sample::

    ... -- python -m experiments.baseline_collection.score_modernbert_useful join
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time

import fsspec
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from haliax.partitioning import ResourceAxis, set_mesh
from jax.sharding import Mesh
from levanter.main.train_classifier import score_texts
from levanter.models.classification import load_eqx_config
from levanter.models.funnelbert import FunnelBertConfig, load_funnelbert_classifier
from levanter.models.pooled_transformer import PooledTransformerConfig, load_pooled_transformer_classifier
from transformers import AutoTokenizer

from experiments.baseline_collection.comparison_sample import SCORES_DIR, ClassifierInput, read_sample
from experiments.baseline_collection.extract_text_shards import EMPTY_PLACEHOLDER
from experiments.baseline_collection.score_modernbert_useful import TOKENIZER_REF
from experiments.baseline_collection.score_pooled_frozen_eval import MODERNBERT_VOCAB_SIZE

logger = logging.getLogger(__name__)

CKPT_NS = "checkpoints/modernbert-useful"

# col -> (checkpoint run-id, classifier input). All target llm_pipeline_v1_1 (lpv11), not the 8B run.
MODELS: dict[str, tuple[str, ClassifierInput]] = {
    "pooled_lpv11_prob_1M_r5": ("mb-clf-lpv11-pooled-1M-r5", ClassifierInput.HTML),
    "pooled_lpv11_prob_10M": ("mb-clf-lpv11-pooled-10M", ClassifierInput.HTML),
    "pooled_lpv11_prob_10M_e3": ("mb-clf-lpv11-pooled-10M-e3", ClassifierInput.HTML),
    "pooled_lpv11_prob_big_10M": ("mb-clf-lpv11-pooledbig-10M", ClassifierInput.HTML),
    # 90M HTML completes the pooled 3x2 (1M/10M/90M x HTML/TEXT) grid; trained in us-central2, hf/ mirrored.
    "pooled_lpv11_prob_90M": ("mb-clf-lpv11-pooled-90M", ClassifierInput.HTML),
    # Arch-sweep 1M survivor (HTML), funnelbert = 4 warm-started ModernBERT layers -> 8x mean-pool -> 4 fresh.
    "funnelbert_lpv11_prob_1M": ("mb-clf-lpv11-funnelbert-1M", ClassifierInput.HTML),
    # TEXT-trained (input = resiliparse-rs main content of the body_strip HTML). 90M was trained in
    # us-central2; its hf/ is mirrored to us-east5 so this job reads in-region.
    "pooled_lpv11_text_prob_1M": ("mb-clf-lpv11-text-pooled-1M", ClassifierInput.TEXT),
    "pooled_lpv11_text_prob_10M": ("mb-clf-lpv11-text-pooled-10M", ClassifierInput.TEXT),
    "pooled_lpv11_text_prob_90M": ("mb-clf-lpv11-text-pooled-90M", ClassifierInput.TEXT),
    # Deployment input (one extraction on raw HTML, normalized after) — what the planner prices; see
    # score_modernbert_useful.MODELS for the verification note.
    "pooled_lpv11_textraw_prob_1M": ("mb-clf-lpv11-text-pooled-1M", ClassifierInput.TEXT_FROM_RAW),
    "pooled_lpv11_textraw_prob_10M": ("mb-clf-lpv11-text-pooled-10M", ClassifierInput.TEXT_FROM_RAW),
    "pooled_lpv11_textraw_prob_90M": ("mb-clf-lpv11-text-pooled-90M", ClassifierInput.TEXT_FROM_RAW),
}

# Config dataclass -> loader. Adding an eqx arch = one entry here (config parsing, incl. enum
# restoration, is `levanter.models.classification.load_eqx_config`, shared with every other consumer).
_EQX_LOADERS = {
    PooledTransformerConfig: load_pooled_transformer_classifier,
    FunnelBertConfig: load_funnelbert_classifier,
}


def load_eqx_arch(ckpt: str, ctx: int | None):
    """(config, loader) for an eqx checkpoint dir; ``ctx`` overrides the eval context length."""
    config = load_eqx_config(ckpt)
    loader = _EQX_LOADERS.get(type(config))
    if loader is None:
        raise ValueError(
            f"{ckpt}: {type(config).__name__} has no loader here; known: {[c.__name__ for c in _EQX_LOADERS]}"
        )
    return (config if ctx is None else dataclasses.replace(config, max_seq_len=ctx)), loader


def run_score(col: str, batch_size: int, ctx: int | None, limit: int | None, input_bucket: str) -> None:
    run_id, kind = MODELS[col]
    ckpt = f"gs://{input_bucket}/{CKPT_NS}/{run_id}/hf"
    config, loader = load_eqx_arch(ckpt, ctx)
    logger.info("col=%s ckpt=%s ctx=%d input=%s devices=%s", col, ckpt, config.max_seq_len, kind, jax.devices())

    # Neural TEXT corpora carry EMPTY_PLACEHOLDER for empty extractions; feed the same at inference.
    ids, texts = read_sample(input_bucket, kind, empty_text=EMPTY_PLACEHOLDER)
    if limit:
        ids, texts = ids[:limit], texts[:limit]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)

    # set_mesh (not `with mesh:`) so haliax-internal init paths see the resource axes.
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with set_mesh(mesh):
        model = loader(config, ckpt, vocab_size=MODERNBERT_VOCAB_SIZE)
        t0 = time.monotonic()
        probs = score_texts(model, texts, tokenizer, config.max_Pos, config.pad_token_id, batch_size=batch_size)
        elapsed = time.monotonic() - t0

    logger.info(
        "TIMING col=%s: scored %d docs in %.1fs = %.1f docs/s = %.1f docs/chip/s",
        col,
        len(texts),
        elapsed,
        len(texts) / elapsed,
        len(texts) / elapsed / n_dev,
    )

    # A --limit run must NEVER land on the real column path: a smoke finishing after a full run would
    # silently truncate the column to N rows, surfacing much later as a column of nulls. Same flat
    # `{col}_smoke.parquet` convention score_modernbert_useful uses, which the join glob ignores.
    if limit is not None:
        out_path = f"{SCORES_DIR}/{col}_smoke.parquet"
    else:
        out_path = f"{SCORES_DIR}/{col}/part-000-of-001.parquet"
    table = pa.table({"warc_record_id": ids, col: [float(p) for p in probs]})
    with fsspec.open(out_path, "wb") as fh:
        pq.write_table(table, fh)
    logger.info("wrote %d scores -> %s (mean P=%.4f)", len(ids), out_path, float(np.mean(probs)))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS), help="Which checkpoint/column to score.")
    p.add_argument("--batch-size", type=int, default=64, help="Padded to a fixed size for static shapes.")
    p.add_argument("--ctx", type=int, default=None, help="Eval context length; default = the training ctx.")
    p.add_argument("--limit", type=int, default=None, help="Smoke: score only the first N docs.")
    p.add_argument("--input-bucket", default="marin-us-east5", help="Bucket holding the checkpoints + sample.")
    args = p.parse_args()
    run_score(args.model, args.batch_size, args.ctx, args.limit, args.input_bucket)


if __name__ == "__main__":
    main()
