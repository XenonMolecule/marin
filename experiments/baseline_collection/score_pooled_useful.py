# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a trained ``pooled_transformer`` useful-classifier over the 100k comparison sample.

The cascade planner needs one scalar ``P(useful)`` column per candidate stage, over the SAME 100k
docs in the SAME order as every other classifier. This is the pooled-transformer counterpart to
``score_modernbert_useful`` (HF/safetensors) and ``score_fasttext_useful`` (CPU) — it reuses their
sample reader and preprocessing verbatim, so doc *i* is doc *i* across all three families.

Checkpoints are the generic equinox format written by ``save_eqx_classifier`` (``model.eqx`` +
``config.json``), NOT HF — the architecture is read back out of ``config.json``, so any pooled run id
works. Output lands in the shared ``model_scores/<col>/`` layout that
``score_modernbert_useful join`` merges onto the sample.

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
from levanter.models.pooled_transformer import load_pooled_transformer_classifier
from transformers import AutoTokenizer

from experiments.baseline_collection.score_modernbert_useful import SCORES_DIR, TOKENIZER_REF, _read_sample
from experiments.baseline_collection.score_pooled_frozen_eval import MODERNBERT_VOCAB_SIZE, load_config

logger = logging.getLogger(__name__)

CKPT_NS = "checkpoints/modernbert-useful"

# col -> checkpoint run-id. All target llm_pipeline_v1_1 (lpv11), not the 8B high_quality run.
MODELS: dict[str, str] = {
    "pooled_lpv11_prob_1M_r5": "mb-clf-lpv11-pooled-1M-r5",
    "pooled_lpv11_prob_10M": "mb-clf-lpv11-pooled-10M",
    "pooled_lpv11_prob_10M_e3": "mb-clf-lpv11-pooled-10M-e3",
    "pooled_lpv11_prob_big_10M": "mb-clf-lpv11-pooledbig-10M",
}


def run_score(col: str, batch_size: int, ctx: int | None, limit: int | None, input_bucket: str) -> None:
    ckpt = f"gs://{input_bucket}/{CKPT_NS}/{MODELS[col]}/hf"
    config = load_config(ckpt, ctx)
    logger.info("col=%s ckpt=%s ctx=%d devices=%s", col, ckpt, config.max_seq_len, jax.devices())

    ids, texts = _read_sample(input_bucket)
    if limit:
        ids, texts = ids[:limit], texts[:limit]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REF)

    # set_mesh (not `with mesh:`) so haliax-internal init paths see the resource axes.
    n_dev = len(jax.devices())
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))
    with set_mesh(mesh):
        model = load_pooled_transformer_classifier(config, ckpt, vocab_size=MODERNBERT_VOCAB_SIZE)
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
