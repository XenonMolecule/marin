# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the loss-masked sysprompt arm (AM) as prebuilt token shards.

Arm A trains on ``conditioned_text`` = ``[S] wrapper + prompt + <|eot_id|> + document``
with loss on every token, so ~6% of its gradient goes to generating the prompt. Arm AM
keeps the identical token stream and zeroes the loss on the prefix, i.e. it is exactly
``p(D | S)``.

Each ``ab-*`` shard becomes ``masked/prebuilt-*.jsonl.gz`` with ``input_ids`` and
``loss_weights``. Tokenization reproduces Levanter's ``BatchTokenizer`` byte for byte:
``[BOS] + encode(text + " " + eos_token, add_special_tokens=False)``, so the AM cache's
``total_tokens`` must equal A's (assert this after tokenizing). ``loss_weights[i]`` gates
the prediction of token ``i+1`` (Levanter's convention), so it is 0 for every position
whose target is in the prefix through ``<|eot_id|>`` and 1 afterwards, including the
position that predicts the next document's BOS, as in A.

Usage (CPU job on Iris, us-central1)::

    uv run --no-sync iris --cluster marin job run --no-wait \\
        --region us-central1 --cpu 8 --memory 14GB --disk 50GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --job-name build-sp30b-masked-447m-9e18 -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/build_sysprompt30b_masked.py --tag 447m_9e18 --workers 8
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import multiprocessing as mp
from typing import cast

import fsspec
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from experiments.scaling_law_sweeps.tokenize_sysprompt30b import BUILD, LLAMA3_TOKENIZER, ab_shards

logger = logging.getLogger(__name__)

TEXT_KEY = "conditioned_text"
START_HEADER_ID = 128006  # <|start_header_id|>
EOT_ID = 128009  # <|eot_id|>


def mask_prefix(ids: list[int]) -> list[int]:
    """1 for positions whose next token is document text, 0 through the [S] prefix."""
    if len(ids) < 3 or ids[1] != START_HEADER_ID:
        raise ValueError(f"record does not start with BOS + <|start_header_id|>: {ids[:4]}")
    eot = ids.index(EOT_ID)
    # positions 0..eot-1 predict tokens 1..eot (the prefix incl. <|eot_id|>) -> 0;
    # position eot predicts the first document token -> 1.
    return [0] * eot + [1] * (len(ids) - eot)


def convert_shard(args: tuple[str, str]) -> tuple[str, int, int, int]:
    src, dst = args
    tok = cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(LLAMA3_TOKENIZER))
    bos, eos = tok.bos_token_id, tok.eos_token
    with fsspec.open(src, "rb") as f, gzip.open(f, "rt", encoding="utf-8") as g:
        records = [json.loads(line) for line in g if line.strip()]
    texts = [r[TEXT_KEY] + " " + eos for r in records]
    encoded = tok(texts, add_special_tokens=False)["input_ids"]
    buf = io.StringIO()
    n_tok = n_masked = 0
    for r, ids in zip(records, encoded, strict=True):
        ids = [bos, *ids]
        w = mask_prefix(ids)
        n_tok += len(ids)
        n_masked += len(w) - sum(w)
        buf.write(json.dumps({"doc_id": r["doc_id"], "input_ids": ids, "loss_weights": w}, separators=(",", ":")))
        buf.write("\n")
    with fsspec.open(dst, "wb") as f, gzip.open(f, "wt", encoding="utf-8") as g:
        g.write(buf.getvalue())
    return dst, len(records), n_tok, n_masked


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    shards = ab_shards(a.tag)
    out_dir = f"{BUILD}/{a.tag}/masked"
    jobs = [(src, f"{out_dir}/prebuilt-{src.rsplit('/', 1)[1].split('-', 1)[1]}") for src in shards]
    logger.info("%d shards -> %s", len(jobs), out_dir)
    docs = toks = masked = 0
    with mp.Pool(a.workers) as pool:
        for i, (_, n_docs, n_tok, n_masked) in enumerate(pool.imap_unordered(convert_shard, jobs), 1):
            docs += n_docs
            toks += n_tok
            masked += n_masked
            if i % 50 == 0 or i == len(jobs):
                logger.info(
                    "%d/%d shards | docs %d | tokens %d | masked %d (%.2f%%)",
                    i,
                    len(jobs),
                    docs,
                    toks,
                    masked,
                    100 * masked / toks,
                )
    manifest = {
        "tag": a.tag,
        "shards": len(jobs),
        "docs": docs,
        "tokens": toks,
        "masked_tokens": masked,
        "text_key": TEXT_KEY,
    }
    with fsspec.open(f"{out_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("DONE %s", json.dumps(manifest))


if __name__ == "__main__":
    main()
