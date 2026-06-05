# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Token-weighted Nemotron-CC quality distribution (matches the paper's Table 2 basis).

The Nemotron-CC paper reports quality-tier ratios by TOKENS. Our earlier pass used
character counts as a proxy; this tokenizes EVERY document with the Llama-3 tokenizer
(the one our caches use) and tallies ACTUAL token counts per (nemotron_quality,
nemotron_kind). It also keeps char + record counts so we can quantify how far the
char proxy drifted from true token weighting.

Tokens are counted with add_special_tokens=False (content tokens only — no BOS/EOS,
which would add a doc-count-correlated constant and slightly inflate buckets with
many short docs). organic (kind=actual) is the apples-to-apples comparison to the
paper's organic Table 2.

Resumable: writes one JSON per source as soon as it finishes, and skips sources whose
output already exists — so a preemption mid-run only re-does the in-flight source.
Single in-region job: a thread pool reads shards while each thread batch-tokenizes its
own docs single-threaded (TOKENIZERS_PARALLELISM=false) so the N threads give N-way
parallelism without rust-thread oversubscription.

Usage (Iris CPU job, us-central2):
    python experiments/baseline_collection/nemotron_token_dist.py \\
        --out-dir gs://marin-us-central2/metadata/nemotron_quality_dist/tokens \\
        --source biased=gs://marin-us-central2/filtered/baseline_nemotron_full-347dfe \\
        --source random=gs://marin-us-central2/filtered/baseline_nemotron_full-2775c6 \\
        --source tenk=gs://marin-us-central2/filtered/dclm_400m_1x_10k_nemotron_full-96bad9
"""

from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # parallelize across shards, not within

import argparse
import gzip
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOKENIZER_ID = "meta-llama/Meta-Llama-3.1-8B"
ALLOWED_BUCKET = "marin-us-central2"
QUALITY_ORDER = ["high", "medium-high", "medium", "medium-low", "low"]
TOKENIZE_BATCH = 1000

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return _TOKENIZER


def _scan_shard(path: str) -> dict:
    tok = _tokenizer()
    tokens: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    records: dict[str, int] = defaultdict(int)
    batch_texts: list[str] = []
    batch_keys: list[str] = []

    def _flush():
        if not batch_texts:
            return
        encs = tok(batch_texts, add_special_tokens=False)["input_ids"]
        for key, enc in zip(batch_keys, encs, strict=True):
            tokens[key] += len(enc)
        batch_texts.clear()
        batch_keys.clear()

    try:
        with fsspec.open(path, "rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                q = rec.get("nemotron_quality") or "unknown"
                k = rec.get("nemotron_kind") or "unknown"
                text = rec.get("text") or ""
                key = f"{q}|{k}"
                chars[key] += len(text)
                records[key] += 1
                batch_texts.append(text)
                batch_keys.append(key)
                if len(batch_texts) >= TOKENIZE_BATCH:
                    _flush()
            _flush()
    except Exception as e:
        return {"__err__": str(e)}
    return {"tokens": dict(tokens), "chars": dict(chars), "records": dict(records)}


def _tally(label: str, root: str, max_workers: int) -> dict:
    if ALLOWED_BUCKET not in root:
        raise ValueError(f"cross-region read blocked (expected {ALLOWED_BUCKET}): {root}")
    fs = fsspec.filesystem("gcs")
    shards = [p if p.startswith("gs://") else f"gs://{p}" for p in fs.glob(f"{root}/*.jsonl.gz")]
    logger.info("[%s] tokenizing %d shards under %s", label, len(shards), root)

    tokens: dict[str, int] = defaultdict(int)
    chars: dict[str, int] = defaultdict(int)
    records: dict[str, int] = defaultdict(int)
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_scan_shard, s) for s in shards]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            if "__err__" in r:
                errors += 1
                logger.warning("[%s] shard error: %s", label, r["__err__"])
                continue
            for key, v in r["tokens"].items():
                tokens[key] += v
            for key, v in r["chars"].items():
                chars[key] += v
            for key, v in r["records"].items():
                records[key] += v
            done += 1
            if done % 1000 == 0:
                logger.info("[%s] %d/%d shards tokenized", label, done, len(shards))

    total_tokens = sum(tokens.values())
    org_tokens = {q: tokens.get(f"{q}|actual", 0) for q in QUALITY_ORDER}
    org_total = sum(org_tokens.values())
    logger.info("[%s] total tokens=%d (organic=%d) errors=%d", label, total_tokens, org_total, errors)
    for q in QUALITY_ORDER:
        pct = 100 * org_tokens[q] / org_total if org_total else 0.0
        logger.info("  [%s] %-12s organic_tokens=%-14d (%5.1f%%)", label, q, org_tokens[q], pct)

    return {
        "label": label,
        "root": root,
        "tokenizer": TOKENIZER_ID,
        "add_special_tokens": False,
        "n_shards": len(shards),
        "errors": errors,
        "by_quality_kind_tokens": dict(tokens),
        "by_quality_kind_chars": dict(chars),
        "by_quality_kind_records": dict(records),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", action="append", required=True, help="label=gs://...root (repeatable)")
    p.add_argument("--out-dir", required=True, help="gs:// dir for per-source JSON results")
    p.add_argument("--max-workers", type=int, default=16)
    args = p.parse_args()

    fs = fsspec.filesystem("gcs")
    # Smallest first so quick results checkpoint early; the big 10k source runs last.
    for spec in args.source:
        label, root = spec.split("=", 1)
        out_path = f"{args.out_dir.rstrip('/')}/tokens_{label}.json"
        if fs.exists(out_path.replace("gs://", "")):
            logger.info("[%s] result already exists at %s — skipping (resume)", label, out_path)
            continue
        result = _tally(label, root.rstrip("/"), args.max_workers)
        with fsspec.open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("[%s] wrote %s", label, out_path)

    logger.info("ALL SOURCES DONE")


if __name__ == "__main__":
    main()
