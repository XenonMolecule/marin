# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Aggregate per-batch token stats from .tokens.gz sidecar files.

For each batch file, emits a per-batch summary row with:
  - Per-status token sums (input, thinking, response, count)
  - Exact FLOP accounting per batch using inference_flops_gold
  - Convenience top-level fields for projection

Launch via ray_run on a regional cluster with --buckets set to that region's
bucket only. Designed to be re-run as more batches come in.
"""

import argparse
import gzip
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

logger = logging.getLogger(__name__)

DEFAULT_BUCKETS = [
    "marin-us-central1",
    "marin-us-east5",
    "marin-eu-west4",
]
OUTPUT_SUBDIR = "documents/baseline_llm_extraction"

# Qwen3-8B config (from experiments/qwen3.py + HF config)
QWEN3_8B = dict(
    hidden_dim=4096,
    intermediate_dim=12288,
    num_layers=36,
    num_heads=32,
    num_kv_heads=8,
    vocab_size=151936,
    glu=True,
)


def inference_flops(prompt_len: int, output_len: int) -> float:
    """Exact inference FLOPs for one request (prefill + decode), closed-form.

    Decomposes lm_flops_per_token into position-independent (MLP, projections,
    LM head) and position-dependent (attention) terms, then uses an arithmetic
    series for the decode attention sum.
    """
    cfg = QWEN3_8B
    h = cfg["hidden_dim"]
    d = h / cfg["num_heads"]
    n_h = cfg["num_heads"]
    n_kv = cfg["num_kv_heads"]
    n_l = cfg["num_layers"]
    V = cfg["vocab_size"]
    inter = cfg["intermediate_dim"]

    mlp = 2 * 3 * h * inter
    qkv_proj = 2 * h * (n_h * d + 2 * n_kv * d)
    dense_proj = 2 * h * h
    lm_head = 2 * h * V
    non_attn_per_token = n_l * (mlp + qkv_proj + dense_proj) + lm_head

    attn_per_ctx = n_l * (2 * n_h * d + 3 * n_h + 2 * d * n_h)

    prefill = non_attn_per_token * prompt_len + attn_per_ctx * prompt_len * prompt_len
    total_ctx = output_len * prompt_len + output_len * (output_len - 1) / 2
    decode = non_attn_per_token * output_len + attn_per_ctx * total_ctx

    return prefill + decode


def list_token_files(bucket: str) -> list[str]:
    fs = fsspec.filesystem("gcs")
    pattern = f"{bucket}/{OUTPUT_SUBDIR}/**/batch_*.tokens.gz"
    paths = fs.glob(pattern)
    return [p if p.startswith("gs://") else f"gs://{p}" for p in paths]


def aggregate_one(path: str) -> dict:
    try:
        with fsspec.open(path, "rb") as f:
            with gzip.open(f, "rt") as gz:
                records = [json.loads(line) for line in gz]
    except Exception as e:
        return {"batch_path": path, "error": str(e)}

    tokens_by_status: dict[str, dict[str, int]] = {}
    batch_flops_all = 0.0
    batch_flops_kept = 0.0

    for r in records:
        s = r.get("status", "unknown")
        bucket = tokens_by_status.setdefault(s, {"input": 0, "thinking": 0, "response": 0, "count": 0})
        in_toks = r.get("input_tokens", 0)
        think_toks = r.get("thinking_tokens", 0)
        resp_toks = r.get("response_tokens", 0)

        bucket["input"] += in_toks
        bucket["thinking"] += think_toks
        bucket["response"] += resp_toks
        bucket["count"] += 1

        out_toks = think_toks + resp_toks
        if in_toks > 0 and out_toks > 0:
            flops = inference_flops(in_toks, out_toks)
        elif in_toks > 0:
            flops = inference_flops(in_toks, 0)
        else:
            flops = 0.0
        batch_flops_all += flops
        if s == "kept":
            batch_flops_kept += flops

    kept = tokens_by_status.get("kept", {"input": 0, "thinking": 0, "response": 0, "count": 0})

    return {
        "batch_path": path,
        "total_records": len(records),
        "by_status": dict(Counter(r.get("status", "unknown") for r in records)),
        "tokens_by_status": tokens_by_status,
        "kept_records": kept["count"],
        "sum_response_tokens_kept": kept["response"],
        "sum_thinking_tokens_kept": kept["thinking"],
        "sum_input_tokens_kept": kept["input"],
        "sum_input_tokens_all": sum(v["input"] for v in tokens_by_status.values()),
        "sum_thinking_tokens_all": sum(v["thinking"] for v in tokens_by_status.values()),
        "sum_response_tokens_all": sum(v["response"] for v in tokens_by_status.values()),
        "flops_all": batch_flops_all,
        "flops_kept": batch_flops_kept,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="GCS path for aggregate JSONL.gz")
    parser.add_argument("--buckets", default=",".join(DEFAULT_BUCKETS), help="Comma-separated bucket names")
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    buckets = args.buckets.split(",")

    all_paths: list[str] = []
    for bucket in buckets:
        t0 = time.monotonic()
        paths = list_token_files(bucket)
        logger.info("Listed %d files in %s (%.1fs)", len(paths), bucket, time.monotonic() - t0)
        all_paths.extend(paths)
    logger.info("Total .tokens.gz files: %d", len(all_paths))

    if args.limit:
        all_paths = all_paths[: args.limit]
        logger.info("Limited to %d files", len(all_paths))

    t0 = time.monotonic()
    summaries: list[dict] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(aggregate_one, p): p for p in all_paths}
        for i, fut in enumerate(as_completed(futures), start=1):
            s = fut.result()
            if "error" in s:
                errors += 1
            else:
                summaries.append(s)
            if i % 1000 == 0:
                elapsed = time.monotonic() - t0
                rate = i / elapsed
                eta = (len(all_paths) - i) / max(rate, 1e-6)
                logger.info("Processed %d/%d (%.0f files/s, ETA %.0fs)", i, len(all_paths), rate, eta)

    logger.info("Done: %d success, %d errors, %.1fs", len(summaries), errors, time.monotonic() - t0)

    with fsspec.open(args.output, "wb") as f:
        with gzip.open(f, "wt", encoding="utf-8") as gz:
            for s in summaries:
                gz.write(json.dumps(s) + "\n")
    logger.info("Wrote %d summaries to %s", len(summaries), args.output)


if __name__ == "__main__":
    main()
