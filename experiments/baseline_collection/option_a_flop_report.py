# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Option A FLOP/token estimate for the llm_curated extraction.

Reads canonical sidecar paths (one path per unique resolved batch with a
.tokens.gz sidecar) from a paths file, aggregates per-batch token sums and
exact Qwen3-8B inference FLOPs, and writes:

  - aggregate.jsonl.gz: per-batch row dump (for re-running projection later).
  - summary.json: per-metric stats (mean, std, CV%, bootstrap 95% CI on the
    mean, projected total to N_TOTAL unique resolved batches), plus per-
    region per-batch means for missingness-bias diagnosis.
  - summary.txt: human-readable version of summary.json.

Designed to run in-region (us-central1) via Iris with no cross-region
traffic. CPU-bound on JSON decode; I/O-bound on tiny GCS reads — favors
many threads over many CPUs.
"""

import argparse
import gzip
import io
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

QWEN3_8B = dict(
    hidden_dim=4096,
    intermediate_dim=12288,
    num_layers=36,
    num_heads=32,
    num_kv_heads=8,
    vocab_size=151936,
)


def inference_flops(prompt_len: int, output_len: int) -> float:
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


def parse_region_from_path(path: str) -> str:
    # gs://marin-us-central1/documents/.../by_region/{region}/data-.../batch_NNNN.tokens.gz
    parts = path.split("/by_region/", 1)
    if len(parts) != 2:
        return "unknown"
    return parts[1].split("/", 1)[0]


def aggregate_one(path: str) -> dict:
    try:
        with fsspec.open(path, "rb") as f:
            data = f.read()
        with gzip.open(io.BytesIO(data), "rt") as gz:
            records = [json.loads(line) for line in gz]
    except Exception as e:
        return {"error": str(e), "path": path}

    sums = {
        "input_all": 0,
        "thinking_all": 0,
        "response_all": 0,
        "input_kept": 0,
        "thinking_kept": 0,
        "response_kept": 0,
        "flops_all": 0.0,
        "flops_kept": 0.0,
        "total_records": len(records),
        "kept_records": 0,
    }
    by_status: dict[str, int] = defaultdict(int)
    for r in records:
        s = r.get("status", "unknown")
        by_status[s] += 1
        in_t = r.get("input_tokens", 0)
        th_t = r.get("thinking_tokens", 0)
        rs_t = r.get("response_tokens", 0)
        out_t = th_t + rs_t

        sums["input_all"] += in_t
        sums["thinking_all"] += th_t
        sums["response_all"] += rs_t

        if in_t > 0 and out_t > 0:
            f_ = inference_flops(in_t, out_t)
        elif in_t > 0:
            f_ = inference_flops(in_t, 0)
        else:
            f_ = 0.0
        sums["flops_all"] += f_

        if s == "kept":
            sums["input_kept"] += in_t
            sums["thinking_kept"] += th_t
            sums["response_kept"] += rs_t
            sums["kept_records"] += 1
            sums["flops_kept"] += f_

    sums["region"] = parse_region_from_path(path)
    sums["status_counts"] = dict(by_status)
    sums["batch_path"] = path
    return sums


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(values.mean()), float(values.std(ddof=1)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths-file", required=True, help="GCS or local path with one .tokens.gz path per line")
    ap.add_argument("--output-dir", required=True, help="GCS dir to write aggregate.jsonl.gz, summary.json, summary.txt")
    ap.add_argument("--n-total", type=int, default=281_254, help="Total unique resolved batches to project to")
    ap.add_argument("--max-workers", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None, help="If set, only process this many paths (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 1. Load paths.
    with fsspec.open(args.paths_file, "r") as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        paths = paths[: args.limit]
    logger.info("Loaded %d canonical paths", len(paths))

    # 2. Parallel aggregate.
    t0 = time.monotonic()
    results: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(aggregate_one, p) for p in paths]
        for i, fut in enumerate(as_completed(futs), start=1):
            s = fut.result()
            if "error" in s:
                errors.append(s)
            else:
                results.append(s)
            if i % 5000 == 0:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Aggregated %d/%d (%.0f files/s, ETA %.0fs, %d errors)",
                    i,
                    len(paths),
                    i / elapsed,
                    (len(paths) - i) / max(i / elapsed, 1e-6),
                    len(errors),
                )
    logger.info("Aggregation done in %.1fs (%d ok, %d errors)", time.monotonic() - t0, len(results), len(errors))

    # 3. Write per-batch aggregate (small, but let's keep it for reuse).
    output_dir = args.output_dir.rstrip("/")
    agg_path = f"{output_dir}/aggregate.jsonl.gz"
    with fsspec.open(agg_path, "wb") as f:
        with gzip.open(f, "wt") as gz:
            for r in results:
                gz.write(json.dumps(r) + "\n")
    logger.info("Wrote %s", agg_path)

    # 4. Bootstrap + per-region stats.
    metrics = [
        ("input_tokens_all", "input_all"),
        ("thinking_tokens_all", "thinking_all"),
        ("response_tokens_all", "response_all"),
        ("input_tokens_kept", "input_kept"),
        ("thinking_tokens_kept", "thinking_kept"),
        ("response_tokens_kept", "response_kept"),
        ("flops_all", "flops_all"),
        ("flops_kept", "flops_kept"),
        ("total_records", "total_records"),
        ("kept_records", "kept_records"),
    ]

    summary: dict = {
        "n_observed_batches": len(results),
        "n_errors": len(errors),
        "n_projected_batches": args.n_total,
        "qwen3_8b_config": QWEN3_8B,
        "per_metric": {},
        "per_region": {},
    }

    N = args.n_total
    for label, key in metrics:
        vals = np.array([r[key] for r in results], dtype=np.float64)
        mean, std, lo, hi = bootstrap_ci(vals, n_boot=10000, seed=0)
        cv_pct = 100 * std / max(mean, 1e-12)
        summary["per_metric"][label] = {
            "per_batch_mean": mean,
            "per_batch_std": std,
            "per_batch_cv_pct": cv_pct,
            "per_batch_mean_ci_lo": lo,
            "per_batch_mean_ci_hi": hi,
            "projected_total": mean * N,
            "projected_total_ci_lo": lo * N,
            "projected_total_ci_hi": hi * N,
            "projected_total_relerr_pct": 100 * (hi - lo) / 2 / max(mean * N, 1e-12),
        }

    # Per-region per-batch means.
    by_region: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_region[r["region"]].append(r)
    for region, rs in by_region.items():
        if not rs:
            continue
        summary["per_region"][region] = {
            "n_batches": len(rs),
            "mean_input_all": float(np.mean([r["input_all"] for r in rs])),
            "mean_thinking_all": float(np.mean([r["thinking_all"] for r in rs])),
            "mean_response_all": float(np.mean([r["response_all"] for r in rs])),
            "mean_flops_all": float(np.mean([r["flops_all"] for r in rs])),
            "mean_total_records": float(np.mean([r["total_records"] for r in rs])),
            "mean_kept_records": float(np.mean([r["kept_records"] for r in rs])),
        }

    # Aggregate status-count distribution.
    status_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for s, c in r.get("status_counts", {}).items():
            status_totals[s] += c
    summary["status_counts_observed"] = dict(status_totals)

    # 5. Write summary outputs.
    summary_json = f"{output_dir}/summary.json"
    with fsspec.open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", summary_json)

    # Human-readable text.
    lines = []
    lines.append("Option A FLOP/token estimate for llm_curated extraction (Qwen3-8B inference)")
    lines.append(f"Observed batches: {summary['n_observed_batches']:,}")
    lines.append(f"Projected to:     {summary['n_projected_batches']:,} unique resolved batches (Option A: assume MAR)")
    lines.append(f"Errors:           {summary['n_errors']}")
    lines.append("")
    lines.append("PER-METRIC (per-batch dispersion + projected total)")
    lines.append("-" * 110)
    lines.append(f"{'metric':<22} {'mean':>14} {'std':>14} {'CV %':>7}  {'projected total':>22}  {'95% CI':>30}")
    lines.append("-" * 110)
    for label, _ in metrics:
        m = summary["per_metric"][label]
        if "flops" in label:
            ptot = f"{m['projected_total']:.3e}"
            pci = f"[{m['projected_total_ci_lo']:.3e}, {m['projected_total_ci_hi']:.3e}]"
        else:
            ptot = f"{m['projected_total']/1e9:,.2f} B"
            pci = f"[{m['projected_total_ci_lo']/1e9:,.2f} B, {m['projected_total_ci_hi']/1e9:,.2f} B]"
        lines.append(
            f"{label:<22} {m['per_batch_mean']:>14,.0f} {m['per_batch_std']:>14,.0f} "
            f"{m['per_batch_cv_pct']:>6.1f}%  {ptot:>22}  {pci:>30}"
        )

    lines.append("")
    lines.append("PER-REGION PER-BATCH MEANS (sanity check for bias from non-MAR sidecar coverage)")
    lines.append("-" * 110)
    lines.append(
        f"{'region':<14} {'n':>8} {'<input>':>12} {'<think>':>12} {'<resp>':>12} {'<flops>':>14} {'<kept_rec>':>12}"
    )
    for region in sorted(summary["per_region"]):
        r = summary["per_region"][region]
        lines.append(
            f"{region:<14} {r['n_batches']:>8,} {r['mean_input_all']:>12,.0f} {r['mean_thinking_all']:>12,.0f} "
            f"{r['mean_response_all']:>12,.0f} {r['mean_flops_all']:>14.3e} {r['mean_kept_records']:>12,.1f}"
        )

    lines.append("")
    lines.append("STATUS DISTRIBUTION (observed records, sidecar-bearing batches)")
    lines.append("-" * 60)
    total = sum(status_totals.values())
    for s, c in sorted(status_totals.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {s:<35} {c:>12,}  ({100*c/max(total,1):>5.2f}%)")
    lines.append(f"  {'TOTAL':<35} {total:>12,}")

    summary_txt = f"{output_dir}/summary.txt"
    with fsspec.open(summary_txt, "w") as f:
        f.write("\n".join(lines))
    logger.info("Wrote %s", summary_txt)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
