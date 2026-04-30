# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Project total tokens for the full extraction run from per-batch aggregates.

Inputs: three per-region aggregate JSONL.gz files (one row per batch).
Outputs: bootstrapped mean + 95% CI for per-batch quantities, projected to
the full-run batch count.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DEFAULT_INPUTS = [
    "/tmp/aggregates/us_central1.jsonl.gz",
    "/tmp/aggregates/us_east5.jsonl.gz",
    "/tmp/aggregates/eu_west4.jsonl.gz",
]


def load_aggregate(path: str) -> list[dict]:
    with gzip.open(path, "rt") as gz:
        return [json.loads(line) for line in gz]


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, ci: float = 0.95, seed: int = 0):
    """Return (mean, lower, upper) from percentile bootstrap over the sample mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return values.mean(), lo, hi


def project(values: np.ndarray, n_total: int, label: str, n_boot: int = 10000):
    mean, lo, hi = bootstrap_ci(values, n_boot=n_boot)
    total_mean = mean * n_total
    total_lo = lo * n_total
    total_hi = hi * n_total
    print(f"{label}")
    print(f"  Per-batch mean: {mean:,.0f}  [95% CI: {lo:,.0f} – {hi:,.0f}]")
    print(f"  Projected total @ {n_total:,} batches: {total_mean:.3e}")
    print(f"    95% CI: [{total_lo:.3e}, {total_hi:.3e}]")
    pm = (total_hi - total_lo) / 2
    print(f"    = {total_mean/1e9:,.2f}B  ±{pm/1e9:.2f}B ({100*pm/total_mean:.2f}%)")
    print()
    return total_mean, total_lo, total_hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    parser.add_argument("--projected-batches", type=int, default=277265)
    parser.add_argument(
        "--batch-count-uncertainty",
        type=float,
        default=0.02,
        help="Relative uncertainty in projected batch count (e.g. 0.02 = ±2%)",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    # Load all per-batch rows
    all_rows = []
    for inp in args.inputs:
        rows = load_aggregate(inp)
        print(f"Loaded {len(rows):>6,} batches from {Path(inp).name}")
        all_rows.extend(rows)
    print(f"Total observed batches: {len(all_rows):,}")
    print(f"Projected total batches: {args.projected_batches:,} (±{100*args.batch_count_uncertainty:.1f}%)")
    print()

    # Extract per-batch vectors
    # Skip any batches with errors
    rows = [r for r in all_rows if "error" not in r]
    if len(rows) != len(all_rows):
        print(f"NOTE: skipped {len(all_rows) - len(rows)} batches with errors")

    # Primary quantity: kept response tokens
    resp_kept = np.array([r["sum_response_tokens_kept"] for r in rows], dtype=np.int64)
    think_kept = np.array([r["sum_thinking_tokens_kept"] for r in rows], dtype=np.int64)
    input_kept = np.array([r["sum_input_tokens_kept"] for r in rows], dtype=np.int64)
    input_all = np.array([r["sum_input_tokens_all"] for r in rows], dtype=np.int64)
    think_all = np.array([r["sum_thinking_tokens_all"] for r in rows], dtype=np.int64)
    resp_all = np.array([r["sum_response_tokens_all"] for r in rows], dtype=np.int64)
    kept_records = np.array([r["kept_records"] for r in rows], dtype=np.int64)
    total_records = np.array([r["total_records"] for r in rows], dtype=np.int64)

    # FLOP data (only present in v2 aggregates)
    has_flops = "flops_all" in rows[0]
    if has_flops:
        flops_all = np.array([r["flops_all"] for r in rows], dtype=np.float64)
        flops_kept = np.array([r["flops_kept"] for r in rows], dtype=np.float64)

    N = args.projected_batches

    print("=" * 70)
    print("PRIMARY: Kept tokens (what you'll train on)")
    print("=" * 70)
    project(resp_kept, N, "Kept response tokens per batch (option A)", args.n_boot)

    print("=" * 70)
    print("SECONDARY: Other per-batch quantities")
    print("=" * 70)
    project(kept_records, N, "Kept records per batch", args.n_boot)
    project(total_records, N, "Total records processed per batch", args.n_boot)
    project(think_kept, N, "Thinking tokens on kept records per batch", args.n_boot)
    project(input_kept, N, "Input tokens on kept records per batch", args.n_boot)

    print("=" * 70)
    print("TERTIARY: All-records totals (for FLOP accounting)")
    print("=" * 70)
    project(input_all, N, "Input tokens ALL records per batch", args.n_boot)
    project(think_all, N, "Thinking tokens ALL records per batch", args.n_boot)
    project(resp_all, N, "Response tokens ALL records per batch", args.n_boot)

    if has_flops:
        print("=" * 70)
        print("FLOP ACCOUNTING (exact per-record computation, projected)")
        print("=" * 70)
        project(flops_all, N, "Total inference FLOPs (all records)", args.n_boot)
        project(flops_kept, N, "Inference FLOPs on kept records only", args.n_boot)
        wasted = flops_all - flops_kept
        project(wasted, N, "Inference FLOPs on filtered records (wasted)", args.n_boot)

    # Error-bar combining: propagate batch-count uncertainty with per-batch CI.
    print("=" * 70)
    print("COMBINED UNCERTAINTY (includes batch count fluctuation)")
    print("=" * 70)
    mean, lo, hi = bootstrap_ci(resp_kept, n_boot=args.n_boot)
    per_batch_relerr = ((hi - lo) / 2) / mean
    total_relerr = float(np.sqrt(per_batch_relerr**2 + args.batch_count_uncertainty**2))
    total_mean = mean * N
    print("Kept response tokens:")
    print(f"  Per-batch mean relerr: {100*per_batch_relerr:.3f}%")
    print(f"  Batch count relerr:    {100*args.batch_count_uncertainty:.2f}%")
    print(f"  Combined relerr:       {100*total_relerr:.2f}%")
    print(f"  Projected total:       {total_mean/1e9:,.2f}B tokens")
    print(f"  95% CI (combined):     [{total_mean*(1-total_relerr)/1e9:,.2f}B, {total_mean*(1+total_relerr)/1e9:,.2f}B]")
    print(f"  → {total_mean/1e9:,.2f}B ±{total_mean*total_relerr/1e9:.2f}B kept response tokens")

    if has_flops:
        print()
        mean_f, lo_f, hi_f = bootstrap_ci(flops_all, n_boot=args.n_boot)
        per_batch_relerr_f = ((hi_f - lo_f) / 2) / mean_f
        total_relerr_f = float(np.sqrt(per_batch_relerr_f**2 + args.batch_count_uncertainty**2))
        total_mean_f = mean_f * N
        print("Total inference FLOPs:")
        print(f"  Per-batch mean relerr: {100*per_batch_relerr_f:.3f}%")
        print(f"  Combined relerr:       {100*total_relerr_f:.2f}%")
        print(f"  Projected total:       {total_mean_f:.3e} FLOPs")
        print(f"  95% CI (combined):     [{total_mean_f*(1-total_relerr_f):.3e}, {total_mean_f*(1+total_relerr_f):.3e}]")
        print(f"  → {total_mean_f:.3e} ±{total_mean_f*total_relerr_f:.2e} FLOPs")


if __name__ == "__main__":
    main()
