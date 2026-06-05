# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spec-agnostic FLOP/token aggregator for `.tokens.gz` sidecars.

Walks the consolidated archive in us-central1 (mirror of all regions for a
given extraction spec), reads each batch's per-record token sidecar, and
sums exact Qwen3-8B inference FLOPs (same closed-form accounting as
`option_a_flop_report.py`). Writes:

  - <output_dir>/summary.json      — per-region + total sums, status hist
  - <output_dir>/aggregate.jsonl.gz — per-batch dump (one row per sidecar)

Designed to run as an Iris CPU job in us-central1 so all reads are
in-region. Uses google.cloud.storage directly to bypass the
fsspec/aiohttp SSL cert verification bug seen on some local machines.

Example (locally):
    uv run python -m experiments.baseline_collection.aggregate_flop_sidecars \\
        --spec high_quality \\
        --output-dir gs://marin-us-central1/scratch/high_quality_flop_estimate/

Example (Iris):
    iris --config lib/iris/examples/marin.yaml job run \\
        --cpu 4 --memory 8GB --region us-central1 \\
        --job-name flop-agg-high_quality --no-wait \\
        -- python -m experiments.baseline_collection.aggregate_flop_sidecars \\
            --spec high_quality \\
            --output-dir gs://marin-us-central1/scratch/high_quality_flop_estimate/
"""

import argparse
import gzip
import io
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from google.cloud import storage

logger = logging.getLogger(__name__)

# Layout of the consolidated archive (us-central1 mirror of all 5 regions).
CONSOLIDATED_BUCKET = "marin-us-central1"
CONSOLIDATED_PREFIX = "documents/baseline_llm_extraction_consolidated/by_region"
REGIONS = ("us-central1", "us-east1", "us-east5", "us-west4", "europe-west4")
LEGACY_SPEC = "low_quality"  # written at the unprefixed path (no spec subdir)


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int = 151_936


QWEN3_8B = ModelConfig(
    hidden_dim=4096,
    intermediate_dim=12288,
    num_layers=36,
    num_heads=32,
    num_kv_heads=8,
)


def inference_flops(prompt_len: int, output_len: int, cfg: ModelConfig = QWEN3_8B) -> float:
    """Exact closed-form inference FLOPs for a single (prefill, decode) call.

    Mirrors option_a_flop_report.inference_flops. Splits into position-
    independent terms (MLP, projections, lm_head) that scale linearly with
    token count and a position-dependent attention term that scales with
    context length.
    """
    h, inter = cfg.hidden_dim, cfg.intermediate_dim
    d = h / cfg.num_heads
    n_h, n_kv = cfg.num_heads, cfg.num_kv_heads
    n_l, V = cfg.num_layers, cfg.vocab_size
    mlp = 2 * 3 * h * inter
    qkv = 2 * h * (n_h * d + 2 * n_kv * d)
    dense = 2 * h * h
    lm_head = 2 * h * V
    non_attn = n_l * (mlp + qkv + dense) + lm_head
    attn = n_l * (2 * n_h * d + 3 * n_h + 2 * d * n_h)
    prefill = non_attn * prompt_len + attn * prompt_len * prompt_len
    decode = non_attn * output_len + attn * (output_len * prompt_len + output_len * (output_len - 1) / 2)
    return prefill + decode


# Singleton GCS client (thread-safe per google-cloud-storage docs).
_CLIENT: storage.Client | None = None


def get_client() -> storage.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = storage.Client()
    return _CLIENT


def enumerate_sidecars(spec: str) -> dict[str, list[str]]:
    """Enumerate batch_*.tokens.gz sidecars for `spec` in the consolidated archive.

    Returns a dict mapping region → list of full gs:// URIs.
    """
    client = get_client()
    bucket = client.bucket(CONSOLIDATED_BUCKET)
    out: dict[str, list[str]] = {}
    for region in REGIONS:
        if spec == LEGACY_SPEC:
            prefix = f"{CONSOLIDATED_PREFIX}/{region}/"
        else:
            prefix = f"{CONSOLIDATED_PREFIX}/{region}/{spec}/"
        uris: list[str] = []
        # list_blobs is paginated server-side; iterating is cheap in-region.
        for blob in client.list_blobs(bucket, prefix=prefix, match_glob="**/batch_*.tokens.gz"):
            uris.append(f"gs://{CONSOLIDATED_BUCKET}/{blob.name}")
        out[region] = uris
        logger.info("  %s: %d sidecars", region, len(uris))
    return out


def aggregate_one(uri: str) -> dict:
    """Read one sidecar, compute per-batch sums + per-status counts."""
    try:
        assert uri.startswith("gs://")
        rest = uri[5:]
        bucket_name, _, key = rest.partition("/")
        blob = get_client().bucket(bucket_name).blob(key)
        data = blob.download_as_bytes()
        with gzip.open(io.BytesIO(data), "rt") as gz:
            records = [json.loads(line) for line in gz]
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "path": uri}

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

    sums["status_counts"] = dict(by_status)
    sums["batch_path"] = uri
    sums["region"] = _region_from_uri(uri)
    return sums


def _region_from_uri(uri: str) -> str:
    """Extract source region from a consolidated-archive URI."""
    marker = f"/{CONSOLIDATED_PREFIX}/"
    if marker not in uri:
        return "unknown"
    tail = uri.split(marker, 1)[1]
    return tail.split("/", 1)[0]


def write_outputs(output_dir: str, results: list[dict], errors: list[dict], elapsed_sec: float, spec: str) -> None:
    """Write aggregate.jsonl.gz + summary.json to a GCS dir."""
    assert output_dir.startswith("gs://"), output_dir
    rest = output_dir[5:].rstrip("/")
    bucket_name, _, prefix = rest.partition("/")
    bucket = get_client().bucket(bucket_name)

    # Per-batch dump (gzipped JSONL).
    agg_buf = io.BytesIO()
    with gzip.open(agg_buf, "wt") as gz:
        for r in results:
            gz.write(json.dumps(r) + "\n")
    bucket.blob(f"{prefix}/aggregate.jsonl.gz").upload_from_string(
        agg_buf.getvalue(),
        content_type="application/gzip",
    )

    # Per-region + grand-total summary.
    keys = [
        "input_all",
        "thinking_all",
        "response_all",
        "input_kept",
        "thinking_kept",
        "response_kept",
        "flops_all",
        "flops_kept",
        "total_records",
        "kept_records",
    ]
    grand_sums = {k: sum(r[k] for r in results) for k in keys}
    per_region: dict[str, dict] = {}
    for region in REGIONS:
        region_rows = [r for r in results if r["region"] == region]
        if not region_rows:
            continue
        per_region[region] = {
            "n_batches": len(region_rows),
            "sums": {k: sum(r[k] for r in region_rows) for k in keys},
        }

    status_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for s, c in r.get("status_counts", {}).items():
            status_totals[s] += c

    summary = {
        "spec": spec,
        "n_batches_total": len(results),
        "n_errors": len(errors),
        "elapsed_sec": elapsed_sec,
        "qwen3_8b_config": QWEN3_8B.__dict__,
        "grand_total_sums": grand_sums,
        "per_region": per_region,
        "status_counts": dict(status_totals),
    }
    bucket.blob(f"{prefix}/summary.json").upload_from_string(
        json.dumps(summary, indent=2),
        content_type="application/json",
    )


def print_pretty_summary(spec: str, results: list[dict], elapsed_sec: float) -> None:
    keys = [
        "input_all",
        "thinking_all",
        "response_all",
        "input_kept",
        "thinking_kept",
        "response_kept",
        "flops_all",
        "flops_kept",
        "total_records",
        "kept_records",
    ]
    grand_sums = {k: sum(r[k] for r in results) for k in keys}
    status_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for s, c in r.get("status_counts", {}).items():
            status_totals[s] += c

    print()
    print("=" * 72)
    print(f"EXACT FLOP AGGREGATION · spec={spec} · {len(results):,} batches · {elapsed_sec:.0f}s")
    print("=" * 72)
    print(f"Total inference FLOPs:    {grand_sums['flops_all']:.4e}")
    print(f"  on kept records:        {grand_sums['flops_kept']:.4e}")
    print(f"  on filtered (wasted):   {grand_sums['flops_all'] - grand_sums['flops_kept']:.4e}")
    print()
    print(f"Input tokens (all):       {grand_sums['input_all']/1e12:>8.3f} T")
    print(f"Input tokens (kept):      {grand_sums['input_kept']/1e12:>8.3f} T")
    print(f"Thinking tokens (kept):   {grand_sums['thinking_kept']/1e9:>8.2f} B")
    print(f"Response tokens (kept):   {grand_sums['response_kept']/1e9:>8.2f} B")
    print()
    print(
        f"Records: {grand_sums['kept_records']:,} kept / "
        f"{grand_sums['total_records']:,} total "
        f"({100*grand_sums['kept_records']/max(grand_sums['total_records'],1):.2f}%)"
    )
    print()
    print("Status distribution:")
    tot = sum(status_totals.values())
    for s, c in sorted(status_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<36} {c:>14,} ({100*c/max(tot,1):>5.2f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="Extraction spec id (e.g. low_quality, med_quality, high_quality)")
    ap.add_argument("--output-dir", required=True, help="GCS dir to write summary.json + aggregate.jsonl.gz")
    ap.add_argument(
        "--max-workers", type=int, default=16, help="Concurrent GCS readers (default 16; in-region can take 32+)"
    )
    ap.add_argument("--limit", type=int, default=None, help="If set, only process this many sidecars (debug)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Enumerating sidecars for spec=%s in consolidated archive…", args.spec)
    per_region_uris = enumerate_sidecars(args.spec)
    all_uris: list[str] = []
    for region in REGIONS:
        all_uris.extend(per_region_uris.get(region, []))
    logger.info("Found %d sidecars across %d regions", len(all_uris), len(per_region_uris))
    if args.limit:
        all_uris = all_uris[: args.limit]
        logger.info("Limiting to %d (debug)", len(all_uris))
    if not all_uris:
        logger.error("No sidecars found — check spec name and consolidation status.")
        return

    t0 = time.monotonic()
    results: list[dict] = []
    errors: list[dict] = []
    progress_every = max(500, len(all_uris) // 30)
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = [pool.submit(aggregate_one, u) for u in all_uris]
        for i, fut in enumerate(as_completed(futs), start=1):
            r = fut.result()
            if "error" in r:
                errors.append(r)
            else:
                results.append(r)
            if i % progress_every == 0:
                el = time.monotonic() - t0
                rate = i / max(el, 1e-3)
                eta = (len(all_uris) - i) / max(rate, 1e-3)
                logger.info("  %d/%d (%.1f files/s, ETA %.0fs, %d errors)", i, len(all_uris), rate, eta, len(errors))

    elapsed = time.monotonic() - t0
    logger.info("Done in %.1fs (%d ok, %d errors)", elapsed, len(results), len(errors))
    if errors[:1]:
        logger.warning("Example error: %s", errors[0])

    print_pretty_summary(args.spec, results, elapsed)
    write_outputs(args.output_dir, results, errors, elapsed, args.spec)
    logger.info(
        "Wrote: %s/summary.json and %s/aggregate.jsonl.gz", args.output_dir.rstrip("/"), args.output_dir.rstrip("/")
    )


if __name__ == "__main__":
    main()
