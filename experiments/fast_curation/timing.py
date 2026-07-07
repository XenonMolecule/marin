# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end timing + throughput report for a fast-curation run.

Three layers:

1. **Phase sentinels** (authoritative wall-clock): ``_phase1_start.json`` / ``_phase1_end.json``.
2. **Registry timestamps** (honest throughput): per-WARC completion mtimes in the central
   ``marin-us-central1`` registry give TPU WARCs/hr + ETA to the manifest size.
3. **Per-WARC timing JSONs** (compute breakdown, ``--deep``): aggregate ``timing_cpu/`` and
   ``timing_tpu/`` into the doc funnel (in → fastText-pass → survivors → kept) and CPU/TPU
   seconds.

    python -m experiments.fast_curation.timing --spec fastpipe_v1 [--deep]
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import fsspec

from experiments.fast_curation.spec import PipelineSpec, get_spec

logger = logging.getLogger(__name__)


def _read_json(path: str) -> dict | None:
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _registry_mtimes(spec: PipelineSpec) -> list[float]:
    """Epoch mtimes of every completed-WARC marker in the central registry."""
    from google.cloud import storage as gcs_storage

    prefix = f"{spec.subdir()}/_completed/"
    client = gcs_storage.Client()
    out: list[float] = []
    for blob in client.bucket("marin-us-central1").list_blobs(prefix=prefix):
        leaf = blob.name.rsplit("/", 1)[-1]
        if leaf.startswith("data-") and blob.updated is not None:
            out.append(blob.updated.timestamp())
    return out


def _count(prefix: str) -> int:
    """Count parquet objects under a gs:// prefix (one LIST)."""
    fs = fsspec.filesystem("gcs")
    try:
        return sum(1 for p in fs.ls(prefix.replace("gs://", "")) if p.endswith(".parquet"))
    except Exception:
        return 0


def _fmt_epoch(e: float) -> str:
    return datetime.fromtimestamp(e, tz=timezone.utc).isoformat(timespec="seconds")


def _deep_aggregate(spec: PipelineSpec, bucket: str) -> dict:
    """Sum the per-WARC timing JSONs (reads MANY small files — run in-region)."""
    fs = fsspec.filesystem("gcs")
    agg = {
        "cpu_warcs": 0,
        "n_in": 0,
        "n_ft_pass": 0,
        "n_justext_empty": 0,
        "n_survivors_cpu": 0,
        "decode_s": 0.0,
        "fasttext_s": 0.0,
        "justext_s": 0.0,
        "tokenize_cpu_s": 0.0,
        "tpu_warcs": 0,
        "n_survivors_tpu": 0,
        "n_kept": 0,
        "score_s": 0.0,
    }
    for sub, keys in (
        ("timing_cpu", ("n_in", "n_ft_pass", "n_justext_empty", "decode_s", "fasttext_s", "justext_s")),
        ("timing_tpu", ("n_kept", "score_s")),
    ):
        prefix = f"{spec.namespace(bucket)}/{sub}/".replace("gs://", "")
        try:
            paths = [p for p in fs.ls(prefix) if p.endswith(".json")]
        except Exception:
            paths = []
        for p in paths:
            d = _read_json("gs://" + p)
            if not d:
                continue
            if sub == "timing_cpu":
                agg["cpu_warcs"] += 1
                agg["n_survivors_cpu"] += d.get("n_survivors", 0)
                agg["tokenize_cpu_s"] += d.get("tokenize_s", 0.0)
            else:
                agg["tpu_warcs"] += 1
                agg["n_survivors_tpu"] += d.get("n_survivors", 0)
            for k in keys:
                agg[k] += d.get(k, 0)
    return agg


def report(spec: PipelineSpec, bucket: str, manifest_n: int, deep: bool) -> None:
    ns = spec.namespace(bucket)
    print(f"\n=== fast-curation timing: {spec.spec_id} ({spec.version()}) ===")
    print(f"namespace: {ns}")

    p1s = _read_json(f"{ns}/_phase1_start.json")
    p1e = _read_json(f"{ns}/_phase1_end.json")
    n_survivors_files = _count(spec.survivors_prefix(bucket))
    n_kept_files = _count(spec.kept_prefix(bucket))
    reg = sorted(_registry_mtimes(spec))

    print("\n-- Phase 1 (CPU) --")
    if p1s:
        print(f"  start: {_fmt_epoch(p1s['epoch'])}  ({p1s.get('n_warcs', '?')} WARCs)")
    if p1s and p1e:
        wall = p1e["epoch"] - p1s["epoch"]
        print(f"  end:   {_fmt_epoch(p1e['epoch'])}")
        print(f"  wall:  {wall / 3600:.2f} h  ({p1e.get('n_warcs', 0) / (wall / 3600):.0f} WARCs/h)" if wall > 0 else "")
    elif p1s:
        print("  (still running — no _phase1_end yet)")
    print(f"  survivor files written: {n_survivors_files}")

    print("\n-- Phase 2 (TPU) --")
    print(f"  WARCs in registry (done): {len(reg)} / {manifest_n}")
    print(f"  kept files written: {n_kept_files}")
    if len(reg) >= 2:
        span = reg[-1] - reg[0]
        rate = len(reg) / (span / 3600) if span > 0 else float("inf")
        print(f"  first done: {_fmt_epoch(reg[0])}")
        print(f"  last  done: {_fmt_epoch(reg[-1])}")
        print(f"  TPU span: {span / 3600:.2f} h  ({rate:.0f} WARCs/h)")
        remaining = manifest_n - len(reg)
        if remaining > 0 and rate > 0:
            print(f"  ETA for remaining {remaining}: {remaining / rate:.2f} h")

    if p1s and reg:
        e2e = reg[-1] - p1s["epoch"]
        print(f"\n-- End-to-end (overlapped) --\n  phase1_start -> last TPU done: {e2e / 3600:.2f} h")

    if deep:
        print("\n-- Doc funnel (deep; reads per-WARC timing JSONs) --")
        a = _deep_aggregate(spec, bucket)
        print(f"  CPU WARCs timed: {a['cpu_warcs']}")
        print(f"  docs in:            {a['n_in']:,}")
        print(f"  fastText-pass:      {a['n_ft_pass']:,}")
        print(f"  JustText-empty drop:{a['n_justext_empty']:,}")
        print(f"  survivors (CPU):    {a['n_survivors_cpu']:,}")
        print(f"  kept (TPU):         {a['n_kept']:,}")
        if a["n_in"]:
            print(f"  end-to-end keep rate: {100.0 * a['n_kept'] / a['n_in']:.2f}% of input docs")
        print(
            f"  CPU compute-s: decode={a['decode_s']:.0f} fasttext={a['fasttext_s']:.0f} "
            f"justext={a['justext_s']:.0f} tokenize={a['tokenize_cpu_s']:.0f}"
        )
        print(f"  TPU compute-s: score={a['score_s']:.0f}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--deep", action="store_true", help="Aggregate per-WARC timing JSONs (many reads).")
    args = ap.parse_args()

    spec = get_spec(args.spec)
    with fsspec.open(args.manifest, "r") as f:
        manifest_n = sum(1 for line in f if line.strip() and not line.startswith("#"))
    report(spec, args.bucket, manifest_n, args.deep)


if __name__ == "__main__":
    main()
