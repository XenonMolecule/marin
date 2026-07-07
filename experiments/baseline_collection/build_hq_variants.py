# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build document-level VARIANTS of the high_quality corpus for causal ablations.

Two variants of the trained hq corpus (the 19.97M-doc decon+dedup HF export in
us-central1), for the knowledge-gap dilution experiment:

  ``dense``  = docs whose URL-category is fact-bearing expository
              (reference_wiki, academic, blog, news, qa_help). ~12% of tokens.
              Used as the UPWEIGHT component in the reweight variant (mixed with
              the full hq cache at weight 0.144 → lifts dense char-share 12%→25%,
              matching DCLM, at fixed token budget).

  ``epoch``  = a uniform hash-subsample of ALL hq docs down to ~DCLM's token
              count (~34.4% → 7.33B tokens), composition unchanged. Isolates the
              "fewer unique tokens / more epochs" confound from composition.

Both are written as jsonl.gz (one shard per input parquet, skip-existing) to
us-central1, then tokenized by ``default_tokenize`` into Levanter caches and
registered in ``curation_plan``.

Runs in-region on us-central1 (where the hq HF export lives). Category is a pure
function of URL (no cross-region join). Launch:

    uv run iris --cluster=marin job run --region us-central1 --enable-extra-resources \\
        --cpu 32 --memory 96GB --extra cpu \\
        -- python experiments/baseline_collection/build_hq_variants.py --variant both --threads 64
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import sys

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.provenance_audit_10k import (
    _assert_no_cross_region,
    _gcs_ls_glob,
    _parallel_map,
    categorize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("build_hq_variants")

HQ_JOINED = "gs://marin-us-central1/documents/baseline_high_quality_hf_export/10364warcs/joined"
OUT_ROOT = "gs://marin-us-central1/documents/hq_variants"
DENSE_OUT = f"{OUT_ROOT}/dense"
EPOCH_OUT = f"{OUT_ROOT}/epoch_sub344"

# Fact-bearing expository registers to concentrate (the reweight target bucket).
DENSE_CATEGORIES = frozenset({"reference_wiki", "academic", "blog", "news", "qa_help"})
# Uniform subsample fraction for the epoch-control variant: 7.33B / 21.30B (=DCLM).
EPOCH_KEEP_FRACTION = 0.344
_EPOCH_SALT = "hq_epoch_v1"
_HASH_DENOM = float(1 << 64)


def _url_unit(url: str, salt: str) -> float:
    """Deterministic uniform [0,1) hash of a URL — stable subsampling across shards."""
    h = hashlib.md5(f"{salt}:{url}".encode()).digest()[:8]
    return int.from_bytes(h, "big") / _HASH_DENOM


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    _assert_no_cross_region(path)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in rows:
            gz.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n")
    with fsspec.open(path, "wb") as out:
        out.write(buf.getvalue())


def _process_shard(shard_path: str, variant: str, fs) -> tuple[int, int]:
    """Emit the dense and/or epoch subsets for one HF-export parquet shard.

    Returns (n_dense, n_epoch) rows written for this shard.
    """
    _assert_no_cross_region(shard_path)
    name = shard_path.rsplit("/", 1)[-1].removesuffix(".parquet")
    dense_path = f"{DENSE_OUT}/{name}.jsonl.gz"
    epoch_path = f"{EPOCH_OUT}/{name}.jsonl.gz"
    want_dense = variant in ("dense", "both") and not fs.exists(dense_path[len("gs://") :])
    want_epoch = variant in ("epoch", "both") and not fs.exists(epoch_path[len("gs://") :])
    if not want_dense and not want_epoch:
        return (-1, -1)

    with fs.open(shard_path, "rb") as fh:
        table = pq.read_table(fh, columns=["url", "text"])
    dense_rows: list[dict] = []
    epoch_rows: list[dict] = []
    for rec in table.to_pylist():
        url, text = rec.get("url"), rec.get("text")
        if not url or not text:
            continue
        if want_dense and categorize(url) in DENSE_CATEGORIES:
            dense_rows.append({"text": text, "url": url})
        if want_epoch and _url_unit(url, _EPOCH_SALT) < EPOCH_KEEP_FRACTION:
            epoch_rows.append({"text": text, "url": url})

    if want_dense:
        _write_jsonl_gz(dense_path, dense_rows)
    if want_epoch:
        _write_jsonl_gz(epoch_path, epoch_rows)
    return (len(dense_rows) if want_dense else -1, len(epoch_rows) if want_epoch else -1)


def run(args: argparse.Namespace) -> int:
    fs = fsspec.filesystem("gcs")
    shards = _gcs_ls_glob(f"{HQ_JOINED}/*.parquet")
    logger.info("[build] %d HF-export shards; variant=%s", len(shards), args.variant)
    results = _parallel_map(
        lambda p: _process_shard(p, args.variant, fs), shards, f"build {args.variant}", max_workers=args.threads
    )
    n_dense = sum(d for d, _ in results if d >= 0)
    n_epoch = sum(e for _, e in results if e >= 0)
    logger.info("[build] wrote dense=%d docs → %s", n_dense, DENSE_OUT)
    logger.info("[build] wrote epoch=%d docs → %s", n_epoch, EPOCH_OUT)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["dense", "epoch", "both"], default="both")
    p.add_argument("--threads", type=int, default=64)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
