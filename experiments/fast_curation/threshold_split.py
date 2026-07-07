# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Split the deduped+deconned fastpipe corpus into ModernBERT-score quality bands (keep top X% of docs).

Reads the deconned tree ``baseline_{spec}_decon_deduped/{n}warcs/deduped/*.jsonl.gz`` (records carry
``{text, modernbert_prob}``), computes the doc-percentile ``modernbert_prob`` thresholds, and writes one
filtered jsonl.gz tree per band at ``baseline_{spec}_{X}_decon_deduped/{n}warcs/deduped/`` — WHOLE
documents only (a doc is entirely in or out of a band). The 100% band IS the deconned input (no split
needed; tokenize it directly with ``--spec {spec}_decon``). Each band tree is a drop-in for
``tokenize_deduped_extracted.py --spec {spec}_{X}_decon``.

Region-local (run as an in-region iris CPU job) -> zero egress. Two passes: (1) a fine histogram over
all shards -> the percentile thresholds, (2) stream each doc into every band it qualifies for. Bands
are nested (80% ⊇ 60% ⊇ 40% ⊇ 20%). Idempotent per output shard (skip-existing).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import logging
import multiprocessing as mp

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

N_BINS = 100_000  # prob resolution 1e-5 over [0, 1]
BAND_PCTS = (80, 60, 40, 20)  # each band keeps the top X% of docs by modernbert_prob


def _read_jsonl_gz(path: str):
    with fsspec.open(path, "rb") as f, gzip.open(f, "rt") as g:
        for line in g:
            if line.strip():
                yield json.loads(line)


def _hist_one(path: str) -> tuple[np.ndarray, int]:
    h = np.zeros(N_BINS, dtype=np.int64)
    n = 0
    for r in _read_jsonl_gz(path):
        p = r.get("modernbert_prob")
        if p is None:
            continue
        h[min(N_BINS - 1, int(float(p) * N_BINS))] += 1
        n += 1
    return h, n


def _compute_thresholds(shards: list[str], procs: int) -> tuple[dict[int, float], int]:
    with mp.get_context("spawn").Pool(procs) as pool:
        res = pool.map(_hist_one, shards)
    total_h = np.sum([h for h, _ in res], axis=0)
    total_n = int(sum(n for _, n in res))
    cum = np.cumsum(total_h)
    thresholds: dict[int, float] = {}
    for pct in BAND_PCTS:
        # keep top pct% -> drop the bottom (100-pct)% -> threshold is the (100-pct) percentile of prob
        target = (100 - pct) / 100.0 * total_n
        thresholds[pct] = int(np.searchsorted(cum, target)) / N_BINS
    return thresholds, total_n


def _filter_one(args: tuple[str, dict[int, str], dict[int, float]]) -> dict[int, int]:
    path, outs, thresholds = args
    if all(fsspec.filesystem("gcs").exists(op) for op in outs.values()):
        return {pct: -1 for pct in thresholds}  # -1 = skipped (already present)
    counts = {pct: 0 for pct in thresholds}
    with contextlib.ExitStack() as stack:
        writers = {
            pct: stack.enter_context(gzip.open(stack.enter_context(fsspec.open(outs[pct], "wb")), "wt"))
            for pct in thresholds
        }
        for r in _read_jsonl_gz(path):
            p = r.get("modernbert_prob")
            if p is None:
                continue
            line = json.dumps(r) + "\n"
            pf = float(p)
            for pct, t in thresholds.items():
                if pf >= t:
                    writers[pct].write(line)
                    counts[pct] += 1
    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--n", type=int, default=10364)
    ap.add_argument("--region", default="us-east5")
    ap.add_argument("--num-procs", type=int, default=16)
    args = ap.parse_args()

    base = f"gs://marin-{args.region}/documents"
    in_dir = f"{base}/baseline_{args.spec}_decon_deduped/{args.n}warcs/deduped"
    listing = fsspec.filesystem("gcs").ls(in_dir.removeprefix("gs://"))
    shards = sorted(f"gs://{p}" for p in listing if p.endswith(".jsonl.gz"))
    if not shards:
        raise RuntimeError(f"no deconned shards at {in_dir}")
    logger.info("threshold-split over %d deconned shards", len(shards))

    thresholds, total_n = _compute_thresholds(shards, args.num_procs)
    logger.info("total docs (100%% band) = %d", total_n)
    for pct in BAND_PCTS:
        logger.info("  band %d%%: keep modernbert_prob >= %.6f", pct, thresholds[pct])

    def _out(pct: int, shard_path: str) -> str:
        bn = shard_path.rsplit("/", 1)[1]
        return f"{base}/baseline_{args.spec}_{pct}_decon_deduped/{args.n}warcs/deduped/{bn}"

    jobs = [(s, {pct: _out(pct, s) for pct in BAND_PCTS}, thresholds) for s in shards]
    with mp.get_context("spawn").Pool(args.num_procs) as pool:
        results = pool.map(_filter_one, jobs)

    band_totals = {pct: sum(c[pct] for c in results if c[pct] >= 0) for pct in BAND_PCTS}
    logger.info("=== band doc counts (100%% = %d) ===", total_n)
    for pct in BAND_PCTS:
        logger.info("  band %d%%: threshold>=%.6f -> %d docs (%.1f%% of 100%%)",
                    pct, thresholds[pct], band_totals[pct], 100.0 * band_totals[pct] / max(total_n, 1))

    # Persist the split summary next to the bands so tokenize/registration can read it back.
    summary = {
        "spec": args.spec,
        "n_warcs": args.n,
        "region": args.region,
        "total_docs_100pct": total_n,
        "bands": {str(pct): {"threshold": thresholds[pct], "docs": band_totals[pct]} for pct in BAND_PCTS},
    }
    with fsspec.open(f"{base}/baseline_{args.spec}_decon_deduped/{args.n}warcs/threshold_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> threshold_split_summary.json")


if __name__ == "__main__":
    main()
