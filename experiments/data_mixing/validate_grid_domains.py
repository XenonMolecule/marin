# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readiness gate: can a mixture actually open every cell, in the training region?

This deliberately does NOT duplicate
``experiments/baseline_collection/verify_grid_store.py``, which is the authority on
store-vs-inputs consistency (cell membership recomputed from the attribute tables, per-cell
row counts, doc and token totals against the tokenized parquets, and a ``TreeCache.load`` of
every cell). Run that first; it answers "did the shuffle produce the right store?".

This answers a different question that verifier cannot: **"will a mixture in region R open
region-R bytes?"** It loads `bucket.path` verbatim from the artifact, and a store copied
between regions keeps the ORIGINAL paths — the us-east5 dclm artifact records every cell as
``gs://marin-us-central1/...`` even though the bytes are in us-east5. So running that
verifier against the east5 store silently validates the central1 copy, and a training job
that trusted those paths would read every token cross-region while believing it was local.

So we check the paths `olmix_domains` actually hands to Levanter, after rebasing:
region-locality, loadability, and doc counts matching the artifact.

Usage:
    python -m experiments.data_mixing.validate_grid_domains \\
        --corpus dclm_10k --region us-east5 [--output gs://.../cells.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
from levanter.store.cache import TreeCache

from experiments.data_mixing.olmix_domains import GridDomain, load_grid_domains
from experiments.data_mixing.olmix_plan import MIXTURE_BLOCK_SIZE
from experiments.scaling_law_sweeps.region_tracker import REGION_TO_BUCKET

logger = logging.getLogger(__name__)

EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32)}
LOAD_PARALLELISM = 16


def _check_cell(domain: GridDomain, split: str = "train") -> tuple[str, int, str | None]:
    """Load one cell the way Levanter will, and confirm its doc count.

    Levanter resolves a component as ``<cache_dir>/<split>``, so we load exactly that
    rather than the artifact's recorded path -- otherwise we would not be testing the
    thing the trainer does.
    """
    path = f"{domain.cache_dir.rstrip('/')}/{split}"
    try:
        cache = TreeCache.load(path, EXEMPLAR)
    except Exception as exc:
        return domain.name, 0, f"load failed at {path}: {type(exc).__name__}: {exc}"
    rows = len(cache)
    if rows != domain.docs:
        return domain.name, rows, f"{path}: cache has {rows} rows, artifact claims {domain.docs}"
    return domain.name, rows, None


def validate(corpus: str, region: str, block_size: int = MIXTURE_BLOCK_SIZE) -> dict:
    domains = load_grid_domains(corpus, region)  # rebases onto `region`, asserts the split level
    expected_prefix = REGION_TO_BUCKET[region] + "/"

    offenders = [d.name for d in domains if not d.cache_dir.startswith(expected_prefix)]
    if offenders:
        raise RuntimeError(f"{len(offenders)} cells are not region-local to {region}: {offenders[:5]}")

    with ThreadPoolExecutor(max_workers=LOAD_PARALLELISM) as pool:
        results = list(pool.map(_check_cell, domains))

    failures = [msg for _, _, msg in results if msg]
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(domains)} cells failed to load:\n  " + "\n  ".join(failures[:10]))

    tokens = np.array([d.tokens for d in domains], dtype=float)
    floor = 1.0 / block_size
    prior = tokens / tokens.sum()
    summary = {
        "corpus": corpus,
        "region": region,
        "cells": len(domains),
        "total_tokens": int(tokens.sum()),
        "total_docs": sum(d.docs for d in domains),
        "block_size": block_size,
        "block_floor": floor,
        "cells_below_floor_at_natural_prior": int((prior < floor).sum()),
        "smallest_cell_tokens": int(tokens.min()),
        "largest_cell_tokens": int(tokens.max()),
        "prior_dynamic_range": float(prior.max() / prior.min()),
        "domains": [
            {
                "name": d.name,
                "cluster": d.cluster_id,
                "quality": d.quality_bucket,
                "tokens": d.tokens,
                "docs": d.docs,
                "cache_dir": d.cache_dir,
            }
            for d in domains
        ],
    }
    logger.info(
        "%s @ %s: %d/%d cells loaded, %d tokens, %d docs; %d cells below the block floor at natural prior",
        corpus,
        region,
        len(domains),
        len(domains),
        summary["total_tokens"],
        summary["total_docs"],
        summary["cells_below_floor_at_natural_prior"],
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--region", required=True, choices=sorted(REGION_TO_BUCKET))
    parser.add_argument("--block-size", type=int, default=MIXTURE_BLOCK_SIZE)
    parser.add_argument("--output", default=None, help="Optional gs:// or local path for the summary JSON.")
    args = parser.parse_args()

    summary = validate(args.corpus, args.region, block_size=args.block_size)
    if args.output:
        with fsspec.open(args.output, "w") as fh:
            fh.write(json.dumps(summary, indent=2))
        logger.info("wrote %s", args.output)
    else:
        print(json.dumps({k: v for k, v in summary.items() if k != "domains"}, indent=2))


if __name__ == "__main__":
    main()
