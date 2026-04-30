# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Region-local runner for the quality-stratified Nemotron filter + tokenize.

Purpose
-------
After mirroring the upstream filtered JSONL to a region's bucket (see
`mirror_filtered_nemotron.sh`), this script does the CPU-bound filter +
tokenize work locally in that region. No cross-region reads.

Usage
-----
Run once per region inside an Iris CPU job (or locally for testing):

    python experiments/baseline_collection/run_quality_filter_region.py \\
        --preset high \\
        --region us-central1 \\
        --output-hash c0ffee01

    python experiments/baseline_collection/run_quality_filter_region.py \\
        --preset medplus \\
        --region us-central1 \\
        --output-hash babe1234

The ``--output-hash`` is chosen by the caller and must be identical across
regions for the determinism check to be meaningful. Use e.g. a short sha of
the (preset, input_hash, tokenizer) triple. The resulting paths are:

    Filtered JSONL: gs://marin-{region}/filtered/baseline_nemotron_q{preset}-{output-hash}/
    Tokenized cache: gs://marin-{region}/tokenized/baseline_nemotron_q{preset}-{output-hash}/

Determinism
-----------
- The filter script (filter_nemotron_quality.py) is already deterministic:
  sorted shard order, preserved row order, mtime=0 gzip header.
- Levanter's tokenize is deterministic given the input JSONL byte-order and
  tokenizer config.
- Run this in two regions, then compare shard sha256 sums. If any differ,
  investigate before registering the datasets in curation_plan.METHODS.
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.baseline_collection.filter_nemotron_quality import (
    QUALITY_HIGH,
    QUALITY_MEDPLUS,
    FilterNemotronQualityConfig,
    filter_nemotron_quality,
)

logger = logging.getLogger(__name__)

PRESETS = {
    "high": QUALITY_HIGH,
    "medplus": QUALITY_MEDPLUS,
}

# Source dataset produced by the existing `filter_nemotron_full` step.
# Hardcoded here so the caller can't accidentally point at a different
# upstream. The hash (-347dfe) is pinned; if upstream changes, bump this.
UPSTREAM_DATASET = "filtered/baseline_nemotron_full-347dfe"

# Marin region -> GCS bucket map. Mirrors `region_tracker.REGION_TO_BUCKET`.
# Note: iris region label for europe-west4 is the full "europe-west4" string,
# NOT the bucket short-name "eu-west4". Keep the bucket alias as a secondary
# key so runners invoked with either name still resolve.
REGION_TO_BUCKET = {
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "us-west4": "gs://marin-us-west4",
    "europe-west4": "gs://marin-eu-west4",
    "eu-west4": "gs://marin-eu-west4",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--preset", required=True, choices=sorted(PRESETS.keys()))
    p.add_argument(
        "--region",
        required=True,
        choices=sorted(REGION_TO_BUCKET.keys()),
        help="Region to read inputs from and write outputs to. No cross-region I/O.",
    )
    p.add_argument(
        "--output-hash",
        required=True,
        help="Short hex string identifying the output dataset version. Keep identical across regions.",
    )
    p.add_argument(
        "--skip-tokenize",
        action="store_true",
        help="Run the JSONL filter only, not the downstream tokenize. Useful for iterating on determinism.",
    )
    p.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip the JSONL filter step. Assumes the filtered output already exists at the expected path.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    bucket = REGION_TO_BUCKET[args.region]
    input_dir = f"{bucket}/{UPSTREAM_DATASET}"
    filtered_out = f"{bucket}/filtered/baseline_nemotron_q{args.preset}-{args.output_hash}"

    cfg = FilterNemotronQualityConfig(
        input_dir=input_dir,
        output_path=filtered_out,
        allowed_quality=PRESETS[args.preset],
    )
    if args.skip_filter:
        logger.info("Skipping filter step (--skip-filter). Expecting output at %s", filtered_out)
    else:
        logger.info("Running quality-filter: %s", cfg)
        filter_nemotron_quality(cfg)

    if args.skip_tokenize:
        logger.info("Skipping tokenize step (--skip-tokenize).")
        return

    # Tokenize via Marin's default_tokenize helper. Resolve locally to keep
    # the executor hash stable, but write the cache to the region bucket so
    # training jobs in this region can read it without egress.
    from experiments.defaults import default_tokenize  # local import: heavy deps
    from experiments.llama import llama3_tokenizer
    from marin.execution.executor import executor_main

    tokenize_out_name = f"baseline_nemotron_q{args.preset}-{args.output_hash}"
    tok_step = default_tokenize(
        name=tokenize_out_name,
        dataset=f"{filtered_out}/*.jsonl.gz",
        tokenizer=llama3_tokenizer,
    )
    # Pin the final cache path to the same region's bucket so there are no
    # cross-region reads when training runs fetch tokens.
    final_tokenize_path = f"{bucket}/tokenized/{tokenize_out_name}"
    tok_step = tok_step.with_output_path(final_tokenize_path)

    logger.info("Tokenize step final path=%s", final_tokenize_path)
    # executor_main is @draccus.wrap()'d and re-parses sys.argv; our argparse
    # flags (--preset/--region/--output-hash) are unknown to it and would
    # crash. Replace argv with just the script name for the executor call.
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        executor_main(steps=[tok_step], description=f"Quality-filter + tokenize ({args.preset}) in {args.region}")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Run failed: %s", e)
        sys.exit(1)
