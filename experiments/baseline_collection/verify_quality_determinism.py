# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify cross-region determinism for the quality-filter + tokenize pipeline.

Compares the `_quality_filter_stats.json` (filter-level) and `train/.stats.json`
(tokenize-level) across all regions for each preset. If every region produces
byte-identical stats for a preset, determinism holds and the dataset is safe
to register as a single `baseline_nemotron_q{preset}-v1` method.

Usage:
    python experiments/baseline_collection/verify_quality_determinism.py \\
        --presets high medplus \\
        --regions us-central1 us-east1 us-east5 eu-west4 \\
        --output-hash v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys

import fsspec

logger = logging.getLogger(__name__)

REGION_TO_BUCKET = {
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "europe-west4": "gs://marin-eu-west4",
    "eu-west4": "gs://marin-eu-west4",
}


def _read_json(path: str) -> dict | None:
    try:
        with fsspec.open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def _json_hash(obj: dict | None) -> str:
    if obj is None:
        return "<missing>"
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def check_preset(preset: str, regions: list[str], output_hash: str) -> bool:
    """Return True iff all regions produced byte-identical stats for this preset."""
    filter_stats: dict[str, dict | None] = {}
    tokenize_stats: dict[str, dict | None] = {}

    for region in regions:
        bucket = REGION_TO_BUCKET[region]
        filter_path = f"{bucket}/filtered/baseline_nemotron_q{preset}-{output_hash}/_quality_filter_stats.json"
        tokenize_path = f"{bucket}/tokenized/baseline_nemotron_q{preset}-{output_hash}/train/.stats.json"
        filter_stats[region] = _read_json(filter_path)
        tokenize_stats[region] = _read_json(tokenize_path)

    ok = True

    # Filter-level comparison: strip region-dependent fields (input_dir,
    # output_dir) before comparing, since those differ by region even if the
    # contents match.
    filter_core = {
        region: {k: v for k, v in stats.items() if k not in ("input_dir", "output_dir")} if stats is not None else None
        for region, stats in filter_stats.items()
    }
    filter_hashes = {region: _json_hash(filter_core[region]) for region in regions}
    logger.info("=== preset=%s FILTER stats ===", preset)
    for region in regions:
        stats = filter_stats[region]
        rows_out = stats.get("total_output_rows") if stats else None
        logger.info(
            "  %-12s hash=%s rows_out=%s",
            region,
            filter_hashes[region],
            rows_out,
        )
    if len({h for h in filter_hashes.values() if h != "<missing>"}) > 1:
        logger.error("  MISMATCH: filter stats differ across regions for preset=%s", preset)
        ok = False
    elif "<missing>" in filter_hashes.values():
        logger.warning(
            "  INCOMPLETE: filter stats missing in %s",
            [r for r, h in filter_hashes.items() if h == "<missing>"],
        )
        ok = False
    else:
        logger.info("  OK: all %d regions match for preset=%s", len(regions), preset)

    # Tokenize-level: Levanter's .stats.json should match exactly. Strip any
    # region-dependent path fields if present.
    tok_hashes = {region: _json_hash(tokenize_stats[region]) for region in regions}
    logger.info("=== preset=%s TOKENIZE stats ===", preset)
    for region in regions:
        stats = tokenize_stats[region]
        total_tokens = stats.get("total_tokens") if stats else None
        total_elements = stats.get("total_elements") if stats else None
        logger.info(
            "  %-12s hash=%s total_tokens=%s total_elements=%s",
            region,
            tok_hashes[region],
            total_tokens,
            total_elements,
        )
    if len({h for h in tok_hashes.values() if h != "<missing>"}) > 1:
        logger.error("  MISMATCH: tokenize stats differ across regions for preset=%s", preset)
        ok = False
    elif "<missing>" in tok_hashes.values():
        logger.warning(
            "  INCOMPLETE: tokenize stats missing in %s",
            [r for r, h in tok_hashes.items() if h == "<missing>"],
        )
        ok = False
    else:
        logger.info("  OK: all %d regions match for preset=%s", len(regions), preset)

    return ok


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--presets", nargs="+", default=["high", "medplus"])
    p.add_argument("--regions", nargs="+", default=["us-central1", "us-east1", "us-east5", "europe-west4"])
    p.add_argument("--output-hash", required=True)
    args = p.parse_args(argv)

    all_ok = True
    for preset in args.presets:
        if not check_preset(preset, args.regions, args.output_hash):
            all_ok = False

    if all_ok:
        logger.info("ALL PRESETS MATCH: safe to register in curation_plan.METHODS")
        sys.exit(0)
    else:
        logger.error("Determinism failed -- do NOT register until resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
