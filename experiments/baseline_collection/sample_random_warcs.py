# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Draw a reproducible uniform random sample of WARC paths from a pool manifest.

The DCLM 400m-1x pool manifest (``experiments/distill/dclm_400m_1x.txt``) is
SORTED by crawl date, so taking the first N lines is a date-biased head, not a
random sample. This script draws ``N`` WARCs WITHOUT replacement uniformly
across the WHOLE pool using a fixed seed, so the result is reproducible and
byte-identical on every run.

Manifest filtering mirrors ``download_warcs._load_manifest``: blank lines and
lines starting with ``#`` are ignored, and each kept line is ``str.strip``-ed.

Usage::

    python experiments/baseline_collection/sample_random_warcs.py \\
        --pool experiments/distill/dclm_400m_1x.txt \\
        --n 3000 --seed 0 \\
        --output experiments/distill/subsets/baseline_warcs_3000_random.txt
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

# Default seed baked into the output filename's provenance. Changing this
# changes the sample, so it must stay fixed for the 3000-random manifest.
DEFAULT_SEED = 0
DEFAULT_N = 3000
WARC_PREFIX = "s3://commoncrawl/crawl-data/"
WARC_SUFFIX = ".warc.gz"


@dataclass(frozen=True)
class SampleConfig:
    pool_path: Path
    output_path: Path
    n: int
    seed: int


def load_pool(pool_path: Path) -> list[str]:
    """Load WARC paths, applying the same filtering as ``_load_manifest``.

    Blank lines and ``#`` comments are dropped; remaining lines are stripped.
    """
    with open(pool_path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def sample_warcs(pool: list[str], n: int, seed: int) -> list[str]:
    """Uniform sample of ``n`` WARCs without replacement across the whole pool.

    Output is sorted for stable diffs. ``random.Random(seed).sample`` over a
    deduplicated, deterministically-ordered population guarantees byte-identical
    results across runs.
    """
    assert len(pool) == len(set(pool)), "pool contains duplicate WARC paths"
    assert n <= len(pool), f"requested n={n} exceeds pool size {len(pool)}"
    # Sort the population first so sampling is independent of input file order.
    population = sorted(pool)
    picks = random.Random(seed).sample(population, n)
    return sorted(picks)


def write_manifest(config: SampleConfig, warcs: list[str]) -> None:
    """Write the sampled manifest with a provenance header comment."""
    header = (
        f"# Random sample of {config.n} WARCs from {config.pool_path.name}\n"
        f"# method: random.Random(seed).sample over the sorted, comment/blank-stripped pool\n"
        f"# seed: {config.seed}\n"
        f"# pool size: drawn without replacement across the WHOLE sorted pool (NOT a head)\n"
    )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_path, "w") as f:
        f.write(header)
        for w in warcs:
            f.write(w + "\n")


def parse_args() -> SampleConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Pool manifest file")
    parser.add_argument("--output", type=Path, required=True, help="Output manifest file")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Number of WARCs to sample")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()
    return SampleConfig(pool_path=args.pool, output_path=args.output, n=args.n, seed=args.seed)


def main() -> None:
    config = parse_args()
    pool = load_pool(config.pool_path)
    assert pool, f"pool {config.pool_path} is empty after filtering"
    warcs = sample_warcs(pool, config.n, config.seed)
    write_manifest(config, warcs)
    print(f"Sampled {len(warcs)} WARCs from {len(pool)} (seed={config.seed}) -> {config.output_path}")


if __name__ == "__main__":
    main()
