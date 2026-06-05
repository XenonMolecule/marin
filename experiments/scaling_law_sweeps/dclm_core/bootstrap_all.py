# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Driver: bootstrap a list of CORE runs in sequence.

Reads run_names from CLI args or a manifest file, derives partial-dir and
output-json paths from --partial-prefix / --output-prefix, and calls
`bootstrap_core.run_one` for each. Skips runs whose output JSON already
exists unless --force.

Intended to be the entrypoint for a single Iris job that processes the
whole 12-cell paper table in one go, so the worker only needs to spin
up once. Run from a us-central1 worker so the per-example sample reads
stay free.

Usage:
    python -m experiments.scaling_law_sweeps.dclm_core.bootstrap_all \\
        --partial-prefix gs://marin-us-central1/metadata/data_curation_core_results/partial \\
        --output-prefix  gs://marin-us-central1/metadata/data_curation_core_bootstrap \\
        --run-names \\
            curation-dclm-expFM_natural-9e+18-d512-L6-B64 \\
            curation-dclm-expFM_natural-9e+19-d1536-L16-B64 \\
            ...
"""

from __future__ import annotations

import argparse
import logging

from experiments.scaling_law_sweeps.dclm_core.bootstrap_core import run_one

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--partial-prefix", required=True, help="Prefix under which each <run_name>/ partial dir lives.")
    p.add_argument("--output-prefix", required=True, help="Prefix under which to write each <run_name>_bootstrap.json.")
    p.add_argument("--run-names", nargs="+", required=True, help="Whitespace-separated run names.")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    n_ok = 0
    n_skipped = 0
    n_failed = 0
    for rn in args.run_names:
        run_input = f"{args.partial_prefix.rstrip('/')}/{rn}/"
        output_json = f"{args.output_prefix.rstrip('/')}/{rn}_bootstrap.json"
        try:
            result = run_one(
                run_input=run_input,
                output_json=output_json,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                run_name=rn,
                force=args.force,
            )
            if result is None:
                n_skipped += 1
            else:
                n_ok += 1
        except Exception:
            logger.exception("[%s] bootstrap failed", rn)
            n_failed += 1

    logger.info("Done. ok=%d skipped=%d failed=%d", n_ok, n_skipped, n_failed)
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
