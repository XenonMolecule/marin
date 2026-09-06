# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendor a solved OLMIX mixture from GCS into the repo, in the shape training sweeps read.

`run_olmix_fit` writes a rich solve artifact to GCS; `GridMixCurationMethod.load_weights`
(``scaling_law_sweeps/data_curation_math.py``) reads a *vendored* file from the repo. The two are
not the same document, and the difference is a silent failure waiting to happen:

* the solve calls the weight map ``mixture``; the loader reads ``weights``. Copying the GCS file
  straight in raises ``KeyError: 'weights'`` at sweep-planning time.
* the solve carries diagnostics -- ``interaction_matrix`` (n_tasks x n_domains), ``log_c``,
  ``domains``, ``dead_domains`` -- that must not enter git.
* the vendored file adds provenance the solve cannot know: ``source`` (the exact gs:// path it
  came from) and ``swarm_runs`` (how many proxy runs backed the fit).

Naming is ``{grid_corpus}_R3e10_k20{mixture_tag}.json`` because ``curation_plan._grid_mix_method``
builds that path from ``grid_corpus`` and ``mixture_tag``.

Usage:

    python -m experiments.data_mixing.vendor_olmix_mixture \\
        --src gs://marin-us-east5/metadata/olmix/dclm_10k/fit_olmix_exact_kl0p01/mix_R3e+10_k20.json \\
        --grid-corpus dclm_10k --mixture-tag _olmixexact_lambda0p01
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import fsspec

logger = logging.getLogger(__name__)

MIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "mixtures"

# Carried from the solve verbatim. Everything else in the solve is diagnostics: keeping
# `interaction_matrix` alone would add an n_tasks x n_domains float grid to git per mixture.
CARRIED_FIELDS = (
    "corpus",
    "kl_reg",
    "live_domains",
    "natural",
    "regression_fit",
    "repetition_factor",
    "requested_tokens",
)

# Levanter's largest legal mixture block; the loader drops cells below 1/block and renormalises.
# Checked here too so a mixture that would lose real mass is caught at vendor time, not mid-sweep.
MAX_MIXTURE_BLOCK_SIZE = 65535
DROPPED_MASS_WARN = 0.001  # 0.1%


def vendor_mixture(src: str, grid_corpus: str, mixture_tag: str) -> pathlib.Path:
    """Convert one solve artifact into its vendored form and write it. Returns the path."""
    with fsspec.open(src, "rt") as f:
        solve = json.load(f)

    if "mixture" not in solve:
        raise KeyError(f"{src} has no 'mixture'; is it a solve artifact? keys={sorted(solve)}")
    weights = solve["mixture"]

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{src}: mixture sums to {total!r}, expected 1.0")

    floor = 1.0 / MAX_MIXTURE_BLOCK_SIZE
    dropped = {k: v for k, v in weights.items() if v < floor}
    dropped_mass = sum(dropped.values())
    logger.info(
        "%s: %d/%d cells clear the %.2e floor; %d sub-floor cells hold %.4f%% of mass",
        grid_corpus,
        len(weights) - len(dropped),
        len(weights),
        floor,
        len(dropped),
        100.0 * dropped_mass,
    )
    if dropped_mass > DROPPED_MASS_WARN:
        logger.warning(
            "%s: sub-floor cells hold %.4f%% of mass, above the %.2f%% we have seen; the executed "
            "mixture will differ from this file by that much",
            grid_corpus,
            100.0 * dropped_mass,
            100.0 * DROPPED_MASS_WARN,
        )

    vendored = {k: solve[k] for k in CARRIED_FIELDS if k in solve}
    vendored["source"] = src
    vendored["swarm_runs"] = solve.get("runs")
    vendored["weights"] = weights

    out = MIXTURES_DIR / f"{grid_corpus}_R3e10_k20{mixture_tag}.json"
    out.write_text(json.dumps(vendored, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s (%d cells, %d swarm runs)", out, len(weights), vendored["swarm_runs"] or -1)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="gs:// path to the solve's mix_*.json.")
    p.add_argument("--grid-corpus", required=True, help="e.g. dclm_10k; must match curation_plan's grid_corpus.")
    p.add_argument(
        "--mixture-tag", default="", help="e.g. _olmixexact_lambda0p01; empty = the default lambda=0.05 solve."
    )
    args = p.parse_args()
    vendor_mixture(args.src, args.grid_corpus, args.mixture_tag)


if __name__ == "__main__":
    main()
