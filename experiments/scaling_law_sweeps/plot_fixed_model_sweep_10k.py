# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10k-WARC variant of the fixed-model sweep plots.

Thin wrapper over `plot_fixed_model_sweep.main` with the 10k-natural defaults
baked in (see `launch_10k_natural.py`): the 10k results bucket, the six
10k-natural methods (which include fineweb_cc and fineweb_edu), the 10364-WARC
title text, and a separate output dir so 10k plots don't clobber the 3k ones.

The 10k-natural runs share the `expFM_natural` experiment tag with the 3k
fixed-model sweep -- the method names (`*_10k`) are what disambiguate them, and
`plot_fixed_model_sweep` renames each `_10k` method to its base name (so they
reuse the base colors/labels). Widths come from the data, so all five 10k widths
(512/1024/1536/2432/3584) are drawn.

Any extra CLI args are forwarded to the underlying parser and override these
defaults, e.g.:

    # Pull the 10k summaries locally first (avoids gcsfs/SSL flakiness), then plot:
    gcloud storage cp \\
        "gs://marin-us-central1/metadata/data_curation_10k_natural_results/*.json" \\
        scratch/fm10k_summaries/
    uv run python experiments/scaling_law_sweeps/plot_fixed_model_sweep_10k.py \\
        --results-prefix scratch/fm10k_summaries/

    # Single metric:
    uv run python experiments/scaling_law_sweeps/plot_fixed_model_sweep_10k.py \\
        --metrics paloma_macro_loss
"""

from __future__ import annotations

import sys

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX as TENK_RESULTS_PREFIX,
)
from experiments.scaling_law_sweeps.launch_10k_natural import (
    METHOD_NAMES as TENK_METHODS,
)
from experiments.scaling_law_sweeps.launch_10k_natural import WIDTHS as TENK_WIDTHS
from experiments.scaling_law_sweeps.plot_fixed_model_sweep import main

TENK_WARC_COUNT = 10364


def run(argv: list[str] | None = None) -> None:
    """Invoke the shared plotter with 10k defaults, letting the caller override any."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    # Only inject a default when the flag isn't already present, so the user can
    # override any of them (e.g. point --results-prefix at a local pulled copy).
    def _default(flag: str, *values: str) -> list[str]:
        return [] if flag in argv else [flag, *values]

    defaults = [
        *_default("--results-prefix", TENK_RESULTS_PREFIX),
        *_default("--methods", *TENK_METHODS),
        *_default("--warc-count", str(TENK_WARC_COUNT)),
        *_default("--output-dir", "scratch/plots/fixed_model_10k"),
        *_default("--side-by-side-hidden-sizes", *(str(w) for w in TENK_WIDTHS)),
        # No _bos_fixed runs in the 10k sweep, and the default --prefer-bos-fixed
        # would drop any (renamed) nemotron_full/llm_curated summaries. Disable it.
        *(["--no-prefer-bos-fixed"] if "--prefer-bos-fixed" not in argv else []),
    ]
    main(defaults + argv)


if __name__ == "__main__":
    run()
