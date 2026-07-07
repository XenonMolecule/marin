# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""10k-WARC variant of the loss-vs-tokens paper figure (native PNG + PDF).

Thin wrapper over `plot_loss_vs_tokens_by_size.main` with the 10k-natural
defaults baked in (see `launch_10k_natural.py`): the 10k results bucket, the six
10k-natural methods (which add fineweb_cc and fineweb_edu to the canonical four),
all five 10k widths as side-by-side panels, and a separate output dir so the 10k
figures don't clobber the 3k ones.

The 10k runs share the `expFM_natural` tag with the 3k sweep and carry LIMA +
uncheatable evals inline, so no LIMA sidecar pull is needed. Methods are loaded
by their `_10k` names (so the files are found) and renamed to base names before
plotting, reusing the base METHOD_DISPLAY entries.

Usage:

    # Pull the 10k summaries locally, then render PNG + PDF for all three metrics:
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size_10k.py --pull

    # Re-use already-pulled summaries (fast inner loop), single metric:
    uv run --with matplotlib --with numpy python \\
        experiments/scaling_law_sweeps/plot_loss_vs_tokens_by_size_10k.py \\
        --metrics uncheatable_macro_loss

Any extra CLI args override these defaults.
"""

from __future__ import annotations

import sys

from experiments.scaling_law_sweeps.launch_10k_natural import (
    DEFAULT_RESULTS_PREFIX as TENK_RESULTS_GS,
)
from experiments.scaling_law_sweeps.launch_10k_natural import (
    METHOD_NAMES as TENK_METHODS,
)
from experiments.scaling_law_sweeps.launch_10k_natural import WIDTHS as TENK_WIDTHS
from experiments.scaling_law_sweeps.plot_loss_vs_tokens_by_size import main

TENK_OURS_METHOD = "high_quality_10k"
TENK_LOCAL_PREFIX = "scratch/fm10k_summaries/"
TENK_OUTPUT_DIR = "scratch/plots/loss_vs_tokens_10k"


def run(argv: list[str] | None = None) -> None:
    """Invoke the shared renderer with 10k defaults, letting the caller override any."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)

    def _default(flag: str, *values: str) -> list[str]:
        return [] if flag in argv else [flag, *values]

    defaults = [
        *_default("--methods", *TENK_METHODS),
        *_default("--ours-method", TENK_OURS_METHOD),
        *_default("--hidden-sizes", *(str(w) for w in TENK_WIDTHS)),
        *_default("--results-prefix", TENK_LOCAL_PREFIX),
        *_default("--results-gs", TENK_RESULTS_GS),
        # 10k summaries carry eval/lima/* inline (no separate sidecar bucket), so
        # point the LIMA sidecar at the same 10k source/dir. This keeps --pull from
        # fetching the irrelevant 3k LIMA bucket cross-region; merge_lima_sidecars
        # then just re-reads the inline LIMA values from the summaries themselves.
        *_default("--lima-sidecar-prefix", TENK_LOCAL_PREFIX),
        *_default("--lima-sidecar-gs", TENK_RESULTS_GS),
        *_default("--output-dir", TENK_OUTPUT_DIR),
        # No _bos_fixed runs in the 10k sweep; the default --prefer-bos-fixed would
        # also expand load_methods with `*_bos_fixed` names that don't exist here.
        *(["--no-prefer-bos-fixed"] if "--prefer-bos-fixed" not in argv else []),
    ]
    main(defaults + argv)


if __name__ == "__main__":
    run()
