# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical launcher for the 10k-WARC natural-epoching curation sweep.

This is the NATURAL-EPOCH replacement for the old ExpC (`expC_T33T`) 10k sweep,
which used simulated epoching (sliced to a uniform 43.14B-token cap) and is NOT
comparable to the natural-epoch WARC-scaling curves at N=100..3000. Here every
run trains with natural epoching (target = `t_exp * s`, like expFM_natural /
expWARC_natural), so the 10k point sits on the same curve as the rest.

WHY A DEDICATED LAUNCHER (read before changing anything):
The grid below is FROZEN ON PURPOSE. dclm_10k and nemotron_10k (and any future
10k method) must launch on the *identical* (width x budget x batch_divisor) grid
so the per-method comparison is fair. Do not parameterize the grid via CLI -- if
the science needs a different grid, edit these constants in one place and every
method re-launches identically. Widths/budgets are NOT exposed as flags for
exactly this reason.

GRID (per method; see scoping in the run registry notes):
  - Widths d in {512, 1024, 1536, 2432, 3584}  (157M, 447M, 998M, 2.90B, 8.11B)
  - Base budget ladder (all widths, full cartesian, batch_divisor=1):
        3e16, 1e17, 3e17, 9e17, 1.8e18, 3e18, 9e18, 1.8e19, 3e19, 9e19, 1.8e20, 3e20
    Cells that would need batch < 8 auto-drop, giving the natural sliding window
    (small models cover the low end, big models the high end).
  - High-end extension (ONLY the two biggest widths, 2432 & 3584):
        9e20   batch_divisor=1  -> v5p-128
        1.8e21 batch_divisor=2  -> v5p-64    (keeps peak slice at v5p-128)
        9e21   batch_divisor=4  -> v5p-128   DROPPED 2026-05-31 (burned big-slice
                                              capacity for ~0% progress; see EXTENSION).
  - Corner policy: drop cells with tokens/param < MIN_TOKENS_PER_PARAM (0.3) so
    we don't burn slices on near-untrained points (e.g. 8.11B @ 9e17).
  => 38 cells/method, peak slice v5p-128. Registered methods: dclm_10k,
     nemotron_10k, high_quality_10k, fineweb_cc_10k, fineweb_edu_10k, resiliparse_10k.

DATA / REGIONS: dclm_10k/nemotron_10k caches are mirrored to all 6 Marin regions,
so those children float freely. high_quality_10k (us-central1), fineweb_cc_10k
(us-central2), fineweb_edu_10k (us-central2) and resiliparse_10k (us-central2) are
single-region for now; their pin_region (curation_plan.METHODS) hard-pins children
to the cache region until mirrored.

USAGE (CPU coordinator on Iris; parent batch is safe -- CPU jobs are non-preemptible):
    iris --cluster marin job run --priority batch --no-wait \\
        --memory 2GB --cpu 4 --job-name 10k-natural-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <token> \\
        -- python experiments/scaling_law_sweeps/launch_10k_natural.py --child-priority batch

    # dry-run locally to inspect the grid:
    python experiments/scaling_law_sweeps/launch_10k_natural.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import time

from iris.client.client import IrisClient

from experiments.scaling_law_sweeps import curation_plan, fixed_model_plan
from experiments.scaling_law_sweeps.completed_adamh import SEQ_LEN, completed_adamh_heuristic
from experiments.scaling_law_sweeps.launch_curation_sweep import PRIORITY_BAND_MAP, submit_all

logger = logging.getLogger(__name__)

# --- FROZEN canonical grid (do not turn these into CLI flags) ----------------
# dclm_10k/nemotron_10k caches are mirrored to all 6 regions (float freely).
# high_quality_10k/fineweb_cc_10k/fineweb_edu_10k caches are single-region for now
# (pin_region in curation_plan.METHODS hard-pins their children to the cache region
# until they're mirrored). Same frozen grid for every method -> fair comparison.
METHOD_NAMES: tuple[str, ...] = (
    "dclm_10k",
    "nemotron_10k",
    "high_quality_10k",
    "fineweb_cc_10k",
    "fineweb_edu_10k",
    "resiliparse_10k",
    "fastpipe_v3_100",
    "fastpipe_v3_80",
    "fastpipe_v3_60",
    "fastpipe_v3_40",
    "fastpipe_v3_20",
)
WIDTHS: tuple[int, ...] = (512, 1024, 1536, 2432, 3584)
BASE_BUDGETS: tuple[float, ...] = (
    3e16,
    1e17,
    3e17,
    9e17,
    1.8e18,
    3e18,
    9e18,
    1.8e19,
    3e19,
    9e19,
    1.8e20,
    3e20,
)
BIG_WIDTHS: tuple[int, ...] = (2432, 3584)
# High-end extension for the big widths only: budget -> batch_divisor.
# Divisors chosen so every extension cell lands on <= v5p-128 (the frozen peak):
# 9e20 fits at div=1; 1.8e21 shrinks x2 (-> v5p-64); 9e21 shrinks x4 (-> v5p-128).
# 9e21 is the aspirational top anchor (8.11B ~21 tok/param) -- a very long run that
# may not finish; tracked anyway. Net peak slice across the whole sweep = v5p-128.
#
# 9e21 IS PROVISIONAL. If monitoring shows it's unreasonable (see the WATCH note on
# the registry rows dclm_10k__pool-10000w / nemotron_10k__pool-10000w for the drop
# criteria), there are two clean, independent actions:
#   1. Stop the in-flight 9e21 jobs WITHOUT disturbing the coordinator or any other
#      cell:  python experiments/scaling_law_sweeps/kill_10k_9e21.py --confirm
#   2. Prevent a future re-launch from resurrecting them: remove the (9e21, 4) entry
#      below (skip-if-done preserves every other cell).
#
# 2026-05-31: 9e21 DROPPED by user direction — the 8.1B/2.9B @ 9e21 cells were
# burning v5p-64/128 capacity (gang-preempt churn, ~0-10% after a day) better
# spent on the smaller cells. The 4 in-flight 9e21 jobs were killed via
# kill_10k_9e21.py. Re-enable by uncommenting the (9e21, 4) entry below.
EXTENSION: tuple[tuple[float, int], ...] = (
    (9e20, 1),
    (1.8e21, 2),
    # (9e21, 4),  # DROPPED 2026-05-31 (see note above); uncomment to restore.
)

# Corner policy (FROZEN): drop any (width, budget) cell where the model would see
# fewer than this many tokens per parameter. The fixed-model machinery floors
# batch at 8 and keeps every forced corner, which at the full ladder produces
# scientifically useless points (e.g. 8.11B @ 9e17 ~ 0.002 tok/param -- noise on a
# v5p-32). 0.3 trims exactly those while keeping the full under->over-trained span
# (8.11B keeps 4 points 2e20->1.8e21; small-model overtrained corners up to ~2750
# tok/param are kept). Applied identically to every method, so the grid is fair.
MIN_TOKENS_PER_PARAM: float = 0.3

# Distinct namespaces so 10k-natural outputs never collide with the N=3000
# fixed-model sweep (which shares the expFM_natural tag; method name disambiguates
# run_name, but separate prefixes keep analysis clean).
DEFAULT_WANDB_GROUP = "data-curation-10k-natural"
DEFAULT_TRACKER_PREFIX = "gs://marin-us-central1/metadata/region_locks/data_curation_10k_natural/"
DEFAULT_RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"


def _tokens_per_param(hidden_size: int, budget: float) -> float:
    """Natural tokens-per-param for a (width, budget) cell: tokens = budget/(3·fpt)."""
    h = completed_adamh_heuristic
    mc = h._build_model_config(hidden_size, seq_len=SEQ_LEN)
    params = mc.total_trainable_params(h.vocab_size)
    return (budget / (3 * mc.flops_per_token(h.vocab_size, SEQ_LEN))) / params


def enumerate_10k_natural_plans(method_names: tuple[str, ...] = METHOD_NAMES) -> list:
    """Build the frozen 10k-natural plan list: base ladder (all widths) + the
    big-width high-end extension (per-budget batch_divisor), with cells below
    MIN_TOKENS_PER_PARAM dropped.

    Reuses `fixed_model_plan.enumerate_fixed_model_plans` per cell so the HP
    recipe, natural-epoch target, and TPU-shape selection are identical to the
    rest of the natural-epoch sweeps. The tokens/param floor is evaluated on the
    real (unrounded) budget and is method-independent, so both methods get the
    identical grid.
    """
    methods = [curation_plan.METHODS[n] for n in method_names]
    for m in methods:
        if m.d_obs_tokens <= 0:
            raise ValueError(f"Method {m.name!r} has d_obs_tokens<=0 -- its 10k cache is not finalized.")

    # (width, budget, batch_divisor) cells: base ladder for all widths, plus the
    # high-end extension for the big widths only.
    cells = [(d, b, 1) for d in WIDTHS for b in BASE_BUDGETS]
    cells += [(d, b, div) for d in BIG_WIDTHS for (b, div) in EXTENSION]

    plans: list = []
    for d, budget, divisor in cells:
        if _tokens_per_param(d, budget) < MIN_TOKENS_PER_PARAM:
            continue
        plans += fixed_model_plan.enumerate_fixed_model_plans(
            methods, hidden_sizes=(d,), budgets=(budget,), batch_divisor=divisor
        )
    return plans


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_NAMES),
        choices=list(METHOD_NAMES),
        help="Subset of the frozen 10k methods to launch (default: both).",
    )
    p.add_argument("--child-priority", choices=["production", "interactive", "batch", "unspecified"], default="batch")
    p.add_argument(
        "--allowed-regions",
        nargs="+",
        default=None,
        help="HARD-restrict children to these regions. Required when launching a method whose cache "
        "is not in all 6 regions: high_quality_10k lacks us-central2, so pass "
        "`us-central1 us-east1 us-east5 us-west4 eu-west4`. dclm/nemotron/fineweb_* are in all 6 "
        "(default None = float freely). A method's pin_region, if set, still overrides this.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print the grid, do not submit.")
    p.add_argument("--no-skip-if-done", action="store_true", help="Re-submit completed runs.")
    p.add_argument("--max-count", type=int, default=None, help="Cap children submitted (smoke test).")
    p.add_argument("--wandb-project", default="marin")
    p.add_argument("--wandb-entity", default="marin-community")
    p.add_argument("--wandb-group", default=DEFAULT_WANDB_GROUP)
    p.add_argument("--tracker-prefix", default=DEFAULT_TRACKER_PREFIX)
    p.add_argument("--results-prefix", default=DEFAULT_RESULTS_PREFIX)
    p.add_argument("--wandb-mode", choices=["auto", "online", "offline", "offline_no_sync"], default="auto")
    p.add_argument("--no-keep-alive", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    plans = enumerate_10k_natural_plans(tuple(args.methods))
    if args.max_count is not None:
        plans = plans[: args.max_count]

    if args.dry_run:
        curation_plan.print_dry_run(plans)
        logger.info("10k-natural: %d total cells across methods=%s", len(plans), args.methods)
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set -- this coordinator must run inside an Iris job.")
    client = IrisClient.remote(controller_address, bundle_id=os.environ.get("IRIS_BUNDLE_ID"))

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Submitting %d 10k-natural children (methods=%s)...", len(plans), args.methods)
    submitted, skipped = submit_all(
        client,
        plans,
        tracker_prefix=args.tracker_prefix,
        skip_if_done=not args.no_skip_if_done,
        child_priority_band=PRIORITY_BAND_MAP[args.child_priority],
        wandb_api_key=wandb_api_key,
        hf_token=hf_token,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        allowed_regions=args.allowed_regions,  # None = float all regions; set to constrain (see --allowed-regions)
        run_suffix="",
        wandb_mode=args.wandb_mode,
        force_primary_tpu=None,
        results_prefix=args.results_prefix,
    )
    logger.info(
        "Result: %d submitted, %d skipped, %d failed",
        len(submitted),
        len(skipped),
        len(plans) - len(submitted) - len(skipped),
    )

    if args.no_keep_alive:
        return
    logger.info("Coordinator entering keep-alive...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
