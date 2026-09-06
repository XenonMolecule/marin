# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Steps 2-3 of OlmixBase end to end: collect the swarm, fit per-task laws, solve the mixture.

Reads every completed, fully-evaluated swarm run across all regions, fits one log-linear
law per target task, and solves the constrained convex program for the proposed mixture.
Writes the proposed mixture, the interaction matrix, and a Figure-7-style (R, k) sweep.

Run as an Iris CPU job -- the fit needs torch and the solve needs cvxpy:

    iris --cluster marin job run --region us-central1 --extra cpu --extra mixing --cpu 8 \\
        -e WANDB_API_KEY "$WANDB_API_KEY" \\
        -- python -m experiments.data_mixing.run_olmix_fit \\
               --corpus dclm_10k --region us-east5 --region us-central1 \\
               --manifest gs://marin-us-east5/metadata/olmix/dclm_10k/swarm_s42_K363.json \\
               --output-prefix gs://marin-us-east5/metadata/olmix/dclm_10k

**Identifiability is enforced, not assumed.** The fit runs over the *live* support -- domains
that appear in at least one collected mixture -- because a never-sampled domain has an
all-zero design column and its coefficient is data-free (see `olmix_fit`). So the condition
for a unique solution is `rows >= m_live + 1`, NOT `rows >= m + 1` over the full grid. That
distinction is worth stating plainly because it cuts both ways: dropping dead domains lowers
the bar (at m_live~75 a unique fit exists from ~76 runs, not 119), but m_live GROWS as more
runs land, so the bar moves up over time and a fit that was identifiable at one collection
can stop being so at the next. The guard therefore recomputes it on every collection.

There is deliberately NO override flag. `olmix_law.fit_log_linear` enforces the same condition
independently, so an override here could not produce an underdetermined fit anyway -- it would
only fail later with a less informative message. This layer exists to fail early, naming the
live-domain count and the shortfall, not to offer an escape hatch that cannot work.
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec
import numpy as np

from experiments.data_mixing.collect_olmix_swarm import (
    load_swarm_rows,
    sources_from_regions,
    write_metrics_csv,
    write_ratios_csv,
)
from experiments.data_mixing.collect_swarm_blend import BLEND_TASKS
from experiments.data_mixing.collect_swarm_core_v2 import core_v2_tasks
from experiments.data_mixing.olmix_fit import (
    fit_all_tasks,
    regression_fit_quality,
    solve_with_natural_reinsertion,
    split_sampled_domains,
    sweep_constraints,
)
from experiments.data_mixing.olmix_plan import read_manifest
from experiments.data_mixing.olmix_solve import DEFAULT_KL_REG, DEFAULT_REPETITION_FACTOR
from experiments.data_mixing.olmix_tasks import build_olmix_exact_tasks, build_target_tasks
from experiments.data_mixing.run_olmix_swarm_standalone import PROXY_TRAIN_STEPS

logger = logging.getLogger(__name__)

# Figure-7 style post-hoc sweep. `k` and `R` enter only at solve time, so one swarm supports
# the whole grid at no training cost.
SWEEP_REQUESTED_TOKENS = (1.0e9, 3.67e9, 7.34e9, 3.0e10)
SWEEP_REPETITION = (2.0, 3.0, 4.0, 5.0)

# Weights below this are reported as effectively dropped by the proposal.
MIXTURE_DENSITY_THRESHOLD = 1e-4


def swarm_design(rows, manifest, task_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The (K, m) mixture design matrix and (K, n_tasks) BPB matrix, rows aligned."""
    weights = np.array([[row.weights[d] for d in manifest.domains] for row in rows], dtype=float)
    bpb = np.array([[row.bpb[t] for t in task_names] for row in rows], dtype=float)
    return weights, bpb


def fit_and_solve(
    manifest,
    rows,
    task_names: list[str],
    requested_tokens: float | None,
    repetition_factor: float,
    kl_reg: float,
    min_appearances: int = 1,
    num_workers: int = 1,
) -> dict:
    """Fit every task's law over the live support and solve for the proposed mixture."""
    domains = list(manifest.domains)
    weights, bpb = swarm_design(rows, manifest, task_names)
    live, dead = split_sampled_domains(domains, weights, min_appearances)

    required = len(live) + 1
    logger.info(
        "%d runs, %d domains (%d live / %d never sampled); a unique fit needs >= %d",
        len(rows),
        len(domains),
        len(live),
        len(dead),
        required,
    )
    if len(rows) < required:
        raise RuntimeError(
            f"underdetermined: {len(rows)} runs for {len(live)} live domains needs >= {required}. "
            f"Any mixture solved here would be meaningless -- with fewer runs than parameters the "
            f"loss is flat in some coefficients and the solver acts on their initialisation. "
            f"Wait for more completions; there is no override."
        )

    tokens = np.array([manifest.tokens[d] for d in domains], dtype=float)
    natural = tokens / tokens.sum()

    params = fit_all_tasks(weights[:, live], bpb, task_names, num_workers=num_workers)
    quality = regression_fit_quality(params, weights[:, live], bpb)
    logger.info("regression fit (in-sample): %s", quality)

    solution = solve_with_natural_reinsertion(
        params=params,
        domains=domains,
        live=live,
        tokens=tokens,
        natural=natural,
        requested_tokens=requested_tokens,
        repetition_factor=repetition_factor,
        kl_reg=kl_reg,
    )
    mixture = solution["mixture"]
    return {
        "corpus": manifest.corpus,
        "runs": len(rows),
        "domains": len(domains),
        "live_domains": len(live),
        "dead_domains": [domains[i] for i in dead],
        "required_runs_for_unique_fit": required,
        "min_appearances": min_appearances,
        "tasks": task_names,
        "regression_fit": quality,
        "mixture": {d: float(w) for d, w in zip(domains, mixture, strict=True)},
        "natural": {d: float(w) for d, w in zip(domains, natural, strict=True)},
        "interaction_matrix": {
            task: {domains[i]: float(v) for i, v in zip(live, params[j, 1:], strict=True)}
            for j, task in enumerate(task_names)
        },
        "log_c": {task: float(params[j, 0]) for j, task in enumerate(task_names)},
        "objective_loss_term": solution["objective_loss_term"],
        "objective_kl_term": solution["objective_kl_term"],
        "objective_kl_penalty": solution["objective_kl_penalty"],
        "kl_penalty_share": solution["kl_penalty_share"],
        "dead_natural_mass": solution["dead_natural_mass"],
        "requested_tokens": requested_tokens,
        "repetition_factor": repetition_factor,
        "kl_reg": kl_reg,
        "n_effectively_dropped": int((mixture < MIXTURE_DENSITY_THRESHOLD).sum()),
    }


def _write_json(payload: dict, path: str) -> None:
    with fsspec.open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    logger.info("wrote %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", required=True)
    p.add_argument("--region", action="append", required=True, help="Repeatable; merged across regions.")
    p.add_argument("--manifest", required=True, help="gs:// path to the swarm manifest JSON.")
    p.add_argument("--output-prefix", required=True)
    p.add_argument(
        "--requested-tokens",
        type=float,
        default=None,
        help="Target run's token budget R. Omit to solve with no availability caps.",
    )
    p.add_argument("--repetition-factor", type=float, default=DEFAULT_REPETITION_FACTOR)
    p.add_argument("--kl-reg", type=float, default=DEFAULT_KL_REG)
    p.add_argument("--expected-train-steps", type=int, default=PROXY_TRAIN_STEPS)
    p.add_argument("--include-mmlu", action="store_true", default=True)
    p.add_argument("--no-include-mmlu", dest="include_mmlu", action="store_false")
    p.add_argument("--sweep", action="store_true", help="Also sweep (R, k) post-hoc; costs no training.")
    p.add_argument(
        "--min-appearances",
        type=int,
        default=1,
        help="Mixtures a domain must appear in to be fitted. 1 = production. Raise it for a "
        "PRELIMINARY fit before the swarm is complete: it shrinks the support to what the "
        "available rows can identify, instead of relaxing the identifiability guard.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Processes to fan the 42 per-task fits across. Measured at 236 s/task on the "
        "real dclm swarm, so serial is ~2.75 h per corpus; the fan-out is bit-identical "
        "(tests/data_mixing/test_olmix_fit_parallel.py). Set to the core count.",
    )
    p.add_argument(
        "--devset",
        choices=("marin", "olmix_exact"),
        default="marin",
        help="Which bpb devset to optimise (ignored unless --metric bpb). 'marin' = the 42-task "
        "objective with DCLM Core v2 held out. 'olmix_exact' = olmix's own 51-task suite from "
        "the paper's Table 9, which INCLUDES the 10 Core v2 tasks and drops gsm8k -- use it for "
        "comparability with the paper, but do not then report beating natural on Core v2. Both "
        "read the same evals, so running both costs one extra CPU job.",
    )
    p.add_argument("--write-csv", action="store_true", help="Also write olmix-schema ratios.csv/metrics.csv.")
    p.add_argument(
        "--metric",
        choices=("bpb", "core_v2", "blend"),
        default="bpb",
        help="What to optimise. 'bpb' = the 42-task OLMo Base-Easy devset (default). "
        "'core_v2' = the 22 DCLM Core v2 tasks, fit as 1-centered_accuracy so that lower "
        "is better and the convex Minimize solver stays valid. NOTE: Core v2 is the "
        "held-out set for the bpb objective, so optimising it forfeits that comparison. "
        "'blend' = the 16 BLEnD cultural-knowledge tasks as gold bpb -- DIAGNOSTIC ONLY: "
        "the across-swarm variance is ~90% a single general-quality axis, and gold bpb "
        "never sees BLEnD's cross-country distractors, so a mixture fit on it is weakly "
        "identified for anything culture-specific.",
    )
    args = p.parse_args()

    manifest = read_manifest(args.manifest)
    if args.metric == "core_v2":
        task_names = list(core_v2_tasks())
        logger.info("objective: DCLM Core v2, %d tasks, fit as (1 - centered_accuracy)", len(task_names))
    elif args.metric == "blend":
        task_names = list(BLEND_TASKS)
        logger.info("objective: BLEnD cultural bpb, %d country tasks (diagnostic)", len(task_names))
    elif args.devset == "olmix_exact":
        task_names = list(build_olmix_exact_tasks(include_mmlu=args.include_mmlu))
        logger.info(
            "objective: olmix's own suite reproduced exactly (paper Table 9), %d tasks. "
            "Core v2 is INSIDE this objective, so results on it are no longer held out.",
            len(task_names),
        )
    else:
        task_names = list(build_target_tasks(include_mmlu=args.include_mmlu))
        logger.info("objective: OLMo Base-Easy bpb devset, %d tasks", len(task_names))
    rows, skipped = load_swarm_rows(
        sources_from_regions(args.region, args.corpus, args.metric),
        manifest,
        task_names,
        args.expected_train_steps,
    )
    logger.info("collected %d usable runs; skipped=%s", len(rows), skipped)
    if not rows:
        raise RuntimeError(f"no usable swarm rows for {args.corpus} in {args.region}; skipped={skipped}")

    result = fit_and_solve(
        manifest=manifest,
        rows=rows,
        task_names=task_names,
        requested_tokens=args.requested_tokens,
        repetition_factor=args.repetition_factor,
        kl_reg=args.kl_reg,
        min_appearances=args.min_appearances,
        num_workers=args.workers,
    )
    result["skipped"] = skipped

    out = args.output_prefix.rstrip("/")
    tag = "uncapped" if args.requested_tokens is None else f"R{args.requested_tokens:.3g}_k{args.repetition_factor:g}"
    _write_json(result, f"{out}/mix_{tag}.json")

    if args.write_csv:
        for name, payload in (
            ("ratios.csv", write_ratios_csv(rows, list(manifest.domains))),
            ("metrics.csv", write_metrics_csv(rows, task_names)),
        ):
            with fsspec.open(f"{out}/{name}", "w") as fh:
                fh.write(payload)

    if args.sweep:
        domains = list(manifest.domains)
        weights, bpb = swarm_design(rows, manifest, task_names)
        live, _ = split_sampled_domains(domains, weights)
        tokens = np.array([manifest.tokens[d] for d in domains], dtype=float)
        natural = tokens / tokens.sum()
        # Redundant with the fit inside `fit_and_solve` -- the laws are deterministic, so these
        # are the same numbers. It is repeated rather than threaded through because the params
        # leave that function only as task/domain-keyed dicts, and rebuilding the array from
        # them would round-trip float32 through Python floats and perturb the solver. What is
        # NOT acceptable is repeating it *serially*: that took 3.5 h and dwarfed the fit it was
        # duplicating. Fan it out the same way.
        params = fit_all_tasks(weights[:, live], bpb, task_names, num_workers=args.workers)
        sweep = sweep_constraints(
            params=params,
            domains=domains,
            live=live,
            tokens=tokens,
            natural=natural,
            requested_tokens_grid=list(SWEEP_REQUESTED_TOKENS),
            repetition_grid=list(SWEEP_REPETITION),
            kl_reg=args.kl_reg,
        )
        _write_json(
            {
                "corpus": manifest.corpus,
                "runs": len(rows),
                "cells": [
                    {
                        "requested_tokens": c["requested_tokens"],
                        "repetition_factor": c["repetition_factor"],
                        "feasible": c["feasible"],
                        "slack": c["slack"],
                        **(
                            {}
                            if not c["feasible"]
                            else {
                                "n_at_cap": c["n_at_cap"],
                                "kl_penalty_share": c["kl_penalty_share"],
                                "mixture": {d: float(w) for d, w in zip(domains, c["mixture"], strict=True)},
                            }
                        ),
                    }
                    for c in sweep
                ],
            },
            f"{out}/sweep_R_k.json",
        )


if __name__ == "__main__":
    main()
