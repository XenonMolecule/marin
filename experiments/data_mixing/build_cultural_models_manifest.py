# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the eval manifest for trained cultural-target models.

Scans `metadata/data_curation_10k_natural_results` for completed
`*_mix_cultural_clp_lambda0p01` cells, locates each run's checkpoint dir under
`checkpoints/isoflop-curation/<run_name>` in the arm's candidate regions, and writes the
`run_name,region,output_path` CSV that `launch_cultural_models` (and the MMLU/BPB manifest
launchers generally) consume. Re-run each babysit tick; downstream skip-existing makes
repeat submissions idempotent.

    uv run python -m experiments.data_mixing.build_cultural_models_manifest \
        --out experiments/core_eval_manifests/cultural_models.txt
"""

from __future__ import annotations

import argparse
import json
import logging

import fsspec

logger = logging.getLogger(__name__)

RESULTS_ROOT = "marin-us-central1/metadata/data_curation_10k_natural_results"
ARM_SUBSTRING = "_mix_cultural_clp"  # matches every cultural arm variant (lambda/R/k tags differ)
# Candidate checkpoint regions per arm prefix (the sweep's --allowed-regions).
ARM_REGIONS: dict[str, tuple[str, ...]] = {
    "curation-dclm_10k": ("us-east5", "us-central1", "us-central2", "us-west4", "europe-west4"),
    # Keep these in sync with each arm's grid-store copies: a region missing here makes
    # every cell trained there invisible to eval waves (2026-09-22: resiliparse still
    # read us-east5-only after its 09-17 us-central2 copy, hiding 13 trained cells).
    # Extra regions are safe — find_checkpoint still requires the FINAL step export.
    "curation-lpv11_10k": ("us-east5", "us-east1", "us-central1", "us-central2"),
    "curation-nemotron_full_10k": ("us-east5", "us-central1"),
    "curation-resiliparse_10k": ("us-east5", "us-central2"),
    "curation-lpv11_fastpipe_v2_10k": ("us-east5", "us-central1", "us-east1", "europe-west4"),
    "curation-fineweb_cc_10k": ("us-east5", "us-central1", "us-central2"),
    "curation-fineweb_edu_10k": ("us-central2", "us-east5", "us-central1", "europe-west4"),
}


def region_bucket(region: str) -> str:
    return "marin-eu-west4" if region == "europe-west4" else f"marin-{region}"


def find_checkpoint(fs, run_name: str, final_step: int | None = None) -> tuple[str, str] | None:
    """Return (region, gs:// checkpoint dir) whose hf export holds the FINAL step.

    Preemption leaves partial hf exports (step-10000/20000) in the regions where dead
    attempts ran; "hf exists" first-hit picked those and evals scored 40%-trained models
    (2026-09-17 wrong-bucket incident: 52 evals). Require step-(final|final-1) when the
    final step is known; fall back to bare existence only when it is not.
    """
    import re

    regions = next((r for prefix, r in ARM_REGIONS.items() if run_name.startswith(prefix)), None)
    if regions is None:
        raise ValueError(f"run {run_name} matches no registered cultural arm prefix")
    for region in regions:
        path = f"{region_bucket(region)}/checkpoints/isoflop-curation/{run_name}"
        hf = f"{path}/hf"
        if not fs.exists(hf):
            continue
        if final_step is None:
            return region, f"gs://{path}"
        steps = [int(m.group(1)) for q in fs.ls(hf, detail=False) for m in [re.search(r"step-(\d+)$", q)] if m]
        if steps and max(steps) >= final_step - 1:
            return region, f"gs://{path}"
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm-substring", default=ARM_SUBSTRING,
                    help="Run-name filter; default cultural arms. Use '_mix_mmlu_' for the MMLU sweep.")
    ap.add_argument("--include-retired", action="store_true",
                    help="Also include the retired `_clp_lambda0p01` generation (mislabeled-lambda "
                    "arms killed 2026-09-14). Excluded by default: they are never plotted, so "
                    "evaluating them wastes eval capacity (user direction 2026-09-19).")
    args = ap.parse_args()

    fs = fsspec.filesystem("gcs")
    runs = [
        p.rsplit("/", 1)[-1].removesuffix(".json")
        for p in fs.ls(RESULTS_ROOT, detail=False)
        if args.arm_substring in p and p.endswith(".json")
        and (args.include_retired or "_clp_lambda0p01-" not in p)
    ]
    rows, missing = [], []
    for run in sorted(runs):
        with fs.open(f"{RESULTS_ROOT}/{run}.json") as fh:
            final_step = json.load(fh)["plan"]["train_steps"]
        hit = find_checkpoint(fs, run, final_step=final_step)
        if hit is None:
            missing.append(run)
            continue
        region, ckpt = hit
        rows.append(f"{run},{region},{ckpt}")

    with open(args.out, "w") as fh:
        fh.write("run_name,region,output_path\n")
        fh.write("\n".join(rows) + ("\n" if rows else ""))
    logger.info("wrote %d rows to %s (%d completed runs lacked a checkpoint dir)", len(rows), args.out, len(missing))
    for run in missing:
        logger.warning("no checkpoint found for completed run %s", run)


if __name__ == "__main__":
    main()
