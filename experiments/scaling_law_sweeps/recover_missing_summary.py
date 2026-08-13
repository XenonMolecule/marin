# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild the per-run summary for cells that finished training + eval but never wrote one.

A child writes its summary JSON and DONE marker as the LAST step, after training, the final
eval, and the HF export. If it dies in that window -- a gang bounce, a preemption, an Iris
kill -- every expensive artifact survives on GCS but the run still looks unfinished to every
downstream counter, and the naive fix is to retrain it from scratch. Observed on
`dclm_10k_mix 3e+19-d512-L6-B256`: checkpoint at step 41054 (== `train_steps - 1`), a
62-key final-eval row, an `hf/step-41054` directory, and no summary.

That run also showed why the export needs checking rather than assuming: it held a full-size
`model.safetensors` and NONE of the four sidecars, so it looked finished and then failed every
HF-based eval with `FileNotFoundError: .../config.json`. `hf_export_complete` reports it. The
sidecars are architecture-determined and byte-identical across runs of the same shape
(verified across d512-L6 runs), so a broken export is repairable by copying them from a
same-shape sibling IN THE SAME REGION -- no re-export, no egress.

This tool finds those cells and rebuilds the summary from what is already on disk, using the
runner's OWN `_build_summary` and its OWN step-checked eval reader rather than a
reimplementation -- a hand-rolled summary is how wrong numbers enter the scaling plots (see
`repair_stale_final_eval.py` for the last time that happened).

A cell qualifies ONLY if `eval_metrics.jsonl` holds a row at exactly `train_steps - 1`.
Anything less means the run really did die mid-training and must be retrained; this tool
refuses it rather than writing a summary off a non-final eval cycle.

    # report
    python -m experiments.scaling_law_sweeps.recover_missing_summary --methods dclm_10k_mix
    # write summaries + DONE markers
    python -m experiments.scaling_law_sweeps.recover_missing_summary --methods dclm_10k_mix --apply
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging

import fsspec

from experiments.scaling_law_sweeps.curation_plan import METHODS
from experiments.scaling_law_sweeps.launch_10k_natural import enumerate_10k_natural_plans
from experiments.scaling_law_sweeps.run_curation_train_standalone import (
    _build_summary,
    _read_last_eval_metrics_checked,
    _write_summary,
)

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "gs://marin-us-central1/metadata/data_curation_10k_natural_results/"
# A run's artifacts live in exactly one regional bucket; we do not know which without looking.
CANDIDATE_BUCKETS = ("marin-us-central1", "marin-us-east5", "marin-us-east1", "marin-eu-west4", "marin-us-central2")
CHECKPOINT_SUBPATH = "checkpoints/isoflop-curation"
# Sidecars every HF-based eval needs; `model.safetensors` alone is NOT enough.
HF_SIDECAR_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")
# Weights are a SINGLE file only for small models. Anything past ~2B shards into
# `model-0000N-of-0000M.safetensors` + `model.safetensors.index.json`, so requiring the literal
# `model.safetensors` reports every d2432/d3584 export as broken when it is perfectly fine.
HF_WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")


def locate_output_path(fs, run_name: str) -> tuple[str, str] | None:
    """(output_path, region) for a run, found by probing each regional bucket."""
    for bucket in CANDIDATE_BUCKETS:
        root = f"{bucket}/{CHECKPOINT_SUBPATH}/{run_name}"
        if fs.exists(f"{root}/checkpoints/eval_metrics.jsonl"):
            return f"gs://{root}", bucket.removeprefix("marin-")
    return None


def hf_export_complete(fs, output_path: str) -> bool:
    """Whether the newest `hf/step-N` export can actually be loaded.

    A run killed during its HF export leaves a full-size `model.safetensors` and none of the
    sidecars, so the directory exists and looks finished. Recovering such a run writes a
    correct summary (the scaling numbers come from eval_metrics.jsonl and are valid) but every
    HF-based eval suite then fails on it with a FileNotFoundError for `config.json`.
    """
    hf = f"{output_path.rstrip('/')}/hf"
    steps = [p for p in (fs.ls(hf, detail=False) if fs.exists(hf) else []) if "/step-" in p]
    if not steps:
        return False
    newest = max(steps, key=lambda p: int(p.rsplit("-", 1)[-1])).rstrip("/")
    have = {p.rsplit("/", 1)[-1] for p in fs.ls(newest, detail=False)}
    return all(f in have for f in HF_SIDECAR_FILES) and any(f in have for f in HF_WEIGHT_FILES)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--results-prefix", default=RESULTS_PREFIX)
    p.add_argument("--apply", action="store_true", help="Write summaries + DONE markers.")
    args = p.parse_args()

    fs = fsspec.filesystem("gs")
    existing = {x.rsplit("/", 1)[-1].removesuffix(".json") for x in fs.ls(args.results_prefix, detail=False)}
    plans = enumerate_10k_natural_plans(tuple(args.methods))
    logger.info("%d planned cells across %s; %d summaries already present", len(plans), args.methods, len(existing))

    recoverable, needs_retrain, no_artifacts = [], [], []
    for plan in plans:
        run_name = plan.run_name_core
        if run_name in existing:
            continue
        found = locate_output_path(fs, run_name)
        if found is None:
            no_artifacts.append(run_name)
            continue
        output_path, region = found
        expected_step = plan.train_steps - 1
        final_eval = _read_last_eval_metrics_checked(output_path, expected_step)
        if final_eval is None:
            needs_retrain.append(run_name)
            continue
        recoverable.append((plan, run_name, output_path, region, final_eval, hf_export_complete(fs, output_path)))

    logger.info(
        "missing summaries: %d recoverable | %d incomplete (no final eval) | %d no artifacts at all",
        len(recoverable),
        len(needs_retrain),
        len(no_artifacts),
    )
    for _, run_name, _, region, fe, hf_ok in recoverable:
        logger.info(
            "  RECOVERABLE %s (%s) step=%s hf_export=%s",
            run_name,
            region,
            fe.get("step"),
            "complete" if hf_ok else "INCOMPLETE -> HF evals will fail until repaired",
        )
    for run_name in needs_retrain:
        logger.info("  INCOMPLETE %s", run_name)
    if needs_retrain:
        # "no final eval" means training has not reached `train_steps - 1` -- which is equally
        # true of a cell that is RUNNING RIGHT NOW. This tool cannot tell those apart, and
        # relaunching a live cell duplicates it (the sweep coordinator has no claim system).
        # Always intersect this list with the live children before treating any of it as work.
        logger.warning(
            "INCOMPLETE includes cells that are still TRAINING -- check live children before "
            "relaunching any of them, or you will duplicate a running run."
        )

    if not args.apply:
        logger.info("dry run -- pass --apply to write %d summaries", len(recoverable))
        return

    for plan, run_name, output_path, region, final_eval, _hf_ok in recoverable:
        summary = _build_summary(
            plan=plan,
            method=METHODS[plan.method_name],
            region=region,
            run_name=run_name,
            output_path=output_path,
            final_eval=final_eval,
        )
        _write_summary(summary, args.results_prefix, run_name)
        marker = {
            "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "run_name_core": plan.run_name_core,
            "method": plan.method_name,
            "experiment_tag": plan.experiment_tag,
            "region": region,
            "recovered_by": "recover_missing_summary",
        }
        with fs.open(f"{output_path}/.data_curation_DONE", "w") as fh:
            json.dump(marker, fh, indent=2)
        logger.info("recovered %s", run_name)


if __name__ == "__main__":
    main()
