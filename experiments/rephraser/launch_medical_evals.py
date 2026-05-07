# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-native coordinator for medical 0.6B-Base re-run EVALS.

Submits one Iris TPU child per completed training run, each running
`run_medical_eval_standalone.py`. Auto-discovers completed training runs
from the training summary prefix (`medical_sft_base_results/*.json`) so
you don't have to enumerate them by hand.

WHY THIS FILE EXISTS
--------------------
Companion to `medical_extraction_sft_v2_base.py` and
`medical_resiliparse_v2_base.py`. After training lands, we need to evaluate
the resulting HF checkpoints on the medical eval suite. The original
recipe routed evals through the Marin executor → Ray; that path is offline.
This coordinator does the Iris-native equivalent: read training summaries,
submit one vLLM-TPU eval child per run.

REGION ROUTING
--------------
Each training run is region-locked (its summary JSON records which region
it claimed). The eval child MUST land in the SAME region — otherwise vLLM
would download the HF checkpoint cross-region (gigabytes of egress per run).
We propagate the per-run region into the iris job's HARD region constraint.

vLLM-TPU IRIS PATTERN
---------------------
Per user-memory `feedback_iris_vllm_tpu_standalone.md` and
`feedback_iris_eval_launch.md`, vLLM evals on Iris need:
  - `--memory 128GB` (vLLM workers consume a lot of RAM)
  - `--extra marin:vllm` (so vllm-tpu is installed in the worker venv)
  - `MARIN_VLLM_MODE=native` env var
  - char-only filtering implicit via `apply_chat_template=False`
This file sets all of those.

USAGE
-----
The launcher itself runs inside an iris parent CPU job:

    iris --cluster marin job run --priority interactive --no-wait \\
        --memory 2GB --cpu 2 --job-name medical-eval-coord \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/launch_medical_evals.py \\
        --child-priority interactive

To eval a single run only (smoke):

        --filter-name medical-extraction-lr5e-6_bs32-qwen3-0.6b-base-rerun
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import fsspec

from iris.client.client import IrisClient
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    preemptible_constraint,
)
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    tpu_device,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Where the training standalone wrote per-run summaries. Each JSON has
# `run_name`, `region`, `output_path`, `branch`, etc. — everything we need
# to dispatch a per-run eval child.
DEFAULT_TRAINING_RESULTS_PREFIX = "gs://marin-us-central1/metadata/medical_sft_base_results/"

# The standalone child script.
SCRIPT = "experiments/rephraser/run_medical_eval_standalone.py"

# TPU primary + alternatives — same as the training coordinators.
DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


# ---------------------------------------------------------------------------
# Discover completed training runs
# ---------------------------------------------------------------------------
def _discover_training_runs(prefix: str) -> list[dict]:
    """Read every `*.json` summary under the training results prefix.

    Each summary was written by `run_medical_sft_standalone._write_summary`.
    Returns the list of summary dicts in arbitrary order.

    fsspec's GCSFileSystem caches directory listings, so when we re-poll the
    same prefix from a long-lived process we'd otherwise miss any files that
    landed after the first listing. Invalidate the cache before each scan.
    """
    fs, urlpath = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
    try:
        fs.invalidate_cache(urlpath)
    except Exception:
        pass  # fsspec backends without cache invalidation are no-ops here.
    summaries: list[dict] = []
    if not fs.exists(urlpath):
        logger.warning("Training results prefix %s does not exist yet — no completed runs to evaluate.", prefix)
        return summaries
    for path in fs.ls(urlpath, detail=False):
        if not path.endswith(".json"):
            continue
        try:
            with fs.open(path, "r") as f:
                summaries.append(json.load(f))
        except Exception as e:
            logger.warning("Failed to read %s: %s", path, e)
    return summaries


def _eval_already_done(eval_output_path: str) -> bool:
    """Check the DONE marker the eval standalone writes on success."""
    done_marker = f"{eval_output_path.rstrip('/')}/.medical_eval_DONE"
    try:
        fs, urlpath = fsspec.core.url_to_fs(done_marker)
        return fs.exists(urlpath)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Submit one eval child
# ---------------------------------------------------------------------------
def submit_one(
    client: IrisClient,
    *,
    summary: dict,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...] = (),
) -> str:
    """Submit one eval child for one completed training run."""
    run_name = summary["run_name"]
    region = summary["region"]
    output_path = summary["output_path"]
    # The HF export from the training side lives at `<output_path>/hf/step-{N}/`.
    # Levanter writes the export to a STEP-suffixed subdirectory (matching its
    # checkpoint layout), so pointing vLLM at just `<output_path>/hf/` fails
    # with "Invalid repository ID or local directory" — vLLM sees only an
    # empty parent dir with the `step-N/` subdir nested inside.
    # We need the step suffix. Levanter writes at `hf_save_steps=num_train_steps`,
    # so the step number is `tokens.num_train_steps` from the training summary.
    bucket_prefix = output_path.split("/", 3)
    rel_after_bucket = bucket_prefix[3] if len(bucket_prefix) >= 4 else output_path
    num_train_steps = summary.get("tokens", {}).get("num_train_steps")
    if num_train_steps is None:
        raise ValueError(
            f"Training summary for {run_name} missing tokens.num_train_steps; "
            f"cannot compute HF export path."
        )
    # Levanter saves the HF export at step (N-1) where N=num_train_steps,
    # because training loops 0..N-1 and the export runs at the LAST completed
    # step. Empirically verified 2026-05-04: bs=32 cells (N=4277) saved at
    # step-4276; bs=64 cells (N=2139) saved at step-2138.
    saved_step = num_train_steps - 1
    model_rel_path = f"{rel_after_bucket.rstrip('/')}/hf/step-{saved_step}"
    model_name = run_name  # same name everywhere keeps W&B tidy

    cmd_args = [
        "python",
        SCRIPT,
        "--model-rel-path",
        model_rel_path,
        "--model-name",
        model_name,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        # MARIN_VLLM_MODE=native is required for vLLM-TPU on Iris
        # (see user-memory feedback_iris_vllm_tpu_standalone.md). The
        # `vllm` extra below pulls vllm-tpu into the worker venv.
        "MARIN_VLLM_MODE": "native",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    # Region constraint: HARD pin to the region that owns this run's checkpoint.
    # Anywhere else would force vLLM to download the HF checkpoint cross-region.
    region_constraint = Constraint(
        key=WellKnownAttribute.REGION,
        op=ConstraintOp.IN,
        values=(region,),
    )
    constraints = [
        preemptible_constraint(True),
        region_constraint,
    ]
    all_variants = (tpu_variant, *tpu_alternatives)
    if len(set(all_variants)) > 1:
        constraints.append(device_variant_constraint(list(all_variants)))

    job = client.submit(
        entrypoint=Entrypoint.from_command(*cmd_args),
        name=f"medical-eval-{run_name}"[:200],
        resources=ResourceSpec(
            cpu=8,
            # vLLM-TPU on Iris needs 128GB per the standalone-vLLM pattern in
            # user-memory feedback_iris_vllm_tpu_standalone.md. Anything below
            # OOMs the lm-eval workers on long-context generation.
            memory="128GB",
            disk="100GB",
            device=tpu_device(tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu", "vllm", "eval"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=100,
        max_retries_failure=10,
        priority_band=child_priority_band,
    )
    logger.info("Submitted eval %s -> %s (region=%s)", run_name, job.job_id, region)
    return str(job.job_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--training-results-prefix",
        default=DEFAULT_TRAINING_RESULTS_PREFIX,
        help="GCS prefix where the training standalone writes summary JSONs.",
    )
    parser.add_argument(
        "--filter-name",
        default=None,
        help=(
            "If set, only evaluate runs whose run_name CONTAINS this substring. "
            "Useful for smoke-testing the eval pipeline against one specific "
            "training run."
        ),
    )
    parser.add_argument(
        "--tpu-variant",
        default=DEFAULT_TPU_VARIANT,
        help="Primary TPU shape per child (v5p-8 default, single-host).",
    )
    parser.add_argument(
        "--tpu-alternatives",
        nargs="*",
        default=list(DEFAULT_TPU_ALTERNATIVES),
        help="Alternative TPU shapes (default ['v6e-4'] — matches the training side).",
    )
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument(
        "--skip-if-done",
        action="store_true",
        default=True,
        help="Skip runs whose eval DONE marker already exists. Default ON.",
    )
    parser.add_argument(
        "--no-skip-if-done",
        dest="skip_if_done",
        action="store_false",
        help="Submit eval children even if a prior eval already finished.",
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=None,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=300,
        help="How often to re-scan the training summaries prefix for new completions. Default 300s (5 min).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.dry_run:
        summaries = _discover_training_runs(args.training_results_prefix)
        if args.filter_name is not None:
            summaries = [s for s in summaries if args.filter_name in s.get("run_name", "")]
        logger.info("Dry run: would evaluate %d runs.", len(summaries))
        for s in summaries:
            logger.info("  %s (region=%s)", s.get("run_name"), s.get("region"))
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — this coordinator must run inside an iris job.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required (set via -e on iris job run).")
    hf_token = os.environ.get("HF_TOKEN")

    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    # Poll the training-summaries prefix on a fixed cadence and submit eval
    # children for newly-landed runs. Loops forever (or one-shot if
    # --no-keep-alive) so one coordinator handles all training cells regardless
    # of when each one lands. Dedupe by run_name + DONE marker.
    submitted_in_process: set[str] = set()
    poll_seconds = args.poll_seconds
    iteration = 0
    while True:
        iteration += 1
        summaries = _discover_training_runs(args.training_results_prefix)
        if args.filter_name is not None:
            summaries = [s for s in summaries if args.filter_name in s.get("run_name", "")]
        if args.max_count is not None:
            summaries = summaries[: args.max_count]

        new_submitted = 0
        for s in summaries:
            run_name = s.get("run_name", "<unknown>")
            if run_name in submitted_in_process:
                continue
            # The eval standalone writes its DONE marker at
            # `<model_path>/eval/lm_eval_harness/.medical_eval_DONE`, where
            # model_path = `<output_path>/hf/step-{N-1}` (Levanter's HF export
            # is step-suffixed). Earlier code missed the step suffix and
            # always re-dispatched.
            num_train_steps = s.get("tokens", {}).get("num_train_steps")
            if num_train_steps is None:
                eval_output_path = f"{s.get('output_path', '').rstrip('/')}/hf/eval/lm_eval_harness"
            else:
                eval_output_path = (
                    f"{s.get('output_path', '').rstrip('/')}/hf/step-{num_train_steps - 1}"
                    f"/eval/lm_eval_harness"
                )
            if args.skip_if_done and _eval_already_done(eval_output_path):
                submitted_in_process.add(run_name)  # treat as done; never resubmit
                continue
            try:
                submit_one(
                    client,
                    summary=s,
                    child_priority_band=child_priority_band,
                    wandb_api_key=wandb_api_key,
                    hf_token=hf_token,
                    tpu_variant=args.tpu_variant,
                    tpu_alternatives=tuple(args.tpu_alternatives),
                )
                submitted_in_process.add(run_name)
                new_submitted += 1
            except Exception as e:
                logger.exception("Failed to submit eval for %s: %s", run_name, e)

        logger.info(
            "Poll #%d: discovered %d runs, dispatched %d new evals "
            "(total handled this process: %d).",
            iteration,
            len(summaries),
            new_submitted,
            len(submitted_in_process),
        )

        if args.no_keep_alive:
            return
        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
