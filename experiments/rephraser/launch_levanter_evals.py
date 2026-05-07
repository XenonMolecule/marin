# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polling Levanter-backend eval coordinator for medical-logprob across all sizes.

Watches both the 0.6B summary prefix (`medical_sft_base_results/`) and the
8B summary prefix (`sft_base_results/`), filters to medical runs, and
dispatches one Levanter-backed `run_eval_levanter_standalone.py --domain
medical-logprob` child per run. Companion to the vLLM-based
`launch_medical_evals.py` + `launch_8b_evals.py` — they continue to drive the
generative-MMLU pipeline; this one drives the logprob fix.

Why a separate coord rather than extending the existing two:
  - Different SCRIPT (`run_eval_levanter_standalone.py` vs
    `run_eval_standalone.py`)
  - Different extras (`tpu+eval` vs `tpu+vllm+eval`); Levanter doesn't need
    vLLM
  - Different DONE marker (`.medical-logprob_eval_DONE` under
    `lm_eval_harness_levanter/` vs `.medical_eval_DONE` under
    `lm_eval_harness/`) — the two coords' dedupe logic stays clean as
    independent state machines
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


SCRIPT = "experiments/rephraser/run_eval_levanter_standalone.py"

# Watch both 0.6B and 8B summary prefixes. 14B has its own legacy Ray-executor
# eval outputs and isn't auto-evaluated here — relaunch 14B cells via the
# single-shot launcher if needed.
DEFAULT_TRAINING_RESULTS_PREFIXES: tuple[str, ...] = (
    "gs://marin-us-central1/metadata/medical_sft_base_results/",  # 0.6B
    "gs://marin-us-central1/metadata/sft_base_results/",  # 8B (and future)
)

DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def _is_medical_run(run_name: str) -> bool:
    return run_name.lower().startswith("medical-")


def _discover_training_runs(prefixes: tuple[str, ...]) -> list[dict]:
    """Read every `medical-*.json` summary across all prefixes."""
    summaries: list[dict] = []
    seen_run_names: set[str] = set()
    for prefix in prefixes:
        fs, urlpath = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
        try:
            fs.invalidate_cache(urlpath)
        except Exception:
            pass
        if not fs.exists(urlpath):
            continue
        for path in fs.ls(urlpath, detail=False):
            if not path.endswith(".json"):
                continue
            try:
                with fs.open(path, "r") as f:
                    s = json.load(f)
                run_name = s.get("run_name", "")
                if not _is_medical_run(run_name):
                    continue
                if run_name in seen_run_names:
                    continue
                seen_run_names.add(run_name)
                summaries.append(s)
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
    return summaries


def _eval_already_done(eval_output_path: str) -> bool:
    """run_eval_levanter_standalone writes .medical-logprob_eval_DONE on success."""
    done_marker = f"{eval_output_path.rstrip('/')}/.medical-logprob_eval_DONE"
    try:
        fs, urlpath = fsspec.core.url_to_fs(done_marker)
        return fs.exists(urlpath)
    except Exception:
        return False


def submit_one(
    client: IrisClient,
    *,
    summary: dict,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...] = (),
    max_retries_failure: int = 2,
) -> str:
    run_name = summary["run_name"]
    region = summary["region"]
    output_path = summary["output_path"]
    num_train_steps = summary.get("tokens", {}).get("num_train_steps")
    if num_train_steps is None:
        raise ValueError(f"Training summary for {run_name} missing tokens.num_train_steps")

    bucket_prefix = output_path.split("/", 3)
    rel_after_bucket = bucket_prefix[3] if len(bucket_prefix) >= 4 else output_path
    saved_step = num_train_steps - 1
    model_rel_path = f"{rel_after_bucket.rstrip('/')}/hf/step-{saved_step}"
    model_name = run_name + "-medical-logprob"  # disambiguates W&B from the generative run

    cmd_args = [
        "python",
        SCRIPT,
        "--domain",
        "medical-logprob",
        "--model-rel-path",
        model_rel_path,
        "--model-name",
        model_name,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        "HF_DATASETS_TRUST_REMOTE_CODE": "1",
        "HF_ALLOW_CODE_EVAL": "1",
    }
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

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
        name=f"levanter-eval-medical-logprob-{run_name}"[:200],
        resources=ResourceSpec(
            cpu=8,
            memory="64GB",  # Levanter loglikelihood doesn't need vLLM's 128GB.
            disk="50GB",
            device=tpu_device(tpu_variant),
        ),
        environment=EnvironmentSpec(
            extras=["tpu", "eval"],
            env_vars=env_vars,
        ),
        constraints=constraints,
        max_retries_preemption=100,
        max_retries_failure=max_retries_failure,
        priority_band=child_priority_band,
    )
    logger.info("Submitted Levanter logprob eval %s -> %s (region=%s)", run_name, job.job_id, region)
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-results-prefixes", nargs="+", default=list(DEFAULT_TRAINING_RESULTS_PREFIXES))
    parser.add_argument("--filter-name", default=None)
    parser.add_argument("--tpu-variant", default=DEFAULT_TPU_VARIANT)
    parser.add_argument("--tpu-alternatives", nargs="*", default=list(DEFAULT_TPU_ALTERNATIVES))
    parser.add_argument(
        "--child-priority",
        choices=["production", "interactive", "batch", "unspecified"],
        default="batch",
    )
    parser.add_argument("--skip-if-done", action="store_true", default=True)
    parser.add_argument("--no-skip-if-done", dest="skip_if_done", action="store_false")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-count", type=int, default=None)
    parser.add_argument(
        "--submit-delay-seconds",
        type=int,
        default=30,
        help="Sleep this long between consecutive child submissions in a single poll. "
             "Levanter's `HFCheckpointConverter.from_hf` pings HF for every registered "
             "model class on each worker startup (~10-20 API calls per child). Firing 21 "
             "children in a 3-sec window — combined with iris's max_retries_failure=10 "
             "retry-bombing on transient errors — blew through HF's 1000/5min rate limit "
             "and hard-failed the whole fan-out. 30s/submit is conservative but safe.",
    )
    parser.add_argument(
        "--max-retries-failure",
        type=int,
        default=2,
        help="Pass-through to iris job submit; lower than the iris default of 10 to avoid "
             "retry-bombing transient HF rate limits.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.dry_run:
        summaries = _discover_training_runs(tuple(args.training_results_prefixes))
        if args.filter_name:
            summaries = [s for s in summaries if args.filter_name in s.get("run_name", "")]
        logger.info("Dry run: would evaluate %d medical runs.", len(summaries))
        for s in summaries:
            logger.info("  %s (region=%s)", s.get("run_name"), s.get("region"))
        return

    controller_address = os.environ.get("IRIS_CONTROLLER_ADDRESS")
    if not controller_address:
        raise RuntimeError("IRIS_CONTROLLER_ADDRESS not set — must run inside an iris job.")
    bundle_id = os.environ.get("IRIS_BUNDLE_ID")
    client = IrisClient.remote(controller_address, bundle_id=bundle_id)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY env var required.")
    hf_token = os.environ.get("HF_TOKEN")
    child_priority_band = PRIORITY_BAND_MAP[args.child_priority]

    submitted_in_process: set[str] = set()
    iteration = 0
    while True:
        iteration += 1
        summaries = _discover_training_runs(tuple(args.training_results_prefixes))
        if args.filter_name:
            summaries = [s for s in summaries if args.filter_name in s.get("run_name", "")]
        if args.max_count:
            summaries = summaries[: args.max_count]

        new_submitted = 0
        for s in summaries:
            run_name = s.get("run_name", "<unknown>")
            if run_name in submitted_in_process:
                continue
            num_train_steps = s.get("tokens", {}).get("num_train_steps")
            output_path = s.get("output_path", "")
            if num_train_steps is None:
                eval_output_path = f"{output_path.rstrip('/')}/hf/eval/lm_eval_harness_levanter"
            else:
                eval_output_path = (
                    f"{output_path.rstrip('/')}/hf/step-{num_train_steps - 1}"
                    f"/eval/lm_eval_harness_levanter"
                )
            if args.skip_if_done and _eval_already_done(eval_output_path):
                submitted_in_process.add(run_name)
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
                    max_retries_failure=args.max_retries_failure,
                )
                submitted_in_process.add(run_name)
                new_submitted += 1
                # Stagger HF API requests across worker startups.
                if args.submit_delay_seconds > 0:
                    time.sleep(args.submit_delay_seconds)
            except Exception as e:
                logger.exception("Failed to submit Levanter eval for %s: %s", run_name, e)

        logger.info(
            "Poll #%d: discovered %d medical runs, dispatched %d new evals (total handled: %d).",
            iteration, len(summaries), new_submitted, len(submitted_in_process),
        )

        if args.no_keep_alive:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
