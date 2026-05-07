# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Polling eval coordinator for 8B SFT cells across code/math/medical.

The 8B training cells (run via `launch_sft_sweep.py` → `run_sft_standalone.py`)
write summaries to `gs://marin-us-central1/metadata/sft_base_results/`. This
coordinator polls that prefix, parses the per-run domain from the run_name,
and dispatches one `run_eval_standalone.py --domain <code|math|medical>`
child per run.

Mirrors `launch_medical_evals.py` (which only handles the 0.6B medical prefix
`medical_sft_base_results/`). Both coordinators can run side-by-side without
stepping on each other.

USAGE
-----
    iris --cluster marin job run --priority interactive --no-wait \\
        --memory 2GB --cpu 2 --job-name eval-coord-8b \\
        -e WANDB_API_KEY <key> -e HF_TOKEN <hf_token> \\
        -- python experiments/rephraser/launch_8b_evals.py \\
        --child-priority interactive --poll-seconds 300
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


DEFAULT_TRAINING_RESULTS_PREFIX = "gs://marin-us-central1/metadata/sft_base_results/"
SCRIPT = "experiments/rephraser/run_eval_standalone.py"

DEFAULT_TPU_VARIANT = "v5p-8"
DEFAULT_TPU_ALTERNATIVES: tuple[str, ...] = ("v6e-4",)

PRIORITY_BAND_MAP = {
    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
    "batch": job_pb2.PRIORITY_BAND_BATCH,
    "unspecified": job_pb2.PRIORITY_BAND_UNSPECIFIED,
}


def _domain_from_run_name(run_name: str, medical_task_set: str = "generative") -> str:
    """Run names look like `<domain>-<branch>-<cell>-qwen3-8b-base-...`.

    For medical runs, the caller may select a non-default task set via
    `medical_task_set` (e.g. "logprob" -> "medical-logprob"). Only the medical
    domain currently has multiple task sets.
    """
    head = run_name.split("-", 1)[0].lower()
    if head not in ("code", "math", "medical"):
        raise ValueError(f"Cannot parse domain from run_name={run_name!r}")
    if head == "medical" and medical_task_set != "generative":
        return f"medical-{medical_task_set}"
    return head


def _domain_summary_prefix(domain: str) -> str:
    return f"gs://marin-us-central1/metadata/{domain}_sft_base_eval_results/"


def _discover_training_runs(prefix: str) -> list[dict]:
    fs, urlpath = fsspec.core.url_to_fs(prefix.rstrip("/") + "/")
    try:
        fs.invalidate_cache(urlpath)
    except Exception:
        pass
    summaries: list[dict] = []
    if not fs.exists(urlpath):
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


def _eval_already_done(eval_output_path: str, domain: str) -> bool:
    """run_eval_standalone.py writes `.{domain}_eval_DONE` on success."""
    done_marker = f"{eval_output_path.rstrip('/')}/.{domain}_eval_DONE"
    try:
        fs, urlpath = fsspec.core.url_to_fs(done_marker)
        return fs.exists(urlpath)
    except Exception:
        return False


def submit_one(
    client: IrisClient,
    *,
    summary: dict,
    domain: str,
    child_priority_band: int,
    wandb_api_key: str,
    hf_token: str | None,
    tpu_variant: str,
    tpu_alternatives: tuple[str, ...] = (),
) -> str:
    run_name = summary["run_name"]
    region = summary["region"]
    output_path = summary["output_path"]
    num_train_steps = summary.get("tokens", {}).get("num_train_steps")
    if num_train_steps is None:
        raise ValueError(f"Training summary for {run_name} missing tokens.num_train_steps")

    # Mirrors launch_medical_evals.submit_one — Levanter exports HF at step (N-1).
    bucket_prefix = output_path.split("/", 3)
    rel_after_bucket = bucket_prefix[3] if len(bucket_prefix) >= 4 else output_path
    saved_step = num_train_steps - 1
    model_rel_path = f"{rel_after_bucket.rstrip('/')}/hf/step-{saved_step}"
    model_name = run_name

    cmd_args = [
        "python",
        SCRIPT,
        "--domain",
        domain,
        "--model-rel-path",
        model_rel_path,
        "--model-name",
        model_name,
    ]

    env_vars = {
        "WANDB_API_KEY": wandb_api_key,
        "PYTHONUNBUFFERED": "1",
        "WANDB_INIT_TIMEOUT": "300",
        "MARIN_VLLM_MODE": "native",
        # Required for HumanEval/MBPP code_eval metric.
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
        name=f"eval-{domain}-{run_name}"[:200],
        resources=ResourceSpec(
            cpu=8,
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
    logger.info("Submitted eval %s (%s) -> %s (region=%s)", run_name, domain, job.job_id, region)
    return str(job.job_id)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--training-results-prefix", default=DEFAULT_TRAINING_RESULTS_PREFIX)
    parser.add_argument("--filter-name", default=None, help="Only evaluate runs whose run_name contains this substring.")
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-keep-alive", action="store_true")
    parser.add_argument(
        "--domain-filter",
        nargs="+",
        choices=["code", "math", "medical"],
        default=None,
        help="Only evaluate runs in these domains. Default: all three.",
    )
    parser.add_argument(
        "--medical-task-set",
        choices=["generative", "logprob"],
        default="generative",
        help="Which MMLU medical task variant to evaluate against. 'logprob' uses "
        "single-token loglikelihood scoring (avoids the format-following collapse "
        "we saw with the generative variant on Base models).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)

    if args.dry_run:
        summaries = _discover_training_runs(args.training_results_prefix)
        logger.info("Dry run: would evaluate %d runs.", len(summaries))
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
            try:
                domain = _domain_from_run_name(run_name, medical_task_set=args.medical_task_set)
            except ValueError as e:
                logger.warning("Skipping %s: %s", run_name, e)
                submitted_in_process.add(run_name)
                continue
            if args.domain_filter and domain not in args.domain_filter:
                submitted_in_process.add(run_name)
                continue

            num_train_steps = s.get("tokens", {}).get("num_train_steps")
            output_path = s.get("output_path", "")
            if num_train_steps is None:
                eval_output_path = f"{output_path.rstrip('/')}/hf/eval/lm_eval_harness"
            else:
                eval_output_path = f"{output_path.rstrip('/')}/hf/step-{num_train_steps - 1}/eval/lm_eval_harness"
            if args.skip_if_done and _eval_already_done(eval_output_path, domain):
                submitted_in_process.add(run_name)
                continue
            try:
                submit_one(
                    client,
                    summary=s,
                    domain=domain,
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
            "Poll #%d: discovered %d runs, dispatched %d new evals " "(total handled this process: %d).",
            iteration,
            len(summaries),
            new_submitted,
            len(submitted_in_process),
        )

        if args.no_keep_alive:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
