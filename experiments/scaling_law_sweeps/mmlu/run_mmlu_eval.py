# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the MMLU (`mmlu_sl_verb`) suite on one HF checkpoint, offline.

Structurally identical to `core_tasks/run_core_tasks_eval.py` and
`olmes_base/run_olmes_eval.py`: offline dataset + hub caches wired in before any HF
import, then Levanter's harness in-process via `levanter_lm_eval.run_levanter_lm_eval`
(the marin `evaluate()` wrapper was removed by the 2026-08-13 upstream sync; the sibling
runners still reference it). Differences here:

  * the task is `mmlu_sl_verb` at ONE shot count (see mmlu_tasks_set for why sl_verb,
    and why 0-shot and 5-shot must be separate runs rather than one `evals=[...]`);
  * datasets come from the `mmlu_hf_cache` prefix, while the model-config hub cache is
    REUSED from `core_tasks_hub_cache` (identical checkpoints);
  * no custom-task install: sl_verb's `hails/mmlu_no_train` is parquet already.

Usage (as an iris child; launcher supplies HF_HUB_OFFLINE + region pin):

    python -m experiments.scaling_law_sweeps.mmlu.run_mmlu_eval \\
        --hf-checkpoint gs://.../hf/step-N/ \\
        --output-dir gs://.../metadata/mmlu_sl_verb_results/<shots>shot/<run_name>/ \\
        --run-name <run_name> \\
        --num-fewshot 5 \\
        --dataset-cache-gcs gs://<same-bucket>/eval_datasets/mmlu_hf_cache/ \\
        --hub-cache-gcs gs://<same-bucket>/eval_datasets/core_tasks_hub_cache/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

from experiments.scaling_law_sweeps.core_tasks.run_core_tasks_eval import _sync_cache

logger = logging.getLogger(__name__)

_DATASET_CACHE_MARKER = "/eval_datasets/mmlu_hf_cache/"
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep


def _prepare_hub_cache(gcs_cache: str, local_dir: str = "/tmp/mmlu_hf_home") -> int:
    n = _sync_cache(gcs_cache, _HUB_CACHE_MARKER, local_dir)
    os.environ["HF_HOME"] = local_dir
    logger.info("Synced %d hub-cache files -> %s; HF_HOME set", n, local_dir)
    return n


def _prepare_offline_dataset_cache(gcs_cache: str, local_dir: str = "/tmp/mmlu_hf_cache") -> int:
    n = _sync_cache(gcs_cache, _DATASET_CACHE_MARKER, local_dir)
    os.environ["HF_DATASETS_CACHE"] = local_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    logger.info("Synced %d dataset-cache files -> %s; HF_DATASETS_OFFLINE=1", n, local_dir)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF checkpoint dir (gs://...); config.json + safetensors.")
    p.add_argument("--output-dir", required=True, help="GCS dir to write results.json into.")
    p.add_argument("--run-name", default=None, help="Run name for wandb / imputed model name.")
    # Validated by `task_for_shots` below rather than argparse `choices`: importing the
    # task set up here would drag in levanter/transformers before the offline env is set.
    p.add_argument(
        "--num-fewshot",
        type=int,
        required=True,
        help="Shot count for this run (0 or 5). 5 = canonical MMLU (dev split, first_n sampler); 0 = the "
        "cheap soft-metric setting. One shot count per run: both share the task name and would collide.",
    )
    p.add_argument("--dataset-cache-gcs", required=True, help="GCS prefix of the MMLU HF-datasets cache in THIS region.")
    p.add_argument("--hub-cache-gcs", required=True, help="GCS prefix of the model-config hub cache in THIS region.")
    p.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke test). None = full.")
    args = p.parse_args()

    # Make every HF surface offline+local BEFORE any HF library import.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _prepare_hub_cache(args.hub_cache_gcs)
    _prepare_offline_dataset_cache(args.dataset_cache_gcs)

    from experiments.scaling_law_sweeps.levanter_lm_eval import run_levanter_lm_eval  # noqa: PLC0415
    from experiments.scaling_law_sweeps.mmlu.mmlu_tasks_set import task_for_shots  # noqa: PLC0415

    task = task_for_shots(args.num_fewshot)
    checkpoint = args.hf_checkpoint.rstrip("/")
    logger.info("MMLU eval: checkpoint=%s task=%s num_fewshot=%d", checkpoint, task.name, task.num_fewshot)

    hf_env = {
        k: os.environ.get(k)
        for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_CACHE", "HF_HOME")
    }
    logger.info("HF offline env: %s", hf_env)

    # lm-eval results.json carries no checkpoint provenance, which made the 2026-09-17
    # stale-checkpoint audit unprovable for old evals. Record what THIS eval scores,
    # before scoring, so verify_eval_integrity can hard-verify every future datapoint.
    from rigging.filesystem import filesystem as marin_filesystem  # noqa: PLC0415

    provenance = {
        "checkpoint": checkpoint,
        "started_at": time.time(),
        "task": task.name,
        "num_fewshot": task.num_fewshot,
    }
    # Sideline any prior attempt's results BEFORE the sidecar lands, so results.json
    # always postdates the sidecar that describes it (verify_eval_integrity's invariant;
    # a preemption between sidecar and results otherwise leaves a mismatched pair).
    prior = f"{args.output_dir.rstrip('/')}/results.json"
    if marin_filesystem("gcs").exists(prior):
        marin_filesystem("gcs").mv(prior, f"{args.output_dir.rstrip('/')}/results.superseded.json")
    with marin_filesystem("gcs").open(f"{args.output_dir.rstrip('/')}/provenance.json", "w") as f:
        json.dump(provenance, f)

    try:
        run_levanter_lm_eval(checkpoint, [task], args.output_dir, args.run_name, max_eval_instances=args.limit)
    except Exception:
        import traceback  # noqa: PLC0415

        from rigging.filesystem import filesystem as marin_filesystem  # noqa: PLC0415

        err = f"HF env: {hf_env}\n\n{traceback.format_exc()}"
        try:
            with marin_filesystem("gcs").open(f"{args.output_dir.rstrip('/')}/_error.txt", "w") as f:
                f.write(err)
        except Exception as write_err:
            logger.error("Failed to write _error.txt: %s", write_err)
        raise
    logger.info("Done; results.json written under %s", args.output_dir)


if __name__ == "__main__":
    main()
