# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the cultural suite (`cultural_sl_verb`) on one HF checkpoint, offline.

Structurally identical to `mmlu/run_mmlu_eval.py`: hub cache synced and HF forced
offline before any HF import, then `levanter_lm_eval.run_levanter_lm_eval` (Levanter's
TPU-native harness, in-process). Differences:

  * the dataset cache is the staged parquet under `eval_datasets/cultural_sl_verb/`, and
    the task YAMLs are generated here with absolute `data_files` paths, then registered
    via `lm_eval_include_path` so the harness's no-arg `TaskManager()` finds them;
  * the model-config hub cache is REUSED from `core_tasks_hub_cache` (same checkpoints).

Usage (as an iris child; the launcher supplies the region pin):

    python -m experiments.scaling_law_sweeps.cultural.run_cultural_eval \\
        --hf-checkpoint gs://.../hf/step-N/ \\
        --output-dir gs://.../metadata/cultural_sl_verb_results/5shot/<run_name>/ \\
        --run-name <run_name> --num-fewshot 5 \\
        --dataset-cache-gcs gs://<same-bucket>/eval_datasets/cultural_sl_verb/ \\
        --hub-cache-gcs gs://<same-bucket>/eval_datasets/core_tasks_hub_cache/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import traceback

from rigging.filesystem import filesystem as marin_filesystem

from experiments.scaling_law_sweeps.core_tasks.run_core_tasks_eval import _sync_cache

logger = logging.getLogger(__name__)

_DATASET_CACHE_MARKER = "/eval_datasets/cultural_sl_verb/"
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep
_LOCAL_DATA_DIR = "/tmp/cultural_sl_verb"
_LOCAL_TASKS_DIR = "/tmp/cultural_sl_verb_tasks"


def _prepare_hub_cache(gcs_cache: str, local_dir: str = "/tmp/cultural_hf_home") -> int:
    n = _sync_cache(gcs_cache, _HUB_CACHE_MARKER, local_dir)
    os.environ["HF_HOME"] = local_dir
    logger.info("Synced %d hub-cache files -> %s; HF_HOME set", n, local_dir)
    return n


def _prepare_staged_data(gcs_cache: str, local_dir: str = _LOCAL_DATA_DIR) -> int:
    n = _sync_cache(gcs_cache, _DATASET_CACHE_MARKER, local_dir)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    logger.info("Synced %d staged files -> %s; HF_DATASETS_OFFLINE=1", n, local_dir)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF checkpoint dir (gs://...); config.json + safetensors.")
    p.add_argument("--output-dir", required=True, help="GCS dir to write results.json into.")
    p.add_argument("--run-name", default=None, help="Run name for wandb / imputed model name.")
    p.add_argument("--num-fewshot", type=int, required=True, help="Shot count (5 is the only blessed value).")
    p.add_argument("--dataset-cache-gcs", required=True, help="GCS prefix of the staged suite in THIS region.")
    p.add_argument("--hub-cache-gcs", required=True, help="GCS prefix of the model-config hub cache in THIS region.")
    p.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke test). None = full.")
    args = p.parse_args()

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _prepare_hub_cache(args.hub_cache_gcs)
    _prepare_staged_data(args.dataset_cache_gcs)

    # Everything below imports transformers / lm_eval transitively; the offline env and
    # HF_HOME above must be in place first, hence the function-local imports.
    from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import (  # noqa: PLC0415
        task_for_shots,
        write_task_yamls,
    )
    from experiments.scaling_law_sweeps.lm_eval_include_path import install_include_paths  # noqa: PLC0415

    written = write_task_yamls(_LOCAL_DATA_DIR, _LOCAL_TASKS_DIR)
    install_include_paths(_LOCAL_TASKS_DIR)
    logger.info("Generated %d task YAMLs under %s", len(written), _LOCAL_TASKS_DIR)

    from experiments.scaling_law_sweeps.levanter_lm_eval import run_levanter_lm_eval  # noqa: PLC0415

    task = task_for_shots(args.num_fewshot)
    checkpoint = args.hf_checkpoint.rstrip("/")
    logger.info("Cultural eval: checkpoint=%s task=%s num_fewshot=%d", checkpoint, task.name, task.num_fewshot)

    hf_env = {
        k: os.environ.get(k)
        for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_CACHE", "HF_HOME")
    }
    logger.info("HF offline env: %s", hf_env)

    # Same provenance sidecar as run_mmlu_eval: record what this eval scores, before
    # scoring, so verify_eval_integrity can hard-verify the datapoint.
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
