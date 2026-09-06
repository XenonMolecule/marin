# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the STANDARD pretraining CORE_TASKS on one HF checkpoint, offline.

This is the standalone, rate-limit-proof equivalent of `default_eval` for the
in-loop CORE_TASKS suite. It does NOT reimplement the harness: after wiring the
offline dataset cache into the environment, it hands a `levanter_lm_evaluation_harness`
`EvaluationConfig` to `marin.evaluation.run.evaluate` — the exact worker function
`default_eval`'s ExecutorStep invokes. So the tasks, shot counts, prompts, scoring
and wandb logging are identical to the in-loop suite; the only additions are:

  * `_prepare_offline_dataset_cache`: sync the in-region CORE_TASKS HF-datasets
    cache to local disk and set `HF_DATASETS_OFFLINE=1` BEFORE anything imports
    `datasets`, so task loading never hits the HF Hub.
  * `HF_HUB_OFFLINE=1` (set by the launcher) makes transformers load the model
    config/tokenizer from the local checkpoint instead of doing a Hub metadata
    check — the two together make the eval immune to HF-token rate limiting.

The task set is `CORE_TASKS_RUNNABLE` (the 12 in-loop entries that load under
marin's datasets>=3 pin; wsc273's stock script-based loader is excluded — see
core_tasks_set).

Usage (as an iris child; the launcher supplies HF_HUB_OFFLINE + region pin):

    python -m experiments.scaling_law_sweeps.core_tasks.run_core_tasks_eval \\
        --hf-checkpoint gs://.../hf/step-N/ \\
        --output-dir gs://.../data_curation_10k_core_tasks_results/<run_name>/ \\
        --run-name <run_name> \\
        --dataset-cache-gcs gs://<same-bucket>/eval_datasets/core_tasks_hf_cache/
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)

_DATASET_CACHE_MARKER = "/eval_datasets/core_tasks_hf_cache/"
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"


def _sync_cache(gcs_cache: str, marker: str, local_dir: str) -> int:
    """Sync a GCS cache prefix to a local dir, preserving the sub-tree after `marker`."""
    from rigging.filesystem import filesystem as marin_filesystem

    base = gcs_cache.rstrip("/")
    fs = marin_filesystem("gcs")
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for remote in fs.find(base):
        key = remote.split(marker, 1)[-1]
        dest = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = remote if remote.startswith("gs://") else f"gs://{remote}"
        with fs.open(src, "rb") as r, open(dest, "wb") as w:
            w.write(r.read())
        n += 1
    return n


def _prepare_hub_cache(gcs_cache: str, local_dir: str = "/tmp/core_tasks_hf_home") -> int:
    """Sync the model-config hub cache and point HF_HOME at it, so from_hf's
    reference-config probe (gpt2, ...) resolves offline. Must run before any
    transformers import."""
    n = _sync_cache(gcs_cache, _HUB_CACHE_MARKER, local_dir)
    os.environ["HF_HOME"] = local_dir
    logger.info("Synced %d hub-cache files -> %s; HF_HOME set", n, local_dir)
    return n


def _prepare_offline_dataset_cache(gcs_cache: str, local_dir: str = "/tmp/core_tasks_hf_cache") -> int:
    """Sync the canonical CORE_TASKS HF-datasets cache from in-region GCS to local
    disk and switch `datasets` to OFFLINE. Must run before any `datasets`/lm-eval
    import reads the env. Returns files synced.

    Mirrors dclm_core.run_dclm_core_eval._prepare_offline_dataset_cache but for the
    standard CORE_TASKS cache path.
    """
    from rigging.filesystem import filesystem as marin_filesystem

    base = gcs_cache.rstrip("/")
    fs = marin_filesystem("gcs")
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    for remote in fs.find(base):
        key = remote.split(_DATASET_CACHE_MARKER, 1)[-1]
        dest = os.path.join(local_dir, key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        src = remote if remote.startswith("gs://") else f"gs://{remote}"
        with fs.open(src, "rb") as r, open(dest, "wb") as w:
            w.write(r.read())
        n += 1
    os.environ["HF_DATASETS_CACHE"] = local_dir
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    logger.info("Synced %d dataset-cache files %s -> %s; HF_DATASETS_OFFLINE=1", n, base, local_dir)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--hf-checkpoint", required=True, help="HF checkpoint dir (gs://...); contains config.json + safetensors."
    )
    p.add_argument(
        "--output-dir", required=True, help="GCS dir to write results.json into (the levanter evaluator's output_path)."
    )
    p.add_argument("--run-name", default=None, help="Run name for wandb / imputed model name.")
    p.add_argument(
        "--dataset-cache-gcs",
        required=True,
        help="GCS prefix of the CORE_TASKS HF-datasets cache in THIS checkpoint's region.",
    )
    p.add_argument(
        "--hub-cache-gcs",
        required=True,
        help="GCS prefix of the CORE_TASKS model-config hub cache (gpt2 + reference configs) in THIS region.",
    )
    p.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke test). None = full eval.")
    args = p.parse_args()

    # Make EVERY HF surface offline+local BEFORE any HF library is imported, so the
    # eval is fully hermetic and immune to shared-token rate limiting:
    #   * TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE — no Hub metadata checks.
    #   * _prepare_hub_cache sets HF_HOME to the synced model-config cache so
    #     from_hf's reference-config probe (gpt2, ...) resolves offline.
    #   * _prepare_offline_dataset_cache sets HF_DATASETS_CACHE + HF_DATASETS_OFFLINE.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    _prepare_hub_cache(args.hub_cache_gcs)
    _prepare_offline_dataset_cache(args.dataset_cache_gcs)

    from fray.cluster import ResourceConfig
    from marin.evaluation.evaluation_config import EvaluationConfig
    from marin.evaluation.run import evaluate

    from experiments.scaling_law_sweeps.core_tasks.core_tasks_set import CORE_TASKS_RUNNABLE

    checkpoint = args.hf_checkpoint.rstrip("/")
    logger.info("CORE_TASKS eval: checkpoint=%s tasks=%d", checkpoint, len(CORE_TASKS_RUNNABLE))

    hf_env = {
        k: os.environ.get(k)
        for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_CACHE", "HF_TOKEN")
    }
    hf_env["HF_TOKEN"] = "set" if hf_env["HF_TOKEN"] else None
    logger.info("HF offline env: %s", hf_env)

    config = EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        # resource_config is only consumed by the @remote decorator at step build
        # time; at runtime (here) it is inert, but the dataclass requires it.
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        model_name=args.run_name,  # None -> imputed from path, matching default_eval
        model_path=checkpoint,
        evaluation_path=args.output_dir.rstrip("/"),
        evals=list(CORE_TASKS_RUNNABLE),
        discover_latest_checkpoint=False,  # launcher already resolves the final step dir
        max_eval_instances=args.limit,
    )
    try:
        evaluate(config)
    except Exception:
        # The iris log plane is unreliable; persist the full traceback + env to GCS
        # so failures are diagnosable regardless.
        import traceback

        from rigging.filesystem import filesystem as marin_filesystem

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
