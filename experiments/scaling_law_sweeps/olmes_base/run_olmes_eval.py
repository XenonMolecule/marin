# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the OLMES base-easy suite on one HF checkpoint, offline.

Structurally identical to `core_tasks/run_core_tasks_eval.py` (offline dataset +
hub caches wired in before any HF import, then `marin.evaluation.run.evaluate` with a
`levanter_lm_evaluation_harness` config — the exact `default_eval` worker fn). The
only additions here are:

  * the task set is `OLMES_BASE_EASY_RUNNABLE`;
  * the datasets come from the `olmes_base_hf_cache` prefix, while the model-config
    hub cache is REUSED from `core_tasks_hub_cache` (identical checkpoints);
  * `install_custom_task_path()` is called before `evaluate` so the harness's no-arg
    `TaskManager()` resolves our parquet-backed `social_iqa` (the stock script loader
    is dead under datasets>=3).

Usage (as an iris child; launcher supplies HF_HUB_OFFLINE + region pin):

    python -m experiments.scaling_law_sweeps.olmes_base.run_olmes_eval \\
        --hf-checkpoint gs://.../hf/step-N/ \\
        --output-dir gs://.../metadata/olmes_base_results/<run_name>/ \\
        --run-name <run_name> \\
        --dataset-cache-gcs gs://<same-bucket>/eval_datasets/olmes_base_hf_cache/ \\
        --hub-cache-gcs gs://<same-bucket>/eval_datasets/core_tasks_hub_cache/
"""

from __future__ import annotations

import argparse
import logging
import os

from experiments.scaling_law_sweeps.core_tasks.run_core_tasks_eval import _sync_cache

logger = logging.getLogger(__name__)

_DATASET_CACHE_MARKER = "/eval_datasets/olmes_base_hf_cache/"
_HUB_CACHE_MARKER = "/eval_datasets/core_tasks_hub_cache/"  # reused from the CORE sweep


def _prepare_hub_cache(gcs_cache: str, local_dir: str = "/tmp/olmes_hf_home") -> int:
    # Clear first: reused workers keep stale HF `.no_exist` markers from a prior run whose
    # cache lacked a reference config, and _sync_cache (overwrite-only) never removes them,
    # so huggingface_hub refuses to load that config offline even once it's cached.
    import shutil

    shutil.rmtree(local_dir, ignore_errors=True)
    n = _sync_cache(gcs_cache, _HUB_CACHE_MARKER, local_dir)
    os.environ["HF_HOME"] = local_dir
    logger.info("Synced %d hub-cache files -> %s; HF_HOME set", n, local_dir)
    return n


def _install_resilient_from_hf() -> None:
    """Make Levanter's ``HFCheckpointConverter.from_hf`` offline-resilient.

    ``from_hf`` probes EVERY registered config's default HF reference to name-match the
    checkpoint's arch; under ``HF_HUB_OFFLINE`` a NON-matching config whose default repo
    isn't in the local cache (gpt2/gemma/... — which the eval container's transformers can't
    resolve offline even when cached) crashes the probe. Fall back to resolving the config
    directly from the checkpoint's own ``model_type`` and building the converter against the
    checkpoint itself (no default-repo fetch).
    """
    from levanter.compat import hf_checkpoints as _hc

    _orig = _hc.HFCheckpointConverter.from_hf

    def _resilient(model_name_or_path, trust_remote_code: bool = False):
        try:
            return _orig(model_name_or_path, trust_remote_code)
        except Exception:
            import json

            import fsspec

            from levanter.models.lm_model import LmConfig

            ckpt = str(model_name_or_path).rstrip("/")
            with fsspec.open(f"{ckpt}/config.json", "r") as f:
                model_type = json.load(f)["model_type"]
            return LmConfig.get_known_choices()[model_type]().hf_checkpoint_converter(ref_checkpoint=ckpt)

    _hc.HFCheckpointConverter.from_hf = staticmethod(_resilient)


def _prepare_offline_dataset_cache(gcs_cache: str, local_dir: str = "/tmp/olmes_hf_cache") -> int:
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
    p.add_argument("--dataset-cache-gcs", required=True, help="GCS prefix of the OLMES HF-datasets cache in THIS region.")
    p.add_argument("--hub-cache-gcs", required=True, help="GCS prefix of the model-config hub cache in THIS region.")
    p.add_argument("--limit", type=int, default=None, help="Cap each task to N examples (smoke test). None = full.")
    args = p.parse_args()

    # Model config/tokenizer load ONLINE (like CORE v2): the container's transformers can't
    # reliably resolve a few repos (gpt2, mistral-regex fix) OFFLINE. The DATASETS stay offline
    # (_prepare_offline_dataset_cache sets HF_DATASETS_OFFLINE=1) so the staged eval data is used
    # without re-download. HF_HOME (set in _prepare_hub_cache) still accelerates hub lookups.
    _prepare_hub_cache(args.hub_cache_gcs)
    _prepare_offline_dataset_cache(args.dataset_cache_gcs)

    # Shadow the dead stock social_iqa with our parquet-backed YAML. Must run before
    # evaluate() constructs the harness's TaskManager().
    from experiments.scaling_law_sweeps.olmes_base.olmes_custom_tasks import install_custom_task_path

    install_custom_task_path()

    from fray.cluster import ResourceConfig
    from marin.evaluation.evaluation_config import EvaluationConfig
    from marin.evaluation.run import evaluate

    from experiments.scaling_law_sweeps.olmes_base.olmes_tasks_set import OLMES_BASE_EASY_RUNNABLE

    checkpoint = args.hf_checkpoint.rstrip("/")
    logger.info("OLMES eval: checkpoint=%s tasks=%d", checkpoint, len(OLMES_BASE_EASY_RUNNABLE))

    hf_env = {k: os.environ.get(k) for k in
              ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_CACHE", "HF_HOME")}
    logger.info("HF offline env: %s", hf_env)

    _install_resilient_from_hf()  # evaluate() loads the model via from_hf's offline-fragile scan

    config = EvaluationConfig(
        evaluator="levanter_lm_evaluation_harness",
        resource_config=ResourceConfig.with_tpu("v5p-8"),
        model_name=args.run_name,
        model_path=checkpoint,
        evaluation_path=args.output_dir.rstrip("/"),
        evals=list(OLMES_BASE_EASY_RUNNABLE),
        discover_latest_checkpoint=False,
        max_eval_instances=args.limit,
    )
    try:
        evaluate(config)
    except Exception:
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
