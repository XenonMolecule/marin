# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a canonical, offline HF-datasets cache for the STANDARD 13-entry CORE_TASKS.

This is the sibling of ``dclm_core/build_eval_dataset_cache.py`` but for the
pretraining in-loop suite (`experiments.evals.task_configs.CORE_TASKS`) that
`default_train` / `default_eval` run — NOT the DCLM Core v2 suite. The two differ:
DCLM installs `custom_tasks/` which SHADOWS several standard lm-eval tasks
(`commonsense_qa`, `agieval_lsat_ar`, `wsc273`/`winograd`) with bespoke local-JSONL
reprocessings. Reusing the DCLM cache would therefore evaluate the wrong data for
those tasks. Here we deliberately do NOT install any custom task path, so every
task resolves to its stock lm-eval definition — the exact dataset `default_eval`
requests at eval time.

The cache is populated by instantiating each CORE task through lm-eval's own
TaskManager (which triggers the dataset download into `HF_DATASETS_CACHE`), then
uploaded byte-identically to a GCS prefix. It is mirrored to every regional bucket
so the 245 cross-region eval jobs read datasets locally under `HF_DATASETS_OFFLINE=1`
instead of hammering the HF Hub API (rate-limited at 1000 req/5min per token).

CRITICAL: the cache must be IDENTICAL in every region — the 245 models span 5
regions and their CORE scores are only comparable if every job sees byte-identical
eval data. Snapshot ONCE, copy that exact snapshot everywhere; never re-download
per region.

Usage (as an iris job, clean HF + GCS reachability):

    iris --cluster marin job run --region us-central1 --memory 16GB --enable-extra-resources \\
        --extra eval -e HF_TOKEN <token> \\
        -- python -m experiments.scaling_law_sweeps.core_tasks.build_core_tasks_dataset_cache \\
               --cache-dir /tmp/core_tasks_hf_cache \\
               --upload-to gs://marin-us-central1/eval_datasets/core_tasks_hf_cache/
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def _unique_core_task_names() -> list[str]:
    """Distinct lm-eval task names behind the runnable CORE_TASKS (excludes wsc273,
    whose stock loader is broken under datasets>=3; see core_tasks_set)."""
    from experiments.scaling_law_sweeps.core_tasks.core_tasks_set import unique_dataset_task_names

    return unique_dataset_task_names()


def build_cache(cache_dir: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Instantiate every CORE task with the STOCK lm-eval TaskManager (no custom
    task path) so each dataset downloads into cache_dir.

    Returns (downloaded_ok, failures) where failures is a list of (task, error).
    """
    # Must be set before `datasets` is imported anywhere.
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    # NOTE: intentionally NO _install_custom_task_path() — we want stock lm-eval
    # definitions, matching what default_eval resolves at eval time.
    import lm_eval.tasks as lm_eval_tasks
    from tenacity import retry, stop_after_attempt, wait_exponential

    task_manager = lm_eval_tasks.TaskManager()

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=5, max=120), reraise=True)
    def _cache_one(task_name: str) -> None:
        # get_task_dict instantiates the task, triggering the dataset download
        # into HF_DATASETS_CACHE. Retry rides out transient HF Hub throttling.
        lm_eval_tasks.get_task_dict([task_name], task_manager)

    ok: list[str] = []
    failures: list[tuple[str, str]] = []
    for name in _unique_core_task_names():
        try:
            _cache_one(name)
            logger.info("cached: %s", name)
            ok.append(name)
        except Exception as e:
            logger.warning("FAILED to cache %s: %s", name, e)
            failures.append((name, str(e)))
    return ok, failures


def warm_hub_cache(warm_checkpoint: str) -> None:
    """Populate the HF hub cache (HF_HOME/hub) with the reference model configs that
    levanter's `HFCheckpointConverter.from_hf` model-type probe loads (e.g. `gpt2`).

    from_hf iterates every registered Levanter model and, for some (gpt2), builds a
    converter whose `reference_checkpoint` is a HUB repo id — loading that repo's
    config from the Hub. That probe is unavoidable and Hub-dependent. Running it ONCE
    here (online, retried) caches those configs so all 245 eval jobs resolve them
    OFFLINE from the mirrored cache and never touch the Hub. HF_HOME must already be
    set (in main) before transformers was imported.
    """
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=5, max=120), reraise=True)
    def _warm() -> None:
        from levanter.compat.hf_checkpoints import HFCheckpointConverter

        HFCheckpointConverter.from_hf(warm_checkpoint.rstrip("/"))

    logger.info("Warming HF hub cache via from_hf(%s) ...", warm_checkpoint)
    _warm()
    logger.info("Hub cache warmed at HF_HOME=%s", os.environ.get("HF_HOME"))


def write_report(gcs_dir: str, ok: list[str], failures: list[tuple[str, str]]) -> None:
    """Write an outcome JSON next to the cache so results are readable even when
    the iris log plane is down."""
    import json

    from rigging.filesystem import filesystem

    report = {"ok": ok, "failures": [{"task": t, "error": e[:500]} for t, e in failures]}
    path = f"{gcs_dir.rstrip('/')}/_build_report.json"
    with filesystem("gcs").open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote build report to %s", path)


def upload_cache(cache_dir: str, gcs_dir: str) -> int:
    """Recursively upload the populated cache dir to a GCS prefix. Returns file count."""
    from rigging.filesystem import filesystem

    fs = filesystem("gcs")
    dest = gcs_dir.rstrip("/")
    n = 0
    for root, _dirs, files in os.walk(cache_dir):
        rel = os.path.relpath(root, cache_dir)
        for f in files:
            local = os.path.join(root, f)
            remote = f"{dest}/{f}" if rel == "." else f"{dest}/{rel}/{f}"
            fs.put(local, remote)
            n += 1
    logger.info("Uploaded %d cache files to %s", n, dest)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, help="Local dir to populate as the HF_DATASETS_CACHE snapshot.")
    ap.add_argument(
        "--upload-to", default=None, help="If set, recursively upload the populated dataset cache to this GCS prefix."
    )
    ap.add_argument("--hub-cache-dir", default="/tmp/core_tasks_hf_home", help="Local HF_HOME for the model-config hub cache.")
    ap.add_argument("--warm-checkpoint", default=None, help="A gs:// hf/step-N/ dir; from_hf(it) warms the hub cache.")
    ap.add_argument("--upload-hub-to", default=None, help="If set, upload the warmed HF_HOME hub cache to this GCS prefix.")
    args = ap.parse_args()

    # Set HF_HOME before any transformers import so the model-config hub cache lands
    # in a known, uploadable dir (datasets go to HF_DATASETS_CACHE, set separately).
    os.environ["HF_HOME"] = args.hub_cache_dir

    ok, failures = build_cache(args.cache_dir)
    logger.info("Cached %d/%d CORE task datasets into %s", len(ok), len(ok) + len(failures), args.cache_dir)
    if failures:
        logger.error("Failures (%d):", len(failures))
        for name, err in failures:
            logger.error("  %s: %s", name, err[:200])
    if args.upload_to:
        # Always write the outcome report (readable via gcloud even if the log
        # plane is down), then upload the cache only when every task succeeded —
        # a partial cache dir could carry half-written dataset files.
        write_report(args.upload_to, ok, failures)
        if not failures:
            upload_cache(args.cache_dir, args.upload_to)
        else:
            logger.error("NOT uploading cache: %d task(s) failed; see _build_report.json.", len(failures))

    # Warm + upload the model-config hub cache so eval jobs resolve from_hf's
    # reference configs (gpt2, ...) offline.
    if args.warm_checkpoint:
        warm_hub_cache(args.warm_checkpoint)
        if args.upload_hub_to:
            n = upload_cache(args.hub_cache_dir, args.upload_hub_to)
            logger.info("Uploaded %d hub-cache files to %s", n, args.upload_hub_to)


if __name__ == "__main__":
    main()
