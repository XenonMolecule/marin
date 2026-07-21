# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the offline HF-datasets cache for the MMLU (`mmlu_sl_verb`) suite.

Sibling of `core_tasks/build_core_tasks_dataset_cache.py` and
`olmes_base/build_olmes_dataset_cache.py`. Differences:
  * Task set is `mmlu_tasks_set` -> the single group name `mmlu_sl_verb`, which fans
    out to 56 subject configs of `hails/mmlu_no_train` (parquet, no loading script).
  * NO custom_tasks include_path: unlike OLMES's social_iqa, nothing here needs
    shadowing — sl_verb's dataset is already parquet.
  * NO model-config hub warming: the eval REUSES the already-mirrored
    `core_tasks_hub_cache` (same checkpoints -> same reference configs). Build that
    with the core builder if it is ever missing.

The datasets download into `HF_DATASETS_CACHE`, then the snapshot is uploaded
byte-identically to a GCS prefix and mirrored to every regional bucket, so eval jobs
read datasets locally under `HF_DATASETS_OFFLINE=1` instead of hammering the Hub.

Usage (iris job; needs torch -> `--extra tpu --extra eval`):

    iris --cluster marin job run --region us-east5 --memory 16GB --enable-extra-resources \\
        --extra tpu --extra eval -e HF_TOKEN <token> \\
        -- python -m experiments.scaling_law_sweeps.mmlu.build_mmlu_dataset_cache \\
               --cache-dir /tmp/mmlu_hf_cache \\
               --upload-to gs://marin-us-east5/eval_datasets/mmlu_hf_cache/
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def build_cache(cache_dir: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Instantiate every MMLU task through lm-eval's TaskManager so each subject
    config's parquet downloads into cache_dir.

    Returns (downloaded_ok, failures) where failures is a list of (task, error).
    """
    # Must be set before `datasets` is imported anywhere.
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    import lm_eval.tasks as lm_eval_tasks
    from tenacity import retry, stop_after_attempt, wait_exponential

    from experiments.scaling_law_sweeps.mmlu.mmlu_tasks_set import unique_dataset_task_names

    task_manager = lm_eval_tasks.TaskManager()

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=5, max=120), reraise=True)
    def _cache_one(task_name: str) -> None:
        lm_eval_tasks.get_task_dict([task_name], task_manager)

    ok: list[str] = []
    failures: list[tuple[str, str]] = []
    for name in unique_dataset_task_names():
        try:
            _cache_one(name)
            logger.info("cached: %s", name)
            ok.append(name)
        except Exception as e:
            logger.warning("FAILED to cache %s: %s", name, e)
            failures.append((name, str(e)))
    return ok, failures


def write_report(gcs_dir: str, ok: list[str], failures: list[tuple[str, str]]) -> None:
    import json

    from rigging.filesystem import filesystem

    report = {"ok": ok, "failures": [{"task": t, "error": e[:500]} for t, e in failures]}
    path = f"{gcs_dir.rstrip('/')}/_build_report.json"
    with filesystem("gcs").open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote build report to %s", path)


def upload_cache(cache_dir: str, gcs_dir: str) -> int:
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
    ap.add_argument("--upload-to", default=None, help="If set, recursively upload the populated dataset cache here.")
    args = ap.parse_args()

    ok, failures = build_cache(args.cache_dir)
    logger.info("Cached %d/%d MMLU task datasets into %s", len(ok), len(ok) + len(failures), args.cache_dir)
    if failures:
        logger.error("Failures (%d):", len(failures))
        for name, err in failures:
            logger.error("  %s: %s", name, err[:200])
    if args.upload_to:
        write_report(args.upload_to, ok, failures)
        if not failures:
            upload_cache(args.cache_dir, args.upload_to)
        else:
            logger.error("NOT uploading cache: %d task(s) failed; see _build_report.json.", len(failures))


if __name__ == "__main__":
    main()
