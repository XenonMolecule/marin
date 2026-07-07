# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a canonical, offline HF-datasets cache for the 22 DCLM CORE tasks.

Populates a local `HF_DATASETS_CACHE` directory by instantiating each CORE task
through lm-eval's OWN task machinery — so the cached dataset fingerprints match
exactly what `run_dclm_core_eval.py` requests at eval time. That cache is then
mirrored byte-identically to every regional bucket (see
`mirror_eval_dataset_cache.py`), and the eval runs with `HF_DATASETS_OFFLINE=1`,
reading datasets from the in-region cache instead of hammering the HF Hub API
(which rate-limits at 1000 requests / 5 min per token and forced us to throttle
launches).

CRITICAL: the cache must be IDENTICAL in every region — the 245 models are
evaluated across 6 regions and their CORE scores are only comparable if every
job sees byte-identical eval data. So we snapshot ONCE here and copy that exact
snapshot everywhere; we never re-download per region.

Usage (local, needs HF reachability once):

    HF_TOKEN=... python -m experiments.scaling_law_sweeps.dclm_core.build_eval_dataset_cache \\
        --cache-dir /tmp/dclm_core_hf_cache
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)


def _unique_lm_eval_tasks() -> list[str]:
    """The distinct lm-eval task names behind the 22 CORE tasks (shot count does
    not change the dataset, so we dedupe on task name)."""
    from experiments.scaling_law_sweeps.dclm_core.task_mapping import CORE_TASK_MAP

    seen: dict[str, None] = {}
    for entry in CORE_TASK_MAP:
        seen.setdefault(entry.lm_eval, None)
    return list(seen)


def build_cache(cache_dir: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Instantiate every CORE task so its dataset downloads into cache_dir.

    Returns (downloaded_ok, failures) where failures is a list of (task, error).
    """
    # Must be set before `datasets` is imported anywhere.
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    from experiments.scaling_law_sweeps.dclm_core.run_dclm_core_eval import _install_custom_task_path

    _install_custom_task_path()

    import lm_eval.tasks as lm_eval_tasks

    task_manager = lm_eval_tasks.TaskManager()

    ok: list[str] = []
    failures: list[tuple[str, str]] = []
    for name in _unique_lm_eval_tasks():
        try:
            # get_task_dict instantiates the task, which triggers the dataset
            # download into HF_DATASETS_CACHE.
            lm_eval_tasks.get_task_dict([name], task_manager)
            logger.info("cached: %s", name)
            ok.append(name)
        except Exception as e:
            logger.warning("FAILED to cache %s: %s", name, e)
            failures.append((name, str(e)))
    return ok, failures


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
        "--upload-to", default=None, help="If set, recursively upload the populated cache to this GCS prefix."
    )
    args = ap.parse_args()

    ok, failures = build_cache(args.cache_dir)
    logger.info("Cached %d/%d CORE task datasets into %s", len(ok), len(ok) + len(failures), args.cache_dir)
    if failures:
        logger.error("Failures (%d):", len(failures))
        for name, err in failures:
            logger.error("  %s: %s", name, err[:200])
    if args.upload_to and not failures:
        upload_cache(args.cache_dir, args.upload_to)
    elif args.upload_to:
        logger.error("NOT uploading: %d task(s) failed to cache; fix before mirroring.", len(failures))


if __name__ == "__main__":
    main()
