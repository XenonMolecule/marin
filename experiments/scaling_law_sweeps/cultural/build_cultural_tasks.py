# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the cultural suite: download the four pinned sources, apply the cleaning rules,
write `<output-dir>/<task>/{dev,test}.parquet` + `_manifest.json`, optionally upload.

Sibling of `mmlu/build_mmlu_dataset_cache.py`, but the staged artifact is plain parquet
rather than an HF datasets cache: the task YAMLs generated at eval time point at these
files directly (`cultural_tasks_set.write_task_yamls`), so eval children never touch the
Hub for data. The `_manifest.json` records source revisions and per-task document counts.

Usage (CPU job; reads HF + GitHub only):

    iris --cluster marin job run --region us-east5 --cpu 2 --memory 8GB --enable-extra-resources \\
        --extra eval -e HF_TOKEN "$HF_TOKEN" \\
        -- python -m experiments.scaling_law_sweeps.cultural.build_cultural_tasks \\
               --output-dir /tmp/cultural_sl_verb \\
               --upload-to gs://marin-us-east5/eval_datasets/cultural_sl_verb/ \\
                           gs://marin-us-central1/eval_datasets/cultural_sl_verb/ ...
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import urllib.request
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download
from rigging.filesystem import filesystem

from experiments.scaling_law_sweeps.cultural.cultural_sources import (
    BLEND_REVISION,
    CULTURALBENCH_FILE,
    CULTURALBENCH_HF_REPO,
    CULTURALBENCH_REVISION,
    FORK_URL,
    NORMAD_FILE,
    NORMAD_HF_REPO,
    NORMAD_REVISION,
    SOURCE_PINS,
    CulturalTask,
    blend_tasks,
    culturalbench_task,
    fork_task,
    normad_task,
)
from experiments.scaling_law_sweeps.cultural.cultural_tasks_set import (
    DEV_SPLIT,
    MANIFEST_FILENAME,
    NUM_SHOTS,
    TASK_SPECS,
    TEST_SPLIT,
)
from experiments.scaling_law_sweeps.olmo_bpb.build_blend_bpb_requests import BLEND_HF_REPO, BLEND_MC_FILE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceRows:
    blend: list[dict]
    culturalbench: list[dict]
    fork: list[dict]
    normad: list[dict]


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def download_sources() -> SourceRows:
    blend_path = hf_hub_download(BLEND_HF_REPO, BLEND_MC_FILE, repo_type="dataset", revision=BLEND_REVISION)
    with open(blend_path, encoding="utf-8") as f:
        blend = json.load(f)
    cb_path = hf_hub_download(
        CULTURALBENCH_HF_REPO, CULTURALBENCH_FILE, repo_type="dataset", revision=CULTURALBENCH_REVISION
    )
    normad_path = hf_hub_download(NORMAD_HF_REPO, NORMAD_FILE, repo_type="dataset", revision=NORMAD_REVISION)
    with urllib.request.urlopen(FORK_URL, timeout=60) as resp:
        fork = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]
    return SourceRows(blend=blend, culturalbench=_read_csv(cb_path), fork=fork, normad=_read_csv(normad_path))


def build_all(sources: SourceRows) -> list[CulturalTask]:
    """Every task of the suite, in `TASK_SPECS` order; fails if any spec is left unbuilt."""
    built = {t.spec.name: t for t in blend_tasks(sources.blend)}
    for task in (culturalbench_task(sources.culturalbench), fork_task(sources.fork), normad_task(sources.normad)):
        built[task.spec.name] = task
    missing = [s.name for s in TASK_SPECS if s.name not in built]
    if missing:
        raise ValueError(f"Unbuilt tasks: {missing}")
    return [built[s.name] for s in TASK_SPECS]


def write_task(output_dir: str, task: CulturalTask) -> dict:
    """Write dev/test parquet for one task; returns its manifest entry."""
    dev, test = task.split()
    task_dir = os.path.join(output_dir, task.spec.name)
    os.makedirs(task_dir, exist_ok=True)
    for split, docs in ((DEV_SPLIT, dev), (TEST_SPLIT, test)):
        pd.DataFrame([d.as_record() for d in docs]).to_parquet(os.path.join(task_dir, f"{split}.parquet"), index=False)
    return {
        "benchmark": task.spec.benchmark,
        "num_choices": task.spec.num_choices,
        "n_dev": len(dev),
        "n_test": len(test),
        "fewshot_ids": [d.native_id for d in dev],
        "countries": sorted({d.country for d in test}),
    }


def write_all(output_dir: str, tasks: list[CulturalTask]) -> dict:
    manifest = {
        "num_shots": NUM_SHOTS,
        "sources": SOURCE_PINS,
        "tasks": {t.spec.name: write_task(output_dir, t) for t in tasks},
    }
    with open(os.path.join(output_dir, MANIFEST_FILENAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def upload(output_dir: str, gcs_prefix: str) -> int:
    fs = filesystem("gcs")
    dest = gcs_prefix.rstrip("/")
    n = 0
    for root, _dirs, files in os.walk(output_dir):
        rel = os.path.relpath(root, output_dir)
        for name in files:
            remote = f"{dest}/{name}" if rel == "." else f"{dest}/{rel}/{name}"
            fs.put(os.path.join(root, name), remote)
            n += 1
    logger.info("Uploaded %d files to %s", n, dest)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="Local dir to write <task>/{dev,test}.parquet into.")
    ap.add_argument("--upload-to", nargs="*", default=[], help="GCS prefixes (one per region) to mirror the dir to.")
    args = ap.parse_args()

    tasks = build_all(download_sources())
    manifest = write_all(args.output_dir, tasks)
    for name, entry in manifest["tasks"].items():
        logger.info("%-40s dev=%d test=%d", name, entry["n_dev"], entry["n_test"])
    logger.info("Total test docs: %d", sum(e["n_test"] for e in manifest["tasks"].values()))
    for prefix in args.upload_to:
        upload(args.output_dir, prefix)


if __name__ == "__main__":
    main()
