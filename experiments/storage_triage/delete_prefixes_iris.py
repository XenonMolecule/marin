#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delete a reviewed list of GCS prefixes as an Iris job (survives laptop disconnect).

The list (one ``gs://bucket/prefix/`` per line, produced by ``build_delete_manifest.py``)
is uploaded to ``--log-dir`` first; the job then deletes each prefix recursively with
gcsfs and writes progress as numbered ``<log-dir>/done-NNNNN.txt`` / ``failed-NNNNN.txt`` chunks
(GCS has no append; resumable: prefixes already in any done chunk are skipped on relaunch). Same protection guard as the shell
scripts: the job refuses to start if any line matches a protected pattern.

Usage:
    uv run experiments/storage_triage/delete_prefixes_iris.py --list rm_pilot.txt \
        --log-dir gs://marin-us-central2/scratch/michaelryan/storage_triage/deletes/pilot
"""

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import fsspec
import gcsfs
from iris.cli.connect import open_iris_client
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTECTED_RE = re.compile(
    r"lpv11_fastpipe_v1|olmix-swarm/[^/]*resiliparse|mb-clf-lpv11|pooled|classifiers/|presharded|_clf_token_cache"
    r"|datasets/(high_quality_3000_distill|llm_pipeline_v1_1_3000_distill)/|/tmp/|documents/baseline_llm_extraction/"
    r"|kept_text|tombstones|/deduped/$"
)


def _delete_stage(list_url: str, log_dir: str, workers: int) -> None:
    fs = gcsfs.GCSFileSystem()
    with fsspec.open(list_url, "r") as f:
        prefixes = [line.strip() for line in f if line.strip()]
    bad = [p for p in prefixes if PROTECTED_RE.search(p) or not p.startswith("gs://marin-")]
    if bad:
        raise RuntimeError(f"REFUSING: {len(bad)} protected/invalid lines, e.g. {bad[:3]}")
    # GCS objects are immutable (no append): progress is written as numbered chunk files.
    already: set[str] = set()
    for chunk in fs.glob(f"{log_dir.replace('gs://', '')}/done-*.txt"):
        with fs.open(chunk, "r") as f:
            already |= {line.split(" ", 1)[-1].strip() for line in f if line.strip()}
    seq = [len(fs.glob(f"{log_dir.replace('gs://', '')}/done-*.txt"))]
    todo = [p for p in prefixes if p not in already]
    print(f"{len(prefixes)} prefixes, {len(already)} already done, {len(todo)} to delete", file=sys.stderr)

    done_lines: list[str] = []
    failed_lines: list[str] = []

    def one(p: str) -> None:
        path = p.replace("gs://", "")
        try:
            if fs.exists(path):
                fs.rm(path, recursive=True)
            done_lines.append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {p}")
        except Exception as e:  # noqa: BLE001 - record and continue; the failed list is the retry input
            failed_lines.append(f"{p}\t{type(e).__name__}: {e}")

    def flush() -> None:
        if done_lines:
            with fsspec.open(f"{log_dir}/done-{seq[0]:05d}.txt", "w") as f:
                f.write("\n".join(done_lines) + "\n")
            done_lines.clear()
        if failed_lines:
            with fsspec.open(f"{log_dir}/failed-{seq[0]:05d}.txt", "w") as f:
                f.write("\n".join(failed_lines) + "\n")
            failed_lines.clear()
        seq[0] += 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0:
                flush()
                print(f"progress {i}/{len(todo)}", file=sys.stderr)
    flush()
    print("done", file=sys.stderr)


@click.command()
@click.option("--list", "list_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--log-dir", required=True, help="gs:// dir for the uploaded list + done/failed logs")
@click.option("--workers", default=4, show_default=True)
@click.option("--cluster", default="marin", show_default=True)
def main(list_path: Path, log_dir: str, workers: int, cluster: str) -> None:
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    bad = [p for p in lines if PROTECTED_RE.search(p)]
    if bad:
        raise SystemExit(f"REFUSING locally: protected lines {bad[:3]}")
    list_url = f"{log_dir.rstrip('/')}/{list_path.name}"
    fs, _ = fsspec.core.url_to_fs(list_url)
    fs.put_file(str(list_path), list_url)
    print(f"uploaded {len(lines)} prefixes -> {list_url}", file=sys.stderr)
    with open_iris_client(cluster_name=cluster, workspace=REPO_ROOT) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_delete_stage, list_url, log_dir.rstrip("/"), workers),
            name=f"storage-delete-{list_path.stem}",
            resources=ResourceSpec(cpu=2, memory="4GB", disk="10GB"),
            environment=EnvironmentSpec(env_vars={}),
        )
        print(f"Submitted {job.job_id}; logs: {log_dir}/done-*.txt, failed-*.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
