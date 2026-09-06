#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run several reviewed delete lists back-to-back as ONE Iris job, with verification gates.

For each batch, in order:
  1. STOP check — if ``<log-root>/STOP`` exists the chain halts before starting the batch
     (and it re-checks every 25 prefixes while deleting).
  2. Guard — refuse if any line in the list matches a protected pattern.
  3. Delete every prefix (recursive), recording ``done-*.txt`` / ``failed-*.txt`` chunks.
  4. Verify — every listed prefix is gone; every *witness* path for the batch still exists
     (checkpoint batches: the kept final ``hf/step-<max>`` of each touched run; data batches:
     the sibling that must survive, e.g. ``kept_text/``, ``deduped/``, the kept cache copy).
  5. Write ``status.json`` (per-batch counts + verification). Any failure or missing witness
     writes ``ABORT`` and stops the chain — later batches never start.

The witnesses are computed locally at submit time (from ``keep_final.tsv`` /
``tokenized_mirror_plan.csv`` / list structure) and uploaded with the lists, so the job
verifies against a frozen, reviewed set.

Usage:
    uv run experiments/storage_triage/chain_delete_iris.py --manifest-dir <dir> \
        --log-root gs://marin-us-central2/scratch/michaelryan/storage_triage/deletes/chain_<date> \
        --batch ckpt_intermediates_rest --batch ckpt_whole_runs --batch fast_curation ...
Stop it:  create ``<log-root>/STOP`` (any content).
"""

import csv
import json
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
STEP_DIR_RE = re.compile(r"^(gs://.+?)/(hf|checkpoints)/step-\d+/$")


# ---------------------------------------------------------------- witnesses (submit-side)


def _keep_finals(manifest_dir: Path) -> dict[str, str]:
    """run url -> kept final path (hf/step-N or checkpoints/step-N), from keep_final.tsv."""
    out: dict[str, str] = {}
    for line in (manifest_dir / "keep_final.tsv").read_text().splitlines():
        if not line.strip():
            continue
        u = line.split("\t")[0]
        m = re.match(r"(gs://.+?)/(hf|checkpoints)/step-\d+$", u)
        if m:
            out[m.group(1)] = u
    return out


def witnesses_for(batch: str, prefixes: list[str], manifest_dir: Path) -> list[str]:
    """Paths that MUST still exist after the batch (checked with fs.exists)."""
    finals = _keep_finals(manifest_dir)
    w: set[str] = set()
    if batch.startswith("ckpt_intermediates"):
        for p in prefixes:
            m = STEP_DIR_RE.match(p)
            if not m:
                raise ValueError(f"unexpected shape in {batch}: {p}")
            run = m.group(1)
            fin = finals[run]  # KeyError = a run without a kept final: refuse at submit time
            w.add(fin + "/config.json" if "/hf/" in fin else fin)
    elif batch == "ckpt_whole_runs":
        # Orphans: the DONE final of the same run in another bucket must survive.
        by_run: dict[str, list[str]] = {}
        for run_url, fin in finals.items():
            by_run.setdefault(re.sub(r"^gs://marin-[a-z0-9-]+/", "", run_url), []).append(fin)
        for p in prefixes:
            rel = re.sub(r"^gs://marin-[a-z0-9-]+/", "", p.rstrip("/"))
            for fin in by_run.get(rel, []):
                if not fin.startswith(p.rstrip("/")):
                    w.add(fin + "/config.json" if "/hf/" in fin else fin)
    elif batch == "fast_curation":
        for p in prefixes:
            ns = re.match(r"^(gs://[^/]+/documents/fast_curation/fastpipe_v3-da3893385e/)", p)
            if ns:
                w.add(ns.group(1) + "kept_text/")
                w.add(ns.group(1) + "kept/")
    elif batch.startswith("dedup_intermediates"):
        for p in prefixes:
            m = re.match(r"^(gs://[^/]+/documents/[^/]+/\d+warcs/)", p)
            if not m:
                raise ValueError(f"unexpected shape in {batch}: {p}")
            w.add(m.group(1) + "deduped/")
    elif batch == "tokenized_mirrors":
        plan = list(csv.DictReader(open(manifest_dir / "tokenized_mirror_plan.csv")))
        keep_by_cache = {r["cache"]: r["keep_bucket"].split(",") for r in plan}
        for p in prefixes:
            m = re.match(r"^gs://marin-[a-z0-9-]+/tokenized/([^/]+)/$", p)
            if not m:
                raise ValueError(f"unexpected shape in {batch}: {p}")
            for b in keep_by_cache.get(m.group(1), []):
                if b:
                    w.add(f"gs://marin-{b}/tokenized/{m.group(1)}/")
    elif batch.startswith("random_pool_and_negatives"):
        pass
    else:
        raise ValueError(f"unknown batch {batch}")
    return sorted(w)


# ---------------------------------------------------------------- Iris-side chain


def _chain_stage(log_root: str, batches: list[str], workers: int) -> None:
    fs = gcsfs.GCSFileSystem()
    root = log_root.replace("gs://", "")
    status_url = f"{log_root}/status.json"
    status: dict = {"batches": {}, "state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def save() -> None:
        with fsspec.open(status_url, "w") as f:
            json.dump(status, f, indent=1)

    def stop_requested() -> bool:
        return fs.exists(f"{root}/STOP")

    def abort(msg: str) -> None:
        status["state"] = "ABORTED"
        status["abort_reason"] = msg
        save()
        with fsspec.open(f"{log_root}/ABORT", "w") as f:
            f.write(msg + "\n")
        print("ABORT:", msg, file=sys.stderr)

    save()
    for batch in batches:
        if stop_requested():
            status["state"] = "STOPPED"
            save()
            return
        bdir = f"{log_root}/{batch}"
        with fsspec.open(f"{bdir}/list.txt", "r") as f:
            prefixes = [line.strip() for line in f if line.strip()]
        with fsspec.open(f"{bdir}/witnesses.txt", "r") as f:
            witnesses = [line.strip() for line in f if line.strip()]
        bad = [p for p in prefixes if PROTECTED_RE.search(p) or not p.startswith("gs://marin-")]
        if bad:
            abort(f"{batch}: protected/invalid lines {bad[:3]}")
            return
        # Pre-check witnesses BEFORE deleting anything in this batch.
        with ThreadPoolExecutor(workers) as ex:
            missing_pre = [
                w for w, ok in zip(witnesses, ex.map(lambda w: fs.exists(w.replace("gs://", "")), witnesses)) if not ok
            ]
        if missing_pre:
            abort(f"{batch}: {len(missing_pre)} witnesses missing BEFORE deletion, e.g. {missing_pre[:3]}")
            return
        st = {"total": len(prefixes), "witnesses": len(witnesses), "deleted": 0, "failed": 0, "phase": "deleting"}
        status["batches"][batch] = st
        save()

        done_lines: list[str] = []
        failed_lines: list[str] = []
        seq = [0]

        def flush() -> None:
            if done_lines:
                with fsspec.open(f"{bdir}/done-{seq[0]:05d}.txt", "w") as f:
                    f.write("\n".join(done_lines) + "\n")
                done_lines.clear()
            if failed_lines:
                with fsspec.open(f"{bdir}/failed-{seq[0]:05d}.txt", "w") as f:
                    f.write("\n".join(failed_lines) + "\n")
                failed_lines.clear()
            seq[0] += 1
            save()

        def one(p: str) -> None:
            path = p.replace("gs://", "")
            try:
                if fs.exists(path):
                    fs.rm(path, recursive=True)
                done_lines.append(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {p}")
                st["deleted"] += 1
            except Exception as e:
                failed_lines.append(f"{p}\t{type(e).__name__}: {e}")
                st["failed"] += 1

        stopped = False
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(prefixes), 25):
                if stop_requested():
                    stopped = True
                    break
                list(ex.map(one, prefixes[i : i + 25]))
                flush()
                print(f"{batch}: {min(i + 25, len(prefixes))}/{len(prefixes)}", file=sys.stderr)
        flush()
        if stopped:
            st["phase"] = "stopped"
            status["state"] = "STOPPED"
            save()
            return

        st["phase"] = "verifying"
        save()
        with ThreadPoolExecutor(workers) as ex:
            remaining = [p for p, e in zip(prefixes, ex.map(lambda p: fs.exists(p.replace("gs://", "")), prefixes)) if e]
            missing = [
                w for w, ok in zip(witnesses, ex.map(lambda w: fs.exists(w.replace("gs://", "")), witnesses)) if not ok
            ]
        st.update(remaining=len(remaining), witnesses_missing=len(missing), phase="done")
        save()
        if st["failed"] or remaining or missing:
            abort(
                f"{batch}: failed={st['failed']} remaining={len(remaining)} witnesses_missing={len(missing)} e.g. {missing[:3]}"
            )
            return
    status["state"] = "COMPLETED"
    save()


@click.command()
@click.option("--manifest-dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--log-root", required=True)
@click.option(
    "--batch", "batches", multiple=True, required=True, help="batch names, in order (rm_<name>.txt must exist)"
)
@click.option("--workers", default=4, show_default=True)
@click.option("--cluster", default="marin", show_default=True)
def main(manifest_dir: Path, log_root: str, batches: tuple[str, ...], workers: int, cluster: str) -> None:
    log_root = log_root.rstrip("/")
    fs, _ = fsspec.core.url_to_fs(log_root)
    if fs.exists(f"{log_root}/status.json".replace("gs://", "")):
        raise SystemExit(f"{log_root} already used; pick a fresh --log-root")
    for b in batches:
        lines = [line.strip() for line in (manifest_dir / f"rm_{b}.txt").read_text().splitlines() if line.strip()]
        bad = [p for p in lines if PROTECTED_RE.search(p)]
        if bad:
            raise SystemExit(f"REFUSING {b}: protected lines {bad[:3]}")
        w = witnesses_for(b, lines, manifest_dir)
        with fsspec.open(f"{log_root}/{b}/list.txt", "w") as f:
            f.write("\n".join(lines) + "\n")
        with fsspec.open(f"{log_root}/{b}/witnesses.txt", "w") as f:
            f.write("\n".join(w) + "\n")
        print(f"{b}: {len(lines)} prefixes, {len(w)} witnesses", file=sys.stderr)
    with open_iris_client(cluster_name=cluster, workspace=REPO_ROOT) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_chain_stage, log_root, list(batches), workers),
            name="storage-delete-chain",
            resources=ResourceSpec(cpu=2, memory="4GB", disk="10GB"),
            environment=EnvironmentSpec(env_vars={}),
        )
        print(f"Submitted {job.job_id}; status: {log_root}/status.json ; STOP switch: {log_root}/STOP", file=sys.stderr)


if __name__ == "__main__":
    main()
