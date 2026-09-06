#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll up the weekly storage-scan object listing for one owner's prefixes.

The ops storage report (``scripts/ops/storage``) leaves a deduped full object
listing (~9 GB parquet, one row per object with size, storage class, created,
updated) at ``gs://marin-us-central2/tmp/storage-scan/deduped/``. Pulling that
to a laptop is slow and egress-billed, so this submits a small Iris job pinned
to us-central2 that filters the listing to a prefix regex and writes two small
parquets back:

* ``rollup.parquet`` — one row per (bucket, depth, prefix) for depths 2..7:
  object count, total bytes, bytes per storage class, oldest/newest ``created``,
  newest ``updated``.
* ``ckpt_objects.parquet`` — the raw object rows for the checkpoint trees
  (``checkpoints/``, ``cooldown-*``, ``models/``, ``exp2166-*``), which are few
  enough (~200k) to keep at file granularity for per-step keep/delete decisions.

Usage:
    uv run experiments/storage_triage/rollup_objects.py \
        --out-dir gs://marin-us-central2/scratch/michaelryan/storage_triage/2026-08-10
"""

import re
import sys
from pathlib import Path

import click
import duckdb
import fsspec
from iris.cli.connect import open_iris_client
from iris.cluster.constraints import region_constraint
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

from scripts.ops.storage.render_report import _download_gcs_parquet

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DEDUPED_DIR = "gs://marin-us-central2/tmp/storage-scan/deduped"
SCAN_REGION = "us-central2"
MAX_DEPTH = 7

# Roots that michaelryan's experiments write to (see
# .agents/projects/user_namespaced_storage_migration.md §2/§4). Deliberately broad:
# whole `metadata/`, `documents/`, `datasets/`, `classifiers/` are included and
# non-owned subtrees are dropped at triage time, not here.
OWNER_PREFIX_RE = (
    r"^("
    r"checkpoints/(isoflop-curation|olmix-swarm|modernbert-useful|qwen3-useful|qwen3-[^/]*rephraser|qwen35-[^/]*rephraser"
    r"|qwen3-[^/]*hq-distill|qwen3-scaletest|medical|code-|math-|mathhelpforum|gsm8k|resili|sft-base|medical-sft-base"
    r"|dclm|sysprompt|arch|clf|useful)"
    r"|cooldown-|short-cooldown-|exp2166-scaling|models/qwen3-8b-extraction"
    r"|tokenized/(baseline_|resiliparse|fastpipe|high_quality|low_quality|med_quality|fineweb_cc|dclm_400m|lpv|hq_distill"
    r"|hq_dense|hq_epoch|rephraser|nemotron_cooldown|dclm_baseline_100m|llm_curated|llm_pipeline|sysprompt|random_|quality_"
    r"|nemotron_full|nemotron_10k|dclm_10k|fineweb_edu_10k|paloma|uncheatable_eval|lima_text)"
    r"|datakit/(store|quality|cluster_assign|tokenize)/[^/]*gridv1"
    r"|mirror/grid_v1|resources/datakit|resources/dclm|resources/tokenizers|artifacts/resiliparse_rs"
    r"|documents/|datasets/|classifiers/|metadata/|filtered/|filtered_subsets/|extracted/|deduped/|manifests/"
    r"|raw/(commoncrawl|baseline-dataset-collection|rephraser|paloma|uncheatable_eval|lima_text)"
    r"|cdx/|downloaded/|mathhelpforum/|distill/|sysprompt_pretrain/|decontamination/|devset/|benchmarks/|eval_datasets/"
    r"|infinigram_indices/|bm25_indices/|bm25_qtest_results/|url_index/|spec_explorer/"
    r"|scratch/(provenance|baseline_compare|nemotron|curation_arena|random_internet|eval_attribution|bootstrap|medical)"
    r"|tmp/(extraction|ttl=2d/warc-cache|ttl=30d|ttl=14d/checkpoints-temp)"
    r"|users/"
    r")"
)
CKPT_PREFIX_RE = r"^(checkpoints/|cooldown-|short-cooldown-|exp2166-scaling|models/qwen3-8b-extraction)"


def _rollup_stage(deduped_dir: str, out_dir: str, owner_re: str, ckpt_re: str) -> None:
    """Iris-side entrypoint: filter the listing and write rollup + checkpoint objects."""
    local_dir = _download_gcs_parquet(deduped_dir, Path("/tmp/storage-scan-cache/deduped"))
    files = sorted(str(p) for p in local_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet under {deduped_dir}")

    conn = duckdb.connect(":memory:")
    conn.execute("SET threads TO 4")
    conn.execute("SET memory_limit = '20GB'")
    conn.execute("SET temp_directory = '/tmp/duckdb-spill'")

    conn.execute(
        """
        CREATE TABLE mine AS
        SELECT bucket, name, size_bytes, storage_class_id, created, updated
        FROM read_parquet($files)
        WHERE regexp_matches(name, $re)
        """,
        {"files": files, "re": owner_re},
    )
    n = conn.execute("SELECT count(*), sum(size_bytes) FROM mine").fetchone()
    print(f"filtered rows={n[0]} bytes={n[1]}", file=sys.stderr)

    depth_selects = []
    for d in range(2, MAX_DEPTH + 1):
        depth_selects.append(
            f"""
            SELECT bucket, {d} AS depth,
                   array_to_string(list_slice(string_split(name, '/'), 1, {d}), '/') AS prefix,
                   count(*) AS object_count,
                   sum(size_bytes) AS total_bytes,
                   sum(CASE WHEN storage_class_id = 1 THEN size_bytes ELSE 0 END) AS standard_bytes,
                   sum(CASE WHEN storage_class_id = 2 THEN size_bytes ELSE 0 END) AS nearline_bytes,
                   sum(CASE WHEN storage_class_id = 3 THEN size_bytes ELSE 0 END) AS coldline_bytes,
                   sum(CASE WHEN storage_class_id = 4 THEN size_bytes ELSE 0 END) AS archive_bytes,
                   min(created) AS oldest_created,
                   max(created) AS newest_created,
                   max(updated) AS newest_updated
            FROM mine
            WHERE len(string_split(name, '/')) > {d}
            GROUP BY bucket, prefix
            """
        )
    union = " UNION ALL ".join(depth_selects)
    local_out = Path("/tmp/storage-triage-out")
    local_out.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY ({union}) TO '{local_out}/rollup.parquet' (FORMAT PARQUET)")
    conn.execute(
        f"COPY (SELECT * FROM mine WHERE regexp_matches(name, $re)) TO '{local_out}/ckpt_objects.parquet' (FORMAT PARQUET)",
        {"re": ckpt_re},
    )
    fs, _ = fsspec.core.url_to_fs(out_dir)
    for fname in ("rollup.parquet", "ckpt_objects.parquet"):
        dest = f"{out_dir.rstrip('/')}/{fname}"
        fs.put_file(str(local_out / fname), dest)
        print(f"wrote {dest} ({(local_out / fname).stat().st_size} bytes)", file=sys.stderr)


@click.command()
@click.option("--deduped-dir", default=DEFAULT_DEDUPED_DIR, show_default=True)
@click.option("--out-dir", required=True, help="gs:// dir for rollup.parquet + ckpt_objects.parquet")
@click.option("--cluster", default="marin", show_default=True)
def main(deduped_dir: str, out_dir: str, cluster: str) -> None:
    re.compile(OWNER_PREFIX_RE)
    with open_iris_client(cluster_name=cluster, workspace=REPO_ROOT) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_callable(_rollup_stage, deduped_dir, out_dir, OWNER_PREFIX_RE, CKPT_PREFIX_RE),
            name="storage-triage-rollup",
            resources=ResourceSpec(cpu=4, memory="32GB", disk="80GB"),
            environment=EnvironmentSpec(env_vars={}),
            constraints=[region_constraint([SCAN_REGION])],
        )
        print(f"Submitted {job.job_id}", file=sys.stderr)
        job.wait(stream_logs=True, timeout=float("inf"))


if __name__ == "__main__":
    main()
