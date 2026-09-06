#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print triage views over the rollup produced by ``rollup_objects.py``.

Reads ``rollup.parquet`` (per bucket/depth/prefix aggregates) and
``ckpt_objects.parquet`` (per-object rows for checkpoint trees) from a local
dir and prints the views used to build the keep/delete manifest:

* checkpoint runs: per run, final ``hf/step-N`` vs other hf steps vs optimizer
  ``checkpoints/step-N`` bytes, DONE marker, newest write.
* stage breakdowns for ``documents/fast_curation/<ns>``, ``documents/*_deduped/<n>warcs``,
  ``datasets/*``, ``classifiers/*``, ``raw/commoncrawl/*``.
* tokenized caches with their per-bucket mirrors.

Usage:
    uv run experiments/storage_triage/triage_report.py LOCAL_DIR [--view all|ckpt|stages|tokenized]
"""

import re
from pathlib import Path

import click
import pandas as pd

TB = 1e12
GB = 1e9

STEP_RE = re.compile(r"/(hf|checkpoints)/step-(\d+)/")


def _load(local_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    roll = pd.read_parquet(local_dir / "rollup.parquet")
    ck = pd.read_parquet(local_dir / "ckpt_objects.parquet")
    for df in (roll,):
        df["TB"] = df.total_bytes / TB
        df["cold_frac"] = (df.coldline_bytes + df.archive_bytes + df.nearline_bytes) / df.total_bytes.clip(lower=1)
    return roll, ck


def _run_root(name: str) -> str | None:
    """Run directory for a checkpoint object: ``checkpoints/<family>/<run>`` or top-level ``cooldown-*/``."""
    parts = name.split("/")
    if parts[0] == "checkpoints":
        if len(parts) < 3:
            return None
        # families with a sub-namespace (isoflop-curation/<run>, olmix-swarm/<run>, modernbert-useful/<run>)
        if parts[1] in {
            "isoflop-curation",
            "olmix-swarm",
            "modernbert-useful",
            "qwen3-useful",
            "sft-base",
            "medical-sft-base",
        }:
            return "/".join(parts[:3])
        return "/".join(parts[:2])
    return parts[0]


def ckpt_view(ck: pd.DataFrame) -> pd.DataFrame:
    ck = ck.copy()
    ck["run"] = ck.name.map(_run_root)
    ck = ck[ck.run.notna()]
    m = ck.name.str.extract(r"/(hf|checkpoints)/step-(\d+)/")
    ck["kind"] = m[0]
    ck["step"] = pd.to_numeric(m[1], errors="coerce")
    ck["is_done"] = ck.name.str.endswith(".data_curation_DONE")
    ck["is_temp"] = ck.name.str.startswith("tmp/")

    rows = []
    for (bucket, run), g in ck.groupby(["bucket", "run"]):
        hf = g[g.kind == "hf"]
        opt = g[g.kind == "checkpoints"]
        hf_max = hf.step.max() if len(hf) else None
        opt_max = opt.step.max() if len(opt) else None
        rows.append(
            dict(
                bucket=bucket.replace("marin-", ""),
                run=run,
                total_GB=g.size_bytes.sum() / GB,
                hf_final_GB=hf[hf.step == hf_max].size_bytes.sum() / GB if hf_max is not None else 0.0,
                hf_final_step=hf_max,
                hf_other_GB=hf[hf.step != hf_max].size_bytes.sum() / GB if hf_max is not None else 0.0,
                n_hf_steps=hf.step.nunique(),
                opt_GB=opt.size_bytes.sum() / GB,
                opt_max_step=opt_max,
                n_opt_steps=opt.step.nunique(),
                other_GB=g[g.kind.isna()].size_bytes.sum() / GB,
                done=bool(g.is_done.any()),
                objects=len(g),
                newest=g.updated.max(),
                oldest=g.created.min(),
                cold_frac=g.loc[g.storage_class_id > 1, "size_bytes"].sum() / max(g.size_bytes.sum(), 1),
            )
        )
    return pd.DataFrame(rows).sort_values("total_GB", ascending=False)


def stage_view(roll: pd.DataFrame, root_re: str, depth: int) -> pd.DataFrame:
    x = roll[(roll.depth == depth) & roll.prefix.str.match(root_re)]
    return x.sort_values("TB", ascending=False)[
        ["bucket", "prefix", "TB", "object_count", "cold_frac", "oldest_created", "newest_updated"]
    ]


@click.command()
@click.argument("local_dir", type=click.Path(exists=True, path_type=Path))
@click.option("--view", default="all", type=click.Choice(["all", "ckpt", "stages", "tokenized"]))
def main(local_dir: Path, view: str) -> None:
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 2000)
    pd.set_option("display.max_colwidth", 110)
    roll, ck = _load(local_dir)

    if view in ("all", "ckpt"):
        cv = ckpt_view(ck)
        cv.to_csv(local_dir / "ckpt_runs.csv", index=False)
        fam = cv.run.str.split("/").str[:2].str.join("/")
        agg = cv.groupby(fam).agg(
            runs=("run", "count"),
            total_TB=("total_GB", lambda s: s.sum() / 1000),
            hf_final_TB=("hf_final_GB", lambda s: s.sum() / 1000),
            hf_other_TB=("hf_other_GB", lambda s: s.sum() / 1000),
            opt_TB=("opt_GB", lambda s: s.sum() / 1000),
            done=("done", "sum"),
        )
        print("=== checkpoint families (all buckets summed)")
        print(agg.sort_values("total_TB", ascending=False).round(2).to_string())
        print("\n=== top 60 runs by size")
        print(cv.head(60).round(1).to_string(index=False))

    if view in ("all", "stages"):
        for title, root, depth in [
            ("documents/fast_curation/<ns>/<stage>", r"^documents/fast_curation/", 4),
            ("documents/baseline_*_deduped/<n>warcs/<stage>", r"^documents/baseline_[^/]*deduped[^/]*/", 4),
            ("documents/baseline_llm_extraction*/<spec-or-region>", r"^documents/baseline_llm_extraction", 4),
            ("documents/* (depth 2)", r"^documents/", 2),
            ("datasets/*/<sub>", r"^datasets/", 3),
            ("classifiers/*/<sub>", r"^classifiers/", 3),
            ("raw/commoncrawl/*", r"^raw/", 3),
            (
                "cdx|downloaded|mathhelpforum|distill|sysprompt_pretrain (depth 2)",
                r"^(cdx|downloaded|mathhelpforum|distill|sysprompt_pretrain|decontamination|devset|benchmarks|eval_datasets|scratch)/",
                2,
            ),
            ("datakit grid (depth 3)", r"^(datakit|mirror|resources|artifacts)/", 3),
            ("metadata/* (depth 2)", r"^metadata/", 2),
            ("indices (depth 2)", r"^(infinigram_indices|bm25_indices|url_index|spec_explorer)/", 2),
            ("tmp (depth 3)", r"^tmp/", 3),
            ("users (depth 3)", r"^users/", 3),
        ]:
            x = stage_view(roll, root, depth)
            print(f"\n=== {title}  [{len(x)} rows, {x.TB.sum():.2f} TB]")
            print(x.head(80).round(3).to_string(index=False))

    if view in ("all", "tokenized"):
        x = roll[(roll.depth == 2) & roll.prefix.str.startswith("tokenized/")]
        piv = x.pivot_table(index="prefix", columns="bucket", values="TB", aggfunc="sum").fillna(0)
        piv["total"] = piv.sum(axis=1)
        piv["n_buckets"] = (piv.drop(columns="total") > 0).sum(axis=1)
        print("\n=== tokenized caches × bucket (TB)")
        print(piv.sort_values("total", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
