#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build keep/delete lists for michaelryan's checkpoint trees and cache intermediates.

READ-ONLY: lists GCS live (metadata ops), classifies, and writes text manifests.
It never deletes. Run the emitted ``rm_*.txt`` lists yourself after review.

Rules (agreed 2026-08-16, see KEEP_DELETE_MANIFEST.md):
* A run's *final model* is ``hf/step-<max>`` (must contain ``config.json``); if a run
  has no hf export at all, its final model is ``checkpoints/step-<max>``.
* DONE runs (``.data_curation_DONE`` present): delete every non-final ``hf/step-N`` and
  every ``checkpoints/step-N`` (optimizer state), regardless of age.
* Non-DONE runs: HOLD if written within ``--hold-days`` or the name matches a
  protected family (lpv11_fastpipe_v1 cells, the live resiliparse OLMIX swarm).
  Otherwise: if the same run is DONE in another bucket → delete the whole copy
  (orphan); if it has no hf and only optimizer state → delete the whole dir
  (abandoned partial); else strip non-final hf + all optimizer state, keep the final hf.
* Runs with no usable final model (no hf, and we would delete their only weights)
  go to ``review_no_final.txt`` — never auto-deleted.
* Explicit protections: ``exp2166-…-9563f0/checkpoints/step-35000`` (cooldown base).
* Cache intermediates: fast_curation staging stages and dedup pipeline intermediates
  are listed as whole-prefix deletes (``rm_fast_curation.txt``, ``rm_dedup_intermediates.txt``).

Outputs (in ``--out-dir``): ``rm_ckpt_intermediates.txt`` (dir prefixes),
``rm_ckpt_whole_runs.txt``, ``review_no_final.txt``, ``hold.txt``, ``keep_final.tsv``,
``rm_fast_curation.txt``, ``rm_dedup_intermediates.txt``, ``summary.md``.

Usage:
    uv run experiments/storage_triage/build_delete_manifest.py --out-dir <dir> [--hold-days 7]
"""

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import gcsfs

BUCKETS = [
    "marin-us-east5",
    "marin-us-central1",
    "marin-us-central2",
    "marin-us-east1",
    "marin-eu-west4",
    "marin-us-west4",
]

# Trees whose children are run dirs.
RUN_PARENT_PREFIXES = [
    "checkpoints/isoflop-curation/",
    "checkpoints/olmix-swarm/",
    "checkpoints/modernbert-useful/",
    "checkpoints/qwen3-useful/",
    "checkpoints/medical-sft-base/",
    "checkpoints/sft-base/",
]
# Top-level run dirs matched by name (depth-1 under checkpoints/ or bucket root).
TOP_RUN_RE = re.compile(
    r"^checkpoints/(qwen3-[^/]*rephraser[^/]*|qwen35-[^/]*rephraser[^/]*|qwen3-[^/]*hq-distill[^/]*|qwen3-scaletest[^/]*"
    r"|code-v3-[^/]*|medical-14b-[^/]*|math-[^/]*|mathhelpforum-[^/]*|gsm8k-[^/]*|resili[^/]*)/$"
    r"|^(cooldown-[^/]*|short-cooldown-[^/]*|exp2166-scaling-ladder-nemotron-validation-optimal-1e\+20-9563f0)/$"
)
PROTECTED_RUN_RE = re.compile(
    r"lpv11_fastpipe_v1|olmix-swarm/[^/]*resiliparse|modernbert-useful/mb-clf-lpv11|modernbert-useful/[^/]*pooled"
)
PROTECTED_STEP_DIRS = {"exp2166-scaling-ladder-nemotron-validation-optimal-1e+20-9563f0/checkpoints/step-35000"}
DONE_MARKER = ".data_curation_DONE"
STEP_RE = re.compile(r"^(hf|checkpoints)/step-(\d+)/")

FAST_CURATION_NS_DELETE_STAGES = [
    "a_presurvivors",
    "a_chunks",
    "kept_chunks",
    "_claims_a",
    "_claims_b",
    "_claims_c",
    "_claims_a_rescue",
    "_heartbeats",
    "timing_a",
    "timing_b",
    "timing_c",
    "_xla_cache",
]
FAST_CURATION_KEEP_NS = "documents/fast_curation/fastpipe_v3-da3893385e/"
FAST_CURATION_DROP_NS = [
    "documents/fast_curation/fastpipe_v3-6855733850/",
    "documents/fast_curation/fastpipe_v2-f78c2b2b7a/",
    "documents/fast_curation/fastpipe_v1-9b5c93de91/",
]
DEDUP_INTERMEDIATE_STAGES = {"normalize", "fuzzy", "reshape", "minhash"}
DEDUP_ROOT_RE = re.compile(r"^documents/baseline_[^/]*deduped[^/]*/$")


@dataclass
class RunInfo:
    bucket: str
    run: str  # relative dir without trailing slash
    files: dict[str, tuple[int, datetime]] = field(default_factory=dict)  # rel path -> (size, updated)

    @property
    def total(self) -> int:
        return sum(s for s, _ in self.files.values())

    @property
    def newest(self) -> datetime:
        return max((u for _, u in self.files.values()), default=datetime(1970, 1, 1, tzinfo=UTC))

    @property
    def done(self) -> bool:
        return DONE_MARKER in self.files

    def steps(self, kind: str) -> dict[int, int]:
        out: dict[int, int] = defaultdict(int)
        for rel, (size, _) in self.files.items():
            m = STEP_RE.match(rel)
            if m and m.group(1) == kind:
                out[int(m.group(2))] += size
        return dict(out)

    def has(self, rel: str) -> bool:
        return rel in self.files


def _updated(info: dict) -> datetime:
    u = info.get("updated") or info.get("mtime")
    if isinstance(u, datetime):
        return u if u.tzinfo else u.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(u).replace("Z", "+00:00"))


def list_run(fs: gcsfs.GCSFileSystem, bucket: str, run: str) -> RunInfo:
    ri = RunInfo(bucket=bucket, run=run)
    prefix = f"{bucket}/{run}/"
    for path, info in fs.find(prefix, detail=True).items():
        if info.get("type") == "directory":
            continue
        ri.files[path[len(prefix) :]] = (int(info.get("size", 0)), _updated(info))
    return ri


def discover_runs(fs: gcsfs.GCSFileSystem, bucket: str) -> list[str]:
    runs: list[str] = []
    for parent in RUN_PARENT_PREFIXES:
        try:
            for p in fs.ls(f"{bucket}/{parent}", detail=False):
                rel = p[len(bucket) + 1 :].rstrip("/")
                if rel and rel != parent.rstrip("/"):
                    runs.append(rel)
        except FileNotFoundError:
            continue
    for parent in ("checkpoints/", ""):
        try:
            entries = fs.ls(f"{bucket}/{parent}", detail=False)
        except FileNotFoundError:
            continue
        for p in entries:
            rel = p[len(bucket) + 1 :]
            if not rel.endswith("/"):
                rel += "/"
            if TOP_RUN_RE.match(rel):
                runs.append(rel.rstrip("/"))
    return sorted(set(runs))


@dataclass
class Plan:
    rm_prefixes: list[tuple[str, int, str]] = field(default_factory=list)  # (gs url, bytes, reason)
    rm_whole: list[tuple[str, int, str]] = field(default_factory=list)
    review: list[tuple[str, int, str]] = field(default_factory=list)
    hold: list[tuple[str, int, str]] = field(default_factory=list)
    keep: list[tuple[str, int, str]] = field(default_factory=list)


def classify(ri: RunInfo, done_elsewhere: set[str], hold_before: datetime, plan: Plan) -> None:
    url = f"gs://{ri.bucket}/{ri.run}"
    hf = ri.steps("hf")
    opt = ri.steps("checkpoints")
    hf_max = max(hf) if hf else None
    opt_max = max(opt) if opt else None
    final_ok = hf_max is not None and ri.has(f"hf/step-{hf_max}/config.json")

    if not ri.done:
        if PROTECTED_RUN_RE.search(ri.run) or ri.newest >= hold_before:
            plan.hold.append((url, ri.total, f"protected/recent (newest {ri.newest:%Y-%m-%d})"))
            return
        if ri.run in done_elsewhere:
            plan.rm_whole.append((url, ri.total, "orphan copy; run DONE in another bucket"))
            return
        if not hf and opt and ri.run.startswith("checkpoints/isoflop-curation/"):
            plan.rm_whole.append(
                (url, ri.total, f"stale partial, optimizer only ({len(opt)} steps, newest {ri.newest:%Y-%m-%d})")
            )
            return
        if not hf and opt:
            # Outside the sweep tree an opt-only run may be a real model (no hf export); keep its
            # last step below and surface it for a human decision.
            plan.review.append(
                (
                    url,
                    ri.total,
                    f"opt-only run outside isoflop-curation ({len(opt)} steps, newest {ri.newest:%Y-%m-%d}); keeping step-{opt_max}",
                )
            )

    if hf and not final_ok:
        plan.review.append((url, ri.total, f"hf/step-{hf_max} lacks config.json"))
        return
    if not hf:
        # No hf export: the last optimizer step is the only weights. Keep it, drop the rest.
        if opt_max is None:
            return
        plan.keep.append((f"{url}/checkpoints/step-{opt_max}", opt[opt_max], "final weights (no hf export)"))
        for s, b in opt.items():
            if s != opt_max:
                plan.rm_prefixes.append((f"{url}/checkpoints/step-{s}/", b, "non-final optimizer step (no hf)"))
        return

    plan.keep.append((f"{url}/hf/step-{hf_max}", hf[hf_max], "final hf" + (" [DONE]" if ri.done else "")))
    for s, b in hf.items():
        if s != hf_max:
            plan.rm_prefixes.append((f"{url}/hf/step-{s}/", b, "intermediate hf export"))
    for s, b in opt.items():
        rel = f"{ri.run}/checkpoints/step-{s}"
        if rel in PROTECTED_STEP_DIRS:
            plan.keep.append((f"{url}/checkpoints/step-{s}", b, "protected base checkpoint"))
            continue
        plan.rm_prefixes.append((f"{url}/checkpoints/step-{s}/", b, "optimizer state" + (" [DONE]" if ri.done else "")))


def cache_lists(fs: gcsfs.GCSFileSystem, out_dir: Path) -> tuple[list[str], list[str]]:
    fc: list[str] = []
    dd: list[str] = []
    for bucket in BUCKETS:
        for stage in FAST_CURATION_NS_DELETE_STAGES:
            p = f"{bucket}/{FAST_CURATION_KEEP_NS}{stage}/"
            if fs.exists(p):
                fc.append(f"gs://{p}")
        for ns in FAST_CURATION_DROP_NS:
            if fs.exists(f"{bucket}/{ns}"):
                fc.append(f"gs://{bucket}/{ns}")
        try:
            roots = fs.ls(f"{bucket}/documents/", detail=False)
        except FileNotFoundError:
            continue
        for r in roots:
            rel = r[len(bucket) + 1 :]
            rel = rel if rel.endswith("/") else rel + "/"
            if not DEDUP_ROOT_RE.match(rel):
                continue
            for nd in fs.ls(f"{bucket}/{rel}", detail=False):
                for st in fs.ls(nd, detail=False):
                    name = st.rstrip("/").split("/")[-1]
                    if name in DEDUP_INTERMEDIATE_STAGES or name.startswith("reshape_"):
                        dd.append(f"gs://{st.rstrip('/')}/")
    (out_dir / "rm_fast_curation.txt").write_text("\n".join(fc) + "\n")
    (out_dir / "rm_dedup_intermediates.txt").write_text("\n".join(dd) + "\n")
    return fc, dd


def _write(out_dir: Path, name: str, rows: list[tuple[str, int, str]]) -> None:
    with open(out_dir / name, "w") as f:
        for url, b, reason in rows:
            f.write(f"{url}\t{b}\t{reason}\n")


@click.command()
@click.option("--out-dir", required=True, type=click.Path(path_type=Path))
@click.option("--hold-days", default=7, show_default=True)
@click.option("--buckets", default=",".join(BUCKETS), show_default=True)
def main(out_dir: Path, hold_days: int, buckets: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = gcsfs.GCSFileSystem()
    hold_before = datetime.now(UTC) - timedelta(days=hold_days)

    runs: dict[str, list[RunInfo]] = defaultdict(list)
    for bucket in buckets.split(","):
        names = discover_runs(fs, bucket)
        print(f"{bucket}: {len(names)} run dirs", file=sys.stderr)
        by_run: dict[str, RunInfo] = {}
        for parent in RUN_PARENT_PREFIXES:
            prefix = f"{bucket}/{parent}"
            try:
                found = fs.find(prefix, detail=True)
            except FileNotFoundError:
                continue
            for path, info in found.items():
                if info.get("type") == "directory":
                    continue
                rel = path[len(bucket) + 1 :]
                run = "/".join(rel.split("/")[:3])
                ri = by_run.setdefault(run, RunInfo(bucket=bucket, run=run))
                ri.files[rel[len(run) + 1 :]] = (int(info.get("size", 0)), _updated(info))
            print(f"  {bucket}/{parent}: {len(found)} objects", file=sys.stderr)
        for run in names:
            if run in by_run or any(run.startswith(p) for p in RUN_PARENT_PREFIXES):
                continue
            ri = list_run(fs, bucket, run)
            if ri.files:
                by_run[run] = ri
        for run, ri in by_run.items():
            runs[run].append(ri)

    done_elsewhere = {run for run, copies in runs.items() if any(c.done for c in copies)}
    plan = Plan()
    for run, copies in runs.items():
        for ri in copies:
            classify(ri, done_elsewhere, hold_before, plan)

    _write(out_dir, "rm_ckpt_intermediates.tsv", plan.rm_prefixes)
    _write(out_dir, "rm_ckpt_whole_runs.tsv", plan.rm_whole)
    _write(out_dir, "review_no_final.tsv", plan.review)
    _write(out_dir, "hold.tsv", plan.hold)
    _write(out_dir, "keep_final.tsv", plan.keep)
    (out_dir / "rm_ckpt_intermediates.txt").write_text("\n".join(u for u, _, _ in plan.rm_prefixes) + "\n")
    (out_dir / "rm_ckpt_whole_runs.txt").write_text("\n".join(u for u, _, _ in plan.rm_whole) + "\n")
    fc, dd = cache_lists(fs, out_dir)

    tb = lambda rows: sum(b for _, b, _ in rows) / 1e12  # noqa: E731
    lines = [
        f"# delete manifest {datetime.now(UTC):%Y-%m-%d %H:%M}Z (hold-days={hold_days})",
        f"- checkpoint intermediates to delete: {len(plan.rm_prefixes)} prefixes, {tb(plan.rm_prefixes):.2f} TB",
        f"- whole run dirs to delete (orphans / opt-only partials): {len(plan.rm_whole)}, {tb(plan.rm_whole):.2f} TB",
        f"- keep (final models): {len(plan.keep)}, {tb(plan.keep):.2f} TB",
        f"- hold (recent/protected): {len(plan.hold)}, {tb(plan.hold):.2f} TB",
        f"- review (no usable final): {len(plan.review)}, {tb(plan.review):.2f} TB",
        f"- fast_curation staging prefixes: {len(fc)}; dedup intermediate prefixes: {len(dd)} (sizes: see rollup)",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
