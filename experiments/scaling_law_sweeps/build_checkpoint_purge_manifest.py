# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a DELETION MANIFEST of purgeable curation-sweep checkpoints (READ-ONLY).

This script NEVER deletes anything. It enumerates every run directory under
`checkpoints/isoflop-curation/` in all six Marin region buckets, sizes each one
via `gcloud storage du`, classifies it, and writes a manifest of paths that are
SAFE to delete to recover storage cost.

Checkpoint layout per run:
  * `<run>/hf/step-N/`           -- HuggingFace exports, written at intervals
                                    (step-10000, 20000, ... and the final step).
                                    Weights only; this is all an eval/inference
                                    consumer needs.
  * `<run>/checkpoints/step-N/`  -- Levanter checkpoints WITH optimizer state
                                    (~3x the weights); only needed to RESUME.
  * `<run>/.data_curation_DONE`  -- completion marker.

Safety rules (see the storage-cleanup protocol):
  * A run is "complete" iff its `.data_curation_DONE` marker exists in that bucket.
  * A run is "protected" iff its run_name_core is in --running-cores (a RUNNING
    Iris job is writing it). Protected runs are NEVER touched.
  * Decision per (region, run):
      - PROTECTED (running)                         -> KEEP everything.
      - `*_10k-expC_T33T-*` (retired 10k sweep)     -> DELETE the whole run dir
        (user-confirmed abandoned; replaced by expFM_natural 10k).
      - complete (done marker present):
          * if an hf/ export exists  -> KEEP only `hf/step-<max>/`; DELETE every
            other `hf/step-N/` (intermediate exports) AND the ENTIRE
            `checkpoints/` tree (optimizer state -- not needed once training is
            done; an 8B optimizer checkpoint is the single biggest object).
          * else (no hf export)      -> KEEP only the highest `checkpoints/step-N/`
            (the sole copy of the weights); DELETE the rest.
      - incomplete & not running & not retired-10k  -> KEEP everything (could be
        resumed; intermediates are the resume state). Reported separately.

The final hf export (highest step) is the canonical thing kept, plus the run's
tiny bookkeeping files (eval_metrics.jsonl, config, done marker).

Outputs (written to --out-dir):
  * purge_manifest.tsv   -- one row per deletable path: region, run, tag, action,
                            final_step_kept, path, bytes
  * purge_paths.txt      -- bare gs:// paths to delete (feed to a separate, gated
                            `gcloud storage rm -r` step AFTER user review)
  * summary printed to stdout

Usage:
    python experiments/scaling_law_sweeps/build_checkpoint_purge_manifest.py \\
        --running-cores /tmp/running_cores.txt --out-dir /tmp/purge
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from collections import defaultdict

REGION_TO_BUCKET = {
    "us-central1": "gs://marin-us-central1",
    "us-central2": "gs://marin-us-central2",
    "us-east1": "gs://marin-us-east1",
    "us-east5": "gs://marin-us-east5",
    "us-west4": "gs://marin-us-west4",
    "europe-west4": "gs://marin-eu-west4",
}
PREFIX = "checkpoints/isoflop-curation/"
DONE_MARKER = ".data_curation_DONE"
STEP_RE = re.compile(r"^step-(\d+)/$")
TAG_RE = re.compile(r"-(expFM_natural|expC_T33T|expWARC_natural|expWARC|expA_natural)-")


def gib(n: int) -> float:
    return n / 1024**3


def du_lines(bucket: str) -> list[tuple[int, str]]:
    """Return (bytes, path) for every directory-rollup and every done-marker
    object under the bucket's isoflop-curation prefix. gcloud storage du emits a
    rollup line for each directory level, so we keep only those (paths ending in
    '/') plus the done-marker files."""
    url = f"{bucket}/{PREFIX}"
    proc = subprocess.run(["gcloud", "storage", "du", url], capture_output=True, text=True)
    rows: list[tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        size_str, path = parts
        if not size_str.isdigit():
            continue
        if path.endswith("/") or path.endswith("/" + DONE_MARKER):
            rows.append((int(size_str), path))
    return rows


def classify_region(region: str, bucket: str, running: set[str]):
    """Return (manifest_rows, kept_bytes, deletable_bytes, review_bytes) for one region.

    manifest_rows: list of dicts {region, run, tag, action, keep, path, bytes}.
    review_bytes: total size of complete-looking-but-not-done & not-running runs we
    leave untouched (a further, separately-gated opportunity).
    """
    rows = du_lines(bucket)
    run_total: dict[str, int] = {}
    run_done: dict[str, bool] = defaultdict(bool)
    ckpt_steps: dict[str, dict[str, int]] = defaultdict(dict)  # checkpoints/step-N -> bytes (optimizer state)
    hf_steps: dict[str, dict[str, int]] = defaultdict(dict)  # hf/step-N -> bytes (weights export)
    ckpt_total: dict[str, int] = {}  # whole checkpoints/ rollup (covers any non-step files too)

    base = f"{bucket}/{PREFIX}"
    for size, path in rows:
        rel = path.split(base, 1)[1] if base in path else None
        if rel is None:
            continue
        segs = rel.split("/")
        run = segs[0]
        if not run:
            continue
        remainder = "/".join(segs[1:])  # after run/
        if remainder == "":  # run rollup ".../run/"
            run_total[run] = size
        elif remainder == DONE_MARKER:
            run_done[run] = True
        elif remainder == "checkpoints/":  # ".../run/checkpoints/" rollup
            ckpt_total[run] = size
        elif len(segs) == 4 and segs[3] == "" and STEP_RE.match(segs[2] + "/"):
            if segs[1] == "checkpoints":
                ckpt_steps[run][segs[2]] = size
            elif segs[1] == "hf":
                hf_steps[run][segs[2]] = size

    manifest: list[dict] = []
    kept = 0
    deletable = 0
    review = 0

    def add(run, tag, action, keep, path, sz):
        nonlocal deletable
        deletable += sz
        manifest.append(
            {"region": region, "run": run, "tag": tag, "action": action, "keep": keep, "path": path, "bytes": sz}
        )

    for run, total in run_total.items():
        tag_m = TAG_RE.search(run)
        tag = tag_m.group(1) if tag_m else "unknown"
        rp = f"{bucket}/{PREFIX}{run}"
        hf = hf_steps.get(run, {})
        ck = ckpt_steps.get(run, {})

        if run in running:
            kept += total
            continue

        if "_10k-expC_T33T-" in run:  # retired sweep -> whole dir
            add(run, tag, "DELETE_DIR", "", f"{rp}/", total)
            continue

        if not run_done.get(run):  # incomplete & not running -> leave for separate review
            review += total
            continue

        # --- completed run ---
        if hf:
            final_hf = max(hf, key=lambda s: int(s.split("-")[1]))
            kept_run = 0
            for step, sz in hf.items():
                if step == final_hf:
                    kept_run += sz  # keep the final HF export
                else:
                    add(run, tag, "DELETE_HF_INTERMEDIATE", f"hf/{final_hf}", f"{rp}/hf/{step}/", sz)
            # delete the ENTIRE checkpoints/ tree (optimizer state) -- size from its rollup
            ck_bytes = ckpt_total.get(run, sum(ck.values()))
            if ck_bytes > 0:
                add(run, tag, "DELETE_OPTIMIZER_STATE", f"hf/{final_hf}", f"{rp}/checkpoints/", ck_bytes)
            # whatever's left (tiny bookkeeping, non-step hf files) is kept
            kept += max(0, total - sum(s for s in hf.values() if True) - ck_bytes) + kept_run
        elif len(ck) > 1:  # no hf export: keep final levanter ckpt, drop intermediates
            final_ck = max(ck, key=lambda s: int(s.split("-")[1]))
            for step, sz in ck.items():
                if step == final_ck:
                    kept += sz
                else:
                    add(run, tag, "DELETE_INTERMEDIATE", f"checkpoints/{final_ck}", f"{rp}/checkpoints/{step}/", sz)
            kept += max(0, total - sum(ck.values()))
        else:
            kept += total

    return manifest, kept, deletable, review


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--running-cores", required=True, help="File of run_name_cores with a RUNNING Iris job (protected).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regions", nargs="+", default=list(REGION_TO_BUCKET))
    args = ap.parse_args()

    running = {c.strip() for c in open(args.running_cores) if c.strip()}
    os.makedirs(args.out_dir, exist_ok=True)

    all_rows: list[dict] = []
    by_region_del: dict[str, int] = {}
    total_keep = 0
    total_review = 0
    by_tag_del: dict[str, int] = defaultdict(int)
    by_action_del: dict[str, int] = defaultdict(int)

    for region in args.regions:
        bucket = REGION_TO_BUCKET[region]
        rows, kept, deletable, review = classify_region(region, bucket, running)
        all_rows.extend(rows)
        by_region_del[region] = deletable
        total_keep += kept
        total_review += review
        for r in rows:
            by_tag_del[r["tag"]] += r["bytes"]
            by_action_del[r["action"]] += r["bytes"]
        print(
            f"{region:14s} deletable={gib(deletable):10.1f} GiB   keep={gib(kept):9.1f}   review={gib(review):9.1f}   rows={len(rows)}"
        )

    tsv = os.path.join(args.out_dir, "purge_manifest.tsv")
    with open(tsv, "w") as f:
        f.write("region\trun\ttag\taction\tkeep\tpath\tbytes\n")
        for r in all_rows:
            f.write(f"{r['region']}\t{r['run']}\t{r['tag']}\t{r['action']}\t{r['keep']}\t{r['path']}\t{r['bytes']}\n")
    paths = os.path.join(args.out_dir, "purge_paths.txt")
    with open(paths, "w") as f:
        for r in all_rows:
            f.write(r["path"] + "\n")

    total_del = sum(by_region_del.values())
    GCS_RATE = 0.020  # $/GiB-month, GCS standard (regional) list price
    print("\n=== DELETABLE by tag ===")
    for tag, b in sorted(by_tag_del.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:18s} {gib(b):10.1f} GiB")
    print("=== DELETABLE by action ===")
    for act, b in sorted(by_action_del.items(), key=lambda kv: -kv[1]):
        print(f"  {act:22s} {gib(b):10.1f} GiB")
    print(f"\nKEEP (final hf + bookkeeping + running):  {gib(total_keep)/1024:8.2f} TiB")
    print(f"REVIEW (incomplete, not running):         {gib(total_review)/1024:8.2f} TiB")
    print(f"TOTAL DELETABLE NOW:                      {gib(total_del)/1024:8.2f} TiB  ({gib(total_del):.0f} GiB)")
    print(f"  est. storage savings @ ${GCS_RATE:.3f}/GiB-mo:  ${gib(total_del)*GCS_RATE:8.0f}/month")
    print(f"Manifest: {tsv}\nPaths:    {paths}")


if __name__ == "__main__":
    main()
