# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge the pipeline keep/drop decisions (DCLM/Nemotron/FineWeb-Edu/FineWeb-CC/hq) onto the labeled
dev set and score each classifier's per-dataset, per-register F1 against the human labels.

  build  (local) — pull the `pipeline-labels` join output, join to the dev-set human labels + register,
         write `pipeline_labels.jsonl` (one row/doc: gold + every pipeline's keep flag, raw score, and
         extracted length) into the small-rephraser dev-set folder.
  f1     (local) — from that jsonl, print precision/recall/F1 for each pipeline, overall and per
         register. Gold = human label binarized (keep/weak_keep -> KEEP; drop/weak_drop -> DROP);
         restricted to `in_membership` docs (those actually in the 10k pool the pipelines scored).

Compare these baselines against your own spec by adding a `kept_myspec` column to the jsonl (your
extractor's keep decision per url) and re-running `f1` — it scores every `kept_*` column it finds.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter

import fsspec
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PIPELINE_LABELS = "gs://marin-us-central2/scratch/provenance_10k/devset/pipeline_labels.parquet"
DEVSET_JSONL = "scratch/devset_export/devset_1934_html/devset.jsonl"
# small-rephraser dev-set folder (default output location; --out overrides)
DEFAULT_OUT = (
    "/Users/michaelryan/Documents/School/Stanford/Research/small-rephraser/static/warcs/"
    "marin_devset_1934_html/pipeline_labels.jsonl"
)

KEEP_LABELS = {"keep", "weak_keep"}
DROP_LABELS = {"drop", "weak_drop"}
_SCORE_COLS = ("dclm_ft", "nemo_quality", "fineweb_score", "hq_len", "dclm_len", "nemo_len", "fwedu_len")


def _gold(label: str | None) -> bool | None:
    if label in KEEP_LABELS:
        return True
    if label in DROP_LABELS:
        return False
    return None  # unsure / unlabeled -> excluded from scoring


def run_build(out_path: str) -> None:
    fs = fsspec.filesystem("gcs")
    flags = {r["url"]: r for r in pq.read_table(PIPELINE_LABELS, filesystem=fs).to_pylist()}
    devset = [json.loads(line) for line in open(DEVSET_JSONL)]

    rows, missing = [], 0
    for d in devset:
        f = flags.get(d["url"])
        if f is None:
            missing += 1
            continue
        rows.append(
            {
                "url": d["url"],
                "domain": f.get("domain"),
                "register": d["register"],
                "human_label": d["label"],
                "gold_keep": _gold(d["label"]),
                "in_membership": bool(f.get("in_membership")),
                "kept_hq": bool(f.get("kept_hq")),
                "kept_dclm": bool(f.get("kept_dclm")),
                "kept_nemo": bool(f.get("kept_nemo")),
                "kept_fwedu": bool(f.get("kept_fwedu")),
                "kept_fwcc": bool(f.get("kept_fwcc")),
                **{c: f.get(c) for c in _SCORE_COLS},
            }
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    in_mem = sum(r["in_membership"] for r in rows)
    logger.info(
        "wrote %d rows (%d in 10k membership, %d devset urls unmatched) -> %s", len(rows), in_mem, missing, out_path
    )
    print(json.dumps({"rows": len(rows), "in_membership": in_mem, "unmatched": missing}, indent=2))


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def run_f1(jsonl_path: str, by_register: bool) -> None:
    rows = [json.loads(line) for line in open(jsonl_path)]
    scored = [r for r in rows if r["in_membership"] and r["gold_keep"] is not None]
    pipelines = [c[5:] for c in rows[0] if c.startswith("kept_")]  # every kept_* column present
    logger.info("scoring %d docs (in-membership, gold-labeled) across %s", len(scored), pipelines)

    def table(subset: list[dict], title: str) -> None:
        npos = sum(r["gold_keep"] for r in subset)
        print(f"\n{title}  (n={len(subset)}, gold-keep={npos}, gold-drop={len(subset) - npos})")
        print(f"  {'pipeline':<8} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        for ds in pipelines:
            col = f"kept_{ds}"
            tp = sum(r[col] and r["gold_keep"] for r in subset)
            fp = sum(r[col] and not r["gold_keep"] for r in subset)
            fn = sum(not r[col] and r["gold_keep"] for r in subset)
            tn = len(subset) - tp - fp - fn
            p, rc, f1 = _prf(tp, fp, fn)
            print(f"  {ds:<8} {p:>6.3f} {rc:>6.3f} {f1:>6.3f}   {tp:>4} {fp:>4} {fn:>4} {tn:>4}")

    table(scored, "ALL REGISTERS")
    if by_register:
        for reg, _ in Counter(r["register"] for r in scored).most_common():
            sub = [r for r in scored if r["register"] == reg]
            if len(sub) >= 10:  # skip tiny registers where F1 is noise
                table(sub, f"register: {reg}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    b = sub.add_parser("build", help="merge pipeline keeps/scores onto the dev-set labels -> jsonl")
    b.add_argument("--out", default=DEFAULT_OUT)
    f = sub.add_parser("f1", help="print per-pipeline, per-register F1 vs human gold")
    f.add_argument("--jsonl", default=DEFAULT_OUT)
    f.add_argument("--no-by-register", action="store_true")
    args = ap.parse_args()

    if args.mode == "build":
        run_build(args.out)
    else:
        run_f1(args.jsonl, not args.no_by_register)
    return 0


if __name__ == "__main__":
    sys.exit(main())
