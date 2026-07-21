# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the 1,934-doc labeled dev set (+ gold extractions) in the small-rephraser `sample_1k` layout.

The old small-rephraser dev set (`static/warcs/sample_1k_html/`) is a flat folder of raw HTML plus a
sidecar `<file>.html.meta.json` per doc (consumed by `scripts/process_warc.py::read_html_folder`). It
carried neither keep/drop labels nor reference extractions. This exporter reproduces that layout for
our 1,934-doc register-spanning dev set and adds the two new things: a 4-level `label`
(keep/weak_keep/weak_drop/drop) per doc, and a `gold/` folder of reference extractions.

Two stages:
  stage  (Iris CPU, in-region us-central2) — consolidate the gate-verified authoritative HTML for every
         dev url into one parquet, choosing the best source per url from `html_gate_report` (verdict
         `match`, preferring full_warc > fallback > local_pool > cc_refetch).
  build  (local) — pull the staged HTML, join register + 4-level labels (human export primary, judge v3
         fallback) + gold text, and write the export tree under `--out-dir`.

Label source (per the export contract): the raw 4-level HUMAN labels come from the annotation tool's
"export labels" button (`scratch/annotate.html` -> `devset_hard_labels.json`); judge v3 fills docs that
were never hand-labeled. Register comes from the human export (user-corrected), then the gold manifest,
then a domain-based fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from urllib.parse import urlparse

import fsspec
import pyarrow.parquet as pq
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.build_devset import _domain_register
from experiments.baseline_collection.recover_devset_html_cc import GATE_OUT, OUT, WORKSPACE

logger = logging.getLogger(__name__)

EXPORT_STAGE = f"{WORKSPACE}/export_stage"  # parquet {dev_url, html, source} — one authoritative html/url

GOLD_DIR = "scratch/gold_extraction/extract_out"
GOLD_MANIFEST = "scratch/gold_extraction/extract_manifest.json"
JUDGE_LABELS = "scratch/devset_judge/v3_run/v3_labels.json"
DEFAULT_HUMAN_LABELS = os.path.expanduser("~/Downloads/devset_hard_labels.json")
# url -> register: the devset selection's per-doc register (LLM reclassification overrides + the
# domain-based category), consolidated from provenance_10k/devset/hard by build_register_map().
REGISTER_MAP = "scratch/devset_register_map.json"

# Authoritative-source preference when a url has several gate-`match` html candidates. full_warc is the
# original crawl record; local_pool is a re-extraction of the same bytes; cc_refetch is a live refetch.
_SOURCE_PRIORITY = {"full_warc": 0, "fallback": 1, "local_pool": 2, "cc_refetch": 3}


def _hid(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:10]


def _domain_slug(url: str) -> str:
    dom = (urlparse(url).netloc or "unknown").lower()
    return re.sub(r"[^a-z0-9]+", "_", dom).strip("_") or "unknown"


def _best_source_per_url() -> dict[str, str]:
    """From html_gate_report, the single authoritative source per dev url (verdict==match only)."""
    fs = fsspec.filesystem("gcs")
    files = [f"gs://{f}" for f in fs.glob(f"{GATE_OUT}/*.parquet".replace("gs://", ""))]
    best: dict[str, str] = {}
    for f in files:
        t = pq.read_table(f, filesystem=fs, columns=["dev_url", "verdict", "source"]).to_pylist()
        for r in t:
            if r["verdict"] != "match":
                continue
            u, s = r["dev_url"], r["source"]
            if s not in _SOURCE_PRIORITY:
                continue
            if u not in best or _SOURCE_PRIORITY[s] < _SOURCE_PRIORITY[best[u]]:
                best[u] = s
    return best


def run_stage(max_workers: int) -> None:
    """Consolidate one authoritative HTML per dev url into EXPORT_STAGE (in-region Zephyr job)."""
    best = _best_source_per_url()
    logger.info("gate-verified authoritative html for %d dev urls", len(best))

    def keep(r: dict) -> dict | None:
        if r.get("status") == "ok" and r.get("html") and best.get(r.get("dev_url")) == r.get("source"):
            return {"dev_url": r["dev_url"], "html": r["html"], "source": r["source"]}
        return None

    pipeline = (
        Dataset.from_files(f"{OUT}/*.parquet")
        .load_parquet(columns=["dev_url", "html", "source", "status"])
        .map(keep)
        .filter(lambda x: x is not None)
        .reshard(8)
        .write_parquet(f"{EXPORT_STAGE}/s-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=False)
    )
    ZephyrContext(
        name="export-devset-stage-html",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=1, ram="4g", regions=["us-central2"], preemptible=True),
    ).execute(pipeline)
    logger.info("staged authoritative html -> %s", EXPORT_STAGE)


_HARD = f"{WORKSPACE.replace('provenance_10k_devset', 'provenance_10k')}/devset/hard"


def build_register_map() -> None:
    """Consolidate url -> register for the whole dev set into REGISTER_MAP (small in-region reads).

    Priority: the LLM register-reclassification overrides, then the domain-based `category` from the
    selection sample parquets. This is the same register each doc carried in the annotation tool.
    """
    fs = fsspec.filesystem("gcs")
    reg: dict[str, str] = {}
    with fs.open(f"{_HARD}/register_overrides.json".replace("gs://", "")) as f:
        reg.update(json.load(f))
    for name in ("sample.parquet", "sample_full.parquet", "neg_sample.parquet"):
        t = pq.read_table(f"{_HARD}/{name}", filesystem=fs)
        col = "register" if "register" in t.column_names else "category"
        for u, c in zip(t.column("url").to_pylist(), t.column(col).to_pylist(), strict=True):
            if c and u not in reg:
                reg[u] = c
    json.dump(reg, open(REGISTER_MAP, "w"))
    logger.info("wrote register map for %d urls -> %s", len(reg), REGISTER_MAP)


def _load_staged_html() -> dict[str, str]:
    """Pull EXPORT_STAGE; one html per url (keep the longest if a url somehow has duplicates)."""
    fs = fsspec.filesystem("gcs")
    files = [f"gs://{f}" for f in fs.glob(f"{EXPORT_STAGE}/*.parquet".replace("gs://", ""))]
    if not files:
        raise SystemExit(f"no staged html at {EXPORT_STAGE} — run `stage` first")
    out: dict[str, str] = {}
    for f in files:
        for r in pq.read_table(f, filesystem=fs, columns=["dev_url", "html"]).to_pylist():
            u, h = r["dev_url"], r["html"]
            if h and (u not in out or len(h) > len(out[u])):
                out[u] = h
    return out


def _gold_text_path(hid: str) -> str | None:
    """Reference extraction for a gold doc: the verified merge if present, else the Sonnet extraction."""
    for name in (f"{hid}_merged.txt", f"{hid}.txt"):
        p = os.path.join(GOLD_DIR, name)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None


def _unverified_html(exclude: set[str]) -> dict[str, str]:
    """Best recovered HTML for dev urls that never got a gate `match` (no_signature/mismatch).

    These have HTML but the recovered snapshot couldn't be signature-verified against the original doc,
    so they carry `html_verified: false` and should be relabeled against what the HTML actually is.
    """
    fs = fsspec.filesystem("gcs")
    gate_files = [f for f in fs.ls(GATE_OUT.replace("gs://", "")) if f.endswith(".parquet")]
    matched, seen = set(), set()
    for f in gate_files:
        t = pq.read_table(f"gs://{f}", filesystem=fs, columns=["dev_url", "verdict"])
        for u, v in zip(t.column("dev_url").to_pylist(), t.column("verdict").to_pylist(), strict=True):
            seen.add(u)
            if v == "match":
                matched.add(u)
    want = (seen - matched) - exclude
    out: dict[str, str] = {}
    for f in fs.glob(f"{OUT}/*.parquet".replace("gs://", "")):
        try:
            t = pq.read_table(f"gs://{f}", filesystem=fs, columns=["dev_url", "html", "status"])
        except (KeyError, OSError, ValueError):
            continue  # a few refetch shards were written with an empty/variant schema
        for u, h, st in zip(*(t.column(c).to_pylist() for c in ("dev_url", "html", "status")), strict=True):
            if u in want and h and st == "ok" and (u not in out or len(h) > len(out[u])):
                out[u] = h
    return out


def run_build(
    out_dir: str,
    human_labels_path: str,
    include_unverified: bool,
    relabel_path: str | None,
    confirm_unverified: bool,
) -> None:
    html_by_url = _load_staged_html()
    verified_urls = set(html_by_url)  # gate-`match` urls; everything added below is html_verified: false
    logger.info("staged (gate-verified) html for %d urls", len(html_by_url))

    human = json.load(open(human_labels_path)) if os.path.exists(human_labels_path) else {}
    if not human:
        logger.warning("no human labels at %s — labels will be judge-only", human_labels_path)
    judge = json.load(open(JUDGE_LABELS)) if os.path.exists(JUDGE_LABELS) else {}
    # Relabels done against the actual recovered HTML (for the unverified snapshots): highest priority.
    relabel = json.load(open(relabel_path)) if relabel_path and os.path.exists(relabel_path) else {}
    gold_man = {d["url"]: d for d in json.load(open(GOLD_MANIFEST))}
    # A few gold docs never got a gate `match` in the full-set pass; guarantee all gold are present by
    # falling back to the html the gold extraction itself ran on (a local file in the gold manifest).
    for url, d in gold_man.items():
        if url not in html_by_url and d.get("html") and os.path.exists(d["html"]):
            html_by_url[url] = open(d["html"]).read()
    if include_unverified:
        extra = _unverified_html(exclude=set(html_by_url))
        html_by_url.update(extra)
        logger.info("added %d unverified-snapshot urls (html_verified=false)", len(extra))
    reg_map = json.load(open(REGISTER_MAP)) if os.path.exists(REGISTER_MAP) else {}
    if not reg_map:
        logger.warning("no register map at %s — falling back to domain heuristic; run `regmap` first", REGISTER_MAP)

    def register_of(url: str) -> str:
        h = human.get(url) or {}
        return (
            h.get("register")  # user's per-doc correction wins
            or reg_map.get(url)  # devset selection register (LLM reclassification + domain category)
            or gold_man.get(url, {}).get("register")
            or _domain_register(urlparse(url).netloc)
        )

    def label_of(url: str) -> tuple[str | None, str | None, str | None]:
        r = relabel.get(url) or {}
        if r.get("label"):
            return r["label"], "relabel", None
        h = human.get(url) or {}
        if h.get("label"):
            return h["label"], "human", None
        j = judge.get(url) or {}
        if j.get("label"):
            return j["label"], "judge_v3", j.get("confidence")
        return None, None, None

    html_dir = os.path.join(out_dir, "devset_1934_html")
    gold_out = os.path.join(html_dir, "gold")
    os.makedirs(gold_out, exist_ok=True)

    jsonl_rows: list[dict] = []
    label_dist: Counter = Counter()
    reg_dist: Counter = Counter()
    n_gold = 0

    for i, url in enumerate(sorted(html_by_url)):
        hid = _hid(url)
        html = html_by_url[url]
        register = register_of(url)
        label, label_source, confidence = label_of(url)
        h = human.get(url) or {}
        gold_src = _gold_text_path(hid)
        base = f"record_{i:05d}_{_domain_slug(url)}_{hid}"

        with open(os.path.join(html_dir, base + ".html"), "w") as f:
            f.write(html)

        meta = {
            "url": url,
            "hid": hid,
            "register": register,
            "label": label,
            "label_source": label_source,
            "confidence": confidence,
            "benchmark": bool(h.get("benchmark")),
            "top_quality": bool(h.get("top_quality")),
            # gate-fingerprint match, a human relabel against the recovered html, or a completed human
            # review pass over the unverified set (confirm_unverified) — each verifies label⟷content.
            "html_verified": url in verified_urls or url in relabel or confirm_unverified,
            "gold": gold_src is not None,
            "gold_path": f"gold/{hid}.txt" if gold_src else None,
        }
        with open(os.path.join(html_dir, base + ".html.meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if gold_src:
            with open(gold_src) as g, open(os.path.join(gold_out, f"{hid}.txt"), "w") as w:
                w.write(g.read())
            n_gold += 1

        jsonl_rows.append({**meta, "html_file": base + ".html", "html_len": len(html)})
        label_dist[label or "UNLABELED"] += 1
        reg_dist[register] += 1

    with open(os.path.join(html_dir, "devset.jsonl"), "w") as f:
        for r in jsonl_rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "num_docs": len(jsonl_rows),
        "num_gold": n_gold,
        "num_html_verified": sum(1 for r in jsonl_rows if r["html_verified"]),
        "num_html_unverified": sum(1 for r in jsonl_rows if not r["html_verified"]),
        "num_human_labeled": sum(1 for r in jsonl_rows if r["label_source"] == "human"),
        "num_judge_labeled": sum(1 for r in jsonl_rows if r["label_source"] == "judge_v3"),
        "num_relabeled": sum(1 for r in jsonl_rows if r["label_source"] == "relabel"),
        "num_unlabeled": label_dist["UNLABELED"],
        "label_dist": dict(label_dist),
        "register_dist": dict(reg_dist.most_common()),
        "layout": (
            "flat html + <file>.html.meta.json sidecar (small-rephraser sample_1k format); "
            "gold/ = reference extractions; devset.jsonl = consolidated table"
        ),
    }
    with open(os.path.join(html_dir, "devset_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("wrote %d docs (%d gold) -> %s", len(jsonl_rows), n_gold, html_dir)
    print(json.dumps(summary, indent=2))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    st = sub.add_parser("stage", help="Iris CPU: consolidate authoritative html for all dev urls")
    st.add_argument("--max-workers", type=int, default=32)
    sub.add_parser("regmap", help="local: consolidate url->register map from the selection artifacts")
    bl = sub.add_parser("build", help="local: write the export tree from staged html + labels + gold")
    bl.add_argument("--out-dir", default="scratch/devset_export")
    bl.add_argument("--human-labels", default=DEFAULT_HUMAN_LABELS)
    bl.add_argument(
        "--include-unverified",
        action="store_true",
        help="also include dev urls whose recovered snapshot couldn't be gate-verified (html_verified=false)",
    )
    bl.add_argument("--relabels", default=None, help="url->{label} overrides (relabeled vs the recovered html)")
    bl.add_argument(
        "--confirm-unverified",
        action="store_true",
        help="mark all included docs html_verified=true (a human reviewed the full unverified set: "
        "corrected some via --relabels, confirmed the rest as already-correct)",
    )
    args = ap.parse_args()

    if args.mode == "stage":
        run_stage(args.max_workers)
    elif args.mode == "regmap":
        build_register_map()
    else:
        run_build(args.out_dir, args.human_labels, args.include_unverified, args.relabels, args.confirm_unverified)
    return 0


if __name__ == "__main__":
    sys.exit(main())
