# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every arm-B record shard against its catalog: line count, doc_id set, no duplicates.

Run after the swap fleet (and after any worker was stopped mid-rewrite: uploads are atomic, but
this is the proof). Each WARC's shard must hold exactly ``n_written`` records, each with a
non-empty ``text`` of at least ``MIN_TEXT_CHARS``, and their ``doc_id`` set must equal the
allow-list minus the short-dropped and missing ids recorded in the catalog.

    python -m experiments.fast_curation.verify_swap_shards
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import logging

import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.fast_curation.swap_extractor import OUTPUT_ROOT, shard_path
from experiments.fast_curation.upstream_resiliparse import MIN_TEXT_CHARS
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)


def verify_one(catalog_path: str, root: str) -> dict:
    with fsspec.open(catalog_path, "r") as fh:
        catalog = json.load(fh)
    h = catalog["warc_hash"]
    if catalog.get("skipped"):
        return {"warc_hash": h, "ok": True, "n_records": 0, "problems": []}
    problems: list[str] = []
    doc_ids: list[str] = []
    n_bad_text = 0
    with fsspec.open(shard_path(root, h), "rb") as raw, gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            doc_ids.append(rec["doc_id"])
            if len(rec.get("text") or "") < MIN_TEXT_CHARS:
                n_bad_text += 1
    if len(doc_ids) != catalog["n_written"]:
        problems.append(f"records {len(doc_ids)} != catalog n_written {catalog['n_written']}")
    if len(set(doc_ids)) != len(doc_ids):
        problems.append(f"{len(doc_ids) - len(set(doc_ids))} duplicate doc_ids")
    if n_bad_text:
        problems.append(f"{n_bad_text} records with text under {MIN_TEXT_CHARS} chars")
    with fsspec.open(f"{root}/allowlist/data-{h}.parquet", "rb") as fh:
        allowed = set(pq.read_table(fh, columns=["doc_id"]).column("doc_id").to_pylist())
    written = set(doc_ids)
    if not written <= allowed:
        problems.append(f"{len(written - allowed)} written doc_ids not in the allow-list")
    n_dropped = len(allowed) - len(written)
    if n_dropped != catalog["n_short"] + len(catalog["missing"]):
        problems.append(
            f"allow-list minus written = {n_dropped}, catalog short+missing = {catalog['n_short'] + len(catalog['missing'])}"
        )
    return {"warc_hash": h, "ok": not problems, "n_records": len(doc_ids), "problems": problems}


def run(root: str, max_workers: int) -> dict:
    catalogs = sorted(fsspec_glob(f"{root}/catalog/data-*.json"))
    template = f"{root}/verify_summary/data-{{shard:05d}}-of-{{total:05d}}.jsonl"
    pipeline = Dataset.from_iterable(catalogs).map(functools.partial(verify_one, root=root)).write_jsonl(template)
    ctx = ZephyrContext(
        name="swap-verify", max_workers=max_workers, resources=ResourceConfig(cpu=1, ram="4g", disk="4g")
    )
    ctx.execute(pipeline)
    results = []
    for path in fsspec_glob(f"{root}/verify_summary/*.jsonl"):
        with fsspec.open(path, "r") as fh:
            results += [json.loads(line) for line in fh if line.strip()]
    bad = [r for r in results if not r["ok"]]
    summary = {
        "n_catalogs": len(catalogs),
        "n_verified": len(results),
        "n_records": sum(r["n_records"] for r in results),
        "n_bad": len(bad),
        "bad": bad[:50],
    }
    with fsspec.open(f"{root}/verify_summary/_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--max-workers", type=int, default=200)
    args = ap.parse_args()
    summary = run(args.root, args.max_workers)
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_bad"] == 0 and summary["n_verified"] == summary["n_catalogs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
