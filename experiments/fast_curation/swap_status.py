# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage gates for the extractor ablation (arm B), computed from what is on disk.

Every number the runbook's checks depend on comes from here, so monitoring and sign-off read the
same artifacts: the allow-list ``_meta.json``, the per-WARC catalogs, the record shards, and the
per-stage ``_done`` markers of B's label tables. Exit status is non-zero while any gate fails, so
the script doubles as the "is it safe to move on" check.

    python -m experiments.fast_curation.swap_status
    python -m experiments.fast_curation.swap_status --expect-warcs 10364
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
from concurrent.futures import ThreadPoolExecutor

import fsspec

from experiments.baseline_collection.grid_corpora import GRID_SUFFIX, QUALITY_ROOT, TOPIC_ROOT
from experiments.fast_curation.swap_extractor import DATASET, LABEL_BASE, OUTPUT_ROOT
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)


def _read_json(path: str) -> dict:
    with fsspec.open(path, "r") as fh:
        return json.load(fh)


def _stems(paths: list[str]) -> set[str]:
    out = set()
    for p in paths:
        name = posixpath.basename(p)
        for suffix in (".jsonl.gz", ".parquet", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        out.add(name)
    return out


def status(
    root: str, label_base: str, dataset: str, expect_warcs: int | None, expect_labels: bool = False
) -> tuple[dict, list[str]]:
    """Return ``(summary, failures)``; ``failures`` is empty when every gate passes.

    Label-table gates apply only with ``expect_labels`` (after ``attach_swap_labels``); before that
    the topic/quality counts are reported but not enforced.
    """
    meta_path = f"{root}/allowlist/_meta.json"
    fs = fsspec.filesystem("gcs") if root.startswith("gs://") else fsspec.filesystem("file")
    meta = _read_json(meta_path) if fs.exists(meta_path) else None
    catalog_paths = fsspec_glob(f"{root}/catalog/data-*.json")
    with ThreadPoolExecutor(64) as ex:
        catalogs = list(ex.map(_read_json, catalog_paths))
    shard_stems = _stems(fsspec_glob(f"{root}/deduped/data-*.jsonl.gz"))
    topic_done = _stems(fsspec_glob(f"{label_base}/{TOPIC_ROOT}/{dataset}_{GRID_SUFFIX}/_done/*"))
    quality_done = _stems(fsspec_glob(f"{label_base}/{QUALITY_ROOT}/{dataset}_{GRID_SUFFIX}/_done/*"))
    registered = _stems(fsspec_glob(f"{root}/_completed/data-*"))
    claims = fsspec_glob(f"{root}/_claims/data-*/_claimed")

    written = [c for c in catalogs if not c.get("skipped")]
    summary = {
        "allowlist": None if meta is None else {k: meta[k] for k in ("n_rows", "n_warcs", "n_files")},
        "catalogs": len(catalogs),
        "skipped_warcs": len(catalogs) - len(written),
        "n_requested": sum(c["n_requested"] for c in catalogs),
        "n_written": sum(c["n_written"] for c in catalogs),
        "n_short": sum(c["n_short"] for c in catalogs),
        "n_missing": sum(len(c["missing"]) for c in catalogs),
        "warcs_with_missing": sorted(c["warc_hash"] for c in catalogs if c["missing"]),
        "shards": len(shard_stems),
        "topic_done": len(topic_done),
        "quality_done": len(quality_done),
        "registered": len(registered),
        "open_claims": len(claims),
    }

    failures = []
    if meta is None:
        failures.append("allow-list _meta.json missing (build_swap_allowlist has not completed)")
    else:
        if meta["n_files"] != meta["n_warcs"]:
            failures.append("allow-list file count != summarized WARC count")
        if expect_warcs is not None and meta["n_warcs"] > expect_warcs:
            failures.append(f"allow-list covers {meta['n_warcs']} WARCs, more than the {expect_warcs} in the manifest")
    if summary["n_missing"]:
        n_warcs = len(summary["warcs_with_missing"])
        failures.append(f"{summary['n_missing']} allow-listed ids missing from refetched WARCs ({n_warcs} WARCs)")
    if (
        summary["n_requested"] + summary["n_written"]
        and summary["n_written"] + summary["n_short"] + summary["n_missing"] != summary["n_requested"]
    ):
        failures.append("catalog arithmetic broken: written + short + missing != requested")
    expected_stems = {f"data-{c['warc_hash']}" for c in written}
    checked = [("shards", shard_stems)]
    if expect_labels:
        checked += [("topic _done", topic_done), ("quality _done", quality_done)]
    for name, have in checked:
        if have != expected_stems:
            failures.append(f"{name}: {len(have)} present vs {len(expected_stems)} catalogued WARCs with output")
    if registered != {f"data-{c['warc_hash']}" for c in catalogs}:
        failures.append(f"registry ({len(registered)}) != catalogs ({len(catalogs)})")
    if expect_warcs is not None and len(catalogs) != expect_warcs:
        failures.append(f"{len(catalogs)} of {expect_warcs} WARCs catalogued")
    if meta is not None and summary["n_requested"] and len(catalogs) == (meta["n_warcs"] + summary["skipped_warcs"]):
        if summary["n_requested"] != meta["n_rows"]:
            failures.append(f"requested {summary['n_requested']} != allow-list rows {meta['n_rows']}")
    return summary, failures


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=OUTPUT_ROOT)
    ap.add_argument("--label-base", default=LABEL_BASE)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--expect-warcs", type=int, default=None, help="Manifest size; enables the completeness gates.")
    ap.add_argument("--expect-labels", action="store_true", help="Also require B's topic/quality tables (post-attach).")
    args = ap.parse_args()
    summary, failures = status(args.root, args.label_base, args.dataset, args.expect_warcs, args.expect_labels)
    print(json.dumps(summary, indent=2))
    for f in failures:
        print(f"GATE FAIL: {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
