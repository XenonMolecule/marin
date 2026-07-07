#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the HuggingFace export for the high_quality 10364-WARC dataset.

The released dataset = the decontaminated, deduped high_quality survivors (text)
with per-document WARC traceback metadata re-attached. The dedup step
(`dedup_extracted.py`) projected every row down to ``{"text": ...}`` and dropped
``url / warc_record_id / warc_file / snapshot``; decontamination
(`decon_apply.py`) then *dropped whole docs without mutating text*. So each
survivor's text is byte-identical to the cleaned text of exactly one raw
extraction record, and we recover the metadata with an exact-text JOIN.

Join key (must match the dedup reshape exactly, `dedup_extracted.py:199`)::

    text = record.get("text") or record.get("generated_text")

Both sides live entirely in the us-central1 bucket, so this reads no cross-region
data. The raw side is the consolidated archive (covers all 10364 WARCs):
``baseline_llm_extraction_consolidated/by_region/{region}/high_quality/``.

Pipeline (one Iris job, region-pinned us-central1):
  1. self-check the reducer logic in-process (fail fast, no infra),
  2. co-group survivors + raw on text, attach metadata -> parquet shards,
  3. validate the output (match rate, distinct WARCs, token/char totals, samples),
  4. write REPORT.{json,md} + source_warcs.txt to the export prefix, and STOP.

It NEVER uploads to HuggingFace. The upload is a separate, human-approved step
once the report's match rate and cost estimate look right.

Launch (CPU, us-central1)::

    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB \\
        --priority interactive --extra cpu --enable-extra-resources \\
        --region us-central1 --job-name hq-hf-export \\
        -- python experiments/baseline_collection/build_high_quality_hf_export.py --stage full

Pre-flight locally (validates the zephyr execution path, no GCS)::

    .venv/bin/python experiments/baseline_collection/build_high_quality_hf_export.py --stage smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Iterator

from fray import ResourceConfig
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths (all in the us-central1 bucket -> region-local, zero cross-region egress)
# --------------------------------------------------------------------------
BUCKET = "gs://marin-us-central1"
SURVIVORS_PREFIX = f"{BUCKET}/documents/baseline_high_quality_decon_deduped/10364warcs/deduped"
RAW_REGIONS = ("us-central1", "us-east1", "us-east5", "us-west4", "europe-west4")
RAW_BASE = f"{BUCKET}/documents/baseline_llm_extraction_consolidated/by_region"
EXPORT_PREFIX = f"{BUCKET}/documents/baseline_high_quality_hf_export/10364warcs"
JOINED_PREFIX = f"{EXPORT_PREFIX}/joined"

# Marker that distinguishes a survivor file from a raw file in the unified file list.
SURVIVOR_MARKER = "/baseline_high_quality_decon_deduped/"

INTERNET_EGRESS_USD_PER_GB = 0.12  # GCP worldwide internet egress, first 1 TB/month tier

# --------------------------------------------------------------------------
# Join logic
# --------------------------------------------------------------------------


def _join_key(record: dict) -> str | None:
    """The exact text expression the dedup reshape used (`dedup_extracted.py:199`)."""
    text = record.get("text") or record.get("generated_text")
    if not text:
        return None
    return text


def _content_hash(text: str) -> str:
    """16-byte blake2b digest of the join text, hex-encoded.

    Keying the shuffle on this compact hash (instead of the multi-KB document
    text) keeps sort/spill buffers small and avoids carrying the full text on
    the raw side — the original text-as-key approach OOM-killed 16 GiB workers.
    128 bits makes a collision across ~10^8 docs vanishingly unlikely.
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def load_tagged(path: str) -> Iterator[dict]:
    """Load one input file, tagging each record as a survivor (L) or raw (R).

    Both sides key on ``h`` = hash(join text). Survivors carry the actual text
    (it becomes the output); raw records carry only the metadata to ferry.
    Unioning at the file-list level lets a single ``group_by`` co-group both
    sides (one shuffle) without a dataset-level union op.
    """
    is_survivor = SURVIVOR_MARKER in path
    for record in load_file(path):
        key = _join_key(record)
        if key is None:
            continue
        if is_survivor:
            yield {"h": _content_hash(key), "side": "L", "text": key}
        else:
            yield {
                "h": _content_hash(key),
                "side": "R",
                "url": record.get("url") or "",
                "warc_record_id": record.get("warc_record_id") or "",
                "warc_file": record.get("warc_file") or "",
                "snapshot": record.get("snapshot") or "",
            }


def attach(content_hash: str, rows: Iterator[dict]) -> Iterator[dict]:
    """Reducer: emit one row per survivor text with its WARC metadata.

    - Drops hashes that have no survivor (raw-only docs removed by dedup/decon).
    - Collapses cross-WARC duplicate raw rows deterministically (smallest
      ``(warc_file, warc_record_id)`` wins) so provenance is reproducible.
    - A survivor with no raw match (should not happen) is kept with null
      metadata so the output row count always equals the survivor count and the
      validation pass can measure the (expected zero) miss rate.
    """
    text: str | None = None
    best: dict | None = None
    for r in rows:
        if r["side"] == "L":
            text = r["text"]
            continue
        cand = (r["warc_file"], r["warc_record_id"])
        if best is None or cand < (best["warc_file"], best["warc_record_id"]):
            best = r
    if text is None:
        return
    if best is None:
        yield {"text": text, "url": None, "warc_record_id": None, "warc_file": None, "snapshot": None}
    else:
        yield {
            "text": text,
            "url": best["url"],
            "warc_record_id": best["warc_record_id"],
            "warc_file": best["warc_file"],
            "snapshot": best["snapshot"],
        }


def _self_check() -> None:
    """In-process sanity check of `attach` — fails fast with zero infra."""
    raw_only = list(attach("h0", iter([{"side": "R", "warc_file": "wD", "warc_record_id": "r"}])))
    assert raw_only == [], f"raw-only text should be dropped, got {raw_only}"

    orphan = list(attach("h0", iter([{"side": "L", "text": "x"}])))
    assert orphan == [{"text": "x", "url": None, "warc_record_id": None, "warc_file": None, "snapshot": None}], orphan

    matched = list(
        attach(
            "h0",
            iter(
                [
                    {"side": "L", "text": "x"},
                    {"side": "R", "url": "u2", "warc_record_id": "r2", "warc_file": "wB", "snapshot": "s"},
                    {"side": "R", "url": "u1", "warc_record_id": "r1", "warc_file": "wA", "snapshot": "s"},
                ]
            ),
        )
    )
    assert matched == [{"text": "x", "url": "u1", "warc_record_id": "r1", "warc_file": "wA", "snapshot": "s"}], matched
    logger.info("self-check passed: drop-raw-only, null-on-orphan, deterministic dup collapse")


# --------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------


def _glob(pattern: str) -> list[str]:
    fs, path = url_to_fs(pattern)
    return [f"gs://{m}" for m in fs.glob(path)]


def _survivor_files() -> list[str]:
    files = _glob(f"{SURVIVORS_PREFIX}/data-*.jsonl.gz")
    if not files:
        raise RuntimeError(f"no survivor files under {SURVIVORS_PREFIX}")
    return files


def _raw_files() -> list[str]:
    files: list[str] = []
    for region in RAW_REGIONS:
        files.extend(_glob(f"{RAW_BASE}/{region}/high_quality/data-*/batch_*.jsonl.gz"))
    if not files:
        raise RuntimeError(f"no raw batch files under {RAW_BASE}/<region>/high_quality/")
    return files


def _bucket_files(files: list[str], num_buckets: int) -> list[list[str]]:
    """Round-robin files into buckets so each read task handles many files.

    Feeding ~840k individual files to ``from_iterable`` creates one tiny shard
    per file; that shard explosion is what OOM-killed the workers. Bucketing
    (the same trick the dedup pipeline uses) keeps the stage-1 shard count
    bounded while still streaming records through the scatter.
    """
    buckets: list[list[str]] = [[] for _ in range(num_buckets)]
    for i, f in enumerate(files):
        buckets[i % num_buckets].append(f)
    return [b for b in buckets if b]


def load_bucket(paths: list[str]) -> Iterator[dict]:
    for path in paths:
        yield from load_tagged(path)


def _du(prefix: str) -> int:
    fs, path = url_to_fs(prefix)
    return sum(fs.size(m) for m in fs.glob(f"{path}/*.parquet"))


def _write_text(url: str, text: str) -> None:
    fs, path = url_to_fs(url)
    with fs.open(path, "w") as f:
        f.write(text)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def run_join(num_output_shards: int, max_workers: int, num_input_buckets: int) -> None:
    survivor_files = _survivor_files()
    raw_files = _raw_files()
    buckets = _bucket_files(survivor_files + raw_files, num_input_buckets)
    logger.info(
        "join inputs: %d survivor + %d raw files -> %d read buckets",
        len(survivor_files),
        len(raw_files),
        len(buckets),
    )

    pipeline = (
        Dataset.from_iterable(buckets)
        .flat_map(load_bucket)
        .group_by(key=lambda r: r["h"], reducer=attach, num_output_shards=num_output_shards)
        .write_parquet(f"{JOINED_PREFIX}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(
        name="hq-hf-join",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=2, ram="32g", disk="50g"),
    )
    result = ctx.execute(pipeline, verbose=True)
    logger.info("join wrote %d parquet shards -> %s", len(result.results or []), JOINED_PREFIX)


def _warc_reduce(warc_file: str, rows: Iterator[dict]) -> Iterator[dict]:
    docs = 0
    chars = 0
    for r in rows:
        docs += 1
        chars += r["_clen"]
    yield {"warc_file": warc_file, "docs": docs, "chars": chars}


def _samples(n: int = 8) -> list[dict]:
    """Read the first joined shard in-driver (one ~small file) for a schema preview."""
    import pyarrow.parquet as pq

    shard = sorted(_glob(f"{JOINED_PREFIX}/data-*.parquet"))[0]
    fs, path = url_to_fs(shard)
    with fs.open(path, "rb") as f:
        table = pq.read_table(f)
    rows = table.slice(0, n).to_pylist()
    for row in rows:
        row["text"] = (row.get("text") or "")[:240]
    return rows


def run_validate(num_output_shards: int, max_workers: int) -> dict:
    joined = Dataset.from_files(f"{JOINED_PREFIX}/data-*.parquet").load_parquet()
    per_warc = joined.map(lambda r: {"warc_file": r["warc_file"] or "", "_clen": len(r["text"] or "")}).group_by(
        key=lambda r: r["warc_file"], reducer=_warc_reduce, num_output_shards=min(num_output_shards, 128)
    )
    ctx = ZephyrContext(
        name="hq-hf-validate",
        max_workers=max_workers,
        resources=ResourceConfig(cpu=2, ram="8g", disk="10g"),
    )
    rows = ctx.execute(per_warc, verbose=True).results

    total_docs = sum(r["docs"] for r in rows)
    total_chars = sum(r["chars"] for r in rows)
    unmatched = next((r["docs"] for r in rows if r["warc_file"] == ""), 0)
    warc_files = sorted(r["warc_file"] for r in rows if r["warc_file"])
    matched = total_docs - unmatched

    size_bytes = _du(JOINED_PREFIX)
    size_gb = size_bytes / 1e9
    est_tokens = int(total_chars / 4)

    report = {
        "dataset": "high_quality 10364-WARC (decontaminated, deduped) + WARC traceback metadata",
        "inputs": {
            "survivors": SURVIVORS_PREFIX,
            "raw_metadata": f"{RAW_BASE}/<region>/high_quality/",
        },
        "output": JOINED_PREFIX,
        "schema": ["text", "url", "warc_record_id", "warc_file", "snapshot"],
        "total_docs": total_docs,
        "matched_docs": matched,
        "unmatched_docs": unmatched,
        "match_rate": (matched / total_docs) if total_docs else 0.0,
        "distinct_warcs": len(warc_files),
        "total_chars": total_chars,
        "est_tokens_chars_div4": est_tokens,
        "output_size_bytes": size_bytes,
        "output_size_gb": round(size_gb, 2),
        "est_egress_usd": round(size_gb * INTERNET_EGRESS_USD_PER_GB, 2),
        "samples": _samples(),
    }

    _write_text(f"{EXPORT_PREFIX}/source_warcs.txt", "\n".join(warc_files) + "\n")
    _write_text(f"{EXPORT_PREFIX}/REPORT.json", json.dumps(report, indent=2))
    _write_text(f"{EXPORT_PREFIX}/REPORT.md", _render_md(report))
    logger.info(
        "REPORT written -> %s  (docs=%d match_rate=%.4f distinct_warcs=%d size=%.1fGB egress~$%.2f)",
        EXPORT_PREFIX,
        total_docs,
        report["match_rate"],
        report["distinct_warcs"],
        size_gb,
        report["est_egress_usd"],
    )
    return report


def _render_md(r: dict) -> str:
    sample_lines = "\n".join(
        f"- `{s.get('warc_file', '')}` rec=`{s.get('warc_record_id', '')}` "
        f"snap=`{s.get('snapshot', '')}`\n  url: {s.get('url', '')}\n  text: {s.get('text', '')!r}"
        for s in r["samples"]
    )
    return f"""# {r['dataset']}

**STATUS: ready for review — NOT uploaded.** Approve before the HuggingFace push.

## Output
- joined parquet: `{r['output']}`
- schema: `{', '.join(r['schema'])}`
- source WARC manifest: `{EXPORT_PREFIX}/source_warcs.txt`

## Validation
| metric | value |
|---|---|
| total docs | {r['total_docs']:,} |
| matched (have WARC metadata) | {r['matched_docs']:,} |
| unmatched (null metadata) | {r['unmatched_docs']:,} |
| **match rate** | **{r['match_rate']:.4%}** |
| distinct source WARCs | {r['distinct_warcs']:,} |
| total chars | {r['total_chars']:,} |
| est tokens (chars/4) | {r['est_tokens_chars_div4']:,} |

## Upload cost
- on-disk parquet size: **{r['output_size_gb']:.2f} GB**
- est. internet egress to HF (@ ${INTERNET_EGRESS_USD_PER_GB}/GB): **~${r['est_egress_usd']:.2f}**

## Samples
{sample_lines}

## Inputs
- survivors: `{r['inputs']['survivors']}`
- raw metadata: `{r['inputs']['raw_metadata']}`
"""


def run_smoke() -> None:
    """End-to-end zephyr execution check on tiny in-memory data (LocalClient)."""
    _self_check()

    def h(text: str) -> str:
        return _content_hash(text)

    survivors = [
        {"h": h("alpha"), "side": "L", "text": "alpha"},
        {"h": h("beta"), "side": "L", "text": "beta"},
        {"h": h("orphan"), "side": "L", "text": "orphan"},
    ]
    raw = [
        {"h": h("alpha"), "side": "R", "url": "u1b", "warc_record_id": "r1b", "warc_file": "wB", "snapshot": "s1"},
        {"h": h("alpha"), "side": "R", "url": "u1", "warc_record_id": "r1", "warc_file": "wA", "snapshot": "s1"},
        {"h": h("beta"), "side": "R", "url": "u2", "warc_record_id": "r2", "warc_file": "wC", "snapshot": "s2"},
        {"h": h("rawonly"), "side": "R", "url": "u3", "warc_record_id": "r3", "warc_file": "wD", "snapshot": "s3"},
    ]
    ds = Dataset.from_list(survivors + raw).group_by(key=lambda r: r["h"], reducer=attach, num_output_shards=4)
    rows = ZephyrContext(name="hq-hf-smoke").execute(ds).results
    by_text = {r["text"]: r for r in rows}
    assert set(by_text) == {"alpha", "beta", "orphan"}, by_text
    assert by_text["alpha"]["warc_file"] == "wA", by_text["alpha"]
    assert by_text["beta"]["url"] == "u2", by_text["beta"]
    assert by_text["orphan"]["url"] is None, by_text["orphan"]
    logger.info("SMOKE PASSED: join, dup collapse, drop-raw-only, null-on-orphan all correct")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["smoke", "join", "validate", "full"], default="full")
    parser.add_argument("--num-output-shards", type=int, default=512)
    parser.add_argument("--num-input-buckets", type=int, default=2048)
    parser.add_argument("--max-workers", type=int, default=256)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _self_check()

    if args.stage == "smoke":
        run_smoke()
        return
    if args.stage in ("join", "full"):
        run_join(args.num_output_shards, args.max_workers, args.num_input_buckets)
    if args.stage in ("validate", "full"):
        report = run_validate(args.num_output_shards, args.max_workers)
        logger.info("DONE. Review %s/REPORT.md, then run the upload step to ship to HuggingFace.", EXPORT_PREFIX)
        logger.info("match_rate=%.4f egress~$%.2f", report["match_rate"], report["est_egress_usd"])


if __name__ == "__main__":
    main()
