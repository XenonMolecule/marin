# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an HF-ready distillation dataset from the high_quality 3000-WARC corpus.

The dataset pairs, for every deduped high_quality document:

    raw_html        — original page HTML (the teacher's input)
    reasoning_trace — the teacher's chain-of-thought (inside ``<think>…</think>``)
    final_output    — the cleaned extracted document (the teacher's answer)
    url, warc_file, warc_record_id, snapshot — provenance

It is assembled entirely from data already on GCS (no WARC re-download, no
teacher recompute) and is written to ``gs://marin-us-central2/datasets/
high_quality_3000_distill/``. It is NOT pushed to HuggingFace.

All stages are map-only and per-WARC (one durable output shard per WARC, written
with ``skip_existing``). There is no global shuffle: the cluster preempts
aggressively and a shuffle has no durable intermediate, so a single preemption
would reset all progress. Per-WARC outputs make progress monotonic.

``assemble`` (run in us-central1, where the extraction lives)
    One task per WARC: reads that WARC's canonical high_quality extraction
    batches, computes a content id (``xxh3_128`` of the whitespace-compacted
    text — identical to the dedup pipeline's id), parses the ``<think>``
    reasoning, and dedups by content id *within the WARC*. Writes one
    metadata-only file per WARC to the central2 staging dir (the only data that
    crosses regions, ~40 GB).

``htmljoin`` (run in us-central2, where the 2 TB of HTML lives)
    One task per per-WARC metadata file: streams that WARC's HTML shard, joins
    on the (normalized) WARC-Record-ID, and writes one final Parquet shard.

``negatives`` (optional, run in us-central2)
    Recovers the teacher's ``[NO_USEFUL_CONTENT]`` abstentions by set
    difference (fed-but-not-kept HTML records) and writes them to
    ``data_no_useful/`` with the abstention marker and no reasoning — negatives
    for a useful-vs-not classifier.

``card`` (run in us-central2)
    Writes the HuggingFace dataset card from parquet footers (no data scan).

Caveats (documented in the dataset card):
  * Dedup is *within-WARC* exact (content hash) only; ~3% cross-WARC exact
    duplicates may remain, and upstream fuzzy near-dup removal for N=3000 never
    completed.
  * The teacher's reasoning for abstentions was never persisted, so the
    ``negatives`` rows carry HTML + marker only (empty ``reasoning_trace``).

Usage::

    R=us-central1; S=assemble  # or R=us-central2 S in {htmljoin,negatives,card}
    uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \\
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \\
        --enable-extra-resources --region $R --job-name hq-distill-$S \\
        -- python experiments/baseline_collection/build_hq_distill_dataset.py $S
"""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Iterator

from fray import ResourceConfig
from marin.datakit.normalize import DEFAULT_MAX_WHITESPACE_RUN_CHARS, generate_id
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext
from zephyr.readers import load_jsonl

logger = logging.getLogger(__name__)

# --- Inputs ----------------------------------------------------------------

RESOLVED_MANIFEST = (
    "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated/resolved/resolved_high_quality.jsonl.gz"
)
CONSOLIDATED_ROOT = "gs://marin-us-central1/documents/baseline_llm_extraction_consolidated"
HTML_DIR = "gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5"

# A page was actually shown to the teacher only if its HTML fit the prompt-length
# fast-path filter (``max_doc_tokens=28672`` * 6 chars/token) in
# download_and_extract.py. Longer pages were dropped *before* the model, so they
# are not abstentions. The teacher's abstention marker for the rest:
MAX_FED_HTML_CHARS = 28672 * 6
NO_USEFUL_MARKER = "[NO_USEFUL_CONTENT]"

# Source regional bucket -> consolidated by_region/{region} subdir name. The
# manifest records canonical paths in the *origin* regional bucket; the Phase-D
# transfer mirrored every canonical batch into the us-central1 consolidated
# archive under ``by_region/{region}/``. We read those local copies so the
# assemble stage never crosses regions. Note europe's bucket is ``eu-west4``
# but its by_region subdir is ``europe-west4``.
_BUCKET_TO_REGION = {
    "marin-us-central1": "us-central1",
    "marin-us-east1": "us-east1",
    "marin-us-east5": "us-east5",
    "marin-us-west4": "us-west4",
    "marin-eu-west4": "europe-west4",
}
_EXTRACTION_MARKER = "/documents/baseline_llm_extraction/"

# --- Outputs ---------------------------------------------------------------

DATASET_ROOT = "gs://marin-us-central2/datasets/high_quality_3000_distill"


def _dataset_root(tag: str | None) -> str:
    """Root output dir, optionally suffixed by ``tag`` so smoke tests stay
    isolated from the real run (their partial shards must never let the real
    run's skip_existing skip real shards)."""
    return DATASET_ROOT if not tag else f"{DATASET_ROOT}_{tag}"


def _meta_staging(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/_staging_metadata"


def _final_data(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/data"


# --- Pure helpers (unit-tested in test_build_hq_distill_dataset.py) --------

_WS_RUN = re.compile(r"\s{" + str(DEFAULT_MAX_WHITESPACE_RUN_CHARS + 1) + r",}")
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN = re.compile(r"<think>(.*)", re.DOTALL)
_URN_PREFIX = "<urn:uuid:"


def content_id(text: str) -> str:
    """Reproduce the dedup pipeline's content id: xxh3_128 of the text after
    compacting any whitespace run longer than the cap (matches normalize_step)."""
    compacted = _WS_RUN.sub(lambda m: m.group(0)[:DEFAULT_MAX_WHITESPACE_RUN_CHARS], text)
    return generate_id(compacted)


def parse_reasoning(generated_text: str) -> str:
    """Extract the chain-of-thought between ``<think>`` and ``</think>``.

    Falls back to everything after an unclosed ``<think>`` (truncated
    generations); returns ``""`` when there is no think block at all.
    """
    if not generated_text:
        return ""
    m = _THINK.search(generated_text)
    if m:
        return m.group(1).strip()
    m = _THINK_OPEN.search(generated_text)
    return m.group(1).strip() if m else ""


def normalize_record_id(raw_id: str) -> str:
    """``<urn:uuid:abc…>`` → ``abc…`` to match the extraction's warc_record_id."""
    rid = raw_id.strip()
    if rid.startswith(_URN_PREFIX):
        rid = rid[len(_URN_PREFIX) :]
    if rid.endswith(">"):
        rid = rid[:-1]
    return rid


def remap_to_consolidated(path: str) -> str:
    """Rewrite a manifest canonical path (origin regional bucket) to its local
    us-central1 consolidated-archive copy."""
    if not path.startswith("gs://"):
        raise ValueError(f"expected gs:// path, got {path!r}")
    bucket = path.split("/", 3)[2]
    region = _BUCKET_TO_REGION.get(bucket)
    if region is None:
        raise ValueError(f"unknown source bucket {bucket!r} in {path!r}")
    idx = path.find(_EXTRACTION_MARKER)
    if idx < 0:
        raise ValueError(f"path missing {_EXTRACTION_MARKER!r}: {path!r}")
    rest = path[idx + len(_EXTRACTION_MARKER) :]  # high_quality/data-{hash}/batch_{idx}.jsonl.gz
    return f"{CONSOLIDATED_ROOT}/by_region/{region}/{rest}"


def warc_hash_from_path(path: str) -> str:
    """Extract the 12-hex WARC hash from a ``…/data-{hash}/batch_*`` path."""
    m = re.search(r"/data-([0-9a-f]{12})/", path)
    if not m:
        raise ValueError(f"no data-<hash> segment in {path!r}")
    return m.group(1)


def html_path_for(warc_hash: str) -> str:
    return f"{HTML_DIR}/data-{warc_hash}.jsonl.gz"


# --- Manifest --------------------------------------------------------------


def load_canonical_batches() -> list[dict]:
    """Return ``[{"path": <consolidated path>, "warc_hash": <hash>}, …]`` for
    every non-empty canonical high_quality batch."""
    import gzip
    import json

    import fsspec

    out: list[dict] = []
    skipped_empty = 0
    with fsspec.open(RESOLVED_MANIFEST, "rb") as f, gzip.open(f, "rt") as gz:
        for line in gz:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("num_records") or 0) <= 0:
                skipped_empty += 1
                continue
            consolidated = remap_to_consolidated(row["path"])
            out.append({"path": consolidated, "warc_hash": warc_hash_from_path(consolidated)})
    logger.info("Loaded %d non-empty canonical batches (skipped %d empty)", len(out), skipped_empty)
    if not out:
        raise RuntimeError(f"no canonical batches found in {RESOLVED_MANIFEST}")
    return out


def load_warcs(limit_warcs: int | None) -> list[dict]:
    """Group canonical batches by WARC: ``[{"warc_hash": h, "paths": [...]}, …]``,
    sorted by hash so Zephyr shard assignment (and thus skip_existing) is stable
    across reruns."""
    by_hash: dict[str, list[str]] = {}
    for b in load_canonical_batches():
        by_hash.setdefault(b["warc_hash"], []).append(b["path"])
    warcs = [{"warc_hash": h, "paths": sorted(paths)} for h, paths in sorted(by_hash.items())]
    if limit_warcs is not None:
        warcs = warcs[:limit_warcs]
    logger.info("Grouped into %d WARCs (one durable output shard each)", len(warcs))
    return warcs


# --- Stage: assemble -------------------------------------------------------
#
# Map-only, one task per WARC, one durable output file per WARC (skip_existing).
# No global shuffle: the cluster preempts aggressively, and a shuffle has no
# durable intermediate, so each preemption would reset all progress. Per-WARC
# files make progress monotonic. Dedup is therefore *within-WARC* only (exact
# content id); cross-WARC exact duplicates (~3%) are removed downstream only if
# a global dedup is run separately.


def _warc_to_metadata(warc: dict) -> Iterator[dict]:
    """Read all of one WARC's canonical batches; yield deduped metadata rows."""
    warc_hash = warc["warc_hash"]
    seen: set[str] = set()
    for path in warc["paths"]:
        for r in load_jsonl(path):
            text = r.get("text") or ""
            if not text:
                continue
            cid = content_id(text)
            if cid in seen:  # within-WARC exact dedup
                continue
            seen.add(cid)
            yield {
                "content_id": cid,
                "reasoning_trace": parse_reasoning(r.get("generated_text") or ""),
                "final_output": text,
                "url": r.get("url") or "",
                "warc_file": r.get("warc_file") or "",
                "warc_record_id": r.get("warc_record_id") or "",
                "snapshot": r.get("snapshot") or "",
                "warc_hash": warc_hash,
            }


def run_assemble(limit_warcs: int | None, tag: str | None) -> None:
    warcs = load_warcs(limit_warcs)
    meta_staging = _meta_staging(tag)
    out_template = f"{meta_staging}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz"
    pipeline = Dataset.from_iterable(warcs).flat_map(_warc_to_metadata).write_jsonl(out_template, skip_existing=True)
    ctx = ZephyrContext(name="hq-distill-assemble", max_workers=200, resources=ResourceConfig(cpu=2, ram="16g"))
    ctx.execute(pipeline)
    logger.info("assemble done -> %s", meta_staging)


# --- Stage: htmljoin -------------------------------------------------------

_FINAL_COLUMNS = ("raw_html", "reasoning_trace", "final_output", "url", "warc_file", "warc_record_id", "snapshot")


def _enrich(meta: dict, html: str | None) -> dict:
    return {
        "raw_html": html,
        "reasoning_trace": meta["reasoning_trace"],
        "final_output": meta["final_output"],
        "url": meta["url"],
        "warc_file": meta["warc_file"],
        "warc_record_id": meta["warc_record_id"],
        "snapshot": meta["snapshot"],
    }


def _html_join(warc_hash: str, rows: Iterator[dict]) -> Iterator[dict]:
    """Stream one WARC's HTML shard and attach raw HTML to its metadata rows."""
    from zephyr import counters

    need = {r["warc_record_id"]: r for r in rows}
    path = html_path_for(warc_hash)
    try:
        html_records = load_jsonl(path)
    except FileNotFoundError:
        counters.increment("hq_distill_missing_html_file")
        logger.warning("no HTML shard for warc_hash=%s (%s)", warc_hash, path)
        for meta in need.values():
            yield _enrich(meta, None)
        return

    for h in html_records:
        if not need:
            break
        nid = normalize_record_id(h.get("id") or "")
        meta = need.pop(nid, None)
        if meta is not None:
            yield _enrich(meta, h.get("html"))
    # Records whose HTML was not found in the shard (should be rare).
    for meta in need.values():
        counters.increment("hq_distill_missing_html_record")
        yield _enrich(meta, None)


def _join_metadata_file(meta_path: str) -> Iterator[dict]:
    """Join one WARC's metadata file (all rows share a warc_hash) to its HTML shard."""
    rows = list(load_jsonl(meta_path))
    if not rows:
        return
    yield from _html_join(rows[0]["warc_hash"], iter(rows))


def run_htmljoin(tag: str | None) -> None:
    import pyarrow as pa

    # Explicit all-string schema so every shard types identically — otherwise a
    # shard whose raw_html happened to be all-null would infer a null-typed
    # column and break dataset loading across shards.
    schema = pa.schema([(c, pa.string()) for c in _FINAL_COLUMNS])
    # One task per per-WARC metadata file → one parquet shard each. Map-only and
    # durable (skip_existing): preemptions only lose in-flight WARCs.
    out_template = f"{_final_data(tag)}/data-{{shard:05d}}-of-{{total:05d}}.parquet"
    pipeline = (
        Dataset.from_files(f"{_meta_staging(tag)}/*.jsonl.gz")
        .flat_map(_join_metadata_file)
        .write_parquet(out_template, schema=schema, skip_existing=True)
    )
    ctx = ZephyrContext(name="hq-distill-htmljoin", max_workers=256, resources=ResourceConfig(cpu=2, ram="24g"))
    ctx.execute(pipeline)
    logger.info("htmljoin done -> %s", _final_data(tag))


# --- Stage: negatives ------------------------------------------------------
#
# The teacher's [NO_USEFUL_CONTENT] abstentions were never persisted, but we can
# recover them by set difference: a fed page (HTML <= MAX_FED_HTML_CHARS) that is
# NOT in the kept set is an abstention. We attach its raw HTML and the abstention
# marker (no reasoning) — enough to train a useful-vs-not classifier. Same
# durable per-WARC, map-only pattern as the join; runs in us-central2.


def _no_useful_dir(tag: str | None) -> str:
    return f"{_dataset_root(tag)}/data_no_useful"


def _negatives_for_metadata_file(meta_path: str) -> Iterator[dict]:
    """For one WARC: emit fed-but-not-kept HTML records as abstention rows."""
    from zephyr import counters

    meta = list(load_jsonl(meta_path))
    if not meta:
        return
    warc_hash = meta[0]["warc_hash"]
    warc_file = meta[0]["warc_file"]
    snapshot = meta[0]["snapshot"]
    kept = {m["warc_record_id"] for m in meta}
    try:
        html_records = load_jsonl(html_path_for(warc_hash))
    except FileNotFoundError:
        counters.increment("hq_distill_neg_missing_html_file")
        return
    seen_html: set[str] = set()
    for h in html_records:
        html = h.get("html") or ""
        if not html or len(html) > MAX_FED_HTML_CHARS:
            continue  # empty, or never shown to the teacher (length-filtered)
        nid = normalize_record_id(h.get("id") or "")
        if nid in kept:
            continue  # this page was kept — it's a positive
        hid = generate_id(html)
        if hid in seen_html:
            continue  # within-WARC duplicate page
        seen_html.add(hid)
        yield {
            "raw_html": html,
            "reasoning_trace": "",
            "final_output": NO_USEFUL_MARKER,
            "url": h.get("url") or "",
            "warc_file": (h.get("metadata") or {}).get("warc_file") or warc_file,
            "warc_record_id": nid,
            "snapshot": snapshot,
        }


def run_negatives(tag: str | None, limit_files: int | None) -> None:
    import pyarrow as pa
    from marin.utils import fsspec_glob

    schema = pa.schema([(c, pa.string()) for c in _FINAL_COLUMNS])
    all_files = sorted(fsspec_glob(f"{_meta_staging(tag)}/*.jsonl.gz"))
    if not all_files:
        raise RuntimeError(f"no staging metadata under {_meta_staging(tag)}; run assemble first")
    total = len(all_files)
    # ``limit_files`` processes only the first N WARCs (a real smoke whose shards
    # are valid output): shard indices and ``total`` match the full run, so the
    # full run's skip_existing reuses them.
    process = all_files[:limit_files] if limit_files else all_files
    out_template = f"{_no_useful_dir(tag)}/data-{{shard:05d}}-of-{total:05d}.parquet"
    pipeline = (
        Dataset.from_iterable(process)
        .flat_map(_negatives_for_metadata_file)
        .write_parquet(out_template, schema=schema, skip_existing=True)
    )
    ctx = ZephyrContext(name="hq-distill-negatives", max_workers=256, resources=ResourceConfig(cpu=2, ram="24g"))
    ctx.execute(pipeline)
    logger.info("negatives done (%d/%d WARCs) -> %s", len(process), total, _no_useful_dir(tag))


# --- Stage: card -----------------------------------------------------------

_CARD_TEMPLATE = """\
---
license: cc-by-4.0
language:
- en
size_categories:
- 10M<n<100M
task_categories:
- text-generation
tags:
- distillation
- reasoning
- common-crawl
- web-extraction
dataset_info:
  features:
  - name: raw_html
    dtype: string
  - name: reasoning_trace
    dtype: string
  - name: final_output
    dtype: string
  - name: url
    dtype: string
  - name: warc_file
    dtype: string
  - name: warc_record_id
    dtype: string
  - name: snapshot
    dtype: string
  splits:
  - name: train
    num_examples: {num_rows}
  download_size: {size_bytes}
---

# High-Quality Web Extraction — Distillation Set (3000 WARCs)

Teacher traces for distilling a high-quality web-content extractor. Each row pairs
the **raw HTML** of a Common Crawl page (the teacher's input) with the teacher's
**reasoning trace** and **final extracted document** (its output). Built for
distillation with or without the reasoning trace.

## Columns

| column | description |
|---|---|
| `raw_html` | Original page HTML from the source WARC record (teacher input). |
| `reasoning_trace` | Teacher chain-of-thought (the `<think>…</think>` block). |
| `final_output` | Cleaned extracted document (teacher answer). |
| `url` | Original page URL. |
| `warc_file` | Source Common Crawl WARC (S3 path). |
| `warc_record_id` | WARC-Record-ID of the source page. |
| `snapshot` | Common Crawl snapshot (`CC-MAIN-YYYY-WW`). |

## Provenance & method

Pages are the first 3000 WARCs of a Common Crawl draw. An LLM extractor was run
under a strict "high-quality" spec; for each page that cleared the bar it emits a
chain-of-thought and a cleaned main-content document. This dataset re-joins those
teacher outputs to the original page HTML.

## Statistics

- Documents: **{num_rows:,}**
- Parquet shards: {num_shards}
- On-disk size: **{size_gib:.1f} GiB**
- Rows with missing `raw_html`: {null_html:,} ({null_pct:.4f}%)

## Caveats

- **`data/` is useful documents only.** Pages the teacher judged to have no
  useful content (`[NO_USEFUL_CONTENT]`) are not in `data/`. They are provided
  separately in `data_no_useful/` (see below) with their raw HTML and the
  abstention marker, but **without reasoning** — the extraction pipeline
  discarded abstentions before persisting, so their reasoning was never saved.
- **Within-WARC exact dedup.** Documents are deduplicated by exact content hash
  (xxh3_128 of the extracted text) within each source WARC. A small fraction
  (~3%) of cross-WARC exact duplicates may remain, and upstream fuzzy
  near-duplicate removal for the N=3000 corpus did not complete, so
  near-duplicates may also remain.
- Prompt scaffolding (the extraction spec / system prompt) is not included.

## License

Released under CC-BY-4.0. Underlying page content is from Common Crawl; see the
Common Crawl terms of use.
"""


def _parquet_dir_stats(dirpath: str) -> dict | None:
    """Footer-only (rows, bytes, shards, raw_html null_count) for a parquet dir, or None if empty."""
    import fsspec
    import pyarrow.parquet as pq
    from marin.utils import fsspec_glob

    files = sorted(fsspec_glob(f"{dirpath}/*.parquet"))
    if not files:
        return None
    num_rows = size_bytes = null_html = 0
    html_idx = _FINAL_COLUMNS.index("raw_html")
    for path in files:
        fs, resolved = fsspec.core.url_to_fs(path)
        size_bytes += fs.size(resolved)
        md = pq.ParquetFile(path).metadata
        num_rows += md.num_rows
        for rg in range(md.num_row_groups):
            col = md.row_group(rg).column(html_idx)
            if col.is_stats_set and col.statistics is not None:
                null_html += col.statistics.null_count
    return {"num_rows": num_rows, "size_bytes": size_bytes, "num_shards": len(files), "null_html": null_html}


def run_card(tag: str | None) -> None:
    """Write a HuggingFace dataset card with footer-derived stats (no data scan)."""
    import fsspec

    pos = _parquet_dir_stats(_final_data(tag))
    if pos is None:
        raise RuntimeError(f"no parquet shards under {_final_data(tag)}")

    card = _CARD_TEMPLATE.format(
        num_rows=pos["num_rows"],
        size_bytes=pos["size_bytes"],
        size_gib=pos["size_bytes"] / 1024**3,
        num_shards=pos["num_shards"],
        null_html=pos["null_html"],
        null_pct=100 * pos["null_html"] / max(pos["num_rows"], 1),
    )

    neg = _parquet_dir_stats(_no_useful_dir(tag))
    if neg is not None:
        card += (
            "\n## NO_USEFUL classifier negatives (`data_no_useful/`)\n\n"
            "Pages shown to the teacher that it did **not** keep (abstentions), with their raw HTML and "
            f'`final_output = "{NO_USEFUL_MARKER}"` (no reasoning). Use as negative examples for a '
            'useful-vs-not classifier (label = `final_output == "[NO_USEFUL_CONTENT]"`).\n\n'
            f"- Negatives: **{neg['num_rows']:,}**\n"
            f"- Parquet shards: {neg['num_shards']}\n"
            f"- On-disk size: **{neg['size_bytes'] / 1024**3:.1f} GiB**\n"
            f"- Rows with missing `raw_html`: {neg['null_html']:,}\n"
        )

    readme_path = f"{_dataset_root(tag)}/README.md"
    with fsspec.open(readme_path, "w") as f:
        f.write(card)
    logger.info(
        "card written -> %s (positives=%d %.1f GiB; negatives=%s)",
        readme_path,
        pos["num_rows"],
        pos["size_bytes"] / 1024**3,
        f"{neg['num_rows']}" if neg else "none",
    )


# --- Entry point -----------------------------------------------------------


def main() -> None:
    configure_logging(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    p_assemble = sub.add_parser("assemble", help="Stage 1: assemble + dedup metadata (run in us-central1).")
    p_assemble.add_argument(
        "--limit-warcs", type=int, default=None, help="Smoke test: only process the first N WARC hashes."
    )
    p_htmljoin = sub.add_parser("htmljoin", help="Stage 2: join HTML + write parquet (run in us-central2).")
    p_negatives = sub.add_parser(
        "negatives", help="Optional: emit [NO_USEFUL_CONTENT] abstentions to data_no_useful/ (run in us-central2)."
    )
    p_negatives.add_argument(
        "--limit-files", type=int, default=None, help="Smoke test: only process the first N per-WARC metadata files."
    )
    p_card = sub.add_parser("card", help="Stage 3: write the dataset card from parquet footers (run in us-central2).")
    for p in (p_assemble, p_htmljoin, p_negatives, p_card):
        p.add_argument(
            "--tag",
            default=None,
            help="Suffix the output dataset dir (e.g. 'smoke') to isolate test runs from the real dataset.",
        )
    args = parser.parse_args()

    if args.stage == "assemble":
        run_assemble(args.limit_warcs, args.tag)
    elif args.stage == "htmljoin":
        run_htmljoin(args.tag)
    elif args.stage == "negatives":
        run_negatives(args.tag, args.limit_files)
    elif args.stage == "card":
        run_card(args.tag)


if __name__ == "__main__":
    main()
