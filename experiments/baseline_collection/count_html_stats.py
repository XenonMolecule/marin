# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-record stats over the 3000-WARC baseline pool.

Same input pool as `count_html_bytes.py`, but instead of summing bytes we
emit four orthogonal stat families per file:

1. doc-size histogram (16 log-2 buckets from <1 KiB to >=16 MiB)
2. tag-vs-text byte split via resiliparse's `extract_plain_text`
3. CC snapshot tag (CC-MAIN-YYYY-WW) per WARC
4. top-2000 domains per file by record count (aggregated to top-N globally)

Per-file checkpointing makes the job preemption-safe: on restart we list
the checkpoint dir and skip already-done files.

Run as a single Iris CPU job in us-central2 (data locality, no egress):

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --region us-central2 --extra cpu --enable-extra-resources \\
        --cpu 16 --memory 32GB --disk 20GB \\
        --priority batch --max-retries 20 \\
        --job-name count-html-stats-3000warc --no-wait \\
        -- python experiments/baseline_collection/count_html_stats.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import urlparse

import fsspec

logger = logging.getLogger(__name__)

INPUT_DIR = "gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5"
CHECKPOINT_DIR = "gs://marin-us-central2/metadata/baseline_3000_html_stats"
SUMMARY_PATH = f"{CHECKPOINT_DIR}/_summary.json"

# Log-2 byte boundaries: [1 KiB, 2 KiB, ..., 16 MiB, inf]. 16 buckets total.
_BUCKET_BOUNDARIES: list[float] = [1 << b for b in range(10, 25)] + [float("inf")]
NUM_BUCKETS = len(_BUCKET_BOUNDARIES)

_SNAPSHOT_RE = re.compile(r"CC-MAIN-\d{4}-\d{2}")

TOP_DOMAINS_PER_FILE = 2000
GIANT_THRESHOLD = 5 * 1024 * 1024  # 5 MiB — skip parse on monster docs
EMPTY_THRESHOLD = 500  # records with html_utf8_bytes < this are "empty/junk"


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".jsonl.gz")


def _extract_snapshot(warc_path: str) -> str:
    m = _SNAPSHOT_RE.search(warc_path or "")
    return m.group(0) if m else "unknown"


def _bucket_idx(n: int) -> int:
    for i, b in enumerate(_BUCKET_BOUNDARIES):
        if n < b:
            return i
    return NUM_BUCKETS - 1


def _domain_of(url: str) -> str:
    try:
        netloc = urlparse(url).netloc
        return netloc.lower() if netloc else ""
    except Exception:
        return ""


def _list_input_files() -> list[str]:
    fs = fsspec.filesystem("gcs")
    prefix = INPUT_DIR.replace("gs://", "")
    paths = fs.ls(prefix)
    return sorted(f"gs://{p}" for p in paths if p.endswith(".jsonl.gz"))


def _list_done_basenames() -> set[str]:
    fs = fsspec.filesystem("gcs")
    prefix = CHECKPOINT_DIR.replace("gs://", "")
    try:
        paths = fs.ls(prefix)
    except FileNotFoundError:
        return set()
    done: set[str] = set()
    for p in paths:
        bn = p.rsplit("/", 1)[-1]
        if bn.endswith(".json") and not bn.startswith("_"):
            done.add(bn.removesuffix(".json"))
    return done


def _process_file(input_path: str) -> dict:
    """Stream one jsonl.gz, accumulate stats, write per-file checkpoint.

    Runs in a fresh `spawn`-context worker — gcsfs's asyncio state does
    not survive `fork()` (would deadlock on first GCS call).
    """
    from resiliparse.extract.html2text import extract_plain_text
    from resiliparse.parse.html import HTMLTree

    basename = _basename(input_path)
    ckpt_path = f"{CHECKPOINT_DIR}/{basename}.json"
    print(f"start: {basename}", flush=True)

    fs = fsspec.filesystem("gcs")
    if fs.exists(ckpt_path.replace("gs://", "")):
        print(f"skip: {basename}", flush=True)
        return {"basename": basename, "skipped": True}

    n_records = 0
    n_empty = 0
    n_giant = 0
    n_parse_failed = 0
    html_utf8_bytes = 0
    text_utf8_bytes = 0
    parsed_html_utf8_bytes = 0  # html bytes for records we successfully parsed

    size_bucket_counts = [0] * NUM_BUCKETS
    domain_records: Counter[str] = Counter()
    domain_html_bytes: Counter[str] = Counter()

    snapshot: str | None = None

    t0 = time.monotonic()
    with fsspec.open(input_path, "rb") as raw:
        with gzip.open(raw, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                html = rec.get("html", "") or ""
                url = rec.get("url", "") or ""

                if snapshot is None:
                    warc_file = (rec.get("metadata") or {}).get("warc_file", "")
                    snapshot = _extract_snapshot(warc_file)

                html_bytes = len(html.encode("utf-8"))
                n_records += 1
                html_utf8_bytes += html_bytes
                size_bucket_counts[_bucket_idx(html_bytes)] += 1

                if html_bytes < EMPTY_THRESHOLD:
                    n_empty += 1

                domain = _domain_of(url)
                if domain:
                    domain_records[domain] += 1
                    domain_html_bytes[domain] += html_bytes

                if html_bytes > GIANT_THRESHOLD:
                    n_giant += 1
                    continue

                try:
                    tree = HTMLTree.parse(html)
                    text = extract_plain_text(tree, main_content=False)
                    text_utf8_bytes += len(text.encode("utf-8"))
                    parsed_html_utf8_bytes += html_bytes
                except Exception:
                    n_parse_failed += 1
                    continue
    elapsed = time.monotonic() - t0

    # Truncate domain counters to top-N to keep checkpoint small.
    top = domain_records.most_common(TOP_DOMAINS_PER_FILE)
    top_domains = [{"domain": d, "n_records": c, "html_bytes": domain_html_bytes[d]} for d, c in top]

    result = {
        "basename": basename,
        "input_path": input_path,
        "snapshot": snapshot or "unknown",
        "n_records": n_records,
        "n_empty": n_empty,
        "n_giant": n_giant,
        "n_parse_failed": n_parse_failed,
        "html_utf8_bytes": html_utf8_bytes,
        "text_utf8_bytes": text_utf8_bytes,
        "parsed_html_utf8_bytes": parsed_html_utf8_bytes,
        "size_bucket_counts": size_bucket_counts,
        "size_bucket_boundaries": [int(b) if b != float("inf") else None for b in _BUCKET_BOUNDARIES],
        "top_domains": top_domains,
        "elapsed_s": round(elapsed, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    with fsspec.open(ckpt_path, "w") as f:
        json.dump(result, f)
    text_frac = (text_utf8_bytes / parsed_html_utf8_bytes) if parsed_html_utf8_bytes else 0.0
    print(
        f"done: {basename} | n={n_records} | text/html={text_frac:.4f} | empty={n_empty} | giant={n_giant} | t={elapsed:.1f}s",
        flush=True,
    )
    return result


def _aggregate() -> dict:
    fs = fsspec.filesystem("gcs")
    prefix = CHECKPOINT_DIR.replace("gs://", "")
    paths = fs.ls(prefix)

    totals = {
        "n_files": 0,
        "n_records": 0,
        "n_empty": 0,
        "n_giant": 0,
        "n_parse_failed": 0,
        "html_utf8_bytes": 0,
        "text_utf8_bytes": 0,
        "parsed_html_utf8_bytes": 0,
        "size_bucket_counts": [0] * NUM_BUCKETS,
    }
    by_snapshot: dict[str, dict] = {}
    domain_records: Counter[str] = Counter()
    domain_html_bytes: Counter[str] = Counter()

    for p in paths:
        bn = p.rsplit("/", 1)[-1]
        if not bn.endswith(".json") or bn.startswith("_"):
            continue
        with fsspec.open(f"gs://{p}", "r") as f:
            rec = json.load(f)
        totals["n_files"] += 1
        for k in (
            "n_records",
            "n_empty",
            "n_giant",
            "n_parse_failed",
            "html_utf8_bytes",
            "text_utf8_bytes",
            "parsed_html_utf8_bytes",
        ):
            totals[k] += rec.get(k, 0)
        for i, c in enumerate(rec.get("size_bucket_counts", [])):
            totals["size_bucket_counts"][i] += c

        ss = rec.get("snapshot", "unknown")
        s = by_snapshot.setdefault(ss, {"n_files": 0, "n_records": 0, "html_utf8_bytes": 0, "text_utf8_bytes": 0})
        s["n_files"] += 1
        s["n_records"] += rec.get("n_records", 0)
        s["html_utf8_bytes"] += rec.get("html_utf8_bytes", 0)
        s["text_utf8_bytes"] += rec.get("text_utf8_bytes", 0)

        for d in rec.get("top_domains", []):
            domain_records[d["domain"]] += d["n_records"]
            domain_html_bytes[d["domain"]] += d["html_bytes"]

    top_n = 200
    top_by_records = [
        {"domain": d, "n_records": c, "html_bytes": domain_html_bytes[d]} for d, c in domain_records.most_common(top_n)
    ]
    top_by_bytes = [
        {"domain": d, "n_records": domain_records[d], "html_bytes": b} for d, b in domain_html_bytes.most_common(top_n)
    ]

    summary = {
        **totals,
        "size_bucket_boundaries": [int(b) if b != float("inf") else None for b in _BUCKET_BOUNDARIES],
        "by_snapshot": by_snapshot,
        "top_domains_by_records": top_by_records,
        "top_domains_by_bytes": top_by_bytes,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
    }
    with fsspec.open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    p.add_argument("--workers", type=int, default=int(os.environ.get("HTML_STATS_WORKERS", "16")))
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args(argv)

    if args.aggregate_only:
        s = _aggregate()
        logger.info(
            "Summary: %d files, %d records, %.3f TB html, %.3f TB text",
            s["n_files"],
            s["n_records"],
            s["html_utf8_bytes"] / 1e12,
            s["text_utf8_bytes"] / 1e12,
        )
        return

    all_files = _list_input_files()
    done = _list_done_basenames()
    todo = [f for f in all_files if _basename(f) not in done]
    logger.info("Total %d files, %d done, %d remaining", len(all_files), len(done), len(todo))

    if not todo:
        s = _aggregate()
        logger.info(
            "Final summary: %s", json.dumps({k: v for k, v in s.items() if not isinstance(v, (list, dict))}, indent=2)
        )
        return

    completed = skipped = failed = 0
    t_start = time.monotonic()
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn_ctx) as pool:
        futures = {pool.submit(_process_file, f): f for f in todo}
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res.get("skipped"):
                    skipped += 1
                else:
                    completed += 1
                if (completed + skipped) % 25 == 0 or (completed + skipped) == len(todo):
                    elapsed = time.monotonic() - t_start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - completed - skipped) / rate if rate > 0 else float("inf")
                    logger.info(
                        "progress: %d/%d done (%d skip, %d fail) | %.2f f/s | ETA %.1f min",
                        completed,
                        len(todo),
                        skipped,
                        failed,
                        rate,
                        eta / 60,
                    )
            except Exception:
                failed += 1
                logger.exception("file failed: %s", futures[fut])

    s = _aggregate()
    logger.info(
        "Done. n_files=%d records=%d html=%.3f TB text=%.3f TB",
        s["n_files"],
        s["n_records"],
        s["html_utf8_bytes"] / 1e12,
        s["text_utf8_bytes"] / 1e12,
    )


if __name__ == "__main__":
    main()
