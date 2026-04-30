# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count plain-text HTML bytes across the 3000-WARC baseline pool.

The records in `gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5/`
are already de-WARCed (each line is `{id, html, url, metadata}`), so we just
stream-read every JSONL.gz, sum `len(rec["html"].encode("utf-8"))`, and write
a per-file checkpoint to GCS.

Per-file checkpointing makes the job preemption-safe: on restart we list
checkpoint dir and skip files already done. Max work lost on preemption is
one in-progress file (~1-2 min).

Run as a single Iris CPU job in us-central2 (data locality):

    uv run iris --config lib/iris/examples/marin.yaml job run \\
        --region us-central2 --extra cpu --enable-extra-resources \\
        --cpu 16 --memory 32GB --disk 20GB \\
        --priority batch --max-retries 20 \\
        --job-name count-html-bytes-3000warc --no-wait \\
        -- python experiments/baseline_collection/count_html_bytes.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import fsspec

logger = logging.getLogger(__name__)

INPUT_DIR = "gs://marin-us-central2/raw/commoncrawl/baseline_3000-265ff5"
CHECKPOINT_DIR = "gs://marin-us-central2/metadata/baseline_3000_html_byte_counts"
SUMMARY_PATH = f"{CHECKPOINT_DIR}/_summary.json"


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".jsonl.gz")


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
    """Stream a single jsonl.gz and sum HTML bytes/chars/records.

    Runs in a fresh `spawn`-context worker — each worker imports its own
    fsspec/gcsfs from scratch so no async event-loop state is inherited
    from the parent (which would deadlock on first GCS call).
    """
    basename = _basename(input_path)
    ckpt_path = f"{CHECKPOINT_DIR}/{basename}.json"
    print(f"start: {basename}", flush=True)

    # Idempotency guard: another worker may have completed it between the
    # initial listing and now.
    fs = fsspec.filesystem("gcs")
    if fs.exists(ckpt_path.replace("gs://", "")):
        print(f"skip: {basename}", flush=True)
        return {"basename": basename, "skipped": True}

    n_records = 0
    html_chars = 0
    html_utf8_bytes = 0
    url_utf8_bytes = 0
    json_line_bytes = 0

    t0 = time.monotonic()
    with fsspec.open(input_path, "rb") as raw:
        with gzip.open(raw, "rt", encoding="utf-8") as f:
            for line in f:
                json_line_bytes += len(line.encode("utf-8"))
                rec = json.loads(line)
                html = rec.get("html", "") or ""
                url = rec.get("url", "") or ""
                n_records += 1
                html_chars += len(html)
                html_utf8_bytes += len(html.encode("utf-8"))
                url_utf8_bytes += len(url.encode("utf-8"))
    elapsed = time.monotonic() - t0

    result = {
        "basename": basename,
        "input_path": input_path,
        "n_records": n_records,
        "html_chars": html_chars,
        "html_utf8_bytes": html_utf8_bytes,
        "url_utf8_bytes": url_utf8_bytes,
        "uncompressed_jsonl_bytes": json_line_bytes,
        "elapsed_s": round(elapsed, 2),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    with fsspec.open(ckpt_path, "w") as f:
        json.dump(result, f)
    print(
        f"done: {basename} | n={n_records} | html_utf8={html_utf8_bytes} | t={elapsed:.1f}s",
        flush=True,
    )
    return result


def _aggregate() -> dict:
    """Read all per-file checkpoints and emit a summary."""
    fs = fsspec.filesystem("gcs")
    prefix = CHECKPOINT_DIR.replace("gs://", "")
    paths = fs.ls(prefix)
    totals = {
        "n_files": 0,
        "n_records": 0,
        "html_chars": 0,
        "html_utf8_bytes": 0,
        "url_utf8_bytes": 0,
        "uncompressed_jsonl_bytes": 0,
    }
    for p in paths:
        bn = p.rsplit("/", 1)[-1]
        if not bn.endswith(".json") or bn.startswith("_"):
            continue
        with fsspec.open(f"gs://{p}", "r") as f:
            rec = json.load(f)
        totals["n_files"] += 1
        for k in ("n_records", "html_chars", "html_utf8_bytes", "url_utf8_bytes", "uncompressed_jsonl_bytes"):
            totals[k] += rec.get(k, 0)
    totals["aggregated_at"] = datetime.now(timezone.utc).isoformat()
    with fsspec.open(SUMMARY_PATH, "w") as f:
        json.dump(totals, f, indent=2)
    return totals


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--workers", type=int, default=int(os.environ.get("HTML_BYTES_WORKERS", "16")))
    p.add_argument("--aggregate-only", action="store_true", help="Skip processing; just rebuild summary.")
    args = p.parse_args(argv)

    if args.aggregate_only:
        summary = _aggregate()
        logger.info("Aggregate-only summary: %s", json.dumps(summary, indent=2))
        return

    all_files = _list_input_files()
    done = _list_done_basenames()
    todo = [f for f in all_files if _basename(f) not in done]
    logger.info("Total %d files, %d done, %d remaining", len(all_files), len(done), len(todo))

    if not todo:
        logger.info("All files already processed. Aggregating.")
        summary = _aggregate()
        logger.info("Final summary: %s", json.dumps(summary, indent=2))
        return

    completed = 0
    skipped = 0
    failed = 0
    t_start = time.monotonic()
    # `spawn` start method is critical: gcsfs (used by fsspec for GCS) holds
    # async event-loop state that does not survive `fork()`. Forked workers
    # silently deadlock on their first GCS call. spawn rebuilds state cleanly.
    spawn_ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn_ctx) as pool:
        futures = {pool.submit(_process_file, f): f for f in todo}
        for fut in as_completed(futures):
            fpath = futures[fut]
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
                        "progress: %d/%d completed (%d skipped, %d failed) | %.2f files/s | ETA %.1f min",
                        completed,
                        len(todo),
                        skipped,
                        failed,
                        rate,
                        eta / 60.0,
                    )
            except Exception as e:
                failed += 1
                logger.exception("Failed on %s: %s", fpath, e)

    logger.info("Done: %d completed, %d skipped, %d failed", completed, skipped, failed)
    summary = _aggregate()
    logger.info("Final summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
