# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract the 100k comparison sample with the XenonMolecule fork's **Rust** main-content engine.

This is ``resiliparse._extract_rs`` from https://github.com/XenonMolecule/chatnoir-resiliparse — NOT
the ``resiliparse`` package marin depends on. The two are different forks that claim the same import
name, so they cannot coexist in one environment:

  * marin's core dep is krypticmouse's fork (``resiliparse.extract.html2text.extract_simplified_dom``
    -> markdownify), which is what the ``resiliparse`` curation corpus was built with.
  * this fork ships a Rust extractor reporting 0.9050 vs 0.8880 token F1 on the marin devset.

Because the Rust engine needs Rust + vcpkg + cmake + libclang to build, it is NOT built here.
``build_resiliparse_rs.py`` publishes a prebuilt Linux artifact to ``ARTIFACT_PREFIX``; this job just
downloads it, puts it on ``sys.path`` and extracts. No toolchain is required on this worker.

Input is the sample's ``raw_html``, which ``score_justext_redecode`` already re-decoded with the
correct charset order (clean and non-null for all 100k, 0 U+FFFD). Output is a row-aligned side
artifact that ``precompute_pipeline_matrix`` merges, same shape as ``build_lpv11_target_columns``.

Run (CPU, us-east5 where the sample lives; no build toolchain needed)::

    uv run iris --cluster marin job run --region us-east5 --cpu 16 --memory 32GB --disk 20GB \\
      --enable-extra-resources --extra cpu --priority interactive --no-wait \\
      --job-name resiliparse-rs -- \\
      python -m experiments.baseline_collection.score_resiliparse_rs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SAMPLE_DIR = f"{OUT_ROOT}/sample_100k_scored"
TEXT_DIR = f"{OUT_ROOT}/text_resiliparse_rs"
# Pages that hard-crash the Rust extractor, recorded for the fork to debug.
CRASH_REPORT = f"{OUT_ROOT}/text_resiliparse_rs_crashes.json"

ARTIFACT_PREFIX = "gs://marin-us-east5/artifacts/resiliparse_rs/latest"
# NOT /tmp: it is mounted noexec on Iris CPU workers, so dlopen() of the extension fails there.
INSTALL_DIR = "/root/resiliparse_rs"
# lexbor is a dylib on vcpkg's x64-linux triplet; the loader resolves it via an $ORIGIN rpath, so it
# must land in the SAME directory as the extension. Order matters: the extension is imported last.
ARTIFACT_LIBS = ("liblexbor.so.2", "_extract_rs.so")
COLUMN = "text_resiliparse_rs"

# Shards hold ~500 docs, so the chunk must be MUCH smaller than a shard or one worker gets the whole
# shard: no parallelism, and a chunk of uncapped multi-MB HTML pickled into a single process (an
# earlier CHUNK_SIZE=500 died with BrokenProcessPool).
CHUNK_SIZE = 25
MAX_HTML_CHARS = 2_000_000  # applied in the PARENT so multi-MB outliers are never shipped to a worker

_SCHEMA = pa.schema([("warc_record_id", pa.string()), (COLUMN, pa.string())])


def install_extractor(artifact_prefix: str = ARTIFACT_PREFIX, dest: str = INSTALL_DIR) -> str:
    """Download the prebuilt Rust extractor and put its package dir on ``sys.path``.

    Returns the package directory. Layout mirrors the fork's own install: the cdylib is dropped in as
    ``resiliparse/_extract_rs.so`` inside the extracted ``resiliparse-py`` tree.
    """
    os.makedirs(dest, exist_ok=True)
    tarball = os.path.join(dest, "resiliparse_py.tar.gz")
    with fsspec.open(f"{artifact_prefix}/resiliparse_py.tar.gz", "rb") as src, open(tarball, "wb") as dst:
        dst.write(src.read())
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)

    pkg_dir = os.path.join(dest, "resiliparse-py")
    if not os.path.isdir(pkg_dir):
        raise RuntimeError(f"expected {pkg_dir} in {artifact_prefix}/resiliparse_py.tar.gz")
    for lib in ARTIFACT_LIBS:
        with fsspec.open(f"{artifact_prefix}/{lib}", "rb") as src:
            payload = src.read()
        with open(os.path.join(pkg_dir, "resiliparse", lib), "wb") as dst:
            dst.write(payload)

    # Prepend so this fork wins over marin's `resiliparse` core dep, which claims the same name.
    sys.path.insert(0, pkg_dir)
    logger.info("installed Rust extractor (%s) from %s -> %s", ", ".join(ARTIFACT_LIBS), artifact_prefix, pkg_dir)
    return pkg_dir


def _extract(htmls: list[str]) -> list[str]:
    """Markdown main-content extraction for a chunk of pages (runs in a worker process).

    Imported inside the worker because the package is downloaded at runtime, not installed.
    """
    from resiliparse._extract_rs import extract_plain_text

    return [extract_plain_text(h, main_content=True, preserve_formatting="markdown") for h in htmls]


def _extract_shard(
    pool: ProcessPoolExecutor, htmls: list[str], ids: list[str]
) -> tuple[list[str | None], list[str], bool]:
    """Extract one shard. Returns ``(texts, crashed_ids, pool_broken)``.

    A page that segfaults the Rust extractor kills the worker and poisons the whole pool
    (``BrokenProcessPool``), which would abort the run — one such page sits in shard 7. When that
    happens the shard is retried one doc per task so the offending ids are pinned down and RECORDED
    rather than silently dropped; their text is null and they are reported at the end. ``pool_broken``
    tells the caller to rebuild even if isolation happened to find no reproducing doc.
    """
    chunks = [htmls[i : i + CHUNK_SIZE] for i in range(0, len(htmls), CHUNK_SIZE)]
    try:
        return [text for part in pool.map(_extract, chunks) for text in part], [], False
    except BrokenProcessPool:
        logger.warning("extractor crashed on this shard — isolating per doc to find the offending page(s)")

    texts: list[str | None] = []
    crashed: list[str] = []
    for rid, html in zip(ids, htmls, strict=True):
        with ProcessPoolExecutor(max_workers=1) as solo:
            try:
                texts.append(next(solo.map(_extract, [[html]]))[0])
            except BrokenProcessPool:
                texts.append(None)
                crashed.append(rid)
                logger.warning("HARD CRASH on %s (%d chars)", rid, len(html))
    return texts, crashed, True


def run(workers: int, overwrite: bool) -> None:
    install_extractor()

    files = sorted(fsspec_glob(f"{SAMPLE_DIR}/*.parquet"))
    if not files:
        raise RuntimeError(f"no sample parquet under {SAMPLE_DIR}")

    # Resume: these run on preemptible workers and Iris restarts the task from scratch, so without
    # this a preemption throws away every shard done so far (observed: ~176 shards lost twice).
    if not overwrite:
        done = {p.rsplit("/", 1)[-1] for p in fsspec_glob(f"{TEXT_DIR}/*.parquet")}
        remaining = [p for p in files if p.rsplit("/", 1)[-1] not in done]
        logger.info("resume: %d/%d shards already written, %d to do", len(done), len(files), len(remaining))
        files = remaining
        if not files:
            logger.info("nothing to do — all shards present under %s", TEXT_DIR)
            return

    total = 0
    extract_time = 0.0
    all_crashed: list[str] = []
    pool = ProcessPoolExecutor(max_workers=workers)
    for shard_i, path in enumerate(files):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "raw_html"])
        ids = t.column("warc_record_id").to_pylist()
        htmls = [(h or "")[:MAX_HTML_CHARS] for h in t.column("raw_html").to_pylist()]

        t0 = time.monotonic()
        texts, crashed, pool_broken = _extract_shard(pool, htmls, ids)
        extract_time += time.monotonic() - t0
        all_crashed.extend(crashed)
        if pool_broken:
            pool.shutdown(wait=False)
            pool = ProcessPoolExecutor(max_workers=workers)  # the old pool is unusable once broken

        table = pa.Table.from_pydict({"warc_record_id": ids, COLUMN: texts}, schema=_SCHEMA)
        out = f"{TEXT_DIR}/{path.rsplit('/', 1)[-1]}"
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        total += len(ids)
        if shard_i % 25 == 0:
            logger.info("shard %d/%d, %d docs done, %d crashes", shard_i, len(files), total, len(all_crashed))
    pool.shutdown()

    # Suffixed so a resumed run cannot clobber the crash list an earlier attempt found.
    report_path = CRASH_REPORT if total == len(files) else CRASH_REPORT.replace(".json", f"_partial_{total}.json")
    with fsspec.open(report_path, "w") as fh:
        json.dump({"crashed_warc_record_ids": all_crashed, "n_docs": total}, fh, indent=2)

    rate = total / extract_time if extract_time else 0.0
    logger.info(
        "TIMING col=%s: extracted %d docs in %.1fs = %.1f docs/s over %d workers = %.1f docs/s/core",
        COLUMN,
        total,
        extract_time,
        rate,
        workers,
        rate / workers,
    )
    logger.info("crashed on %d/%d docs -> %s", len(all_crashed), total, report_path)
    logger.info("DONE -> %s", TEXT_DIR)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 2) - 1))
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract every shard. Default resumes, skipping shards already in TEXT_DIR.",
    )
    args = p.parse_args()
    run(args.workers, args.overwrite)


if __name__ == "__main__":
    main()
