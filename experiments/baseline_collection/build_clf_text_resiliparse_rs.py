# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize the TEXT-classifier input for the 100k comparison sample (row-aligned side artifact).

TEXT-trained lpv11 classifiers (fastText ``*_TEXT``, ``mb-clf-lpv11-text-*``) were trained on main
content extracted by the XenonMolecule fork's **Rust** ``resiliparse._extract_rs`` from the *body_strip*
HTML the HTML classifiers consume — i.e. HTML that was ALREADY lowercased, whitespace-collapsed and
``<head>``-stripped — and then whitespace-collapsed + lowercased again (``extract_prep_text_rs``).

That is NOT the sample's ``text_resiliparse_rs`` column, which is the extractor-comparison text
(``extract_plain_text(raw_html)``, un-normalized) used for Levenshtein against the targets. Scoring a
TEXT model over that column would be a train/serve mismatch. This job writes the faithful input:

    clf_text_resiliparse_rs = extract_clf_text(preprocess_html(stripped_html))

with the SAME transform function the prep shards were built with, so doc *i* is doc *i* across all
classifier families. Empty / frameset / panicking docs are "" (fastText feeds "" as trained; the
neural TEXT scorers map "" -> ``__empty__``, matching ``extract_text_shards.EMPTY_PLACEHOLDER``).

Output: ``comparison_sample.clf_text_dir(bucket)/sample-XXXXX-of-00200.parquet`` (warc_record_id, col),
shard order == the sample's sorted shard order, so ``precompute_pipeline_matrix._read_side_artifact``
and the scorers can consume it row-for-row. Resumable (skips written shards).

Run (CPU, us-east5 where the sample lives)::

    uv run iris --cluster marin job run --region us-east5 --cpu 16 --memory 32GB \\
      --enable-extra-resources --extra cpu --priority interactive --no-wait \\
      --job-name clf-text-rs -- \\
      python -m experiments.baseline_collection.build_clf_text_resiliparse_rs
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.baseline_collection.comparison_sample import CLF_TEXT_COLUMN, clf_text_dir, preprocess_html, sample_dir
from experiments.baseline_collection.extract_prep_text_rs import _extractor, extract_clf_text, install_extractor
from experiments.baseline_collection.score_resiliparse_rs import ARTIFACT_PREFIX
from experiments.fsspec_paths import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLUMN = CLF_TEXT_COLUMN
PRESERVE_FORMATTING = "markdown"  # what every TEXT track was built with (full_prep_resiliparse_rs stats)
CHUNK_SIZE = 25  # << shard size (~500) so a shard fans out across the pool
_SCHEMA = pa.schema([("warc_record_id", pa.string()), (COLUMN, pa.string())])


def _extract_chunk(htmls: list[str]) -> list[tuple[str, bool, bool]]:
    extract = _extractor(ARTIFACT_PREFIX)
    return [extract_clf_text(h, extract, PRESERVE_FORMATTING) for h in htmls]


def _extract_shard(pool: ProcessPoolExecutor, htmls: list[str], ids: list[str]) -> tuple[list[str], int, int, bool]:
    """Returns (texts, n_frameset, n_panic, pool_broken). A segfault (frameset screen should prevent it)
    poisons the pool; then isolate per doc so one crash costs one doc, recorded as ""."""
    chunks = [htmls[i : i + CHUNK_SIZE] for i in range(0, len(htmls), CHUNK_SIZE)]
    try:
        res = [r for part in pool.map(_extract_chunk, chunks) for r in part]
        return [t for t, _, _ in res], sum(f for _, f, _ in res), sum(p for _, _, p in res), False
    except BrokenProcessPool:
        logger.warning("extractor crashed on this shard — isolating per doc")
    texts: list[str] = []
    n_fs = n_pn = 0
    for rid, html in zip(ids, htmls, strict=True):
        with ProcessPoolExecutor(max_workers=1) as solo:
            try:
                t, fs, pn = next(solo.map(_extract_chunk, [[html]]))[0]
                texts.append(t)
                n_fs += fs
                n_pn += pn
            except BrokenProcessPool:
                texts.append("")
                logger.warning('HARD CRASH on %s (%d chars) -> ""', rid, len(html))
    return texts, n_fs, n_pn, True


def run(input_bucket: str, workers: int, overwrite: bool) -> None:
    # Install ONCE in the parent before the pool exists: forked workers inherit the installed dir and
    # sys.path, so `_extractor` never downloads. Letting each worker install races on the same .so
    # ("file too short" ImportError from a half-written extension).
    install_extractor(ARTIFACT_PREFIX)
    out_dir = clf_text_dir(input_bucket)
    files = sorted(fsspec_glob(f"{sample_dir(input_bucket)}/*.parquet"))
    if not files:
        raise RuntimeError(f"no sample parquet under {sample_dir(input_bucket)}")
    n_shards = len(files)
    todo = list(enumerate(files))
    if not overwrite:
        done = {p.rsplit("/", 1)[-1] for p in fsspec_glob(f"{out_dir}/*.parquet")}
        todo = [(i, p) for i, p in todo if f"sample-{i:05d}-of-{n_shards:05d}.parquet" not in done]
        logger.info("resume: %d/%d shards already written", n_shards - len(todo), n_shards)
        if not todo:
            return

    total = n_fs = n_pn = 0
    extract_time = 0.0
    pool = ProcessPoolExecutor(max_workers=workers)
    for k, (i, path) in enumerate(todo):
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read(columns=["warc_record_id", "stripped_html"])
        ids = t.column("warc_record_id").to_pylist()
        htmls = [preprocess_html(h) for h in t.column("stripped_html").to_pylist()]
        t0 = time.monotonic()
        texts, fs, pn, broken = _extract_shard(pool, htmls, ids)
        extract_time += time.monotonic() - t0
        n_fs += fs
        n_pn += pn
        if broken:
            pool.shutdown(wait=False)
            pool = ProcessPoolExecutor(max_workers=workers)
        table = pa.Table.from_pydict({"warc_record_id": ids, COLUMN: texts}, schema=_SCHEMA)
        with fsspec.open(f"{out_dir}/sample-{i:05d}-of-{n_shards:05d}.parquet", "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        total += len(ids)
        if k % 25 == 0:
            logger.info("shard %d/%d, %d docs, frameset=%d panic=%d", k, len(todo), total, n_fs, n_pn)
    pool.shutdown()
    rate = total / extract_time if extract_time else 0.0
    logger.info(
        "TIMING col=%s: %d docs in %.1fs = %.1f docs/s over %d workers = %.1f docs/s/core; frameset=%d panic=%d",
        COLUMN,
        total,
        extract_time,
        rate,
        workers,
        rate / workers,
        n_fs,
        n_pn,
    )
    logger.info("DONE -> %s", out_dir)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-bucket", default="marin-us-east5")
    p.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 2) - 1))
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    run(a.input_bucket, a.workers, a.overwrite)


if __name__ == "__main__":
    main()
