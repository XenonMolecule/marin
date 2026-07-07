# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""v2 Phase C (CPU): JustText on the ModernBERT-kept docs only -> final kept corpus.

The whole point of v2: JustText (the dominant CPU cost) runs ONLY on docs that already passed
ModernBERT (Phase B), i.e. ~5x fewer than v1 (which JustTexts every fastText-survivor). Reads each
WARC's pre-survivor parquet (raw html) + its Phase-B keeplist ({doc_id, prob}); for docs with
prob >= threshold, runs JustText on the html and writes the final ``kept/data-{warc_hash}.parquet``
(identical KEPT_SCHEMA to v1, so downstream dedup/tokenize is unchanged). The keeplist retains every
doc's prob, so re-thresholding stays free.

Standalone claim-based worker (launch many via ``launch_cpu_c.py``); needs justext.

    python -m experiments.fast_curation.cpu_phase_c --spec fastpipe_v2 --shuffle-seed 0
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import logging
import time

import fsspec
import pyarrow as pa

from experiments.fast_curation import batch_format, preprocess
from experiments.fast_curation.cpu_phase import _gcs_exists, _make_justext_pool, _pin_threads, run_claim_loop
from experiments.fast_curation.spec import PipelineSpec, get_spec

logger = logging.getLogger(__name__)

# Sub-WARC checkpoint granularity. JustText on a contended preemptible host is slow, so bank
# progress every CHUNK_SIZE kept docs to GCS: a preemption loses <=1 chunk, not the whole WARC.
CHUNK_SIZE = 512


def _check_deps() -> None:
    if not importlib.util.find_spec("justext"):
        raise RuntimeError("Missing worker dep 'justext'. Launch with --extra cpu --extra extraction-bakeoff.")


def _safe_utf8(s: str) -> str:
    """Drop characters that can't be UTF-8 encoded (lone surrogates from bad-decoded pages).

    Essential: pyarrow's string->UTF-8 serialization raises ``UnicodeEncodeError`` on such a
    character, which crashes the parquet write for the ENTIRE chunk — so one bad document would kill
    the whole WARC's Phase C (this was ~75% of C failures). The dropped bytes are junk anyway.
    """
    return s.encode("utf-8", "ignore").decode("utf-8")


def _chunk_kept_table(table, cidx: list[int], texts: list[str], doc_ids: list[str], prob_of: dict) -> tuple:
    """Build the KEPT_SCHEMA table for one chunk; returns (table, n_justext_empty)."""
    cols: dict[str, list] = {name: [] for name in batch_format.KEPT_SCHEMA.names}
    n_empty = 0
    for j, i in enumerate(cidx):
        if not texts[j]:
            n_empty += 1
            continue  # passed ModernBERT but JustText found no content -> nothing to train on.
        cols["doc_id"].append(doc_ids[i])
        cols["url"].append(_safe_utf8(table.column("url")[i].as_py()))
        cols["warc_hash"].append(table.column("warc_hash")[i].as_py())
        cols["snapshot"].append(table.column("snapshot")[i].as_py())
        cols["fasttext_score"].append(table.column("fasttext_score")[i].as_py())
        cols["text"].append(_safe_utf8(texts[j]))
        cols["input_ids"].append(table.column("input_ids")[i].as_py())
        cols["n_tokens"].append(table.column("n_tokens")[i].as_py())
        cols["modernbert_prob"].append(prob_of[doc_ids[i]])
    return pa.table(cols, schema=batch_format.KEPT_SCHEMA), n_empty


def _process_one(warc_path: str, warc_hash: str, refresh, *, spec: PipelineSpec, pool, bucket: str) -> dict:
    presurvivor_path = f"{spec.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet"
    keeplist_path = f"{spec.keeplist_prefix(bucket)}/data-{warc_hash}.parquet"
    kept_path = f"{spec.kept_prefix(bucket)}/data-{warc_hash}.parquet"
    chunk_dir = f"{spec.namespace(bucket)}/kept_chunks/data-{warc_hash}"

    t0 = time.monotonic()
    kl = batch_format.read_table(keeplist_path)
    prob_of = dict(zip(kl.column("doc_id").to_pylist(), kl.column("modernbert_prob").to_pylist(), strict=True))

    table = batch_format.read_table(presurvivor_path)
    n = table.num_rows
    doc_ids = table.column("doc_id").to_pylist()
    # Rows that passed ModernBERT (prob >= threshold).
    keep_idx = [i for i in range(n) if prob_of.get(doc_ids[i], 0.0) >= spec.modernbert_threshold]

    # JustText in CHUNK_SIZE batches, checkpointing each chunk to GCS (skip already-done chunks on
    # reclaim). A preempted worker loses at most the in-flight chunk.
    chunk_paths: list[str] = []
    t_jt = 0.0
    n_empty = 0
    for ci, cstart in enumerate(range(0, len(keep_idx), CHUNK_SIZE)):
        cpath = f"{chunk_dir}/chunk_{ci:04d}.parquet"
        chunk_paths.append(cpath)
        if _gcs_exists(cpath):
            continue  # banked by an earlier (pre-preemption) run.
        cidx = keep_idx[cstart : cstart + CHUNK_SIZE]
        htmls = [
            (table.column("html")[i].as_py(), spec.justext_lang, spec.justext_max_html_chars, spec.justext_paragraph_sep)
            for i in cidx
        ]
        s = time.monotonic()
        if pool is not None and htmls:
            texts = pool.run(htmls, spec.justext_timeout)
        else:
            texts = [preprocess.justext_text(h, lang, cap, sep) for h, lang, cap, sep in htmls]
        t_jt += time.monotonic() - s
        ctable, ce = _chunk_kept_table(table, cidx, texts, doc_ids, prob_of)
        n_empty += ce
        batch_format.write_kept(cpath, ctable)
        refresh()  # keep the WARC claim fresh between chunks (long contended WARCs)

    # Merge chunks -> final flat kept parquet (downstream reads the flat kept/ layout unchanged).
    tables = [batch_format.read_table(cp) for cp in chunk_paths]
    merged = pa.concat_tables(tables) if tables else batch_format.KEPT_SCHEMA.empty_table()
    batch_format.write_kept(kept_path, merged)

    try:
        with fsspec.open(f"{spec.namespace(bucket)}/timing_c/data-{warc_hash}.json", "w") as f:
            json.dump(
                {
                    "warc_hash": warc_hash,
                    "n_presurvivors": n,
                    "n_modernbert_keep": len(keep_idx),
                    "n_justext_empty": n_empty,
                    "n_kept": merged.num_rows,
                    "n_chunks": len(chunk_paths),
                    "justext_s": round(t_jt, 3),
                    "wall_s": round(time.monotonic() - t0, 3),
                },
                f,
            )
    except Exception as e:
        logger.warning("phase C timing write failed for %s: %s", warc_hash, e)
    wall = time.monotonic() - t0
    logger.info(
        "C %s: %d mb-keep -> %d kept (%d justext-empty, %d chunks) in %.1fs",
        warc_hash,
        len(keep_idx),
        merged.num_rows,
        n_empty,
        len(chunk_paths),
        wall,
    )
    return {
        "docs_in": len(keep_idx),  # docs ModernBERT kept (JustText input)
        "docs_out": merged.num_rows,  # final kept docs after JustText
        "wall_seconds": wall,
        "compute_seconds": {"justext": t_jt},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v3")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=5)
    ap.add_argument("--justext-procs", type=int, default=8)
    ap.add_argument(
        "--claim-stale-hours",
        type=float,
        default=0.25,
        help="Reclaim a dead worker's WARC after this (15 min: refreshed per-chunk, so safe; short "
        "enough that a crashed peer's WARCs free up inside the endgame-drain window in run_claim_loop).",
    )
    args = ap.parse_args()

    _check_deps()
    spec = get_spec(args.spec)
    preprocess.assert_justext_version(spec.justext_version)
    _pin_threads()
    pool = _make_justext_pool(args.justext_procs)
    process_one = functools.partial(_process_one, spec=spec, pool=pool, bucket=args.bucket)

    def _keeplist_path(h: str) -> str:
        return f"{spec.keeplist_prefix(args.bucket)}/data-{h}.parquet"

    run_claim_loop(
        spec,
        args.manifest,
        args.bucket,
        phase_key="c",
        process_one=process_one,
        shuffle_seed=args.shuffle_seed,
        start=args.start,
        limit=args.limit,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
        upstream_path_fn=_keeplist_path,  # only JustText a WARC once Phase B wrote its keeplist
        upstream_done_path=f"gs://marin-us-central1/{spec.subdir()}/_phase_b_end.json",
        # Per-chunk claim refresh keeps live workers fresh; a dead worker's WARC is reclaimed in 30
        # min (then finished from its banked chunks) instead of the default 3h.
        claim_stale_hours=args.claim_stale_hours,
    )
    if pool is not None:
        pool.close()


if __name__ == "__main__":
    main()
