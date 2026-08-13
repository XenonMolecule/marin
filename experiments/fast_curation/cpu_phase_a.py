# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""v2 Phase A (CPU): decode -> fastText gate -> tokenize. NO JustText (deferred to Phase C).

The v2 reorder runs ModernBERT BEFORE JustText so JustText (the dominant CPU cost) only touches
ModernBERT-survivors. Phase A is therefore cheap per WARC (decode + fastText + tokenize, no
JustText) and writes one ``a_presurvivors/data-{warc_hash}.parquet`` carrying the RAW html (Phase C
runs JustText on it) + the pre-tokenized ids (Phase B scores them).

Standalone claim-based worker (launch many via ``launch_cpu_a.py``); only needs fasttext + the
ModernBERT tokenizer (no extraction-bakeoff/justext extra).

    python -m experiments.fast_curation.cpu_phase_a --spec fastpipe_v2 --shuffle-seed 0
"""

from __future__ import annotations

import os

# Re-enable the Rust tokenizer's cross-document Rayon parallelism for the batched tokenize in
# ``_process_one``. Iris/Fray inject ``TOKENIZERS_PARALLELISM=false`` (a fork-deadlock guard), which
# silently serializes ``encode_batch`` and would erase the batch speedup. Phase A never forks
# (JustText — the only ProcessPool user — is deferred to Phase C), so ``"true"`` is safe here. Must
# be set before the tokenizer's first ``encode_batch``; module import is the earliest safe point.
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import argparse
import functools
import importlib
import json
import logging
import random
import time

import fsspec
import pyarrow as pa

from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc, _load_manifest, _warc_path_hash
from experiments.baseline_collection.run_extract_standalone import _claim_warc_atomic, _load_completed_registry
from experiments.fast_curation import batch_format, preprocess
from experiments.fast_curation.cpu_phase import _gcs_exists, _load_registry_mtimes, run_claim_loop
from experiments.fast_curation.spec import PipelineSpec, get_spec
from experiments.fast_curation.telemetry import Heartbeat, region_from_bucket

logger = logging.getLogger(__name__)

# Sub-WARC checkpoint granularity for Phase A (INPUT records per chunk). Phase A is ~12 min/WARC, so
# without this a preemption re-does the whole WARC; here it resumes from banked chunks. Tune below the
# low percentile of the per-region preemption lifetime (see scratch/preempt_analyze.py).
CHUNK_RECORDS = 5000


def _check_deps() -> None:
    missing = [m for m in ("fasttext", "transformers") if not importlib.util.find_spec(m)]
    if missing:
        raise RuntimeError(f"Missing worker deps {missing}. Launch with --extra cpu --extra dclm.")


def _process_one(
    warc_path: str,
    warc_hash: str,
    refresh,
    *,
    spec: PipelineSpec,
    model,
    tokenizer,
    bucket: str,
    chunk_records: int = CHUNK_RECORDS,
    cleanup_chunks: bool = True,
) -> dict:
    t0 = time.monotonic()
    records = _decode_one_warc(warc_path)
    t_decode = time.monotonic() - t0
    n_in = len(records)
    chunk_dir = f"{spec.namespace(bucket)}/a_chunks/data-{warc_hash}"
    presurvivor_path = f"{spec.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet"

    # Checkpoint every ``chunk_records`` INPUT records: Phase A is ~12 min/WARC, so a preempted worker
    # resumes from banked chunks instead of re-doing the whole WARC. Each chunk still fastText-gates
    # then runs ONE batched tokenize over its survivors (preserving the Rayon speedup).
    chunk_paths: list[str] = []
    t_ft = t_tok = 0.0
    for ci, cstart in enumerate(range(0, n_in, chunk_records)):
        cpath = f"{chunk_dir}/chunk_{ci:05d}.parquet"
        chunk_paths.append(cpath)
        if _gcs_exists(cpath):
            continue  # banked by an earlier (pre-preemption) run
        rows: list[dict] = []
        survivor_texts: list[str] = []
        for r in records[cstart : cstart + chunk_records]:
            s = time.monotonic()
            prob = preprocess.fasttext_useful_prob(model, r["text_body"])
            t_ft += time.monotonic() - s
            if prob < spec.fasttext_threshold:
                continue
            # Parity-safe html cap: a page over the spec's JustText size cap is skipped in Phase C
            # (justext_text returns "" -> dropped as no-content), so storing "" here yields the
            # identical drop while bounding per-row memory + the parquet cell size. With v3's 50MB cap
            # this only fires on genuinely pathological pages, not real long articles.
            html = r["html"] if len(r["html"]) <= spec.justext_max_html_chars else ""
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "url": r["url"],
                    "warc_hash": r["warc_hash"],
                    "snapshot": r["snapshot"],
                    "fasttext_score": float(prob),
                    "html": html,
                }
            )
            survivor_texts.append(r["text_body"])
        # ONE batched tokenize over this chunk's survivors (Rayon-parallel across docs).
        s = time.monotonic()
        ids_list = preprocess.tokenize_trunc_batch(tokenizer, survivor_texts, spec.max_length)
        t_tok += time.monotonic() - s
        for row, ids in zip(rows, ids_list, strict=True):
            row["input_ids"] = ids
            row["n_tokens"] = len(ids)
        batch_format.write_presurvivors(cpath, rows)
        refresh()  # keep the WARC claim fresh between chunks (Phase A is long)
    del records

    # Merge chunks -> final flat presurvivor (downstream reads a_presurvivors/ unchanged).
    tables = [batch_format.read_table(cp) for cp in chunk_paths]
    merged = pa.concat_tables(tables) if tables else batch_format.PRESURVIVOR_SCHEMA.empty_table()
    batch_format.write_table(presurvivor_path, merged)
    n_presurv = merged.num_rows
    del tables, merged
    if cleanup_chunks:
        batch_format.drop_chunk_dir(chunk_dir)

    try:
        with fsspec.open(f"{spec.namespace(bucket)}/timing_a/data-{warc_hash}.json", "w") as f:
            json.dump(
                {
                    "warc_hash": warc_hash,
                    "n_in": n_in,
                    "n_presurvivors": n_presurv,
                    "n_chunks": len(chunk_paths),
                    "decode_s": round(t_decode, 3),
                    "fasttext_s": round(t_ft, 3),
                    "tokenize_s": round(t_tok, 3),
                    "wall_s": round(time.monotonic() - t0, 3),
                },
                f,
            )
    except Exception as e:
        logger.warning("phase A timing write failed for %s: %s", warc_hash, e)
    wall = time.monotonic() - t0
    logger.info(
        "A %s: %d in -> %d presurvivors (%d chunks) in %.1fs", warc_hash, n_in, n_presurv, len(chunk_paths), wall
    )
    return {
        "docs_in": n_in,
        "docs_out": n_presurv,
        "wall_seconds": wall,
        "compute_seconds": {"decode": t_decode, "fasttext": t_ft, "tokenize": t_tok},
    }


RESCUE_STALE_MINUTES = 20.0


def _rescue_loop(
    spec: PipelineSpec,
    manifest_path: str,
    bucket: str,
    *,
    model,
    tokenizer,
    shuffle_seed: int,
    poll_seconds: float,
    max_idle_passes: int,
    stale_minutes: float,
) -> None:
    """Re-decode WARCs that are A-done but still stuck at B (stranded in a no/weak-B region).

    The WARC is re-fetched from Common Crawl S3 (never copied cross-region), so this rescues a
    stranded presurvivor into THIS (B-healthy) region with zero GCS egress — the big HTML is rebuilt
    from source. Central B-claims still ensure each WARC is scored by exactly one B; a duplicate
    presurvivor is harmless (dedup drops it) and CPU is free. A WARC is a candidate only if it has
    been A-done longer than ``stale_minutes`` (so a healthy B about to pick it up isn't pre-empted)
    and has no presurvivor in our own region yet.
    """
    central = f"gs://marin-us-central1/{spec.subdir()}"
    pairs = [(w, _warc_path_hash(w)) for w in _load_manifest(manifest_path)]
    random.Random(shuffle_seed).shuffle(pairs)
    all_hashes = {h for _, h in pairs}
    region = region_from_bucket(bucket)
    hb = Heartbeat(spec, phase="a", region=region, seed=shuffle_seed, kind="cpu")
    logger.info("RESCUE worker (seed=%d region=%s): re-decoding B-stuck WARCs into %s", shuffle_seed, region, bucket)

    idle = 0
    while True:
        a_mt = _load_registry_mtimes(f"{central}/_completed_a")
        b_done = _load_completed_registry(f"{central}/_completed_b")
        cutoff = time.time() - stale_minutes * 60
        stuck = [(w, h) for (w, h) in pairs if h in a_mt and h not in b_done and a_mt[h] < cutoff]
        progressed = 0
        for warc_path, h in stuck:
            if _gcs_exists(f"{spec.presurvivors_prefix(bucket)}/data-{h}.parquet"):
                continue  # already local -> our own B will score it.
            if not _claim_warc_atomic(f"{central}/_claims_a_rescue/data-{h}", stale_hours=1.0):
                continue  # another rescuer owns it.
            logger.info("RESCUE %s: re-decoding from S3 into %s", h, bucket)
            stats = _process_one(warc_path, h, lambda: None, spec=spec, model=model, tokenizer=tokenizer, bucket=bucket)
            if stats:
                hb.record_warc(warc_hash=h, **stats)
            progressed += 1
        if progressed == 0:
            idle += 1
            hb.tick("idle")
            all_b_done = all_hashes <= b_done
            logger.info("rescue idle %d/%d (%d stuck, all_b_done=%s)", idle, max_idle_passes, len(stuck), all_b_done)
            if idle >= max_idle_passes and all_b_done:
                hb.close("done")
                break
            time.sleep(poll_seconds)
        else:
            idle = 0


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
    ap.add_argument(
        "--rescue", action="store_true", help="Rescue mode: re-decode B-stuck WARCs into this region from S3."
    )
    ap.add_argument("--rescue-stale-minutes", type=float, default=RESCUE_STALE_MINUTES)
    ap.add_argument("--chunk-records", type=int, default=CHUNK_RECORDS, help="Sub-WARC checkpoint size (input records).")
    ap.add_argument(
        "--keep-chunks",
        dest="cleanup_chunks",
        action="store_false",
        default=True,
        help="Keep sub-WARC checkpoint chunks after merge (default deletes: ~4 TiB at 10k scale).",
    )
    args = ap.parse_args()

    _check_deps()
    spec = get_spec(args.spec)
    model = preprocess.load_fasttext(spec.fasttext_model_for(args.bucket))  # region-local mirror
    tokenizer = preprocess.load_tokenizer(spec.tokenizer_ref)

    if args.rescue:
        _rescue_loop(
            spec,
            args.manifest,
            args.bucket,
            model=model,
            tokenizer=tokenizer,
            shuffle_seed=args.shuffle_seed,
            poll_seconds=args.poll_seconds,
            max_idle_passes=args.max_idle_passes,
            stale_minutes=args.rescue_stale_minutes,
        )
        return

    process_one = functools.partial(
        _process_one,
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        bucket=args.bucket,
        chunk_records=args.chunk_records,
        cleanup_chunks=args.cleanup_chunks,
    )
    run_claim_loop(
        spec,
        args.manifest,
        args.bucket,
        phase_key="a",
        process_one=process_one,
        shuffle_seed=args.shuffle_seed,
        start=args.start,
        limit=args.limit,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
    )


if __name__ == "__main__":
    main()
