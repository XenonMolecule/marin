# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fused single-phase worker (storage_version=4): the whole cascade on one TPU host.

One process claims a shard and runs BOTH phases: Phase A (decode -> resiliparse-rs extract ->
fastText gate) executes on the host's own CPUs in a pool of subprocesses, feeding extracted
presurvivor rows through RAM to the main thread, which scores them on the chip and writes the
final ``kept/`` + catalog — the byte-identical output contract of the two-phase v3 run.

What this deletes relative to two-phase (the 100k run's measured pain, 2026-08-29..09-01):
no presurvivors on GCS (~72MB of traffic/WARC), no reaper, no ``_a_done``/``_claims_a``, no
cross-phase region stranding (data lives in RAM on whichever node has a chip), and no
A-vs-B host-RAM contention. Benchmark protocol: .agents/projects/fused_worker_benchmark.md.

Concurrency model: ``--a-procs`` subprocesses (each owning its own crash-isolated rust
extraction pool of ``--extract-procs-per-worker``) run whole-WARC Phase A and return rows via
pickling; the parent keeps at most ``--queue-depth`` WARCs in flight (RAM backpressure —
presurvivors are ~40MB/WARC). The chip consumes completed WARCs in arrival order, so A and B
overlap fully; the chip is the intended bottleneck. Preemption loses only in-flight WARCs:
per-WARC ``_b_marks`` (carrying catalog rows) make resumption lossless, same as v3.

    python -m experiments.fast_curation.fused_phase --spec lpv11_fastpipe_v2_1_fused \\
        --bucket gs://marin-us-east5 --a-procs 14 --extract-procs-per-worker 6 \\
        --queue-depth 6 --max-shard 500
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import random
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.baseline_collection.comparison_sample import normalize_text
from experiments.baseline_collection.decode_warcs_clean import _decode_one_warc_stream
from experiments.baseline_collection.run_extract_standalone import (
    _claim_warc_atomic,
    _load_completed_registry,
    _refresh_claim,
    _register_completed_warc,
)
from experiments.fast_curation import batch_format, preprocess, tpu_phase
from experiments.fast_curation import shard_worklist as sw
from experiments.fast_curation.cpu_phase_c import _install_rust_extractor, _safe_utf8
from experiments.fast_curation.spec import (
    MODERNBERT_CLS_TOKEN_ID,
    MODERNBERT_SEP_TOKEN_ID,
    PipelineSpec,
    get_spec,
)
from experiments.fast_curation.telemetry import Heartbeat, region_from_bucket
from experiments.fast_curation.tpu_phase import (
    CLAIM_REFRESH_SECONDS,
    _load_marker_payloads,
    _throttled,
    _write_marker,
)

logger = logging.getLogger(__name__)

# Self-kill when no WARC completes for this long. Fleet-wide silent hangs (workers wedged
# mid-GCS-call with claims stale and jobs still RUNNING) hit the two-phase fleets twice on
# 2026-08-31 and the fused fleet on 09-01; a max-healthy WARC is ~4 min, so 25 min of
# silence means wedged, and exiting lets Iris retry + the coordinator backfill, with
# ``_b_marks`` making the resume lossless.
STALL_EXIT_MINUTES = 25.0
_LAST_PROGRESS = time.monotonic()


def _touch_progress() -> None:
    global _LAST_PROGRESS
    _LAST_PROGRESS = time.monotonic()


def _start_stall_watchdog() -> None:
    def _watch():
        while True:
            time.sleep(60)
            stalled = (time.monotonic() - _LAST_PROGRESS) / 60
            if stalled > STALL_EXIT_MINUTES:
                logger.error("no progress for %.0f min — self-terminating for retry+backfill", stalled)
                os._exit(70)

    threading.Thread(target=_watch, daemon=True, name="stall-watchdog").start()


# Per-subprocess globals, set once by _a_init (a ProcessPoolExecutor initializer).
_A_SPEC: PipelineSpec | None = None
_A_MODEL = None
_A_POOL = None


def _a_init(spec_id: str, resiliparse_artifact: str, extract_procs: int) -> None:
    """Initializer for A subprocesses: fastText model + a private crash-isolated rust pool."""
    global _A_SPEC, _A_MODEL, _A_POOL
    _A_SPEC = get_spec(spec_id)
    _A_MODEL = preprocess.load_fasttext(_A_SPEC.fasttext_model)
    _A_POOL = _install_rust_extractor(
        resiliparse_artifact,
        _A_SPEC,
        extract_procs,
        extract_fn=preprocess.resiliparse_rs_screen_batch,
        crash_result=("", False),
        # Per-PID unpack dir: the A subprocesses initialize concurrently, and racing unpacks
        # into a shared dir yield truncated .so files (silent doc loss via empty extractions).
        dest=f"/root/resiliparse_rs_a{os.getpid()}",
    )


def a_extract_warc(warc_path: str, warc_hash: str) -> tuple[list[dict], dict]:
    """Whole-WARC Phase A in this subprocess: decode -> extract+screen -> fastText gate.

    Returns ``(PRESURVIVOR_V3 rows, timing dict)`` — rows travel back to the parent by pickle
    and never touch GCS. Semantics are identical to ``cpu_phase_a._process_one_text``'s v3 path
    (same population screen, same ``_safe_utf8``-then-normalize classifier input).
    """
    spec, model, pool = _A_SPEC, _A_MODEL, _A_POOL
    t0 = time.monotonic()
    rows: list[dict] = []
    n_in = 0
    n_prefilter = n_empty = 0
    crashed: list[str] = []
    t_decode = t_extract = t_ft = 0.0
    last = t0
    # Streamed decode: only one chunk of decoded html is alive at a time (plus the compressed
    # download), so per-WARC RAM is ~2-4GB instead of the 15-20GB peaks that OOMed v5e hosts.
    for chunk in _decode_one_warc_stream(warc_path, chunk_records=2000):
        now = time.monotonic()
        t_decode += now - last  # time since last chunk finished = decode/download share
        n_in += len(chunk)
        s = time.monotonic()
        results, chunk_crashed = pool.run(
            [r["html"] for r in chunk], [r["doc_id"] for r in chunk], spec.justext_max_html_chars
        )
        t_extract += time.monotonic() - s
        crashed.extend(chunk_crashed)
        s = time.monotonic()
        for r, (text, in_population) in zip(chunk, results, strict=True):
            if not in_population:
                n_prefilter += 1
                continue
            text = _safe_utf8(text)
            if not text:
                n_empty += 1
                continue
            prob = preprocess.fasttext_useful_prob(model, normalize_text(text))
            if prob < spec.fasttext_threshold:
                continue
            rows.append(
                {
                    "doc_id": r["doc_id"],
                    "url": _safe_utf8(r["url"]),
                    "warc_hash": r["warc_hash"],
                    "snapshot": r["snapshot"],
                    "fasttext_score": float(prob),
                    "text": text,
                }
            )
        t_ft += time.monotonic() - s
        last = time.monotonic()
    timing = {
        "a_n_in": n_in,
        "a_n_prefilter_dropped": n_prefilter,
        "a_n_extract_empty": n_empty,
        "a_n_extract_crashed": len(crashed),
        "a_decode_s": round(t_decode, 3),
        "a_extract_s": round(t_extract, 3),
        "a_fasttext_s": round(t_ft, 3),
        "a_wall_s": round(time.monotonic() - t0, 3),
    }
    return rows, timing


def run_fused_worker(
    spec: PipelineSpec,
    bucket: str,
    *,
    resiliparse_artifact: str,
    a_procs: int,
    extract_procs_per_worker: int,
    queue_depth: int,
    batch_size: int,
    bucket_tokens: bool,
    shuffle_seed: int,
    poll_seconds: float,
    max_idle_passes: int,
    claim_stale_hours: float,
    max_shard: int | None,
    any_region: bool = False,
) -> None:
    """Claim shards (region-matched), pipeline A subprocesses into the chip, write kept+catalog."""
    # jax stays a local import (as in tpu_phase) so CPU-only contexts can import this module.
    import jax  # noqa: PLC0415
    from haliax.partitioning import ResourceAxis, set_mesh  # noqa: PLC0415
    from jax.sharding import Mesh  # noqa: PLC0415

    region = region_from_bucket(bucket)
    # On a multihost slice each host runs this worker independently (MULTIHOST_ENV confines
    # jax to local chips); diversify shard order per host so they don't chase the same claims.
    shuffle_seed += int(os.environ.get("TPU_WORKER_ID", "0") or "0") * 1009
    _start_stall_watchdog()
    tpu_phase._assert_ckpt_in_region(spec.modernbert_ckpt_for(bucket), bucket)
    tpu_phase._setup_compile_cache(f"{spec.namespace(bucket)}/_xla_cache")

    tokenizer = preprocess.load_tokenizer(spec.tokenizer_ref)
    gt_tok = preprocess.load_gigatoken(spec.tokenizer_ref)
    special = {"cls_id": MODERNBERT_CLS_TOKEN_ID, "sep_id": MODERNBERT_SEP_TOKEN_ID}
    preprocess.assert_gigatoken_parity(tokenizer, gt_tok, spec.max_length, **special)
    tokenize_batch = functools.partial(
        preprocess.tokenize_trunc_batch_gigatoken, gt_tok, max_length=spec.max_length, **special
    )

    n_dev = len(jax.devices())
    if batch_size % n_dev != 0:
        raise ValueError(f"--batch-size {batch_size} must be a multiple of device count {n_dev}")
    mesh = Mesh(np.array(jax.devices()).reshape(n_dev, 1), (ResourceAxis.DATA, ResourceAxis.MODEL))

    ex = ProcessPoolExecutor(
        max_workers=a_procs,
        initializer=_a_init,
        initargs=(spec.spec_id, resiliparse_artifact, extract_procs_per_worker),
    )
    with set_mesh(mesh):
        model, config = tpu_phase.load_model(spec, mesh, bucket)
        pooled = tpu_phase.load_pooled_model(spec, mesh, bucket)
        score_fn = tpu_phase.make_score_fn()

        # Fused work has NO data gravity (nothing is read from a region bucket before
        # processing; kept/ lands wherever the worker runs), so region-matching is a soft
        # locality preference, not a correctness constraint. Own-region shards come first;
        # with ``any_region`` the worker then drains the global queue instead of idling —
        # shard supply can never starve a pool.
        index = [e for e in sw.load_index(spec) if max_shard is None or e["shard"] < max_shard]
        mine = [e["shard"] for e in index if e["region"] == region]
        others = [e["shard"] for e in index if e["region"] != region] if any_region else []
        random.Random(shuffle_seed).shuffle(mine)
        random.Random(shuffle_seed).shuffle(others)
        my_shards = mine + others
        hb = Heartbeat(spec, phase="fused", region=region, seed=shuffle_seed, kind="tpu")
        logger.info("fused worker (seed=%d region=%s): %d shards", shuffle_seed, region, len(my_shards))

        idle = 0
        while True:
            done = _load_completed_registry(sw.sentinel_prefix(spec, "b"))
            remaining = [s for s in my_shards if f"{s:05d}" not in done]
            if not remaining:
                logger.info("all %d region shards complete; exiting.", len(my_shards))
                break
            progressed = 0
            for s in remaining:
                claim_path = f"{sw.claim_prefix(spec, 'b')}/shard-{s:05d}"
                if not _claim_warc_atomic(claim_path, stale_hours=claim_stale_hours):
                    continue
                _run_one_shard(
                    spec,
                    s,
                    ex,
                    bucket=bucket,
                    claim_path=claim_path,
                    queue_depth=queue_depth,
                    batch_size=batch_size,
                    bucket_tokens=bucket_tokens,
                    tokenize_batch=tokenize_batch,
                    score_fn=score_fn,
                    model=model,
                    config=config,
                    pooled=pooled,
                    hb=hb,
                )
                progressed += 1
            if progressed == 0:
                idle += 1
                hb.tick("idle")
                if idle >= max_idle_passes:
                    logger.warning("no claimable shards for %d passes; exiting.", idle)
                    break
                time.sleep(poll_seconds)
            else:
                idle = 0
        hb.close("done")
    ex.shutdown(wait=False, cancel_futures=True)


def _run_one_shard(spec, s: int, ex, *, bucket, claim_path, queue_depth, **b_kwargs) -> None:
    """Pipeline one claimed shard: A futures windowed at ``queue_depth``, chip consumes FIFO-ish."""
    hb = b_kwargs.pop("hb")

    # Scoring batches count as watchdog progress: a pathological WARC can legitimately take
    # >STALL_EXIT_MINUTES end-to-end, and killing mid-WARC just restarts the same WARC forever
    # (observed 2026-09-03: six shards wedged at n-1 WARCs in kill-retry loops). Only a worker
    # producing NO batches for the window is truly hung.
    def _refresh_and_touch():
        _refresh_claim(claim_path)
        _touch_progress()

    refresh = _throttled(_refresh_and_touch, CLAIM_REFRESH_SECONDS)
    marks_prefix = f"{sw.CENTRAL_BUCKET}/{spec.subdir()}/_b_marks/shard-{s:05d}"
    pairs = sw.load_shard(spec, s)
    marked = _load_marker_payloads(marks_prefix)
    todo = [(w, h) for w, h in pairs if h not in marked]
    logger.info("fused claimed shard %05d: %d WARCs (%d already marked)", s, len(pairs), len(marked))

    _touch_progress()
    pending: dict = {}  # future -> (warc_hash, submit_time)
    it = iter(todo)

    def _fill() -> None:
        while len(pending) < queue_depth:
            try:
                w, h = next(it)
            except StopIteration:
                return
            pending[ex.submit(a_extract_warc, w, h)] = h

    _fill()
    while pending:
        done_futs, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        for fut in done_futs:
            h = pending.pop(fut)
            try:
                rows, a_timing = fut.result()  # a code crash is a real bug: let it propagate
            except RuntimeError as e:
                if "Failed to download" not in str(e):
                    raise
                # An unfetchable WARC (deadline-bounded retries exhausted — the six 2026-09-03
                # giants). Record it EXPLICITLY as failed (never silently zero) so the shard can
                # close; the catalog row carries the reason for later re-audit/re-fetch.
                logger.error("WARC %s unfetchable, recording explicit failure: %s", h, e)
                rows = []
                a_timing = {
                    "a_n_in": 0,
                    "a_n_prefilter_dropped": 0,
                    "a_n_extract_empty": 0,
                    "a_n_extract_crashed": 0,
                    "a_decode_s": 0.0,
                    "a_extract_s": 0.0,
                    "a_fasttext_s": 0.0,
                    "a_wall_s": 0.0,
                    "a_download_failed": str(e)[:300],
                }
            _fill()  # keep the producers busy while the chip scores
            table = (
                pa.Table.from_pylist(rows, schema=batch_format.PRESURVIVOR_V3_SCHEMA)
                if rows
                else batch_format.PRESURVIVOR_V3_SCHEMA.empty_table()
            )
            row = tpu_phase.process_warc_v3(
                spec,
                h,
                b_kwargs["score_fn"],
                b_kwargs["model"],
                b_kwargs["config"],
                bucket=bucket,
                batch_size=b_kwargs["batch_size"],
                bucket_tokens=b_kwargs["bucket_tokens"],
                tokenize_batch=b_kwargs["tokenize_batch"],
                pooled=b_kwargs["pooled"],
                refresh=refresh,
                table=table,
            )
            row.update(a_timing)
            _write_marker(f"{marks_prefix}/data-{h}", row)
            marked[h] = row
            _touch_progress()
            hb.record_warc(
                warc_hash=h,
                docs_in=a_timing["a_n_in"],
                docs_out=row["n_kept"],
                wall_seconds=row["wall_s"] + a_timing["a_wall_s"],
                compute_seconds={
                    "decode": a_timing["a_decode_s"],
                    "extract": a_timing["a_extract_s"],
                    "fasttext": a_timing["a_fasttext_s"],
                    "pooled": row["pooled_s"],
                    "modernbert": row["score_s"],
                },
            )
    catalog = pa.Table.from_pylist([marked[h] for _, h in pairs])
    with fsspec.open(sw.catalog_path(spec, s), "wb") as f:
        pq.write_table(catalog, f, compression="zstd")
    _register_completed_warc(f"{s:05d}", sw.sentinel_prefix(spec, "b"))
    logger.info("fused shard %05d complete: %d WARCs cataloged", s, len(pairs))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="lpv11_fastpipe_v2_1_fused")
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--resiliparse-artifact", default=None, help="Defaults to <bucket>/artifacts/resiliparse_rs/latest.")
    ap.add_argument("--a-procs", type=int, required=True, help="A subprocesses (each a whole-WARC pipeline).")
    ap.add_argument("--extract-procs-per-worker", type=int, default=6)
    ap.add_argument("--queue-depth", type=int, default=6, help="Max WARCs in flight (RAM backpressure).")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--no-bucket-tokens", dest="bucket_tokens", action="store_false", default=True)
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=40)
    ap.add_argument("--claim-stale-hours", type=float, default=0.2)
    ap.add_argument("--max-shard", type=int, default=None)
    ap.add_argument(
        "--any-region",
        action="store_true",
        help="After own-region shards, claim ANY region's shards (fused work has no data gravity).",
    )
    args = ap.parse_args()

    spec = get_spec(args.spec)
    if spec.storage_version != 4:
        raise ValueError(f"{args.spec} is not a storage_version=4 (fused) spec")
    run_fused_worker(
        spec,
        args.bucket,
        resiliparse_artifact=args.resiliparse_artifact or f"{args.bucket}/artifacts/resiliparse_rs/latest",
        a_procs=args.a_procs,
        extract_procs_per_worker=args.extract_procs_per_worker,
        queue_depth=args.queue_depth,
        batch_size=args.batch_size,
        bucket_tokens=args.bucket_tokens,
        shuffle_seed=args.shuffle_seed,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
        claim_stale_hours=args.claim_stale_hours,
        max_shard=args.max_shard,
        any_region=args.any_region,
    )


if __name__ == "__main__":
    main()
