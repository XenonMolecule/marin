# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 (CPU): standalone WARC worker — fastText-filter, JustText-extract, pre-tokenize.

This is a STANDALONE claim-based worker (like ``run_extract_standalone.py``), launched as many
independent top-level Iris jobs by ``launch_cpu.py``. Each top-level job gets its own ``--extra``
deps (fasttext/justext), so this avoids the Zephyr-worker-env-inheritance gap where worker actor
child jobs ran core-only. Workers coordinate purely through GCS: atomic per-WARC claims + a
central completed registry; the per-WARC survivor parquet is the resume marker.

Per WARC:
1. Download + cleanly decode every ``text/html`` record (``decode_warcs_clean._decode_one_warc`` —
   WHATWG charset precedence, never U+FFFD). Each record carries ``html`` (full page, for JustText)
   and ``text_body`` (``body_strip`` + ws-collapse + lower — the fastText/ModernBERT input).
2. fastText useful-filter on ``text_body``: below ``spec.fasttext_threshold`` -> dropped.
3. JustText on the raw ``html`` -> training ``text`` (empty -> dropped: no content to keep).
4. Pre-tokenize ``text_body`` for ModernBERT (truncate to ``spec.max_length``).

Survivors are written to one parquet per WARC (``cpu_survivors/data-{warc_hash}.parquet``).

Run as a standalone Iris CPU job (launch many via ``launch_cpu.py``)::

    uv run iris --cluster marin job run --region us-east5 --cpu 2 --memory 24GB --disk 24GB \\
      --enable-extra-resources --extra cpu --extra dclm --extra extraction-bakeoff \\
      --priority batch --no-wait --job-name fastcur-cpu-0 \\
      -e WANDB_API_KEY <key> -e HF_TOKEN <token> -- \\
      python -m experiments.fast_curation.cpu_phase --spec fastpipe_v1 --shuffle-seed 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

import fsspec

from experiments.baseline_collection.decode_warcs_clean import (
    _decode_one_warc,
    _load_manifest,
    _warc_path_hash,
)
from experiments.baseline_collection.run_extract_standalone import (
    _claim_warc_atomic,
    _list_fresh_claims,
    _load_completed_registry,
    _refresh_claim,
    _register_completed_warc,
)
from experiments.fast_curation import preprocess
from experiments.fast_curation.batch_format import write_survivors
from experiments.fast_curation.spec import PipelineSpec, get_spec
from experiments.fast_curation.telemetry import Heartbeat, region_from_bucket

logger = logging.getLogger(__name__)

# A worker gives up after this many consecutive idle passes even if the upstream phase never signaled
# done — so a stuck upstream straggler (which blocks the phase-end sentinel) can't leave workers
# idling forever. At poll_seconds=30 this is ~30 min: long enough to ride out normal transient
# starvation, short enough that jobs free their resources instead of hanging at the endgame.
HARD_IDLE_PASSES = 60

# JustText's learned tier (sklearn RandomForest + fastText) spawns its own BLAS/OpenMP threads.
# Without pinning, a P-process ProcessPool oversubscribes (P procs x N threads >> cores) and the
# canary got only ~1.9x on 8 cores. Pin every numeric lib to 1 thread/process so P procs map
# cleanly to P cores (~Px). Set in the parent (spawn children inherit) AND re-applied per child.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}


def _pin_threads() -> None:
    for k, v in _SINGLE_THREAD_ENV.items():
        os.environ[k] = v
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass


def _check_worker_deps(spec: PipelineSpec) -> None:
    """Fail fast if a worker dep is missing or the JustText fork drifted from the spec."""
    import importlib

    missing = []
    for mod in ("fasttext", "justext", "transformers"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise RuntimeError(
            f"Missing worker deps {missing}. Launch with "
            f"--extra cpu --extra dclm --extra extraction-bakeoff (see launch_cpu.py)."
        )
    preprocess.assert_justext_version(spec.justext_version)


def _write_cpu_timing(bucket: str, spec: PipelineSpec, warc_hash: str, payload: dict) -> None:
    """Best-effort per-WARC timing sidecar (auxiliary telemetry; never fails the WARC)."""
    path = f"{spec.namespace(bucket)}/timing_cpu/data-{warc_hash}.json"
    try:
        with fsspec.open(path, "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.warning("failed to write cpu timing for %s: %s", warc_hash, e)


def _extract_one_warc(warc_path: str, spec: PipelineSpec, model, tokenizer, bucket: str, pool) -> list[dict]:
    """Decode -> fastText gate -> JustText (parallel) -> tokenize for one WARC. Returns survivor rows.

    Three passes so the dominant cost (JustText, per fastText-survivor) fans out across cores:
    (1) serial fastText gate, (2) parallel JustText over a ProcessPool, (3) serial tokenize.
    """
    warc_hash = _warc_path_hash(warc_path)
    t0 = time.monotonic()
    records = _decode_one_warc(warc_path)  # download + clean decode (bundled)
    t_decode = time.monotonic() - t0
    n_in = len(records)

    # Pass 1: fastText useful-gate (serial; ~1ms/doc).
    s = time.monotonic()
    ft_pass: list[tuple[dict, float]] = []
    for r in records:
        prob = preprocess.fasttext_useful_prob(model, r["text_body"])
        if prob >= spec.fasttext_threshold:
            ft_pass.append((r, prob))
    t_ft = time.monotonic() - s
    n_ft_pass = len(ft_pass)
    logger.info(
        "  %s: %d/%d ft-pass in %.0fs; JustText on %d (procs=%s)...",
        warc_hash,
        n_ft_pass,
        n_in,
        t_ft,
        n_ft_pass,
        getattr(pool, "n_procs", 1),
    )

    # Pass 2: JustText on survivor HTML, fanned out across cores (with the spec's per-doc timeout).
    s = time.monotonic()
    jt_args = [
        (r["html"], spec.justext_lang, spec.justext_max_html_chars, spec.justext_paragraph_sep) for r, _ in ft_pass
    ]
    if pool is not None and jt_args:
        texts = pool.run(jt_args, spec.justext_timeout)
    else:
        texts = [preprocess.justext_text(h, lang, cap, sep) for h, lang, cap, sep in jt_args]
    t_jt = time.monotonic() - s

    # Pass 3: tokenize survivors with non-empty extraction (serial; fast).
    survivors: list[dict] = []
    n_justext_empty = 0
    t_tok = 0.0
    for (r, prob), text in zip(ft_pass, texts, strict=True):
        if not text:
            n_justext_empty += 1
            continue  # fastText said "useful" but JustText found no content -> nothing to train on.
        s = time.monotonic()
        ids = preprocess.tokenize_trunc(tokenizer, r["text_body"], spec.max_length)
        t_tok += time.monotonic() - s
        survivors.append(
            {
                "doc_id": r["doc_id"],
                "url": r["url"],
                "warc_hash": r["warc_hash"],
                "snapshot": r["snapshot"],
                "fasttext_score": float(prob),
                "text": text,
                "input_ids": ids,
                "n_tokens": len(ids),
            }
        )

    _write_cpu_timing(
        bucket,
        spec,
        warc_hash,
        {
            "warc_hash": warc_hash,
            "n_in": len(records),
            "n_ft_pass": n_ft_pass,
            "n_justext_empty": n_justext_empty,
            "n_survivors": len(survivors),
            "decode_s": round(t_decode, 3),
            "fasttext_s": round(t_ft, 3),
            "justext_s": round(t_jt, 3),
            "tokenize_s": round(t_tok, 3),
            "wall_s": round(time.monotonic() - t0, 3),
        },
    )
    logger.info(
        "%s: %d in -> %d ft-pass -> %d survivors (%d justext-empty) in %.1fs",
        warc_hash,
        len(records),
        n_ft_pass,
        len(survivors),
        n_justext_empty,
        time.monotonic() - t0,
    )
    return survivors


def _gcs_exists(path: str) -> bool:
    fs, p = fsspec.core.url_to_fs(path)
    return fs.exists(p)


def _list_upstream_present(upstream_path_fn, pairs) -> set[str] | None:
    """Upstream paths that exist, from ONE fresh listing of their shared parent dir.

    Replaces a per-WARC ``exists()`` probe per manifest entry per pass. All known
    ``upstream_path_fn``s map every hash into one flat directory; this is verified
    on a sample and, if it ever stops holding, returns None so the caller falls
    back to per-WARC probes. ``refresh=True`` bypasses gcsfs's process-lifetime
    dircache so newly-produced inputs are visible.
    """
    sample_parents = {upstream_path_fn(h).rsplit("/", 1)[0] for _, h in pairs[:8]}
    if len(sample_parents) != 1:
        return None
    parent = sample_parents.pop()
    fs, parent_path = fsspec.core.url_to_fs(parent)
    try:
        names = fs.ls(parent_path, refresh=True)
    except FileNotFoundError:
        return set()  # upstream dir not created yet — nothing present
    except Exception as e:
        logger.warning("upstream listing failed for %s: %s", parent, e)
        return None
    return {f"{parent}/{p.rsplit('/', 1)[-1]}" for p in names}


def _load_registry_mtimes(registry_prefix: str) -> dict[str, float]:
    """Like ``_load_completed_registry`` but returns ``hash -> completion-epoch`` (the marker mtime).

    Used by the rescue path to tell a genuinely-stranded WARC (A-done long ago, still no B) from one
    that is merely slow (A-done seconds ago, B will get to it). Best-effort; empty on any error.
    """
    import datetime as _dt

    fs = fsspec.filesystem("gcs")
    out: dict[str, float] = {}
    try:
        # refresh=True: gcsfs caches listings for the life of the process, and the
        # rescue loop relies on this listing actually reflecting new completions.
        for e in fs.ls(registry_prefix.replace("gs://", ""), detail=True, refresh=True):
            name = e["name"].rsplit("/", 1)[-1]
            if not name.startswith("data-"):
                continue
            t = e.get("mtime") or e.get("updated") or e.get("timeCreated")
            if isinstance(t, (int, float)):
                out[name[5:]] = float(t)
            elif hasattr(t, "timestamp"):
                out[name[5:]] = t.timestamp()
            elif isinstance(t, str):
                out[name[5:]] = _dt.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
    except Exception as e:
        logger.warning("rescue: failed to load registry mtimes for %s: %s", registry_prefix, e)
    return out


def _phase_is_complete(phase_end_path: str, manifest_path: str) -> bool:
    """True only if a phase-end sentinel exists AND it was written for THIS manifest.

    The sentinel is a cheap "this phase is globally done, exit now" flag so idle workers don't
    re-list an O(WARCs) registry every poll. But it is namespace-scoped while a run is
    manifest-scoped: a 300-WARC smoke sharing the namespace would finish, write the flag, and then
    every worker of the real 10k run exits ~5 min after start having claimed nothing — silently, as
    SUCCEEDED. That cost hours across three separate launches. Matching on the manifest makes a
    smaller run's sentinel harmless to a larger one.

    A sentinel with no recorded manifest is unattributable, so it is IGNORED rather than honoured.
    Trusting it re-opens the exact trap above from the writer side: ``tpu_phase`` shipped without the
    manifest field, so a straggler of the 300-WARC run wrote an unlabeled ``_phase_b_end.json`` and
    every subsequently-launched B worker of the 10k run exited in ~42 s as SUCCEEDED. The cost of
    ignoring a genuinely-final unlabeled sentinel is bounded (workers idle out after
    ``max_idle_passes`` instead of exiting instantly); the cost of honouring a stale one is a
    silently dead run.
    """
    try:
        with fsspec.open(phase_end_path, "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return False
    except Exception:
        return False  # unreadable sentinel must not wedge a worker
    written_for = payload.get("manifest")
    if not written_for:
        logger.warning("ignoring unlabeled phase-end sentinel %s (no manifest recorded)", phase_end_path)
        return False
    if written_for != manifest_path:
        logger.info(
            "ignoring phase-end sentinel written for a different manifest (%s != %s)",
            written_for,
            manifest_path,
        )
        return False
    return True


def _write_once(path: str, payload: dict) -> None:
    """First-writer-wins sentinel via GCS ``if_generation_match=0`` (ignore 412 if it exists)."""
    from google.cloud import storage as gcs_storage

    bucket_name, blob_path = path.replace("gs://", "").split("/", 1)
    blob = gcs_storage.Client().bucket(bucket_name).blob(blob_path)
    try:
        blob.upload_from_string(json.dumps(payload), if_generation_match=0)
    except Exception as e:
        if "conditionNotMet" not in str(e) and "412" not in str(e):
            raise


class JustextPool:
    """Persistent ``spawn`` process pool for JustText fan-out, with an optional per-doc timeout.

    ``spawn`` (not fork): children re-import cleanly and do NOT inherit the parent's 1.88 GB
    fastText model or its GCS/grpc threads (fork-after-threads can deadlock). Each child loads
    JustText's classifier once and is reused across WARCs.

    ``run(args, timeout)`` bounds each document's wall-clock. jusText DOM-parses via lxml's C
    parser, which a Python signal (SIGALRM) cannot interrupt, so a pathological page can hang a
    worker indefinitely. A doc exceeding ``timeout`` yields ``""`` (dropped as no-content); because
    the stuck worker is wedged in C, the pool is then hard-``terminate()``-d and rebuilt to actually
    kill it. Under the html-size cap, timeouts are rare, so the rebuild cost is negligible.
    """

    def __init__(self, n_procs: int):
        import multiprocessing as mp

        self._ctx = mp.get_context("spawn")
        self.n_procs = n_procs
        self._pool = None
        self._ensure()

    def _ensure(self) -> None:
        if self._pool is None:
            self._pool = self._ctx.Pool(self.n_procs, initializer=_pin_threads)

    def run(self, args: list[tuple], timeout: float | None) -> list[str]:
        """Map ``preprocess._justext_one`` over ``args`` (each ``(html, lang, max_html_chars)``)."""
        import multiprocessing as mp

        self._ensure()
        if timeout is None:
            return list(self._pool.map(preprocess._justext_one, args, chunksize=16))
        # Submit all, then collect with a per-doc deadline; a wedged worker is killed after the pass.
        asyncs = [self._pool.apply_async(preprocess._justext_one, (a,)) for a in args]
        out: list[str] = []
        stuck = False
        for ar in asyncs:
            try:
                out.append(ar.get(timeout=timeout))
            except mp.TimeoutError:
                out.append("")
                stuck = True
        if stuck:
            self.close()  # hard-kill the C-level parse; next run() rebuilds a fresh pool.
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


def _make_justext_pool(n_procs: int):
    """A persistent JustText process pool, or None for serial (``n_procs <= 1``)."""
    if n_procs <= 1:
        return None
    return JustextPool(n_procs)


def run_claim_loop(
    spec: PipelineSpec,
    manifest_path: str,
    bucket: str,
    *,
    phase_key: str,
    process_one,
    shuffle_seed: int,
    start: int = 0,
    limit: int | None = None,
    poll_seconds: float = 30.0,
    max_idle_passes: int = 5,
    upstream_path_fn=None,
    upstream_done_path: str | None = None,
    claim_stale_hours: float = 0.25,
) -> int:
    """Generic standalone claim/registry loop shared by the v2 phases (and reusable by v1).

    Iterates the (shuffled) manifest; for each not-yet-registered WARC it atomically claims and
    calls ``process_one(warc_path, warc_hash)`` (which does the per-WARC compute + writes its
    output), then registers it. ``phase_key`` namespaces the registry (`_completed_{key}`) and
    claims (`_claims_{key}`) so phases don't collide. Returns the number processed by this worker.
    """
    warc_paths = _load_manifest(manifest_path)[start:]
    if limit is not None:
        warc_paths = warc_paths[:limit]
    pairs = [(w, _warc_path_hash(w)) for w in warc_paths]
    random.Random(shuffle_seed).shuffle(pairs)
    my_hashes = {h for _, h in pairs}

    registry_prefix = f"gs://marin-us-central1/{spec.subdir()}/_completed_{phase_key}"
    # CENTRAL claims (in us-central1, like the registry) make claiming global across regions: any
    # region work-steals any undone WARC, first claimer wins, no cross-region double-processing.
    # B/C still only claim WARCs whose local input exists (upstream_path_fn), so the A->B->C chain
    # stays co-located in whichever region claimed the WARC (data locality; presurvivor html is big).
    claim_root = f"gs://marin-us-central1/{spec.subdir()}/_claims_{phase_key}"
    # Phase start/end sentinels are CENTRAL too, so a region that runs only a later phase still
    # sees the global "upstream phase done" signal (first-writer-wins = earliest start / first done).
    central_sub = f"gs://marin-us-central1/{spec.subdir()}"
    _write_once(
        f"{central_sub}/_phase_{phase_key}_start.json",
        {"epoch": time.time(), "n_warcs": len(pairs), "manifest": manifest_path, "start": start},
    )
    logger.info("phase %s worker (seed=%d): %d WARCs", phase_key, shuffle_seed, len(pairs))
    # Best-effort dashboard heartbeat (A/C are CPU). Monotonic across preemption; never fails a WARC.
    hb = Heartbeat(spec, phase=phase_key, region=region_from_bucket(bucket), seed=shuffle_seed, kind="cpu")

    phase_end_path = f"{central_sub}/_phase_{phase_key}_end.json"
    # Endgame drain: once upstream is done the total work is FIXED, but a crashed peer's WARCs stay
    # claimed until claim_stale_hours. Waiting only max_idle_passes (~2.5 min) makes survivors quit
    # before those claims expire, orphaning the last few WARCs so the phase parks at 99.x% needing a
    # manual finisher. Instead wait a full stale window (+margin) so we reclaim them and self-drain.
    endgame_idle_passes = max_idle_passes + int(claim_stale_hours * 3600 / poll_seconds) + 1
    idle = 0
    done_here = 0
    while True:
        # Fast, scalable self-exit: once the phase-end sentinel exists the phase is globally complete,
        # so exit on a single cheap blob check instead of re-listing the O(WARCs) registry every pass
        # (that listing is what left idle workers lingering at 10k and would be untenable at 7M).
        if _phase_is_complete(phase_end_path, manifest_path):
            logger.info("phase %s: %s present — phase complete; exiting.", phase_key, phase_end_path)
            hb.close("done")
            break
        completed = _load_completed_registry(registry_prefix)
        if my_hashes <= completed:
            # Only a FULL-manifest worker may stamp the phase-end sentinel. A sliced worker
            # (--start/--limit, e.g. a canary) shares the manifest path, so its sentinel would pass
            # the manifest-scope check and instantly exit every full-run worker launched after it —
            # the historical namespace-sentinel trap, replayed through the slice door (it cost the
            # first optimized-worker launch: a 3-WARC canary's sentinel exited it in 2 seconds).
            if start == 0 and limit is None:
                _write_once(phase_end_path, {"epoch": time.time(), "manifest": manifest_path})
            logger.info("phase %s: all %d of this worker's WARCs complete; exiting.", phase_key, len(pairs))
            hb.close("done")
            break
        # One listing each for claims and upstream inputs per pass, instead of a
        # per-WARC probe pair against the central bucket per manifest entry (the
        # per-WARC probes were ~20M class-B GETs/hour fleet-wide on us-central1).
        claimed_fresh = _list_fresh_claims(claim_root, claim_stale_hours)
        upstream_present = _list_upstream_present(upstream_path_fn, pairs) if upstream_path_fn is not None else None
        progressed = 0
        for warc_path, h in pairs:
            if h in completed:
                continue
            if upstream_present is not None:
                if upstream_path_fn(h) not in upstream_present:
                    continue  # upstream phase hasn't produced this WARC's input yet.
            elif upstream_path_fn is not None and not _gcs_exists(upstream_path_fn(h)):
                continue  # upstream listing unavailable — per-WARC fallback.
            if h in claimed_fresh:
                continue  # freshly claimed by another worker; can't be won.
            claim_path = f"{claim_root}/data-{h}"
            if not _claim_warc_atomic(claim_path, stale_hours=claim_stale_hours):
                continue
            # Pass a refresh callback so a long (contended) process_one keeps its claim fresh and
            # isn't falsely reclaimed; a DEAD worker's claim still goes stale in claim_stale_hours.
            stats = process_one(warc_path, h, lambda p=claim_path: _refresh_claim(p))
            _register_completed_warc(h, registry_prefix)
            if stats:
                hb.record_warc(warc_hash=h, **stats)
            done_here += 1
            progressed += 1
        if progressed == 0:
            idle += 1
            # Don't exit while the upstream phase is still producing inputs we'll need.
            # Same manifest scoping as our own sentinel: an unlabeled or foreign upstream sentinel
            # must not convince us the feeder is finished and send us into terminal drain.
            upstream_done = upstream_done_path is None or _phase_is_complete(upstream_done_path, manifest_path)
            hb.tick("draining" if upstream_done else "idle")
            logger.info(
                "phase %s idle pass %d/%d (%d in registry, upstream_done=%s)",
                phase_key,
                idle,
                endgame_idle_passes,
                len(completed),
                upstream_done,
            )
            if idle >= endgame_idle_passes and upstream_done:
                logger.info(
                    "phase %s: upstream done and idle %d passes (>= full claim-stale window) — nothing "
                    "left to reclaim; exiting.",
                    phase_key,
                    idle,
                )
                hb.close("done")
                break
            # Hard idle timeout: exit even if the upstream never signaled done. An upstream straggler
            # (a WARC that can't be processed) blocks the phase-end sentinel forever; without this a
            # worker idles indefinitely. After HARD_IDLE_PASSES of zero claimable work, give up so the
            # job actually terminates and frees its resources.
            if idle >= HARD_IDLE_PASSES:
                logger.warning(
                    "phase %s: HARD idle timeout (%d passes, ~%dmin) with no claimable work — exiting "
                    "despite upstream_done=%s (likely a stuck straggler upstream).",
                    phase_key,
                    idle,
                    int(idle * poll_seconds / 60),
                    upstream_done,
                )
                hb.close("done")
                break
            time.sleep(poll_seconds)
        else:
            idle = 0
    return done_here


def run_cpu_worker(
    spec: PipelineSpec,
    manifest_path: str,
    bucket: str,
    *,
    shuffle_seed: int,
    start: int = 0,
    limit: int | None = None,
    poll_seconds: float = 30.0,
    max_idle_passes: int = 5,
    justext_procs: int = 8,
) -> None:
    """Claim-and-process WARCs until all are in the completed registry (resumable)."""
    _check_worker_deps(spec)
    _pin_threads()  # parent env -> inherited by spawn children before their BLAS import
    pool = _make_justext_pool(justext_procs)
    model = preprocess.load_fasttext(spec.fasttext_model)
    tokenizer = preprocess.load_tokenizer(spec.tokenizer_ref)

    warc_paths = _load_manifest(manifest_path)[start:]
    if limit is not None:
        warc_paths = warc_paths[:limit]
    pairs = [(w, _warc_path_hash(w)) for w in warc_paths]
    random.Random(shuffle_seed).shuffle(pairs)
    total = len(pairs)
    my_hashes = {h for _, h in pairs}

    registry_prefix = f"gs://marin-us-central1/{spec.subdir()}/_completed_cpu"
    _write_once(
        f"{spec.namespace(bucket)}/_phase1_start.json",
        {"epoch": time.time(), "n_warcs": total, "manifest": manifest_path, "start": start},
    )
    logger.info("Phase 1 CPU worker (seed=%d): %d WARCs -> %s", shuffle_seed, total, spec.survivors_prefix(bucket))

    idle = 0
    done_here = 0
    while True:
        completed = _load_completed_registry(registry_prefix)
        # Done = all of THIS worker's WARCs are registered (NOT len(completed)>=total, which would
        # false-exit when the registry holds entries from other runs / other workers' shards).
        if my_hashes <= completed:
            _write_once(f"{spec.namespace(bucket)}/_phase1_end.json", {"epoch": time.time(), "n_warcs": total})
            logger.info("all %d of this worker's WARCs complete; exiting.", total)
            break

        # One listing per pass instead of a per-WARC claim probe per manifest entry.
        claimed_fresh = _list_fresh_claims(f"{spec.namespace(bucket)}/_cpu_claims", 3.0)
        progressed = 0
        for warc_path, h in pairs:
            if h in completed:
                continue
            if h in claimed_fresh:
                continue  # freshly claimed by another worker; can't be won.
            claim_dir = f"{spec.namespace(bucket)}/_cpu_claims/data-{h}"
            if not _claim_warc_atomic(claim_dir):
                continue  # another worker owns it (or it's a non-stale done claim).
            survivors = _extract_one_warc(warc_path, spec, model, tokenizer, bucket, pool)
            write_survivors(f"{spec.survivors_prefix(bucket)}/data-{h}.parquet", survivors)
            _register_completed_warc(h, registry_prefix)
            done_here += 1
            progressed += 1

        if progressed == 0:
            idle += 1
            logger.info("idle pass %d/%d (%d/%d in registry)", idle, max_idle_passes, len(completed), total)
            if idle >= max_idle_passes:
                logger.info("no claimable work for %d passes; exiting (other workers may finish the rest).", idle)
                break
            time.sleep(poll_seconds)
        else:
            idle = 0

    if pool is not None:
        pool.close()
    logger.info("CPU worker done: processed %d WARCs.", done_here)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="fastpipe_v1", help="Pipeline spec id (see spec.SPECS).")
    ap.add_argument("--manifest", default="experiments/distill/dclm_400m_1x.txt")
    ap.add_argument("--bucket", default="gs://marin-us-east5", help="Regional output bucket.")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="Per-worker manifest shuffle for claim diversity.")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N WARCs of the manifest.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N WARCs after --start.")
    ap.add_argument("--poll-seconds", type=float, default=30.0)
    ap.add_argument("--max-idle-passes", type=int, default=5)
    ap.add_argument("--justext-procs", type=int, default=8, help="ProcessPool size for JustText fan-out (~= --cpu).")
    args = ap.parse_args()

    spec = get_spec(args.spec)
    run_cpu_worker(
        spec,
        args.manifest,
        args.bucket,
        shuffle_seed=args.shuffle_seed,
        start=args.start,
        limit=args.limit,
        poll_seconds=args.poll_seconds,
        max_idle_passes=args.max_idle_passes,
        justext_procs=args.justext_procs,
    )


if __name__ == "__main__":
    main()
