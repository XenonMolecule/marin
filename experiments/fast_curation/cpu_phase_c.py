# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""v2 Phase C (CPU): extraction on the ModernBERT-kept docs only -> final kept corpus.

The whole point of v2: extraction (the dominant CPU cost) runs ONLY on docs that already passed
ModernBERT (Phase B), i.e. ~5x fewer than v1 (which extracts every fastText-survivor). Reads each
WARC's pre-survivor parquet (raw html) + its Phase-B keeplist ({doc_id, prob}); for docs with
prob >= threshold, extracts text from the html and writes the final ``kept/data-{warc_hash}.parquet``
(identical KEPT_SCHEMA to v1, so downstream dedup/tokenize is unchanged). The keeplist retains every
doc's prob, so re-thresholding stays free.

Which engine runs is ``spec.extraction_engine``:

* :attr:`Extractor.JUSTEXT` (v1-v3) — the XenonMolecule jusText fork, 9.43 docs/s/core.
* :attr:`Extractor.RESILIPARSE_RS` (lpv11 line) — the fork's Rust engine, 291.8 docs/s/core,
  downloaded as a prebuilt artifact at worker startup (no build toolchain, no package dep).

Standalone claim-based worker (launch many via ``launch_cpu_c.py``).

    python -m experiments.fast_curation.cpu_phase_c --spec fastpipe_v3 --shuffle-seed 0
    python -m experiments.fast_curation.cpu_phase_c --spec lpv11_fastpipe_v1 --shuffle-seed 0
"""

from __future__ import annotations

import argparse
import functools
import importlib
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import fsspec
import pyarrow as pa

from experiments.fast_curation import batch_format, preprocess
from experiments.fast_curation.cpu_phase import _gcs_exists, _make_justext_pool, _pin_threads, run_claim_loop
from experiments.fast_curation.spec import RESILIPARSE_RS_ARTIFACT, Extractor, PipelineSpec, get_spec

logger = logging.getLogger(__name__)

# Sub-WARC checkpoint granularity. Extraction on a contended preemptible host is slow, so bank
# progress every CHUNK_SIZE kept docs to GCS: a preemption loses <=1 chunk, not the whole WARC.
CHUNK_SIZE = 512

# --- resiliparse-rs tuning -----------------------------------------------------------------------
# Docs per pool task. Small so (a) uncapped multi-MB pages are not pickled into one giant task and
# (b) the parallelism is real; a segfaulting page costs one task, not the whole checkpoint chunk.
RS_TASK_CHUNK = 25
# Cap the BYTES in one pool task, not just the doc count. A task is pickled into a worker, so a
# fixed doc count lets a batch of large pages blow up a child's memory. That matters more than it
# looks: an OOM-killed child surfaces as BrokenProcessPool, which the isolation path would then
# attribute to a "crashing page" and record as an empty extraction — silently DROPPING a long
# document, the exact data `justext_max_html_chars=50_000_000` exists to preserve. Bounding by size
# removes the failure mode without dropping anything: an oversized page simply gets a task of its own.
RS_TASK_MAX_BYTES = 32 * 1024 * 1024


def _size_aware_chunks(htmls: list[str], task_chunk: int, max_bytes: int = RS_TASK_MAX_BYTES) -> list[list[str]]:
    """Split ``htmls`` into pool tasks of <= ``task_chunk`` docs AND <= ``max_bytes`` total.

    A single document larger than ``max_bytes`` still gets its own task — never skipped.
    """
    out: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for html in htmls:
        size = len(html or "")
        if cur and (len(cur) >= task_chunk or cur_bytes + size > max_bytes):
            out.append(cur)
            cur, cur_bytes = [], 0
        cur.append(html)
        cur_bytes += size
    if cur:
        out.append(cur)
    return out


# A crashing page is ~0.1% of pages. If isolation finds a large fraction of a chunk crashing, the
# install is broken (wrong .so / missing lexbor), so fail loudly instead of writing an empty corpus.
RS_MAX_CRASH_FRACTION = 0.5
RS_MIN_CRASH_SAMPLE = 8
# Cap the ids echoed into the per-WARC timing JSON (the count is always exact).
MAX_REPORTED_CRASH_IDS = 100
# A page with real prose, so a successful extraction is unambiguously non-empty.
SMOKE_HTML = (
    "<html><body><article><h1>Smoke</h1><p>This paragraph exists so the startup smoke test can "
    "assert that the Rust extractor returns real main content rather than an empty string.</p>"
    "</article></body></html>"
)


def _check_deps(spec: PipelineSpec) -> None:
    """Fail fast on a missing/drifted extraction dependency for this spec's engine."""
    if spec.extraction_engine is Extractor.RESILIPARSE_RS:
        return  # not a package dep: the prebuilt artifact is downloaded at startup.
    if not importlib.util.find_spec("justext"):
        raise RuntimeError("Missing worker dep 'justext'. Launch with --extra cpu --extra extraction-bakeoff.")
    preprocess.assert_justext_version(spec.justext_version)


def _assert_artifact_matches_spec(artifact_prefix: str, spec: PipelineSpec) -> None:
    """Fail fast unless the published artifact is the exact build the spec pins.

    The extracted text IS the training text, so a rebuild from a different fork commit silently
    produces a different corpus under the same version hash. The pyo3 extension is also not abi3, so
    a CPython minor mismatch would otherwise surface as an opaque ImportError deep in a worker.
    """
    with fsspec.open(f"{artifact_prefix}/manifest.json", "r") as f:
        manifest = json.load(f)
    built = manifest["commit_sha"]
    if spec.resiliparse_rs_commit is not None and built != spec.resiliparse_rs_commit:
        raise RuntimeError(
            f"{spec.spec_id} pins resiliparse-rs commit {spec.resiliparse_rs_commit} but "
            f"{artifact_prefix} was built from {built}. Publish/point at the pinned build."
        )
    want = ".".join(manifest["python_version"].split(".")[:2])
    have = f"{sys.version_info.major}.{sys.version_info.minor}"
    if want != have:
        raise RuntimeError(
            f"resiliparse-rs artifact is a pyo3 (non-abi3) build for CPython {want}, but this worker "
            f"runs {have}. Launch Phase C on CPython {want}."
        )


def _init_rs_worker(pkg_dir: str) -> None:
    """Pool-child initializer: put the downloaded fork FIRST on ``sys.path`` and pin BLAS threads.

    Prepending is what makes ``import resiliparse`` resolve to this fork instead of marin's core
    ``resiliparse`` dep, which claims the same import name.
    """
    sys.path.insert(0, pkg_dir)
    _pin_threads()


class ResiliparseRsPool:
    """Crash-isolating ``spawn`` process pool for the Rust extractor.

    The engine SIGSEGVs on a small fraction of pages (0.108% measured, all ``<frameset>``). That is a
    hard process death, not a Python exception: it kills the worker and poisons the pool
    (``BrokenProcessPool``), which without isolation would abort the WARC. :meth:`run` survives it by
    re-running the failed batch one document per task on a solo executor, recording the offending
    doc ids and emitting ``""`` for them — already this pipeline's "dropped" convention — then
    continuing. One page can never kill a WARC.

    The solo executor is *reused* across the isolation pass and rebuilt only after an actual death,
    so isolating a 512-doc chunk costs ~1 process spawn plus one per crashing page (not 512).

    ``spawn``, like ``JustextPool``: children re-import cleanly and do not inherit the parent's
    GCS/grpc threads. ``extract_fn`` is injectable so tests can drive the isolation path with a
    process that really dies.
    """

    def __init__(
        self,
        n_procs: int,
        pkg_dir: str,
        *,
        task_chunk: int = RS_TASK_CHUNK,
        extract_fn=preprocess.resiliparse_rs_batch,
        crash_result="",
    ):
        self.n_procs = max(1, n_procs)
        self._ctx = mp.get_context("spawn")
        self._pkg_dir = pkg_dir
        self._task_chunk = task_chunk
        self._fn = extract_fn
        # What a crashed page yields in the results list — "" for the plain extractor; a fused
        # extract_fn with a richer per-doc result passes its own "dropped" value.
        self._crash_result = crash_result
        self._pool: ProcessPoolExecutor | None = None

    def _new_executor(self, n_procs: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=n_procs,
            mp_context=self._ctx,
            initializer=_init_rs_worker,
            initargs=(self._pkg_dir,),
        )

    @staticmethod
    def _chunks(htmls: list[str], task_chunk: int) -> list[list[str]]:
        """Public seam for the size-aware splitter (see ``_size_aware_chunks``)."""
        return _size_aware_chunks(htmls, task_chunk)

    def run(self, htmls: list[str], doc_ids: list[str], max_html_chars: int) -> tuple[list[str], list[str]]:
        """Extract a batch of pages. Returns ``(texts, crashed_doc_ids)``; crashed pages get ``""``."""
        if not htmls:
            return [], []
        if self._pool is None:
            self._pool = self._new_executor(self.n_procs)
        chunks = [(part, max_html_chars) for part in _size_aware_chunks(htmls, self._task_chunk)]
        try:
            return [text for part in self._pool.map(self._fn, chunks) for text in part], []
        except Exception as e:
            # BrokenProcessPool (a segfaulted/OOM-killed child) or a raising page: either way the
            # batch is unusable and the pool may be poisoned, so drop it and isolate per document.
            logger.warning(
                "resiliparse-rs batch of %d failed (%s: %s) — isolating per doc", len(htmls), type(e).__name__, e
            )
        self.close()
        return self._isolate(htmls, doc_ids, max_html_chars)

    def _isolate(self, htmls: list[str], doc_ids: list[str], max_html_chars: int) -> tuple[list[str], list[str]]:
        texts: list[str] = []
        crashed: list[str] = []
        solo = self._new_executor(1)  # reused across docs; only a dead child forces a rebuild.
        try:
            for doc_id, html in zip(doc_ids, htmls, strict=True):
                try:
                    texts.append(solo.submit(self._fn, ([html], max_html_chars)).result()[0])
                    continue
                except BrokenProcessPool:
                    reason = "hard process death (SIGSEGV / OOM-kill)"
                    solo.shutdown(wait=False)  # a dead child poisons the executor.
                    solo = self._new_executor(1)
                except Exception as e:
                    reason = f"{type(e).__name__}: {e}"
                texts.append(self._crash_result)
                crashed.append(doc_id)
                logger.warning("resiliparse-rs dropped %s (%d chars): %s", doc_id, len(html), reason)
        finally:
            solo.shutdown(wait=False)
        if len(htmls) >= RS_MIN_CRASH_SAMPLE and len(crashed) > RS_MAX_CRASH_FRACTION * len(htmls):
            raise RuntimeError(
                f"resiliparse-rs crashed on {len(crashed)}/{len(htmls)} docs — the install is broken "
                "(expected ~0.1%). Refusing to write a corpus of empty extractions."
            )
        return texts, crashed

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


def _install_rust_extractor(
    artifact_prefix: str,
    spec: PipelineSpec,
    n_procs: int,
    *,
    extract_fn=preprocess.resiliparse_rs_batch,
    crash_result="",
    dest: str | None = None,
) -> ResiliparseRsPool:
    """Verify + install the prebuilt Rust extractor and return a smoke-tested pool for it.

    ``dest`` overrides the unpack directory — REQUIRED when several processes on one host
    install concurrently (the fused worker's A subprocesses): racing unpacks into the shared
    default dir produce truncated ``.so`` files ("file too short", observed 2026-09-01).

    ``install_extractor`` is imported here, not at module scope, on purpose: importing
    ``score_resiliparse_rs`` drags in ``marin.utils`` (datasets / rigging / huggingface_hub) and runs
    a module-level ``logging.basicConfig``, neither of which belongs in the live jusText worker.
    """
    from experiments.baseline_collection.score_resiliparse_rs import install_extractor

    _assert_artifact_matches_spec(artifact_prefix, spec)
    # Unpacks under /root by default: /tmp is noexec -> dlopen fails.
    pkg_dir = install_extractor(artifact_prefix) if dest is None else install_extractor(artifact_prefix, dest=dest)
    pool = ResiliparseRsPool(n_procs, pkg_dir, extract_fn=extract_fn, crash_result=crash_result)
    results, crashed = pool.run([SMOKE_HTML], ["_smoke"], spec.justext_max_html_chars)
    smoke_text = results[0][0] if isinstance(results[0], tuple) else results[0]
    if crashed or not smoke_text.strip():
        raise RuntimeError(f"resiliparse-rs smoke extraction failed (crashed={crashed}, result={results[0]!r})")
    logger.info("resiliparse-rs ready (%s, %d procs)", pkg_dir, pool.n_procs)
    return pool


def _safe_utf8(s: str) -> str:
    """Drop characters that can't be UTF-8 encoded (lone surrogates from bad-decoded pages).

    Essential: pyarrow's string->UTF-8 serialization raises ``UnicodeEncodeError`` on such a
    character, which crashes the parquet write for the ENTIRE chunk — so one bad document would kill
    the whole WARC's Phase C (this was ~75% of C failures). The dropped bytes are junk anyway.
    """
    return s.encode("utf-8", "ignore").decode("utf-8")


def _chunk_kept_table(
    table, cidx: list[int], texts: list[str], doc_ids: list[str], prob_of: dict, schema, pooled_of: dict | None = None
) -> tuple:
    """Build the kept table for one chunk; returns (table, n_extract_empty)."""
    cols: dict[str, list] = {name: [] for name in schema.names}
    n_empty = 0
    for j, i in enumerate(cidx):
        if not texts[j]:
            n_empty += 1
            continue  # passed ModernBERT but extraction found no content -> nothing to train on.
        cols["doc_id"].append(doc_ids[i])
        cols["url"].append(_safe_utf8(table.column("url")[i].as_py()))
        cols["warc_hash"].append(table.column("warc_hash")[i].as_py())
        cols["snapshot"].append(table.column("snapshot")[i].as_py())
        cols["fasttext_score"].append(table.column("fasttext_score")[i].as_py())
        cols["text"].append(_safe_utf8(texts[j]))
        cols["input_ids"].append(table.column("input_ids")[i].as_py())
        cols["n_tokens"].append(table.column("n_tokens")[i].as_py())
        cols["modernbert_prob"].append(prob_of[doc_ids[i]])
        if pooled_of is not None:
            cols["pooled_prob"].append(pooled_of.get(doc_ids[i]))
    return pa.table(cols, schema=schema), n_empty


def _extract_chunk(spec: PipelineSpec, pool, htmls: list[str], chunk_doc_ids: list[str]) -> tuple[list[str], list[str]]:
    """Extract one chunk with the spec's engine. Returns ``(texts, crashed_doc_ids)``.

    Only the Rust engine can lose documents to a process death; jusText's own failures are already
    Python exceptions handled inside ``preprocess.justext_text``, so it never reports crashes.
    """
    if spec.extraction_engine is Extractor.RESILIPARSE_RS:
        return pool.run(htmls, chunk_doc_ids, spec.justext_max_html_chars)
    args = [(h, spec.justext_lang, spec.justext_max_html_chars, spec.justext_paragraph_sep) for h in htmls]
    if pool is not None and args:
        return pool.run(args, spec.justext_timeout), []
    return [preprocess.justext_text(h, lang, cap, sep) for h, lang, cap, sep in args], []


def _process_one(
    warc_path: str, warc_hash: str, refresh, *, spec: PipelineSpec, pool, bucket: str, cleanup_chunks: bool = True
) -> dict:
    presurvivor_path = f"{spec.presurvivors_prefix(bucket)}/data-{warc_hash}.parquet"
    keeplist_path = f"{spec.keeplist_prefix(bucket)}/data-{warc_hash}.parquet"
    kept_path = f"{spec.kept_prefix(bucket)}/data-{warc_hash}.parquet"
    chunk_dir = f"{spec.namespace(bucket)}/kept_chunks/data-{warc_hash}"

    t0 = time.monotonic()
    kl = batch_format.read_table(keeplist_path)
    prob_of = dict(zip(kl.column("doc_id").to_pylist(), kl.column("modernbert_prob").to_pylist(), strict=True))
    kept_schema = batch_format.kept_schema_for(spec)
    pooled_of = (
        dict(zip(kl.column("doc_id").to_pylist(), kl.column("pooled_prob").to_pylist(), strict=True))
        if "pooled_prob" in kl.schema.names
        else None
    )

    table = batch_format.read_table(presurvivor_path)
    n = table.num_rows
    doc_ids = table.column("doc_id").to_pylist()
    # Rows that passed ModernBERT (prob >= threshold). On the pooled (lpv11) line a doc the pooled
    # stage dropped carries a NaN prob and is excluded here for free: NaN >= threshold is False.
    keep_idx = [i for i in range(n) if prob_of.get(doc_ids[i], 0.0) >= spec.modernbert_threshold]

    # Extract in CHUNK_SIZE batches, checkpointing each chunk to GCS (skip already-done chunks on
    # reclaim). A preempted worker loses at most the in-flight chunk.
    chunk_paths: list[str] = []
    t_ex = 0.0
    n_empty = 0
    crashed_ids: list[str] = []
    for ci, cstart in enumerate(range(0, len(keep_idx), CHUNK_SIZE)):
        cpath = f"{chunk_dir}/chunk_{ci:04d}.parquet"
        chunk_paths.append(cpath)
        if _gcs_exists(cpath):
            continue  # banked by an earlier (pre-preemption) run.
        cidx = keep_idx[cstart : cstart + CHUNK_SIZE]
        htmls = [table.column("html")[i].as_py() for i in cidx]
        s = time.monotonic()
        texts, crashed = _extract_chunk(spec, pool, htmls, [doc_ids[i] for i in cidx])
        t_ex += time.monotonic() - s
        crashed_ids.extend(crashed)
        ctable, ce = _chunk_kept_table(table, cidx, texts, doc_ids, prob_of, kept_schema, pooled_of)
        n_empty += ce
        batch_format.write_kept(cpath, ctable, kept_schema)
        refresh()  # keep the WARC claim fresh between chunks (long contended WARCs)

    # Merge chunks -> final flat kept parquet (downstream reads the flat kept/ layout unchanged).
    tables = [batch_format.read_table(cp) for cp in chunk_paths]
    merged = pa.concat_tables(tables) if tables else kept_schema.empty_table()
    batch_format.write_kept(kept_path, merged, kept_schema)
    if cleanup_chunks:
        batch_format.drop_chunk_dir(chunk_dir)

    try:
        with fsspec.open(f"{spec.namespace(bucket)}/timing_c/data-{warc_hash}.json", "w") as f:
            json.dump(
                {
                    "warc_hash": warc_hash,
                    "n_presurvivors": n,
                    "n_modernbert_keep": len(keep_idx),
                    # Historic key names (kept stable for already-written v1-v3 sidecars); with the
                    # Rust engine they mean "empty extraction" / "extraction seconds".
                    "n_justext_empty": n_empty,
                    "n_kept": merged.num_rows,
                    "n_chunks": len(chunk_paths),
                    "justext_s": round(t_ex, 3),
                    "extractor": str(spec.extraction_engine),
                    # Docs lost to a hard extractor crash (counted in n_justext_empty too, since a
                    # crash yields ""). Recorded so the loss is visible rather than silent.
                    "n_extract_crashed": len(crashed_ids),
                    "crashed_doc_ids": crashed_ids[:MAX_REPORTED_CRASH_IDS],
                    "wall_s": round(time.monotonic() - t0, 3),
                },
                f,
            )
    except Exception as e:
        logger.warning("phase C timing write failed for %s: %s", warc_hash, e)
    wall = time.monotonic() - t0
    logger.info(
        "C %s [%s]: %d mb-keep -> %d kept (%d empty, %d crashed, %d chunks) in %.1fs",
        warc_hash,
        spec.extraction_engine,
        len(keep_idx),
        merged.num_rows,
        n_empty,
        len(crashed_ids),
        len(chunk_paths),
        wall,
    )
    return {
        "docs_in": len(keep_idx),  # docs ModernBERT kept (extraction input)
        "docs_out": merged.num_rows,  # final kept docs after extraction
        "wall_seconds": wall,
        "compute_seconds": {str(spec.extraction_engine): t_ex},
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
    ap.add_argument(
        "--keep-chunks",
        dest="cleanup_chunks",
        action="store_false",
        default=True,
        help="Keep sub-WARC checkpoint chunks after merge (default deletes: ~4 TiB at 10k scale).",
    )
    ap.add_argument("--justext-procs", type=int, default=8, help="Extraction ProcessPool size (both engines).")
    ap.add_argument(
        "--resiliparse-artifact",
        default=RESILIPARSE_RS_ARTIFACT,
        help="Prebuilt resiliparse-rs artifact prefix (point at a same-region mirror to avoid an "
        "egress-charged read). Ignored unless the spec's extractor is resiliparse_rs.",
    )
    ap.add_argument(
        "--claim-stale-hours",
        type=float,
        default=0.25,
        help="Reclaim a dead worker's WARC after this (15 min: refreshed per-chunk, so safe; short "
        "enough that a crashed peer's WARCs free up inside the endgame-drain window in run_claim_loop).",
    )
    args = ap.parse_args()

    spec = get_spec(args.spec)
    _check_deps(spec)
    _pin_threads()
    if spec.extraction_engine is Extractor.RESILIPARSE_RS:
        pool = _install_rust_extractor(args.resiliparse_artifact, spec, args.justext_procs)
    else:
        pool = _make_justext_pool(args.justext_procs)
    process_one = functools.partial(
        _process_one, spec=spec, pool=pool, bucket=args.bucket, cleanup_chunks=args.cleanup_chunks
    )

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
        upstream_path_fn=_keeplist_path,  # only extract a WARC once Phase B wrote its keeplist
        upstream_done_path=f"gs://marin-us-central1/{spec.subdir()}/_phase_b_end.json",
        # Per-chunk claim refresh keeps live workers fresh; a dead worker's WARC is reclaimed in 30
        # min (then finished from its banked chunks) instead of the default 3h.
        claim_stale_hours=args.claim_stale_hours,
    )
    if pool is not None:
        pool.close()


if __name__ == "__main__":
    main()
