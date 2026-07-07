# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-decode the 100k sample's WARCs with the CORRECT charset decoder, then run jusText 4.2.0
(fastText tier) on the clean raw HTML — producing a clean ``raw_html`` for all 100k docs + a
``text_justext`` extraction column.

Why re-decode: the sample's ``raw_html`` is only present for ~7% (the where-kept design), and we
want jusText on RAW html for all 100k. Critically we decode with ``decode_warcs_clean.decode_payload``
(WHATWG charset order, never ``errors="replace"`` → 0 U+FFFD), NOT the broken ``download_warcs``
decoder the extractors used. See ``WARC_ENCODING.md``.

jusText = the XenonMolecule fork v4.2.0 ``fasttext`` tier (model ``MichaelR207/justext-classifier``,
auto-downloaded from HF; ``JUSTEXT_MODEL=fasttext``). Stages:

  ``extract``  — CPU fan-out, one task per WARC: decode (clean) → keep only the 100k sample docs →
      jusText each → write ``justext_redecode/data-<hash>.parquet`` {warc_record_id, raw_html, text_justext}.
      Logs per-core jusText docs/s + a U+FFFD count (must be 0).
  ``validate`` — scan the staged raw_html for U+FFFD (the encoding gate; must be 0).
  ``join``     — REPLACE ``raw_html`` (clean, all 100k) + ADD ``text_justext`` onto sample_100k_scored.

Run (CPU; HF_TOKEN for the model)::

    iris --cluster marin job run --region us-east5 --cpu 8 --memory 32GB --disk 60GB \\
      --enable-extra-resources --extra cpu --extra extraction-bakeoff --priority interactive --no-wait \\
      --job-name justext-redecode -e HF_TOKEN ... -- \\
      python -m experiments.baseline_collection.score_justext_redecode extract
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import re
import time
from collections.abc import Iterator

import fsspec
from fray import ResourceConfig
from marin.utils import fsspec_glob
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext

from experiments.baseline_collection.decode_warcs_clean import (
    REPLACEMENT,
    _decode_one_warc,
    _load_manifest,
    _warc_path_hash,
)

logger = logging.getLogger(__name__)

OUT_ROOT = "gs://marin-us-east5/documents/extractor_compare/high_quality_200warc"
SAMPLE_DIR = f"{OUT_ROOT}/sample_100k"
SCORED_DIR = f"{OUT_ROOT}/sample_100k_scored"
STAGING = f"{OUT_ROOT}/justext_redecode"
MANIFEST = "experiments/distill/dclm_1p7b_completed_sample200_warcs.txt"
COLUMNS = ("warc_record_id", "raw_html", "text_justext")

_SAMPLE_IDS: set[str] | None = None  # per-worker lazy cache


def _sample_record_ids() -> set[str]:
    global _SAMPLE_IDS
    if _SAMPLE_IDS is None:
        import pyarrow.parquet as pq

        ids: set[str] = set()
        for p in sorted(fsspec_glob(f"{SAMPLE_DIR}/*.parquet")):
            with fsspec.open(p, "rb") as fh:
                ids.update(pq.ParquetFile(fh).read(columns=["warc_record_id"]).column("warc_record_id").to_pylist())
        _SAMPLE_IDS = ids
        logger.info("sample record ids: %d", len(ids))
    return _SAMPLE_IDS


# XML-invalid control chars (NULL + C0 controls except \t \n \r). lxml (jusText's parser) rejects
# these with "All strings must be XML compatible"; they're encoding garbage, not content.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# jusText/lxml can HANG (not raise) on pathological documents — a single malformed doc inside a WARC
# froze the whole endgame. Two guards: (1) a cheap char-size pre-skip, (2) a hard per-doc wall-clock
# TIMEOUT enforced in a child process (the hang is C-level in lxml, so a Python signal can't interrupt
# it — only killing the process can). Both keep the clean raw_html; only text_justext is empty on skip.
MAX_JUSTEXT_HTML_CHARS = 3_000_000
JUSTEXT_DOC_TIMEOUT = 20.0  # seconds per doc; pathological docs are killed + skipped past this


def _justext_extract(html: str, stoplist) -> str:
    """jusText main-content text from clean HTML, IN-PROCESS (no timeout). Used by the bench on already-
    clean staged HTML where no pathological doc is expected; the WARC pass uses the timeout-guarded path."""
    import justext

    if len(html) > MAX_JUSTEXT_HTML_CHARS:
        logger.warning("skipping jusText on %d-char doc (> %d cap); empty extraction", len(html), MAX_JUSTEXT_HTML_CHARS)
        return ""
    try:
        paras = justext.justext(html.encode("utf-8", "ignore"), stoplist)
        return "\n\n".join(p.text for p in paras if not p.is_boilerplate)
    except Exception as e:
        logger.warning("jusText failed on a doc (%s); empty extraction", type(e).__name__)
        return ""


_CHILD_READY = "__READY__"


def _justext_child_loop(conn) -> None:
    """Child process: load jusText (incl. the fastText model) ONCE, signal ready, then extract each html
    received on the pipe. A doc that hangs lxml hangs THIS process — the parent kills it on timeout, so
    the WARC can continue. Model load happens before the ready signal, OUTSIDE the parent's per-doc timeout."""
    os.environ.setdefault("JUSTEXT_MODEL", "fasttext")
    import justext

    stoplist = justext.get_stoplist("English")
    t0 = time.monotonic()
    try:  # warm the (fastText) model now so a slow load/download isn't mistaken for a hung doc
        justext.justext(b"<html><body><p>warmup paragraph for model load.</p></body></html>", stoplist)
    except Exception as e:
        logger.warning("jusText child warmup failed (%s)", type(e).__name__)
    logger.info("jusText child: model loaded + warm in %.1fs", time.monotonic() - t0)
    conn.send(_CHILD_READY)
    while True:
        try:
            html = conn.recv()
        except EOFError:
            return
        try:
            paras = justext.justext(html.encode("utf-8", "ignore"), stoplist)
            text = "\n\n".join(p.text for p in paras if not p.is_boilerplate)
        except Exception:
            text = ""
        conn.send(text)


class JustextTimeoutRunner:
    """Run jusText in a persistent child process with a hard per-doc timeout. On timeout the child is
    killed (defeats C-level lxml hangs) and respawned; the offending doc gets an empty extraction."""

    def __init__(self, timeout: float = JUSTEXT_DOC_TIMEOUT, ready_timeout: float = 300.0):
        self.timeout = timeout
        self.ready_timeout = ready_timeout  # generous: covers first-time fastText model download
        self.n_timeout = 0
        # fork (not spawn): the child inherits the already-imported module — no costly re-import of the
        # heavy zephyr/justext stack per child, and no spawn `__main__` re-execution pitfall.
        self._ctx = multiprocessing.get_context("fork")
        self._start()

    def _start(self) -> None:
        self._conn, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(target=_justext_child_loop, args=(child,), daemon=True)
        self._proc.start()
        # Block (off the per-doc timeout) until the child has loaded its model and is ready to serve.
        if not self._conn.poll(self.ready_timeout) or self._conn.recv() != _CHILD_READY:
            logger.warning("jusText child not ready after %ss; proceeding anyway", self.ready_timeout)

    def extract(self, html: str) -> str:
        if len(html) > MAX_JUSTEXT_HTML_CHARS:
            logger.warning("skipping jusText on %d-char doc (> %d cap)", len(html), MAX_JUSTEXT_HTML_CHARS)
            return ""
        try:
            self._conn.send(html)
            if self._conn.poll(self.timeout):
                return self._conn.recv()
        except (EOFError, BrokenPipeError, OSError):
            pass  # child died mid-doc — treat as a skip, respawn below
        else:
            self.n_timeout += 1
            logger.warning("jusText doc TIMEOUT (>%ss) — killing child, skipping doc", self.timeout)
        self._proc.kill()
        self._proc.join()
        self._start()
        return ""

    def close(self) -> None:
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join()


def _process_warc(warc_path: str) -> Iterator[dict]:
    """Decode one WARC (clean), keep only the sample's docs, jusText each (per-doc timeout-guarded)."""
    ids = _sample_record_ids()
    runner = JustextTimeoutRunner()

    warc_hash = _warc_path_hash(warc_path)
    logger.info("WARC %s: decoding + jusText (sample docs only)...", warc_hash)
    n = ffd = 0
    jt_time = 0.0
    try:
        for page in _decode_one_warc(warc_path):
            rid = page["doc_id"]
            if rid not in ids:
                continue
            html = _CONTROL_RE.sub("", page["html"])  # strip XML-invalid control chars (lxml + clean storage)
            if REPLACEMENT in html:  # the clean decoder should never produce U+FFFD
                ffd += 1
            t = time.monotonic()
            text = runner.extract(html)
            jt_time += time.monotonic() - t
            n += 1
            if n % 100 == 0:  # per-doc progress so a stall is visible long before the WARC finishes
                logger.info(
                    "WARC %s: %d sample docs done (%.1f docs/s, %d timeouts)",
                    warc_hash, n, n / jt_time if jt_time else 0.0, runner.n_timeout,
                )
            yield {"warc_record_id": rid, "raw_html": html, "text_justext": text}
        if n:
            logger.info(
                "WARC %s: %d sample docs | jusText %.1fs = %.2f docs/s/core | U+FFFD=%d | timeouts=%d",
                warc_hash,
                n,
                jt_time,
                n / jt_time if jt_time else 0.0,
                ffd,
                runner.n_timeout,
            )
    finally:
        runner.close()


def run_extract(limit_warcs: int | None, max_workers: int, manifest: str = MANIFEST) -> None:
    import pyarrow as pa

    warcs = _load_manifest(manifest)
    if limit_warcs is not None:
        warcs = warcs[:limit_warcs]
    logger.info("jusText re-decode over %d WARCs (sample docs only) -> %s", len(warcs), STAGING)
    schema = pa.schema([(c, pa.string()) for c in COLUMNS])

    def _out(shard_idx: int, total: int) -> str:
        return f"{STAGING}/data-{_warc_path_hash(warcs[shard_idx])}.parquet"

    pipeline = (
        Dataset.from_list(warcs)
        .reshard(len(warcs))
        .flat_map(_process_warc)
        .write_parquet(_out, schema=schema, skip_existing=True)
    )
    ctx = ZephyrContext(name="justext-redecode", max_workers=max_workers, resources=ResourceConfig(cpu=2, ram="24g"))
    ctx.execute(pipeline)
    logger.info("extract done -> %s", STAGING)


def run_extract_local(manifest: str) -> None:
    """Process WARCs serially IN-PROCESS (no Zephyr) — for re-running a tiny stuck subset without the
    coordinator's heartbeat/lease machinery (which reassigns slow cold-start WARCs in a loop). Per-doc
    jusText timeout still applies via ``_process_warc``. Writes one parquet per WARC, skipping staged ones."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    warcs = _load_manifest(manifest)
    schema = pa.schema([(c, pa.string()) for c in COLUMNS])
    logger.info("LOCAL extract over %d WARCs -> %s", len(warcs), STAGING)
    for w in warcs:
        out = f"{STAGING}/data-{_warc_path_hash(w)}.parquet"
        if fsspec_glob(out):
            logger.info("skip (already staged): %s", out)
            continue
        t0 = time.monotonic()
        rows = list(_process_warc(w))
        table = pa.Table.from_pylist(rows, schema=schema)
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh)
        logger.info("wrote %d rows in %.1fs -> %s", len(rows), time.monotonic() - t0, out)
    logger.info("LOCAL extract done -> %s", STAGING)


def run_validate() -> None:
    """Encoding gate: count U+FFFD across the staged raw_html (a correct decode yields 0)."""
    import pyarrow.parquet as pq

    files = sorted(fsspec_glob(f"{STAGING}/*.parquet"))
    if not files:
        raise RuntimeError(f"no staged parquet under {STAGING}; run extract first")
    rows = bad = 0
    for f in files:
        with fsspec.open(f, "rb") as fh:
            for h in pq.ParquetFile(fh).read(columns=["raw_html"]).column("raw_html").to_pylist():
                rows += 1
                if h and REPLACEMENT in h:
                    bad += 1
    logger.info("VALIDATE: %d shards, %d docs, %d with U+FFFD", len(files), rows, bad)
    if bad:
        raise SystemExit(f"FAIL: {bad} docs contain U+FFFD — decode is wrong, do NOT join")
    logger.info("VALIDATE PASS: zero U+FFFD across %d docs", rows)


def load_staging() -> dict[str, tuple[str | None, str | None]]:
    """{warc_record_id: (clean_raw_html, text_justext)} from the staged re-decode parquet (empty if none).

    Shared so any join that rebuilds ``sample_100k_scored`` (e.g. the BERT score join) can re-apply the
    clean raw_html + text_justext, instead of those existing only in whichever join ran last."""
    import pyarrow.parquet as pq

    staged: dict[str, tuple[str | None, str | None]] = {}
    for f in sorted(fsspec_glob(f"{STAGING}/*.parquet")):
        with fsspec.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read()
        for rid, rh, tj in zip(
            t.column("warc_record_id").to_pylist(),
            t.column("raw_html").to_pylist(),
            t.column("text_justext").to_pylist(),
            strict=True,
        ):
            staged[rid] = (rh, tj)
    return staged


def run_join() -> None:
    """REPLACE raw_html (clean) + ADD text_justext onto sample_100k_scored (in place)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    staged = load_staging()
    logger.info("staged %d re-decoded docs", len(staged))

    shards = sorted(fsspec_glob(f"{SCORED_DIR}/*.parquet"))
    for path in shards:
        with fsspec.open(path, "rb") as fh:
            t = pq.ParquetFile(fh).read()
        rids = t.column("warc_record_id").to_pylist()
        old_raw = t.column("raw_html").to_pylist()
        new_raw = [staged[r][0] if r in staged else old_raw[i] for i, r in enumerate(rids)]
        new_tj = [staged.get(r, (None, None))[1] for r in rids]
        idx = t.column_names.index("raw_html")
        t = t.set_column(idx, "raw_html", pa.array(new_raw, type=pa.string()))
        t = t.append_column("text_justext", pa.array(new_tj, type=pa.string()))
        with fsspec.open(path, "wb") as fh:
            pq.write_table(t, fh)
    logger.info("join done: raw_html replaced + text_justext added on %d shards of %s", len(shards), SCORED_DIR)


def run_bench(n_docs: int, warmup: int) -> None:
    """True single-core jusText speed: read ALREADY-decoded clean ``raw_html`` from staging and
    time ONLY ``justext.justext`` (no decode, no fan-out, no HF fetch in the hot loop). Reports
    pure docs/s/core + p50/p90/p99 per-doc latency — the number the two-stage extract log conflates
    with decode + worker contention."""
    import pyarrow.parquet as pq

    os.environ.setdefault("JUSTEXT_MODEL", "fasttext")
    import justext

    stoplist = justext.get_stoplist("English")

    files = sorted(fsspec_glob(f"{STAGING}/*.parquet"))
    if not files:
        raise RuntimeError(f"no staged parquet under {STAGING}; run extract first")
    htmls: list[str] = []
    for f in files:
        with fsspec.open(f, "rb") as fh:
            for h in pq.ParquetFile(fh).read(columns=["raw_html"]).column("raw_html").to_pylist():
                if h:
                    htmls.append(h)
        if len(htmls) >= n_docs + warmup:
            break
    htmls = htmls[: n_docs + warmup]
    logger.info("bench: %d docs (%d warmup) on 1 core", len(htmls), warmup)

    for h in htmls[:warmup]:  # warm stoplist / fastText model / lxml — discarded
        _justext_extract(h, stoplist)

    lat: list[float] = []
    t0 = time.monotonic()
    for h in htmls[warmup:]:
        t = time.monotonic()
        _justext_extract(h, stoplist)
        lat.append(time.monotonic() - t)
    wall = time.monotonic() - t0
    lat.sort()
    n = len(lat)

    def pct(q: float) -> float:
        return lat[min(n - 1, int(q * n))] * 1000.0

    logger.info(
        "JUSTEXT BENCH (1 core, fastText tier): %d docs in %.1fs = %.2f docs/s/core | "
        "per-doc p50=%.1fms p90=%.1fms p99=%.1fms",
        n,
        wall,
        n / wall if wall else 0.0,
        pct(0.50),
        pct(0.90),
        pct(0.99),
    )


def main() -> None:
    configure_logging(logging.INFO)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    pe = sub.add_parser("extract", help="Decode (clean) + jusText the 100k sample docs.")
    pe.add_argument("--limit-warcs", type=int, default=None, help="Smoke: first N WARCs.")
    pe.add_argument("--max-workers", type=int, default=64)
    pe.add_argument("--manifest", default=MANIFEST, help="WARC manifest (override to re-run a subset).")
    pl = sub.add_parser("extract-local", help="Serial in-process extract (no Zephyr) for a small stuck subset.")
    pl.add_argument("--manifest", default=MANIFEST, help="WARC manifest (override to re-run a subset).")
    sub.add_parser("validate", help="U+FFFD gate over staged raw_html (must be 0).")
    sub.add_parser("join", help="Replace raw_html + add text_justext on sample_100k_scored.")
    pb = sub.add_parser("bench", help="True single-core jusText speed on already-decoded staged HTML.")
    pb.add_argument("--n-docs", type=int, default=2000, help="Timed docs after warmup.")
    pb.add_argument("--warmup", type=int, default=50, help="Discarded warmup docs.")
    args = p.parse_args()
    if args.stage == "extract":
        run_extract(args.limit_warcs, args.max_workers, args.manifest)
    elif args.stage == "extract-local":
        run_extract_local(args.manifest)
    elif args.stage == "validate":
        run_validate()
    elif args.stage == "join":
        run_join()
    elif args.stage == "bench":
        run_bench(args.n_docs, args.warmup)


if __name__ == "__main__":
    main()
