# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert fastText-format classifier shards from ``body_strip`` HTML to extracted TEXT.

Every useful-vs-NO_USEFUL classifier so far trained on ``body_strip`` HTML, where **78.3% of the
characters are markup**: a representative survivor doc is 8192 tokens as HTML but 470 tokens with the
tags gone, so ~94% of the context window is spent on markup and most docs are truncated before the
model sees their content. This produces the TEXT-representation twin of a training/eval set so the
two can be compared head to head.

The extractor is the XenonMolecule fork's **Rust** main-content engine (``resiliparse._extract_rs``),
downloaded as a prebuilt Linux artifact by :func:`score_resiliparse_rs.install_extractor` — see that
module for why it must not be unpacked under ``/tmp``.

Invariants that make the output a drop-in replacement:

* one output line per input line, SAME ORDER, SAME label token — index-aligned score files
  (fastText/pooled cascade scores) and ``--train-rows`` prefixes stay valid;
* newlines inside the extracted markdown are collapsed to spaces (a newline would split one doc into
  several fastText records);
* a doc whose extraction is empty — or which crashes the extractor — still gets a line, holding
  :data:`EMPTY_PLACEHOLDER`, so nothing is ever dropped.

Three ways a page can take a shard down, all contained (the recipe is ``score_resiliparse_rs``'s,
hardened for the ~40x larger corpus):

* the Rust extractor hard-SEGFAULTS on ``<frameset>`` pages — 0.108% of the web, ~275 per shard, far
  too many to absorb by rebuilding the pool, so they are screened inside the worker before the call;
* anything else that kills a worker is isolated one doc per process and written as a placeholder;
* a page whose extraction grows without bound (two exist in the survivor set) would OOM-kill the
  whole task, so workers run under :data:`WORKER_ADDRESS_SPACE_LIMIT` and die alone instead.

Shards are written under ``_partial/`` and moved into place only once complete: SIGTERM unwinding the
writer's ``with`` block finalizes a truncated GCS object, which a resumed run would happily skip.

Launch one job per shard — independent jobs schedule as capacity frees, while a coordinated
N-worker group on this cluster schedules a single actor and crawls::

    for i in $(seq 0 39); do
      uv run iris --cluster marin job run --region us-east5 \\
        --cpu 16 --memory 64GB --disk 100GB --extra cpu --extra dclm --enable-extra-resources \\
        --priority interactive --preemptible --no-wait --job-name xtext-surv-$i -- \\
        python -m experiments.baseline_collection.extract_text_shards \\
          --in-glob 'gs://.../presharded_survivor_w640_10M/train_shard_*.txt.gz' \\
          --out-dir 'gs://.../presharded_survivor_w640_10M_text' \\
          --shard-start $i --shard-end $((i + 1)) --workers 16
    done

``--verify`` re-reads an input/output pair and checks the invariants (line count, label agreement, no
embedded newlines) plus mean chars and mean ``answerdotai/ModernBERT-base`` tokens before vs after.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import re
import resource
import time
from concurrent.futures import ProcessPoolExecutor

import fsspec
from fsspec.core import url_to_fs

from experiments.baseline_collection.score_resiliparse_rs import install_extractor
from experiments.fsspec_paths import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Single whitespace-free token, so an empty extraction costs one fastText token and one wordpiece
# instead of silently shortening the file.
EMPTY_PLACEHOLDER = "__empty__"
# Worker -> parent marker for a page the frameset screen refused to hand to the extractor. NUL bytes
# cannot appear in the extracted text, so this can never collide with a real extraction.
FRAMESET_SKIPPED = "\x00frameset\x00"
# Applied in the PARENT: multi-MB outliers are never pickled to a worker.
MAX_HTML_CHARS = 2_000_000
# Must stay MUCH smaller than a batch or one worker takes the whole batch, and a crash isolates over
# 25 docs instead of thousands.
CHUNK_SIZE = 25
# Docs held in memory per round trip. At ~69 kB/doc this is ~140 MB of HTML in flight.
BATCH_DOCS = 2000
_WHITESPACE = re.compile(r"\s+")
# Heap cap per worker. Two of the 40 survivor shards hold a page whose extraction grows without
# bound: with no cap the node's OOM killer takes down the whole task (state KILLED, exit 0, at a
# byte-identical point on every retry), so the shard could never finish. Capped, the runaway worker
# dies alone, the failure is isolated per doc, and the page gets a placeholder like any other
# unextractable one. Normal extraction of a 25-doc chunk uses a few MB.
#
# RLIMIT_DATA, not RLIMIT_AS: since Linux 4.7 it covers the heap and anonymous mmaps — what actually
# runs away here — without counting the address space CPython and the extension merely RESERVE. An
# RLIMIT_AS of 4 GiB kills every worker at startup on some platforms.
WORKER_HEAP_LIMIT = 6 * 1024**3
# A shard where extraction fails this often is a broken environment (e.g. a memory cap so tight that
# no page survives), not a bad corpus — refuse to publish it. Healthy shards sit near 0.8%.
MAX_PLACEHOLDER_RATE = 0.10
TOKENIZER = "answerdotai/ModernBERT-base"
# The classifiers' context window: docs longer than this are truncated at training/scoring time.
CONTEXT_TOKENS = 8192


def _cap_worker_memory() -> None:
    """Bound this worker's heap so a runaway page cannot OOM the whole task.

    Only the soft limit moves, and never above the inherited hard limit. A platform that refuses the
    call (macOS caps RLIMIT_DATA below what it reports as its hard limit) must not lose its workers
    over it: without the cap the job behaves exactly as it did before the cap existed, and the
    :data:`MAX_PLACEHOLDER_RATE` check still guards the output.
    """
    _, hard = resource.getrlimit(resource.RLIMIT_DATA)
    limit = WORKER_HEAP_LIMIT if hard == resource.RLIM_INFINITY else min(WORKER_HEAP_LIMIT, hard)
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (limit, hard))
    except (ValueError, OSError) as exc:
        logger.warning("could not cap worker heap at %d bytes: %r", limit, exc)


def _new_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, initializer=_cap_worker_memory)


def _extract(htmls: list[str]) -> list[str | None]:
    """Markdown main-content extraction for a chunk of pages (runs in a worker process).

    :data:`FRAMESET_SKIPPED` marks a page the screen refused; ``None`` never comes from here (the
    parent uses it for a page that crashed the process). Imported inside the worker because the
    package is downloaded at runtime, not installed.

    ``<frameset>`` pages are screened here rather than in the parent: the check is a substring scan
    over multi-kB HTML, so it belongs on the worker cores, and it is what keeps the pool alive. At
    ~0.108% of pages — ~275 per 240k-doc shard — recovering from the segfault instead of screening
    for it would mean tearing down and rebuilding the pool hundreds of times per shard. The prep
    HTML is lowercased, so the lowercase test is exact; the uppercase test costs nothing and covers
    a caller that feeds unlowercased HTML.
    """
    from resiliparse._extract_rs import extract_plain_text

    out: list[str | None] = []
    for html in htmls:
        if "<frameset" in html or "<FRAMESET" in html:
            out.append(FRAMESET_SKIPPED)
            continue
        out.append(extract_plain_text(html, main_content=True, preserve_formatting="markdown"))
    return out


def _one_line(text: str | None) -> str:
    """Collapse extracted markdown to a single fastText-safe line."""
    if not text:
        return EMPTY_PLACEHOLDER
    flat = _WHITESPACE.sub(" ", text).strip()
    return flat or EMPTY_PLACEHOLDER


def _isolate(htmls: list[str], offset: int, crashed: list[int]) -> list[str | None]:
    """Re-run a chunk the pool died on, one doc per process, recording the offenders' indices.

    Both failure modes end the same way — the page is unextractable, gets a placeholder, and is
    reported: a segfault kills the worker (``BrokenProcessPool``), and a page that hits the address
    space cap raises ``MemoryError`` (or dies, depending on where the allocation failed).
    """
    texts: list[str | None] = []
    for i, html in enumerate(htmls):
        with _new_pool(1) as solo:
            try:
                texts.append(solo.submit(_extract, [html]).result()[0])
            except Exception as exc:
                texts.append(None)
                crashed.append(offset + i)
                logger.warning("UNEXTRACTABLE doc %d (%d chars): %r", offset + i, len(html), exc)
    return texts


class _Pool:
    """A process pool that can be replaced in place once a segfault has poisoned it."""

    def __init__(self, workers: int):
        self.workers = workers
        self.pool = _new_pool(workers)

    def rebuild(self) -> None:
        self.pool.shutdown(wait=False)
        self.pool = _new_pool(self.workers)

    def close(self) -> None:
        self.pool.shutdown()


def extract_batch(pool: _Pool, htmls: list[str], offset: int, crashed: list[int]) -> list[str | None]:
    """Extract a batch of pages, surviving a worker that dies on one of them.

    Chunks are submitted individually so that a :class:`BrokenProcessPool` only costs the chunks that
    had not returned yet: those are retried on a fresh pool, and a chunk that fails again is split to
    one doc per process so the offending page is pinned down and recorded instead of losing the
    shard.
    """
    chunks = [htmls[i : i + CHUNK_SIZE] for i in range(0, len(htmls), CHUNK_SIZE)]
    results: list[list[str | None] | None] = [None] * len(chunks)
    futures = [pool.pool.submit(_extract, chunk) for chunk in chunks]
    broken = False
    for i, future in enumerate(futures):
        try:
            results[i] = future.result()
        except Exception:
            broken = True
    if broken:
        logger.warning("extractor failed in this batch — retrying %d chunk(s)", sum(r is None for r in results))
        pool.rebuild()
        for i, chunk in enumerate(chunks):
            if results[i] is not None:
                continue
            try:
                results[i] = pool.pool.submit(_extract, chunk).result()
            except Exception:
                pool.rebuild()
                results[i] = _isolate(chunk, offset + i * CHUNK_SIZE, crashed)
    texts: list[str | None] = []
    for chunk_texts in results:
        assert chunk_texts is not None, "every chunk is either extracted or isolated"
        texts.extend(chunk_texts)
    return texts


def _read_lines(path: str):
    """Stream a gzipped fastText file line by line (shards are ~17 GB uncompressed)."""
    with fsspec.open(path, "rb") as raw, gzip.open(raw, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def convert_shard(path: str, out_path: str, pool: _Pool, limit_docs: int | None) -> dict:
    """Convert one shard, streaming input and output. Returns the shard's stats."""
    docs = 0
    empty = 0
    frameset = 0
    crashed: list[int] = []
    html_chars = 0
    text_chars = 0
    t0 = time.monotonic()

    batch_labels: list[str] = []
    batch_htmls: list[str] = []

    with fsspec.open(out_path, "wb") as raw_out, gzip.open(raw_out, "wt", encoding="utf-8") as out:

        def flush() -> None:
            nonlocal docs, empty, frameset, text_chars
            if not batch_htmls:
                return
            texts = extract_batch(pool, batch_htmls, docs, crashed)
            for label, text in zip(batch_labels, texts, strict=True):
                # Three ways to get a placeholder, counted apart: screened frameset, crashed page
                # (already in ``crashed``), and a page the extractor found no main content in.
                if text == FRAMESET_SKIPPED:
                    frameset += 1
                    line = EMPTY_PLACEHOLDER
                else:
                    line = _one_line(text)
                    if line == EMPTY_PLACEHOLDER and text is not None:
                        empty += 1
                text_chars += len(line)
                out.write(f"{label} {line}\n")
            docs += len(batch_htmls)
            batch_labels.clear()
            batch_htmls.clear()

        for line in _read_lines(path):
            label, _, html = line.rstrip("\n").partition(" ")
            html = html[:MAX_HTML_CHARS]
            html_chars += len(html)
            batch_labels.append(label)
            batch_htmls.append(html)
            if limit_docs is not None and docs + len(batch_htmls) >= limit_docs:
                break
            if len(batch_htmls) >= BATCH_DOCS:
                flush()
                logger.info(
                    "%s: %d docs, %d empty, %d frameset, %d crashed",
                    os.path.basename(path),
                    docs,
                    empty,
                    frameset,
                    len(crashed),
                )
        flush()

    elapsed = time.monotonic() - t0
    stats = {
        "shard": os.path.basename(path),
        "docs": docs,
        "empty": empty,
        "frameset_prefiltered": frameset,
        "crashed_indices": crashed,
        "mean_html_chars": html_chars / docs if docs else 0.0,
        "mean_text_chars": text_chars / docs if docs else 0.0,
        "elapsed": elapsed,
        "docs_per_second": docs / elapsed if elapsed else 0.0,
    }
    logger.info("shard done: %s", json.dumps({k: v for k, v in stats.items() if k != "crashed_indices"}))
    placeholders = empty + frameset + len(crashed)
    if docs and placeholders / docs > MAX_PLACEHOLDER_RATE:
        raise RuntimeError(
            f"{stats['shard']}: {placeholders}/{docs} docs unextractable "
            f"(> {MAX_PLACEHOLDER_RATE:.0%}) — refusing to publish a shard this degraded"
        )
    return stats


def run(
    in_glob: str,
    out_dir: str,
    shard_start: int,
    shard_end: int | None,
    workers: int,
    limit_docs: int | None,
) -> None:
    files = sorted(fsspec_glob(in_glob))
    if not files:
        raise ValueError(f"no shards match {in_glob}")
    selected = files[shard_start:shard_end]
    logger.info("%d shard(s) match; this job takes [%s:%s] = %d", len(files), shard_start, shard_end, len(selected))

    install_extractor()
    fs, _ = url_to_fs(out_dir)
    pool = _Pool(workers)
    try:
        for path in selected:
            name = os.path.basename(path)
            out_path = f"{out_dir}/{name}"
            # Resume: preemptible workers restart the task from scratch, so a finished shard must be
            # recognisable. The final name is only ever claimed by the move below, which is why the
            # shard is written under _partial first: SIGTERM on preemption unwinds the `with`, and a
            # closed GCSFile FINALIZES its upload — publishing a truncated shard straight to the
            # final name, which the next attempt would then skip. Observed on 2 of 40 shards.
            if fsspec_exists(out_path):
                logger.info("skip %s — output exists", name)
                continue
            tmp_path = f"{out_dir}/_partial/{name}.{os.getpid()}.{int(time.time())}"
            stats = convert_shard(path, tmp_path, pool, limit_docs)
            fs.mv(tmp_path, out_path)
            with fsspec.open(f"{out_dir}/_stats/{name}.json", "w") as fh:
                json.dump(stats, fh, indent=2)
    finally:
        pool.close()
    logger.info("DONE -> %s", out_dir)


def _token_counts(texts: list[str]) -> dict[str, float]:
    """Mean/median :data:`TOKENIZER` length of a sample, plus the share that a 8192-token classifier
    would truncate — the number the whole HTML-vs-text comparison is about."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    lengths = sorted(len(ids) for ids in tokenizer(texts, truncation=False)["input_ids"])
    return {
        "mean_tokens": sum(lengths) / len(lengths),
        "median_tokens": lengths[len(lengths) // 2],
        "frac_over_8192": sum(1 for n in lengths if n > CONTEXT_TOKENS) / len(lengths),
    }


def verify(in_glob: str, out_dir: str, shard_start: int, shard_end: int | None, token_sample: int) -> None:
    """Check the output invariants against the input and report the size reduction.

    The report is written to ``{out_dir}/_stats/verify_{shard}.json`` as well as logged: job logs on
    this cluster are not durable (finelog answers ``Not Found`` for finished jobs), so a
    log-only report would be unreadable minutes after the job succeeds.
    """
    files = sorted(fsspec_glob(in_glob))[shard_start:shard_end]
    for path in files:
        name = os.path.basename(path)
        out_path = f"{out_dir}/{name}"
        n = 0
        label_mismatch = 0
        empty = 0
        html_chars = 0
        text_chars = 0
        html_sample: list[str] = []
        text_sample: list[str] = []
        for src, dst in zip(_read_lines(path), _read_lines(out_path), strict=True):
            src_label, _, html = src.rstrip("\n").partition(" ")
            dst_label, _, text = dst.rstrip("\n").partition(" ")
            if src_label != dst_label:
                label_mismatch += 1
            if "\n" in text or "\r" in text:
                raise AssertionError(f"{name} line {n}: embedded newline in output")
            if text == EMPTY_PLACEHOLDER:
                empty += 1
            html_chars += len(html)
            text_chars += len(text)
            if len(html_sample) < token_sample:
                html_sample.append(html)
                text_sample.append(text)
            n += 1
        html_tokens = _token_counts(html_sample)
        text_tokens = _token_counts(text_sample)
        report = {
            "shard": name,
            "lines": n,
            "label_mismatch": label_mismatch,
            "empty_placeholders": empty,
            "mean_html_chars": html_chars / n,
            "mean_text_chars": text_chars / n,
            "char_reduction_pct": 100 * (1 - text_chars / html_chars),
            "token_sample": len(html_sample),
            **{f"html_{k}": v for k, v in html_tokens.items()},
            **{f"text_{k}": v for k, v in text_tokens.items()},
        }
        with fsspec.open(f"{out_dir}/_stats/verify_{name}.json", "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("VERIFY %s", json.dumps(report))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-glob", required=True, help="fastText-format gzipped shards to convert")
    p.add_argument("--out-dir", required=True, help="output prefix; shards keep their input filename")
    p.add_argument("--shard-start", type=int, default=0)
    p.add_argument("--shard-end", type=int, default=None, help="exclusive; default = all shards")
    p.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit-docs", type=int, default=None, help="stop after N docs per shard (smoke test)")
    p.add_argument("--verify", action="store_true", help="compare existing output against the input instead")
    p.add_argument("--token-sample", type=int, default=200, help="--verify: docs to tokenize for the token means")
    args = p.parse_args()

    if args.verify:
        verify(args.in_glob, args.out_dir, args.shard_start, args.shard_end, args.token_sample)
        return
    run(args.in_glob, args.out_dir, args.shard_start, args.shard_end, args.workers, args.limit_docs)


if __name__ == "__main__":
    main()
