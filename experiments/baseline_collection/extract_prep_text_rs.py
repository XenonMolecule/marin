# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-materialize fastText ``full_prep`` shards as TEXT instead of HTML.

The whole lpv11 classifier lineup trains on ``body_strip`` HTML, which is ~78% markup by
character count. This script produces the TEXT twin of the prep shards so the identical
document set can be retrained on extracted main content:

    <prep_root>/{train,test}/data-{i:05d}.txt.gz     (body_strip HTML lines)
        -> <out_root>/{train,test}/data-{i:05d}.txt.gz  (extracted-text lines)

Line i of the output corresponds to line i of the input, same label, same order, same count.
Empty extractions are KEPT as a bare ``__label__x `` line so the document set never changes
(``fasttext_useful_classifier.run_eval`` keeps such rows and scores them at the class prior;
fastText itself skips zero-token training examples).

Extractor is the XenonMolecule fork's **Rust** engine (``resiliparse._extract_rs``), NOT marin's
``resiliparse`` core dep — see ``score_resiliparse_rs.py`` and
``.agents/projects/`` notes. The prebuilt Linux artifact is downloaded at runtime; no toolchain
is needed on the worker.

Why read the prep shards rather than the labeled parquet:
  * the prep shards ARE the HTML the incumbent classifiers consume, so extracting from them
    isolates exactly one variable (markup present vs removed) — same docs, same case folding,
    same whitespace policy, same label, same order;
  * 295 GB of gz reads instead of ~780 GB of parquet (the "read the right artifact" lesson);
  * no join, no snapshot reads, no dataset-root dependency.
Consequence to keep in mind: ``body_strip`` already lowercased and whitespace-collapsed the HTML
and dropped ``<head>``, so ``<pre>``/``<title>`` information is gone before extraction. That is
the same information the HTML models never saw, which is the point.

Run (us-central2, 8 disjoint strided slices; see launch_fasttext_text_track.sh)::

    python experiments/baseline_collection/extract_prep_text_rs.py \\
      --prep-root gs://marin-us-central2/classifiers/useful_fasttext_lpv11/full_prep_body_strip \\
      --out-root  gs://marin-us-central2/classifiers/useful_fasttext_lpv11/full_prep_resiliparse_rs \\
      --train-selection .../\\_dryrun_selected_640_stratified.txt \\
      --slice-index 0 --num-slices 8 --workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import fsspec

from experiments.fsspec_paths import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_USEFUL = "__label__useful"
_WS_RE = re.compile(r"\s+")

# Prebuilt Rust extractor. Mirror it into the run's own region first (16 MB) so no job reads
# cross-region; both prefixes hold byte-identical artifacts.
DEFAULT_ARTIFACT_PREFIX = "gs://marin-us-central2/artifacts/resiliparse_rs/latest"
# NOT /tmp: iris CPU workers mount it noexec, so dlopen() of the extension fails there.
INSTALL_DIR = "/root/resiliparse_rs"
# lexbor is resolved through an $ORIGIN rpath, so it must sit beside the extension; import order
# matters only in that the extension is loaded last.
ARTIFACT_LIBS = ("liblexbor.so.2", "_extract_rs.so")

# CONFIRMED upstream bug: the Rust extractor SEGFAULTS on <frameset> documents (0.108% of a 100k
# CC sample; 100% of the crashers contained <frameset>, no non-frameset crasher). A segfault kills
# the worker and poisons the pool, and at this corpus size that would be ~37k crashes. Prep lines
# are already lowercased by body_strip, so a plain substring test is sufficient. These pages are
# redirect/parking framesets whose main content is genuinely empty, so "" is also the right answer.
FRAMESET_MARKER = "<frameset"

_EXTRACT = None  # resolved lazily inside each worker process


def install_extractor(artifact_prefix: str, dest: str = INSTALL_DIR) -> str:
    """Download the prebuilt Rust extractor and prepend its package dir to ``sys.path``."""
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
    logger.info("installed Rust extractor from %s -> %s", artifact_prefix, pkg_dir)
    return pkg_dir


def _extractor(artifact_prefix: str):
    """The fork's ``extract_plain_text``, installed on first use in this process."""
    global _EXTRACT
    if _EXTRACT is None:
        if not os.path.isdir(os.path.join(INSTALL_DIR, "resiliparse-py")):
            install_extractor(artifact_prefix)
        else:
            sys.path.insert(0, os.path.join(INSTALL_DIR, "resiliparse-py"))
        from resiliparse._extract_rs import extract_plain_text

        _EXTRACT = extract_plain_text
    return _EXTRACT


def extract_clf_text(html: str, extract, preserve_formatting: str) -> tuple[str, bool, bool]:
    """body_strip HTML -> classifier-input text. Returns (text, was_frameset_skipped, panicked).

    THE canonical TEXT-classifier input: whatever calls this sees byte-identical text to the fastText
    prep shards (and, modulo the neural corpora's ``__empty__`` placeholder for empty docs, to the
    ModernBERT/pooled TEXT training data). Normalization matches
    ``fasttext_useful_classifier.to_fasttext_text``: collapse all whitespace to single spaces, strip,
    lowercase. Empty, frameset-screened and panicking documents all yield "".

    The Rust extractor can PANIC on a single document (observed: "end byte index 120 is not a char
    boundary; it is inside '☆'" — the fork slices a String at a byte offset that lands mid-codepoint).
    pyo3 raises that as ``PanicException``, which inherits from BaseException, so a plain
    ``except Exception`` does NOT catch it; it then fails to pickle back to the parent and kills the
    whole slice. One bad document must cost one document, so catch it here, in the child.
    """
    if not html:
        return "", False, False
    if FRAMESET_MARKER in html:
        return "", True, False
    try:
        text = extract(html, main_content=True, preserve_formatting=preserve_formatting)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        return "", False, True
    return _WS_RE.sub(" ", text or "").strip().lower(), False, False


def html_to_text_line(label: str, html: str, extract, preserve_formatting: str) -> tuple[str, int, bool, bool]:
    """One prep line -> one text line. Returns (line, n_text_chars, was_frameset_skipped, panicked).

    fastText is line-oriented, so the single-line normalization in ``extract_clf_text`` is mandatory.
    """
    text, was_frameset, panicked = extract_clf_text(html, extract, preserve_formatting)
    return f"{label} {text}", len(text), was_frameset, panicked


def extract_one_shard(spec: dict) -> dict:
    """Extract one prep shard to its text twin. Atomic tmp->rename + skip-existing (resumable)."""
    src = f"{spec['prep_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz"
    out = f"{spec['out_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz"
    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath):
        return {"index": spec["index"], "split": spec["split"], "skipped": True}

    extract = _extractor(spec["artifact_prefix"])
    pf = spec["preserve_formatting"]
    tmp = f"{out}.tmp"
    n_lines = n_pos = n_frameset = n_empty = n_panic = 0
    chars_in = chars_out = 0
    t0 = time.monotonic()
    with (
        fsspec.open(src, "rt", compression="gzip", encoding="utf-8") as fin,
        fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as fout,
    ):
        for raw in fin:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            label, _, html = raw.partition(" ")
            line, n_chars, was_frameset, panicked = html_to_text_line(label, html, extract, pf)
            fout.write(line + "\n")
            n_lines += 1
            n_pos += label == LABEL_USEFUL
            n_frameset += was_frameset
            n_panic += panicked
            n_empty += n_chars == 0
            chars_in += len(html)
            chars_out += n_chars
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    return {
        "index": spec["index"],
        "split": spec["split"],
        "skipped": False,
        "n_lines": n_lines,
        "n_useful": n_pos,
        "n_frameset_skipped": n_frameset,
        "n_panic": n_panic,
        "n_empty_text": n_empty,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "seconds": round(time.monotonic() - t0, 1),
    }


def extract_single_file(
    in_file: str, out_file: str, artifact_prefix: str, preserve_formatting: str, overwrite: bool
) -> dict:
    """Line-for-line text twin of one fastText file (used for the frozen 7k sample).

    Every input line yields exactly one output line at the same position — including lines whose
    extraction is empty — so the two files can be zipped positionally and the frozen document set
    is provably unchanged.
    """
    fs, rpath = fsspec.core.url_to_fs(out_file)
    if fs.exists(rpath) and not overwrite:
        logger.info("exists, skipping: %s", out_file)
        return {"skipped": True}
    extract = _extractor(artifact_prefix)
    tmp = f"{out_file}.tmp"
    n_lines = n_blank = n_frameset = n_empty = n_panic = 0
    chars_in = chars_out = 0
    with (
        fsspec.open(in_file, "rt", compression="gzip", encoding="utf-8") as fin,
        fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as fout,
    ):
        for raw in fin:
            raw = raw.rstrip("\n")
            n_lines += 1
            if not raw:
                # Preserve blank lines verbatim so absolute line numbers stay aligned.
                fout.write("\n")
                n_blank += 1
                continue
            label, _, html = raw.partition(" ")
            line, n_chars, was_frameset, panicked = html_to_text_line(label, html, extract, preserve_formatting)
            fout.write(line + "\n")
            n_frameset += was_frameset
            n_panic += panicked
            n_empty += n_chars == 0
            chars_in += len(html)
            chars_out += n_chars
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    stats = {
        "in_file": in_file,
        "out_file": out_file,
        "n_lines": n_lines,
        "n_blank": n_blank,
        "n_frameset_skipped": n_frameset,
        "n_panic": n_panic,
        "n_empty_text": n_empty,
        "chars_in": chars_in,
        "chars_out": chars_out,
        "text_char_fraction": round(chars_out / max(chars_in, 1), 4),
    }
    logger.info("single-file done: %s", json.dumps(stats))
    return stats


def _read_selection(path: str) -> list[int]:
    with fsspec.open(path, "r") as f:
        return sorted(int(x) for x in f.read().strip().split(",") if x.strip())


def _shard_indices_under(prep_root: str, split: str) -> list[int]:
    paths = fsspec_glob(f"{prep_root}/{split}/data-*.txt.gz")
    return sorted(int(p.rsplit("data-", 1)[1][:5]) for p in paths)


def build_worklist(prep_root: str, out_root: str, train_selection: str, splits: list[str], **common) -> list[dict]:
    specs: list[dict] = []
    for split in splits:
        idxs = _read_selection(train_selection) if split == "train" else _shard_indices_under(prep_root, split)
        specs += [{"index": i, "split": split, "prep_root": prep_root, "out_root": out_root, **common} for i in idxs]
    return specs


def _extract_chunk(payload: tuple[list[tuple[str, str]], str, str]) -> list[tuple[str, int, bool, bool]]:
    """Convert a chunk of (label, html) pairs in a worker. A segfault here kills only this chunk."""
    pairs, artifact_prefix, preserve_formatting = payload
    extract = _extractor(artifact_prefix)
    return [html_to_text_line(label, html, extract, preserve_formatting) for label, html in pairs]


def rescue_one_shard(spec: dict, chunk_size: int = 200) -> dict:
    """Extract a shard the pool segfaults on, charging the crash to ONE document instead of the shard.

    ``extract_one_shard`` hands a whole shard to one worker, so a single unextractable page takes the
    shard with it and it ends up quarantined (6 shards did on the fastText prep run). Here the shard
    is processed in chunks; a chunk that kills its worker is re-run one document per process, and the
    document that crashes gets the same empty-text line the ``<frameset>`` guard emits. Same
    normalization as the bulk path, so a rescued shard is byte-comparable with the rest.
    """
    src = f"{spec['prep_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz"
    out = f"{spec['out_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz"
    fs, rpath = fsspec.core.url_to_fs(out)
    if fs.exists(rpath) and not spec.get("overwrite"):
        return {"index": spec["index"], "split": spec["split"], "skipped": True}

    prefix, pf = spec["artifact_prefix"], spec["preserve_formatting"]
    pairs: list[tuple[str, str]] = []
    with fsspec.open(src, "rt", compression="gzip", encoding="utf-8") as fin:
        for raw in fin:
            raw = raw.rstrip("\n")
            if raw:
                label, _, html = raw.partition(" ")
                pairs.append((label, html))

    results: list[tuple[str, int, bool, bool]] = []
    n_segfault = 0
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        pool = ProcessPoolExecutor(max_workers=1)
        try:
            results.extend(pool.map(_extract_chunk, [(chunk, prefix, pf)]).__next__())
        except BrokenProcessPool:
            logger.warning("chunk at %d segfaulted — isolating %d docs", start, len(chunk))
            for offset, pair in enumerate(chunk):
                solo = ProcessPoolExecutor(max_workers=1)
                try:
                    results.extend(solo.map(_extract_chunk, [([pair], prefix, pf)]).__next__())
                except BrokenProcessPool:
                    n_segfault += 1
                    logger.error("UNEXTRACTABLE doc %d (%d chars)", start + offset, len(pair[1]))
                    results.append((f"{pair[0]} ", 0, False, False))
                finally:
                    solo.shutdown(wait=False)
        finally:
            pool.shutdown(wait=False)

    tmp = f"{out}.tmp"
    with fsspec.open(tmp, "wt", compression="gzip", encoding="utf-8") as fout:
        for line, _, _, _ in results:
            fout.write(line + "\n")
    tfs, trpath = fsspec.core.url_to_fs(tmp)
    tfs.mv(trpath, rpath)
    stats = {
        "index": spec["index"],
        "split": spec["split"],
        "n_lines": len(results),
        "n_segfault": n_segfault,
        "n_panic": sum(1 for r in results if r[3]),
        "n_frameset_skipped": sum(1 for r in results if r[2]),
        "chars_out": sum(r[1] for r in results),
    }
    logger.info("RESCUED %s", json.dumps(stats))
    return stats


def _run_isolated(pending: list[dict], quarantined: list[dict]) -> list[dict]:
    """Process shards one per pool, so a segfault is attributable to exactly one shard.

    A worker segfault kills the process, so the child cannot catch it and a shared pool only reports
    "some worker died" — retrying the whole batch then hits the same poison shard forever (slice 2
    burned its 5 restarts in 20 seconds). One shard per pool turns an unattributable crash into a
    named one, which can be skipped and reported instead of killing the slice.
    """
    done: list[dict] = []
    for spec in pending:
        pool = ProcessPoolExecutor(max_workers=1)
        try:
            done.extend(pool.map(extract_one_shard, [spec]))
        except BrokenProcessPool:
            logger.error(
                "QUARANTINE %s shard %s — the Rust extractor segfaults on it; continuing",
                spec["split"],
                spec["index"],
            )
            quarantined.append({"split": spec["split"], "index": spec["index"]})
        finally:
            pool.shutdown(wait=False)
    return done


def run(args) -> None:
    # Install once in the parent: with fork-based pools the children inherit sys.path and the
    # unpacked tree, so 16 workers never race on the same /root/resiliparse_rs directory.
    install_extractor(args.artifact_prefix)
    if args.in_file:
        extract_single_file(args.in_file, args.out_file, args.artifact_prefix, args.preserve_formatting, args.overwrite)
        return

    if args.rescue_shards:
        # "train:1266,test:387" — the shards a bulk slice quarantined, redone chunk-by-chunk.
        for item in args.rescue_shards.split(","):
            split, _, index = item.strip().partition(":")
            rescue_one_shard(
                {
                    "split": split,
                    "index": int(index),
                    "prep_root": args.prep_root,
                    "out_root": args.out_root,
                    "artifact_prefix": args.artifact_prefix,
                    "preserve_formatting": args.preserve_formatting,
                    "overwrite": args.overwrite,
                },
                chunk_size=args.rescue_chunk_size,
            )
        return

    specs = build_worklist(
        args.prep_root,
        args.out_root,
        args.train_selection,
        [s.strip() for s in args.splits.split(",") if s.strip()],
        artifact_prefix=args.artifact_prefix,
        preserve_formatting=args.preserve_formatting,
    )
    # Strided, not contiguous: slices stay disjoint (the preshard lesson) while big and small
    # shards spread evenly across jobs so no slice becomes the straggler.
    mine = specs[args.slice_index :: args.num_slices]
    logger.info(
        "slice %d/%d: %d of %d shards -> %s", args.slice_index, args.num_slices, len(mine), len(specs), args.out_root
    )

    done: list[dict] = []
    quarantined: list[dict] = []
    pending = list(mine)
    restarts = 0
    while pending:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        try:
            for res in pool.map(extract_one_shard, pending):
                done.append(res)
                if len(done) % 10 == 0:
                    logger.info("%d/%d shards", len(done), len(mine))
            pending = []
        except BrokenProcessPool:
            # A worker segfaulted despite the <frameset> guard. Every finished shard is already
            # durable on GCS, so rebuilding the pool and resubmitting resumes via skip-existing.
            restarts += 1
            logger.warning(
                "BrokenProcessPool (restart %d/%d) — resubmitting unfinished shards", restarts, args.max_pool_restarts
            )
            pending = [s for s in mine if not _output_exists(s)]
            if restarts > args.isolate_after:
                # Repeat crashes mean a poison shard, not bad luck: finish one-per-pool so the
                # offender is named and skipped rather than killing the whole slice.
                logger.warning("isolating %d remaining shards after %d restarts", len(pending), restarts)
                done += _run_isolated(pending, quarantined)
                pending = []
        finally:
            pool.shutdown(wait=False)
    if quarantined:
        logger.error("QUARANTINED %d shard(s): %s", len(quarantined), quarantined)

    chars_in = sum(r.get("chars_in", 0) for r in done)
    chars_out = sum(r.get("chars_out", 0) for r in done)
    summary = {
        "slice_index": args.slice_index,
        "num_slices": args.num_slices,
        "n_shards": len(done),
        "n_skipped_existing": sum(1 for r in done if r.get("skipped")),
        "n_lines": sum(r.get("n_lines", 0) for r in done),
        "n_useful": sum(r.get("n_useful", 0) for r in done),
        "n_frameset_skipped": sum(r.get("n_frameset_skipped", 0) for r in done),
        "n_panic": sum(r.get("n_panic", 0) for r in done),
        "n_empty_text": sum(r.get("n_empty_text", 0) for r in done),
        "chars_in": chars_in,
        "chars_out": chars_out,
        "text_char_fraction": round(chars_out / max(chars_in, 1), 4),
        "pool_restarts": restarts,
        "quarantined": quarantined,
        "preserve_formatting": args.preserve_formatting,
    }
    with fsspec.open(
        f"{args.out_root}/_extract_stats_slice{args.slice_index:02d}of{args.num_slices:02d}.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)
    logger.info("DONE %s", json.dumps(summary))


def _output_exists(spec: dict) -> bool:
    fs, rpath = fsspec.core.url_to_fs(f"{spec['out_root']}/{spec['split']}/data-{spec['index']:05d}.txt.gz")
    return fs.exists(rpath)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prep-root", default="", help="Source full_prep_body_strip root (gs://).")
    p.add_argument("--out-root", default="", help="Destination text prep root (gs://).")
    p.add_argument("--train-selection", default="", help="GCS path to the _dryrun_selected_*.txt CSV of train indices.")
    p.add_argument("--splits", default="train,test", help="Comma-separated splits to extract.")
    p.add_argument("--slice-index", type=int, default=0)
    p.add_argument("--num-slices", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-pool-restarts", type=int, default=5)
    p.add_argument(
        "--isolate-after",
        type=int,
        default=2,
        help="After this many pool restarts, finish one-shard-per-pool so a poison shard is named and skipped.",
    )
    p.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    p.add_argument(
        "--preserve-formatting",
        default="markdown",
        help="Rust extractor formatting mode. MUST match whatever the neural text track uses.",
    )
    p.add_argument("--in-file", default="", help="Single-file mode: source fastText .txt.gz (frozen 7k).")
    p.add_argument("--out-file", default="", help="Single-file mode: destination .txt.gz.")
    p.add_argument("--overwrite", action="store_true", help="Single-file mode: redo even if output exists.")
    p.add_argument(
        "--rescue-shards",
        default="",
        help='Comma-separated "split:index" shards to redo chunk-by-chunk (for shards a slice quarantined).',
    )
    p.add_argument("--rescue-chunk-size", type=int, default=200)
    args = p.parse_args()
    if args.in_file and not args.out_file:
        raise SystemExit("--in-file requires --out-file")
    if args.rescue_shards and not (args.prep_root and args.out_root):
        raise SystemExit("--rescue-shards requires --prep-root and --out-root")
    # Rescue names its shards explicitly, so it needs no selection file.
    if not args.in_file and not args.rescue_shards and not (args.prep_root and args.out_root and args.train_selection):
        raise SystemExit("shard mode requires --prep-root, --out-root and --train-selection")
    run(args)


if __name__ == "__main__":
    main()
