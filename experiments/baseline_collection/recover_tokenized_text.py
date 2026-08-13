# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover source text (and hence urls) from a Levanter token cache.

The 10,364-WARC resiliparse corpus was deduped, decontaminated, tokenized, and
then its document tree was deleted; only ``tokenized/resiliparse_decon_10364warcs``
survives. The quality x domain grid needs ``{text, url}`` at that post-decon layer,
and url lives only in the raw extraction.

Both sides can be rejoined without re-running dedup, because
:func:`marin.datakit.normalize.generate_id` is a pure xxh3_128 content hash and
``normalize_record`` never modifies text: decode a cache row, hash it, and look
the hash up in a ``(content_id -> url)`` map built from the raw extraction.

Row framing
-----------
``BatchTokenizer`` writes each document as::

    <|begin_of_text|>  ...document tokens...  Ġ  <|end_of_text|>

The ``Ġ`` (lone space) before EOS is appended unconditionally. It is dropped at
the TOKEN level rather than by rstrip-ing the decoded string, so a document that
legitimately ends in whitespace keeps its own trailing space. ``decode`` must run
with ``clean_up_tokenization_spaces=False``; the default rewrites " ." -> "." and
silently breaks the hash for ~20% of documents.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Iterator

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from levanter.store.tree_store import TreeStore
from marin.datakit.normalize import generate_id
from marin.utils import fsspec_glob
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def read_jsonl_gz(path: str) -> Iterator[dict]:
    """Stream a gzipped jsonl shard from a local path or object store."""
    with fsspec.open(path, "rb", compression="gzip") as fh:
        for line in fh:
            yield json.loads(line)


TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
SPACE_TOKEN = "Ġ"
READ_BATCH = 4000


class RowFraming:
    """Token ids for the framing BatchTokenizer puts around every document."""

    def __init__(self, tok) -> None:
        self.bos = tok.convert_tokens_to_ids(BOS_TOKEN)
        self.eos = tok.convert_tokens_to_ids(EOS_TOKEN)
        self.space = tok.convert_tokens_to_ids(SPACE_TOKEN)
        for name, value in (("bos", self.bos), ("eos", self.eos), ("space", self.space)):
            if value is None:
                raise ValueError(f"tokenizer {TOKENIZER} has no id for {name} token")


def strip_framing(ids: list[int], framing: RowFraming) -> tuple[list[int], str | None]:
    """Return the document's own tokens, plus an anomaly label if framing is off.

    Returns the tokens unchanged (minus whatever framing was present) rather than
    raising, so a caller can count anomalies across a whole corpus instead of
    dying on the first one.
    """
    anomaly = None
    core = ids
    if core and core[0] == framing.bos:
        core = core[1:]
    else:
        anomaly = "missing-bos"
    if core and core[-1] == framing.eos:
        core = core[:-1]
    else:
        anomaly = anomaly or "missing-eos"
    if core and core[-1] == framing.space:
        core = core[:-1]
    else:
        anomaly = anomaly or "missing-trailing-space"
    return core, anomaly


def decode_rows(rows, tok, framing: RowFraming) -> tuple[list[str], dict[str, int]]:
    texts: list[str] = []
    anomalies: dict[str, int] = {}
    for row in rows:
        ids = np.asarray(row, dtype=np.int64).tolist()
        core, anomaly = strip_framing(ids, framing)
        if anomaly:
            anomalies[anomaly] = anomalies.get(anomaly, 0) + 1
        texts.append(tok.decode(core, skip_special_tokens=False, clean_up_tokenization_spaces=False))
    return texts, anomalies


def validate_against_shard(cache_path: str, shard_path: str, limit: int | None) -> int:
    """Positionally compare a cache part against the document shard that built it.

    Only valid where the cache ledger shows one part per input shard AND the part's
    row count equals the shard's document count; the caller is responsible for
    picking such a pair.
    """
    src = [r["text"] for r in read_jsonl_gz(shard_path) if r.get("text")]
    logger.info("source shard: %d docs", len(src))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    framing = RowFraming(tok)
    store = TreeStore.open({"input_ids": 0}, cache_path, mode="r")
    jagged = store.tree["input_ids"]

    n = min(len(src), limit or len(src))
    exact = 0
    id_match = 0
    all_anomalies: dict[str, int] = {}
    mismatch_example = None

    for start in range(0, n, READ_BATCH):
        stop = min(start + READ_BATCH, n)
        texts, anomalies = decode_rows(jagged.get_batch_sync(list(range(start, stop))), tok, framing)
        for key, count in anomalies.items():
            all_anomalies[key] = all_anomalies.get(key, 0) + count
        for k, text in enumerate(texts):
            source = src[start + k]
            exact += text == source
            id_match += generate_id(text) == generate_id(source)
            if text != source and mismatch_example is None:
                mismatch_example = (start + k, source, text)

    print("=" * 70)
    print(f"rows compared positionally : {n:,}")
    print(f"decoded text == source     : {exact:,}/{n:,}  ({100 * exact / n:.4f}%)")
    print(f"content id == source id    : {id_match:,}/{n:,}  ({100 * id_match / n:.4f}%)")
    print(f"framing anomalies          : {all_anomalies or 'none'}")
    print("=" * 70)
    if mismatch_example:
        i, source, text = mismatch_example
        pos = next((j for j in range(min(len(source), len(text))) if source[j] != text[j]), min(len(source), len(text)))
        print(f"\nfirst mismatch row {i} (src {len(source)} chars, dec {len(text)} chars, diverge@{pos})")
        print(f"  src: {source[max(0, pos - 40) : pos + 40]!r}")
        print(f"  dec: {text[max(0, pos - 40) : pos + 40]!r}")
    return 0 if exact == n else 1


def probe_lineage(cache_path: str, raw_glob: str, n_shards: int, n_rows: int, seed: int) -> int:
    """Test whether a raw extraction is the ancestor of a token cache.

    Builds the content-id set for a random subset of raw shards, then samples cache
    rows uniformly (NOT contiguously — the cache is written one part per input
    shard, so a contiguous run is a sample of size one) and measures how many decode
    to a document in that set. Compares the observed rate against what shared
    lineage predicts.
    """
    shards = sorted(fsspec_glob(raw_glob))
    logger.info("raw extraction: %d shards", len(shards))
    rng = random.Random(seed)
    picked = rng.sample(shards, min(n_shards, len(shards)))

    raw_ids: set[str] = set()
    raw_docs = 0
    for i, path in enumerate(picked, 1):
        for record in read_jsonl_gz(path):
            text = record.get("text")
            if text:
                raw_ids.add(generate_id(text))
                raw_docs += 1
        if i % 25 == 0 or i == len(picked):
            logger.info("  hashed %d/%d shards, %d docs, %d unique ids", i, len(picked), raw_docs, len(raw_ids))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    framing = RowFraming(tok)
    store = TreeStore.open({"input_ids": 0}, cache_path, mode="r")
    jagged = store.tree["input_ids"]
    total_rows = len(store)

    idxs = sorted(rng.sample(range(total_rows), min(n_rows, total_rows)))
    shard_fraction = len(picked) / len(shards)
    survival = total_rows / (raw_docs / shard_fraction)
    expected = shard_fraction * survival * len(idxs)

    logger.info(
        "cache %d rows; sampling %d scattered; raw subset covers %.3f%% of shards; "
        "implied survival %.1f%%; expected hits if same lineage ~%.0f",
        total_rows,
        len(idxs),
        100 * shard_fraction,
        100 * survival,
        expected,
    )

    hits = 0
    checked = 0
    all_anomalies: dict[str, int] = {}
    for start in range(0, len(idxs), READ_BATCH):
        batch = idxs[start : start + READ_BATCH]
        texts, anomalies = decode_rows(jagged.get_batch_sync(batch), tok, framing)
        for key, count in anomalies.items():
            all_anomalies[key] = all_anomalies.get(key, 0) + count
        for text in texts:
            checked += 1
            hits += generate_id(text) in raw_ids
        logger.info("  scanned %d/%d rows, hits %d", checked, len(idxs), hits)

    rate = hits / checked if checked else 0.0
    print("=" * 70)
    print(f"raw subset          : {len(picked)}/{len(shards)} shards, {raw_docs:,} docs, {len(raw_ids):,} unique ids")
    print(f"cache rows sampled  : {checked:,} of {total_rows:,} (scattered)")
    print(f"hits                : {hits:,}   observed rate {100 * rate:.3f}%")
    print(f"expected if same    : ~{expected:,.0f}   ({100 * expected / max(checked, 1):.3f}%)")
    print(f"ratio observed/exp  : {hits / expected if expected else float('nan'):.3f}")
    print(f"framing anomalies   : {all_anomalies or 'none'}")
    print("=" * 70)
    verdict = "SAME LINEAGE" if expected and hits / expected > 0.5 else "NOT THE ANCESTOR"
    print(f"VERDICT: {verdict}")
    return 0 if expected and hits / expected > 0.5 else 1


def cache_parts(cache_path: str) -> list[tuple[str, int, int]]:
    """``(part_name, global_start_row, num_rows)`` for every part, in ledger order.

    Row indices into a ``TreeStore`` are global, so a part is addressed by the
    running total of the parts before it.
    """
    with fsspec.open(f"{cache_path.rstrip('/')}/shard_ledger.json", "r") as fh:
        ledger = json.load(fh)
    parts = []
    start = 0
    for name in ledger["finished_shards"]:
        rows = ledger["shard_rows"][name]
        parts.append((name, start, rows))
        start += rows
    if start != ledger["total_num_rows"]:
        raise ValueError(f"ledger parts sum to {start}, expected {ledger['total_num_rows']}")
    return parts


def extract_survivor_ids(cache_path: str, output_dir: str, num_chunks: int, chunk_idx: int, greedy: bool = False) -> int:
    """Decode every row of this chunk's parts and record its content id.

    One parquet per part, written only after the part is fully decoded, so a
    preempted job resumes by skipping parts whose output already exists — the
    same done-marker discipline the grid stages use.

    ``greedy`` drops the fixed chunk assignment and lets the job work through
    every outstanding part, in an order unique to ``chunk_idx``. Submitting jobs
    to this cluster is far more expensive than running them, so when only a
    handful of a wave lands, the ones that do need to be able to drain the whole
    backlog rather than stopping at their slice. Two greedy jobs can pick the same
    part and both write it; the write is idempotent, so that costs duplicated work
    and nothing else, and the shuffled orders keep collisions rare.
    """
    parts = cache_parts(cache_path)
    if greedy:
        mine = list(parts)
        random.Random(chunk_idx).shuffle(mine)
        logger.info("greedy worker %d: eligible for all %d parts", chunk_idx, len(mine))
    else:
        mine = [p for i, p in enumerate(parts) if i % num_chunks == chunk_idx]
        logger.info("chunk %d/%d owns %d of %d parts", chunk_idx, num_chunks, len(mine), len(parts))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    framing = RowFraming(tok)
    store = TreeStore.open({"input_ids": 0}, cache_path, mode="r")
    jagged = store.tree["input_ids"]
    fs, _ = fsspec.core.url_to_fs(output_dir)

    done = 0
    written_rows = 0
    for name, start, rows in mine:
        out_path = f"{output_dir.rstrip('/')}/{name}.parquet"
        if fs.exists(out_path):
            logger.info("%s already done, skipping", name)
            done += 1
            continue

        ids: list[str] = []
        anomalies: dict[str, int] = {}
        for offset in range(0, rows, READ_BATCH):
            batch = list(range(start + offset, start + min(offset + READ_BATCH, rows)))
            texts, batch_anomalies = decode_rows(jagged.get_batch_sync(batch), tok, framing)
            for key, count in batch_anomalies.items():
                anomalies[key] = anomalies.get(key, 0) + count
            ids.extend(generate_id(t) for t in texts)

        if len(ids) != rows:
            raise ValueError(f"{name}: decoded {len(ids)} rows, ledger says {rows}")
        if anomalies:
            raise ValueError(f"{name}: unexpected row framing {anomalies}")

        table = pa.table({"row": pa.array(range(rows), type=pa.uint32()), "id": pa.array(ids, type=pa.string())})
        with fsspec.open(out_path, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        done += 1
        written_rows += rows
        logger.info("%s: %d rows -> %s  (%d/%d parts)", name, rows, out_path, done, len(mine))

    logger.info("chunk %d complete: %d parts, %d newly written rows", chunk_idx, done, written_rows)
    return 0


def id_bucket(content_id: str, num_buckets: int) -> int:
    """Bucket an id by its leading hex digits so both sides of the join agree."""
    return int(content_id[:8], 16) % num_buckets


def hash_raw(raw_glob: str, output_dir: str, num_chunks: int, chunk_idx: int, num_buckets: int) -> int:
    """Record ``(content_id, shard, line)`` for every raw document, bucketed by id.

    Emits every raw document, not just survivors: membership is settled later
    against the exact 128-bit ids, so nothing here needs the survivor set in
    memory. Text is not carried, which keeps the shuffle at tens of GB instead of
    hundreds.
    """
    shards = sorted(fsspec_glob(raw_glob))
    mine = [(i, p) for i, p in enumerate(shards) if i % num_chunks == chunk_idx]
    logger.info("chunk %d/%d owns %d of %d raw shards", chunk_idx, num_chunks, len(mine), len(shards))

    fs, _ = fsspec.core.url_to_fs(output_dir)
    marker = f"{output_dir.rstrip('/')}/_done/chunk-{chunk_idx:05d}-of-{num_chunks:05d}.json"
    if fs.exists(marker):
        logger.info("chunk %d already complete", chunk_idx)
        return 0

    # Everything this chunk owns is held in memory until the end, so the chunk
    # count sets peak RSS: ~100 bytes/doc, i.e. ~250 MB per chunk at 200 chunks
    # over a 550M-doc corpus. Do not run this with a handful of large chunks.
    buckets: list[dict[str, list]] = [{"id": [], "shard": [], "line": []} for _ in range(num_buckets)]
    total = 0
    for done_shards, (shard_idx, path) in enumerate(mine, 1):
        for line_no, record in enumerate(read_jsonl_gz(path)):
            text = record.get("text")
            if not text:
                continue
            content_id = generate_id(text)
            b = buckets[id_bucket(content_id, num_buckets)]
            b["id"].append(content_id)
            b["shard"].append(shard_idx)
            b["line"].append(line_no)
            total += 1
        if done_shards % 10 == 0 or done_shards == len(mine):
            logger.info("  %d/%d shards, %d docs", done_shards, len(mine), total)

    for b_idx, b in enumerate(buckets):
        table = pa.table(
            {
                "id": pa.array(b["id"], type=pa.string()),
                "shard": pa.array(b["shard"], type=pa.uint32()),
                "line": pa.array(b["line"], type=pa.uint32()),
            }
        )
        out = f"{output_dir.rstrip('/')}/bucket={b_idx:04d}/chunk-{chunk_idx:05d}.parquet"
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")

    with fsspec.open(marker, "w") as fh:
        json.dump({"chunk": chunk_idx, "shards": len(mine), "docs": total}, fh)
    logger.info("chunk %d complete: %d shards, %d docs", chunk_idx, len(mine), total)
    return 0


def verify_tree(
    documents_dir: str, keeplist_dir: str, num_shards: int, expect_total: int, report_path: str | None
) -> int:
    """Reconcile the rebuilt tree against the token cache it was derived from.

    Deliberately avoids re-reading the ~410 GiB of documents: ``materialize``
    already asserts per shard that it wrote exactly its keep-list length, so the
    matched half is the sum of the keep-lists. Only the unmatched shards (~0.3% of
    the corpus) are counted directly. The two must add up to the cache's row count.
    """
    fs, _ = fsspec.core.url_to_fs(documents_dir)
    data_shards = sorted(fsspec_glob(f"{documents_dir.rstrip('/')}/data-*.jsonl.gz"))
    unmatched_shards = sorted(fsspec_glob(f"{documents_dir.rstrip('/')}/unmatched-*.jsonl.gz"))
    logger.info("tree: %d data shards, %d unmatched shards", len(data_shards), len(unmatched_shards))

    matched_docs = 0
    missing_keeplists = []
    for i in range(num_shards):
        path = f"{keeplist_dir.rstrip('/')}/shard-{i:05d}.parquet"
        if not fs.exists(path):
            missing_keeplists.append(i)
            continue
        with fsspec.open(path, "rb") as fh:
            matched_docs += pq.read_metadata(fh).num_rows
        if (i + 1) % 2000 == 0:
            logger.info("  keeplists %d/%d, %d docs", i + 1, num_shards, matched_docs)

    unmatched_docs = 0
    for path in unmatched_shards:
        unmatched_docs += sum(1 for _ in read_jsonl_gz(path))

    total = matched_docs + unmatched_docs
    report = {
        "data_shards": len(data_shards),
        "data_shards_expected": num_shards,
        "unmatched_shards": len(unmatched_shards),
        "matched_docs": matched_docs,
        "unmatched_docs": unmatched_docs,
        "total_docs": total,
        "expected_total": expect_total,
        "reconciles": total == expect_total,
        "missing_keeplists": missing_keeplists[:20],
    }
    if report_path:
        with fsspec.open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

    if len(data_shards) != num_shards:
        raise ValueError(f"expected {num_shards} data shards, found {len(data_shards)}")
    if total != expect_total:
        raise ValueError(f"tree totals {total} docs, expected {expect_total}")
    return 0


def materialize_unmatched(
    survivor_dir: str,
    representatives_dir: str,
    survivor_ids_dir: str,
    cache_path: str,
    output_dir: str,
    num_buckets: int,
    chunk_idxs: list[int],
    num_chunks: int,
    unmatched_ids_path: str | None = None,
) -> int:
    """Write the survivors that no raw document matched, using their decoded text.

    These are documents whose stored tokens carry chunk-boundary corruption, so the
    original bytes are unrecoverable and no url can be attributed to them. Their
    decoded text is nonetheless exactly what the model read, so emitting it keeps
    the rebuilt corpus identical in size and content to the one that was trained on
    — which dropping them would not. ``url`` is empty, and WebOrganizer's
    ``"{url}\\n\\n{text}"`` template degrades gracefully rather than failing.

    Reads only the specific cache rows involved (~0.3% of the corpus), located via
    the per-part survivor tables, so this costs a fraction of a full decode pass.
    """
    # Deriving the unmatched set means differencing 290M ids across 64 buckets —
    # over a GB of parquet and several minutes. Every worker would otherwise repeat
    # it, so it is computed once and cached; the result is only ~0.3% of the corpus.
    fs_cache, _ = fsspec.core.url_to_fs(unmatched_ids_path) if unmatched_ids_path else (None, None)
    if unmatched_ids_path and fs_cache.exists(unmatched_ids_path):
        with fsspec.open(unmatched_ids_path, "rb") as fh:
            unmatched = set(pq.read_table(fh, columns=["id"])["id"].to_pylist())
        logger.info("loaded %d unmatched ids from %s", len(unmatched), unmatched_ids_path)
    else:
        unmatched = set()
        for bucket_idx in range(num_buckets):
            survivors: set[str] = set()
            for path in sorted(fsspec_glob(f"{survivor_dir.rstrip('/')}/bucket={bucket_idx:04d}/*.parquet")):
                with fsspec.open(path, "rb") as fh:
                    survivors.update(pq.read_table(fh, columns=["id"])["id"].to_pylist())
            with fsspec.open(f"{representatives_dir.rstrip('/')}/bucket-{bucket_idx:04d}.parquet", "rb") as fh:
                matched = set(pq.read_table(fh, columns=["id"])["id"].to_pylist())
            unmatched |= survivors - matched
        logger.info("unmatched survivors corpus-wide: %d", len(unmatched))
        if unmatched_ids_path:
            table = pa.table({"id": pa.array(sorted(unmatched), type=pa.string())})
            with fsspec.open(unmatched_ids_path, "wb") as fh:
                pq.write_table(table, fh, compression="zstd")
            logger.info("cached unmatched ids -> %s", unmatched_ids_path)

    parts = cache_parts(cache_path)
    mine = [(i, p) for i, p in enumerate(parts) if i % num_chunks in set(chunk_idxs)]
    logger.info("chunk(s) %s own %d of %d cache parts", chunk_idxs, len(mine), len(parts))

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    framing = RowFraming(tok)
    store = TreeStore.open({"input_ids": 0}, cache_path, mode="r")
    jagged = store.tree["input_ids"]
    fs, _ = fsspec.core.url_to_fs(output_dir)

    written_total = 0
    for _, (name, start, _rows) in mine:
        out = f"{output_dir.rstrip('/')}/unmatched-{name}.jsonl.gz"
        if fs.exists(out):
            continue
        with fsspec.open(f"{survivor_ids_dir.rstrip('/')}/{name}.parquet", "rb") as fh:
            table = pq.read_table(fh)
        targets = [
            (row, cid)
            for row, cid in zip(table["row"].to_pylist(), table["id"].to_pylist(), strict=True)
            if cid in unmatched
        ]
        if not targets:
            with fsspec.open(out, "wb", compression="gzip"):
                pass
            continue

        batch = jagged.get_batch_sync([start + row for row, _ in targets])
        texts, anomalies = decode_rows(batch, tok, framing)
        if anomalies:
            raise ValueError(f"{name}: unexpected row framing {anomalies}")
        with fsspec.open(out, "wb", compression="gzip") as fh:
            for text in texts:
                fh.write(json.dumps({"text": text, "url": ""}, ensure_ascii=False).encode())
                fh.write(b"\n")
        written_total += len(texts)
        logger.info("%s: %d unmatched docs -> %s", name, len(texts), out)

    logger.info("chunk(s) %s complete: %d unmatched docs written", chunk_idxs, written_total)
    return 0


def diagnose_unmatched(
    survivor_dir: str,
    representatives_dir: str,
    survivor_ids_dir: str,
    cache_path: str,
    bucket_idx: int,
    sample: int,
    report_path: str | None = None,
) -> int:
    """Characterise survivors that found no raw document.

    Answers the only question that matters about them: are they a benign long tail
    (e.g. documents whose length or content makes the decode lossy) or a systematic
    slice of the corpus? Reports how their token lengths compare to the corpus and
    prints examples, so the cause is identified rather than assumed.
    """
    survivors: set[str] = set()
    for path in sorted(fsspec_glob(f"{survivor_dir.rstrip('/')}/bucket={bucket_idx:04d}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            survivors.update(pq.read_table(fh, columns=["id"])["id"].to_pylist())

    with fsspec.open(f"{representatives_dir.rstrip('/')}/bucket-{bucket_idx:04d}.parquet", "rb") as fh:
        matched = set(pq.read_table(fh, columns=["id"])["id"].to_pylist())

    unmatched = survivors - matched
    logger.info(
        "bucket %d: %d survivors, %d matched, %d unmatched (%.4f%%)",
        bucket_idx,
        len(survivors),
        len(matched),
        len(unmatched),
        100 * len(unmatched) / max(len(survivors), 1),
    )
    if not unmatched:
        return 0

    # Locate a sample of the unmatched ids back in the per-part survivor tables so
    # their cache rows — and therefore their token lengths — can be recovered.
    parts = cache_parts(cache_path)
    offsets = {name: start for name, start, _ in parts}
    wanted = set(list(unmatched)[:sample])
    found: list[tuple[str, int]] = []
    for path in sorted(fsspec_glob(f"{survivor_ids_dir.rstrip('/')}/*.parquet")):
        name = path.rsplit("/", 1)[-1].removesuffix(".parquet")
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh)
        for row, value in zip(table["row"].to_pylist(), table["id"].to_pylist(), strict=True):
            if value in wanted:
                found.append((value, offsets[name] + row))
                wanted.discard(value)
        if not wanted or len(found) >= sample:
            break
    logger.info("located %d unmatched ids in the cache", len(found))
    if not found:
        return 0

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    framing = RowFraming(tok)
    store = TreeStore.open({"input_ids": 0}, cache_path, mode="r")
    jagged = store.tree["input_ids"]

    rows = jagged.get_batch_sync([r for _, r in found])
    lengths = []
    anomaly_counts: dict[str, int] = {}
    examples = []
    for (_, global_row), row in zip(found, rows, strict=True):
        ids = np.asarray(row, dtype=np.int64).tolist()
        core, anomaly = strip_framing(ids, framing)
        if anomaly:
            anomaly_counts[anomaly] = anomaly_counts.get(anomaly, 0) + 1
        lengths.append(len(core))
        if len(examples) < 5:
            text = tok.decode(core, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            examples.append((global_row, len(core), text[:160].replace("\n", " ")))

    baseline = jagged.get_batch_sync(list(range(0, min(len(store), 200_000), 97)))
    base_lengths = [len(np.asarray(r, dtype=np.int64)) - 3 for r in baseline]

    lengths.sort()
    base_lengths.sort()
    report = {
        "bucket": bucket_idx,
        "survivors": len(survivors),
        "matched": len(matched),
        "unmatched": len(unmatched),
        "unmatched_pct": round(100 * len(unmatched) / max(len(survivors), 1), 4),
        "sampled": len(lengths),
        "unmatched_len": {
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
        "baseline_len": {
            "median": base_lengths[len(base_lengths) // 2],
            "p90": base_lengths[int(0.9 * (len(base_lengths) - 1))],
            "max": base_lengths[-1],
        },
        "framing_anomalies": anomaly_counts,
        "examples_with_replacement_char": sum("�" in t for _, _, t in examples),
        "examples": [{"row": r, "tokens": n, "text": t} for r, n, t in examples],
    }
    # finelog is unreliable on this cluster, so job stdout is frequently
    # unreadable after the fact. Any diagnostic worth running must land its
    # findings in object storage rather than printing them.
    if report_path:
        with fsspec.open(report_path, "w") as fh:
            json.dump(report, fh, indent=2)
        logger.info("report -> %s", report_path)
    print(json.dumps(report, indent=2))
    return 0


def bucket_survivors(
    survivor_dir: str, output_dir: str, num_chunks: int, chunk_idx: int, num_buckets: int, expect_parts: int | None
) -> int:
    """Re-partition the per-part survivor ids by id bucket, so the join reads only its slice.

    ``expect_parts`` guards the one way this stage can quietly produce a wrong
    answer: it globs whatever parts exist and then writes a done marker, so
    starting it before ``survivor-ids`` has finished would bake an incomplete id
    set in and mark it complete. Downstream would notice only as unmatched
    survivors in ``select-representatives``, far from the cause.
    """
    parts = sorted(fsspec_glob(f"{survivor_dir.rstrip('/')}/*.parquet"))
    if expect_parts is not None and len(parts) != expect_parts:
        raise ValueError(
            f"survivor-ids is not finished: found {len(parts)} part files, expected {expect_parts}. "
            "Re-run this stage once every part exists."
        )
    mine = [p for i, p in enumerate(parts) if i % num_chunks == chunk_idx]
    logger.info("chunk %d/%d owns %d of %d survivor files", chunk_idx, num_chunks, len(mine), len(parts))

    fs, _ = fsspec.core.url_to_fs(output_dir)
    marker = f"{output_dir.rstrip('/')}/_done/chunk-{chunk_idx:05d}-of-{num_chunks:05d}.json"
    if fs.exists(marker):
        logger.info("chunk %d already complete", chunk_idx)
        return 0

    buckets: list[list[str]] = [[] for _ in range(num_buckets)]
    total = 0
    for path in mine:
        with fsspec.open(path, "rb") as fh:
            for value in pq.read_table(fh, columns=["id"])["id"].to_pylist():
                buckets[id_bucket(value, num_buckets)].append(value)
                total += 1

    for b_idx, ids in enumerate(buckets):
        table = pa.table({"id": pa.array(ids, type=pa.string())})
        out = f"{output_dir.rstrip('/')}/bucket={b_idx:04d}/chunk-{chunk_idx:05d}.parquet"
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")

    with fsspec.open(marker, "w") as fh:
        json.dump({"chunk": chunk_idx, "files": len(mine), "ids": total}, fh)
    logger.info("chunk %d complete: %d ids", chunk_idx, total)
    return 0


def select_representatives(
    survivor_dir: str, raw_index_dir: str, output_dir: str, num_buckets: int, bucket_idx: int
) -> int:
    """For one id bucket, pick the single raw document that represents each survivor.

    Exact-duplicate texts share a content id, so a surviving id can occur in many
    raw shards while dedup kept exactly one copy. Which copy dedup kept is not
    recoverable (the intermediates are gone), so the lowest ``(shard, line)`` is
    chosen — deterministic, and the alternatives differ only in url since the text
    is identical by construction.
    """
    survivors: set[str] = set()
    for path in sorted(fsspec_glob(f"{survivor_dir.rstrip('/')}/bucket={bucket_idx:04d}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            survivors.update(pq.read_table(fh, columns=["id"])["id"].to_pylist())
    logger.info("bucket %d: %d survivor ids", bucket_idx, len(survivors))

    best: dict[str, tuple[int, int]] = {}
    for path in sorted(fsspec_glob(f"{raw_index_dir.rstrip('/')}/bucket={bucket_idx:04d}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh)
        for content_id, shard, line in zip(
            table["id"].to_pylist(), table["shard"].to_pylist(), table["line"].to_pylist(), strict=True
        ):
            if content_id not in survivors:
                continue
            candidate = (shard, line)
            current = best.get(content_id)
            if current is None or candidate < current:
                best[content_id] = candidate

    missing = len(survivors) - len(best)
    logger.info("bucket %d: matched %d/%d survivors (missing %d)", bucket_idx, len(best), len(survivors), missing)

    shards = pa.array([v[0] for v in best.values()], type=pa.uint32())
    lines = pa.array([v[1] for v in best.values()], type=pa.uint32())
    table = pa.table({"id": pa.array(list(best.keys()), type=pa.string()), "shard": shards, "line": lines})
    out = f"{output_dir.rstrip('/')}/bucket-{bucket_idx:04d}.parquet"
    with fsspec.open(out, "wb") as fh:
        pq.write_table(table, fh, compression="zstd")
    logger.info("bucket %d -> %s", bucket_idx, out)
    return 0 if missing == 0 else 2


def shard_keeplists(
    representatives_dir: str,
    output_dir: str,
    num_shards: int,
    expect_total: int | None,
    allow_missing: int = 0,
) -> int:
    """Invert the id-bucketed representatives into one keep-list per raw shard.

    Runs as a single job on purpose: the whole ``(shard, line)`` table is only
    ~2.3 GB as two uint32 columns, and inverting it in one place avoids writing
    ``num_buckets x num_shards`` fragments that the materialize pass would then
    have to stitch back together.
    """
    shard_parts = []
    line_parts = []
    for path in sorted(fsspec_glob(f"{representatives_dir.rstrip('/')}/*.parquet")):
        with fsspec.open(path, "rb") as fh:
            table = pq.read_table(fh, columns=["shard", "line"])
        shard_parts.append(table["shard"].to_numpy(zero_copy_only=False).astype(np.uint32))
        line_parts.append(table["line"].to_numpy(zero_copy_only=False).astype(np.uint32))
        logger.info("read %s (%d rows)", path, table.num_rows)

    shards = np.concatenate(shard_parts)
    lines = np.concatenate(line_parts)
    del shard_parts, line_parts
    total = len(shards)
    logger.info("total representatives: %d", total)
    # The one global check the whole reconstruction turns on: every row of the token
    # cache must resolve to exactly one raw document, or the rebuilt corpus silently
    # differs from the one the existing resiliparse_10k runs trained on.
    #
    # `allow_missing` exists because a real, understood shortfall was found: ~0.29%
    # of survivors are long documents whose stored tokens carry chunk-boundary
    # corruption, so they decode to text that never existed in raw and cannot be
    # hash-matched (see the OPEN ISSUE section of the project doc). Those documents
    # are recovered separately from the cache. The bound stays deliberately tight so
    # that any NEW loss still fails loudly rather than hiding behind the known one.
    if expect_total is not None:
        missing = expect_total - total
        if missing < 0 or missing > allow_missing:
            raise ValueError(
                f"representatives total {total} vs expected {expect_total} "
                f"(missing {missing}, allowed {allow_missing})"
            )
        if missing:
            logger.warning("proceeding with %d unmatched survivors (bounded by --allow-missing)", missing)

    order = np.lexsort((lines, shards))
    shards = shards[order]
    lines = lines[order]
    boundaries = np.searchsorted(shards, np.arange(num_shards + 1, dtype=np.uint32))

    written = 0
    for shard_idx in range(num_shards):
        start, stop = boundaries[shard_idx], boundaries[shard_idx + 1]
        table = pa.table({"line": pa.array(lines[start:stop], type=pa.uint32())})
        out = f"{output_dir.rstrip('/')}/shard-{shard_idx:05d}.parquet"
        with fsspec.open(out, "wb") as fh:
            pq.write_table(table, fh, compression="zstd")
        written += stop - start
        if (shard_idx + 1) % 500 == 0:
            logger.info("  %d/%d shards written, %d lines", shard_idx + 1, num_shards, written)

    if written != total:
        raise ValueError(f"wrote {written} lines, expected {total}")
    logger.info("keep-lists complete: %d shards, %d lines", num_shards, written)
    return 0


def materialize(raw_glob: str, keeplist_dir: str, output_dir: str, num_chunks: int, chunk_idx: int) -> int:
    """Write the post-decon ``{text, url}`` tree by filtering raw shards to their keep-lists.

    Text is copied from the raw extraction verbatim — no decoded text reaches the
    output. The decode was only ever used to learn *which* documents survived.
    """
    shards = sorted(fsspec_glob(raw_glob))
    mine = [(i, p) for i, p in enumerate(shards) if i % num_chunks == chunk_idx]
    logger.info("chunk %d/%d owns %d of %d raw shards", chunk_idx, num_chunks, len(mine), len(shards))

    fs, _ = fsspec.core.url_to_fs(output_dir)
    total_kept = 0
    for done_shards, (shard_idx, path) in enumerate(mine, 1):
        out = f"{output_dir.rstrip('/')}/data-{shard_idx:05d}-of-{len(shards):05d}.jsonl.gz"
        if fs.exists(out):
            continue

        with fsspec.open(f"{keeplist_dir.rstrip('/')}/shard-{shard_idx:05d}.parquet", "rb") as fh:
            keep = set(pq.read_table(fh, columns=["line"])["line"].to_pylist())

        kept = 0
        with fsspec.open(out, "wb", compression="gzip") as fh:
            for line_no, record in enumerate(read_jsonl_gz(path)):
                if not record.get("text") or line_no not in keep:
                    continue
                fh.write(json.dumps({"text": record["text"], "url": record.get("url", "")}, ensure_ascii=False).encode())
                fh.write(b"\n")
                kept += 1
        if kept != len(keep):
            raise ValueError(f"shard {shard_idx}: kept {kept}, keep-list has {len(keep)}")
        total_kept += kept
        if done_shards % 10 == 0 or done_shards == len(mine):
            logger.info("  %d/%d shards, %d docs written", done_shards, len(mine), total_kept)

    logger.info("chunk %d complete: %d docs", chunk_idx, total_kept)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    k = sub.add_parser("shard-keeplists", help="invert representatives into one keep-list per raw shard")
    k.add_argument("--representatives-dir", required=True)
    k.add_argument("--output-dir", required=True)
    k.add_argument("--num-shards", type=int, default=10364)
    k.add_argument("--expect-total", type=int, default=None, help="cache row count the representatives must equal")
    k.add_argument(
        "--allow-missing",
        type=int,
        default=0,
        help="tolerated shortfall vs --expect-total; keep tight so new loss still fails loudly",
    )

    m = sub.add_parser("materialize", help="write the {text, url} tree from raw using the keep-lists")
    m.add_argument("--raw-glob", required=True)
    m.add_argument("--keeplist-dir", required=True)
    m.add_argument("--output-dir", required=True)
    m.add_argument("--num-chunks", type=int, default=1)
    m.add_argument("--chunk-idx", type=int, nargs="+", default=[0])
    m.add_argument("--greedy", action="store_true", help="work every outstanding unit, not just this chunk")

    h = sub.add_parser("hash-raw", help="record (content_id, shard, line) for every raw document")
    h.add_argument("--raw-glob", required=True)
    h.add_argument("--output-dir", required=True)
    h.add_argument("--num-chunks", type=int, default=1)
    h.add_argument("--chunk-idx", type=int, nargs="+", default=[0])
    h.add_argument("--greedy", action="store_true", help="work every outstanding unit, not just this chunk")
    h.add_argument("--num-buckets", type=int, default=64)

    b = sub.add_parser("bucket-survivors", help="re-partition survivor ids by id bucket")
    b.add_argument("--survivor-dir", required=True)
    b.add_argument("--output-dir", required=True)
    b.add_argument("--num-chunks", type=int, default=1)
    b.add_argument("--chunk-idx", type=int, nargs="+", default=[0])
    b.add_argument("--num-buckets", type=int, default=64)
    b.add_argument("--expect-parts", type=int, default=None, help="survivor part count that must exist first")

    vt = sub.add_parser("verify-tree", help="reconcile the rebuilt tree against the cache row count")
    vt.add_argument("--documents-dir", required=True)
    vt.add_argument("--keeplist-dir", required=True)
    vt.add_argument("--num-shards", type=int, default=10364)
    vt.add_argument("--expect-total", type=int, required=True)
    vt.add_argument("--report-path", default=None)

    u = sub.add_parser("materialize-unmatched", help="write unmatched survivors from their decoded cache text")
    u.add_argument("--survivor-dir", required=True)
    u.add_argument("--representatives-dir", required=True)
    u.add_argument("--survivor-ids-dir", required=True)
    u.add_argument("--cache-path", required=True)
    u.add_argument("--output-dir", required=True)
    u.add_argument("--num-buckets", type=int, default=64)
    u.add_argument("--num-chunks", type=int, default=1)
    u.add_argument("--chunk-idx", type=int, nargs="+", default=[0])
    u.add_argument("--unmatched-ids-path", default=None, help="cache for the derived unmatched id set")

    d = sub.add_parser("diagnose-unmatched", help="characterise survivors with no matching raw document")
    d.add_argument("--survivor-dir", required=True)
    d.add_argument("--representatives-dir", required=True)
    d.add_argument("--survivor-ids-dir", required=True)
    d.add_argument("--cache-path", required=True)
    d.add_argument("--bucket-idx", type=int, default=0)
    d.add_argument("--sample", type=int, default=300)
    d.add_argument("--report-path", default=None, help="GCS path for the JSON report (stdout is unreliable here)")

    r = sub.add_parser("select-representatives", help="pick one raw document per surviving id")
    r.add_argument("--survivor-dir", required=True)
    r.add_argument("--raw-index-dir", required=True)
    r.add_argument("--output-dir", required=True)
    r.add_argument("--num-buckets", type=int, default=64)
    r.add_argument("--bucket-idx", type=int, required=True)

    s = sub.add_parser("survivor-ids", help="decode the cache and record every row's content id")
    s.add_argument("--cache-path", required=True)
    s.add_argument("--output-dir", required=True)
    s.add_argument("--num-chunks", type=int, default=1)
    s.add_argument("--chunk-idx", type=int, nargs="+", default=[0])
    s.add_argument("--greedy", action="store_true", help="work every outstanding unit, not just this chunk")

    v = sub.add_parser("validate", help="positionally compare a cache part to its source shard")
    v.add_argument("--cache-path", required=True)
    v.add_argument("--shard-path", required=True)
    v.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("probe", help="test whether a raw extraction is a cache's ancestor")
    p.add_argument("--cache-path", required=True)
    p.add_argument("--raw-glob", required=True)
    p.add_argument("--n-shards", type=int, default=200)
    p.add_argument("--n-rows", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # `--chunk-idx` takes a list so one Iris job can process several chunks. The
    # chunk COUNT fixes the output partitioning (and, for hash-raw and
    # bucket-survivors, the output filenames), so it must stay constant across
    # runs; how many chunks a single job works through is independent of that and
    # is what keeps the submission count — the real bottleneck — low.
    # `--greedy` widens a job from its assigned chunks to every chunk, walked in an
    # order seeded by its first chunk index. Landing a job on this cluster costs
    # far more than running one, so when a wave only partly lands, the jobs that
    # did land should drain the backlog instead of stopping at their slice. Every
    # unit is guarded by a done marker, so revisiting one is free and two workers
    # racing on the same unit merely duplicates work.
    def units(total: int) -> list[int]:
        if not getattr(args, "greedy", False):
            return list(args.chunk_idx)
        order = list(range(total))
        random.Random(args.chunk_idx[0]).shuffle(order)
        return order

    if args.mode == "validate":
        return validate_against_shard(args.cache_path, args.shard_path, args.limit)
    if args.mode == "survivor-ids":
        for idx in args.chunk_idx if not args.greedy else args.chunk_idx[:1]:
            extract_survivor_ids(args.cache_path, args.output_dir, args.num_chunks, idx, args.greedy)
        return 0
    if args.mode == "shard-keeplists":
        return shard_keeplists(
            args.representatives_dir, args.output_dir, args.num_shards, args.expect_total, args.allow_missing
        )
    if args.mode == "materialize":
        for idx in units(args.num_chunks):
            materialize(args.raw_glob, args.keeplist_dir, args.output_dir, args.num_chunks, idx)
        return 0
    if args.mode == "hash-raw":
        for idx in units(args.num_chunks):
            hash_raw(args.raw_glob, args.output_dir, args.num_chunks, idx, args.num_buckets)
        return 0
    if args.mode == "bucket-survivors":
        for idx in args.chunk_idx:
            bucket_survivors(
                args.survivor_dir, args.output_dir, args.num_chunks, idx, args.num_buckets, args.expect_parts
            )
        return 0
    if args.mode == "verify-tree":
        return verify_tree(args.documents_dir, args.keeplist_dir, args.num_shards, args.expect_total, args.report_path)
    if args.mode == "materialize-unmatched":
        return materialize_unmatched(
            args.survivor_dir,
            args.representatives_dir,
            args.survivor_ids_dir,
            args.cache_path,
            args.output_dir,
            args.num_buckets,
            args.chunk_idx,
            args.num_chunks,
        )
    if args.mode == "diagnose-unmatched":
        return diagnose_unmatched(
            args.survivor_dir,
            args.representatives_dir,
            args.survivor_ids_dir,
            args.cache_path,
            args.bucket_idx,
            args.sample,
        )
    if args.mode == "select-representatives":
        return select_representatives(
            args.survivor_dir, args.raw_index_dir, args.output_dir, args.num_buckets, args.bucket_idx
        )
    return probe_lineage(args.cache_path, args.raw_glob, args.n_shards, args.n_rows, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
