# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a paired (HTML -> 8B gold text) train/dev/test set for local iteration on
rule-based extractors (resiliparse / justext / trafilatura / ...).

Scope: ONLY pages the 8B actually extracted -- the ``data/`` shards of
``high_quality_3000_distill`` (``data_no_useful/`` abstentions are reserved for the
fastText router and excluded here). A page qualifies iff its ``final_output`` is a
non-empty, non-abstention string; no other quality/length/dedup filtering.

The ``html`` field is ``body_strip(raw_html)`` -- the SAME preprocessing the 8B
teacher saw (mirrors small-rephraser's ``preprocess_html_for_extraction``), so the
rule-based extractors are compared on the teacher's actual input. It also drops
scripts/head, shrinking the artifact.

Splits are WARC-disjoint (each WARC -> one split) and dev/test are snapshot-
stratified and frozen independent of train size (reuses ``held_out_indices``), so
train can grow later without contaminating the held-out evals. Per-record
``has_table``/``has_code``/``has_list`` flags let you slice eval by content type
(where extractors diverge most).

Run IN us-central2 (where the parquet lives) and write the compact jsonl.gz here;
then pull locally with a single ``gcloud storage cp`` (only this small artifact
crosses regions). Schema per record:
    {html, final_output, url, warc_record_id, snapshot,
     has_table, has_code, has_list, split}
"""

import argparse
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum

import fsspec
import pyarrow.parquet as pq

from experiments.baseline_collection.fasttext_useful_classifier import (
    USEFUL_DIR,
    _shard_snapshots,
    body_strip,
    held_out_indices,
)
from experiments.fsspec_paths import fsspec_glob

logger = logging.getLogger(__name__)

# Only the columns the extractor task needs (skip the large reasoning_trace).
READ_COLUMNS = ["raw_html", "final_output", "url", "warc_record_id", "snapshot"]
ABSTAIN_MARKER = "[NO_USEFUL_CONTENT]"

DEFAULT_OUTPUT = "gs://marin-us-central2/datasets/extractor_eval_set/medium_12k"

# big_train reserves a prefix of the SAME shuffled train-WARC order the small train
# drew from, so the two share no WARC. The small train consumed <=~32 WARCs (~3400
# kept docs/WARC); 128 is a wide safety margin. Disjointness is also asserted at the
# warc_record_id level before writing.
DEFAULT_BIG_TRAIN_RESERVE = 128
DEFAULT_PER_WARC_CAP = 500  # cap docs taken per WARC so big_train spans many WARCs (diversity)


class HtmlMode(StrEnum):
    """Which HTML to store in the ``html`` field (row selection is identical either way).

    ``BODY_STRIP`` mirrors the 8B teacher's input (``body_strip(raw_html)``): scripts/head
    dropped, ``<body>`` kept. ``RAW`` stores the unmodified ``raw_html`` so extractors run on
    the full page. Both modes select the SAME rows (sampling runs before this transform) and
    keep ``has_table``/``has_code``/``has_list`` computed on the body-stripped text, so the two
    datasets are byte-identical except for the ``html`` field.
    """

    BODY_STRIP = "body_strip"
    RAW = "raw"


HTML_FIELD_DESC = {
    HtmlMode.BODY_STRIP: "body_strip(raw_html) — matches the 8B teacher's input",
    HtmlMode.RAW: "raw_html — unmodified page HTML (same rows as body_strip variant)",
}


def content_flags(html: str) -> dict[str, bool]:
    """Cheap structural flags for slicing eval by content type."""
    lowered = html.lower()
    return {
        "has_table": "<table" in lowered,
        "has_code": "<code" in lowered or "<pre" in lowered,
        "has_list": "<ul" in lowered or "<ol" in lowered,
    }


def to_record(row: dict, split: str, html_mode: HtmlMode) -> dict:
    """A paired example: HTML input + the 8B gold output + provenance.

    ``html_mode`` selects body_strip'd vs raw HTML for the ``html`` field. Structural flags are
    always computed on the body_strip'd text so they match across modes (page-content flags, not
    head/script noise) — the only field that differs between modes is ``html``.
    """
    bs = body_strip(row["raw_html"] or "")
    html = (row["raw_html"] or "") if html_mode is HtmlMode.RAW else bs
    return {
        "html": html,
        "final_output": row["final_output"],
        "url": row.get("url") or "",
        "warc_record_id": row.get("warc_record_id") or "",
        "snapshot": row.get("snapshot") or "",
        **content_flags(bs),
        "split": split,
    }


def _is_extracted(final_output: str | None) -> bool:
    """A page was actually extracted iff it has real, non-abstention output."""
    return bool(final_output) and final_output.strip() != ABSTAIN_MARKER


def read_shard(path: str) -> list[dict]:
    """Read one WARC shard, keeping only actually-extracted rows (pruned columns)."""
    rows: list[dict] = []
    with fsspec.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=READ_COLUMNS, batch_size=1024):
            for r in batch.to_pylist():
                if _is_extracted(r.get("final_output")):
                    rows.append(r)
    return rows


def collect_docs(shards: list[str], target: int, seed: int, stop_early: bool, max_workers: int = 32) -> list[dict]:
    """Uniformly sample ``target`` rows from ``shards`` (seeded, reproducible).

    Shards are read in shuffled order; with ``stop_early`` we stop once the pool
    comfortably exceeds the target (train, where shards are plentiful), otherwise
    we read every shard (dev/test holdout, few shards) -- then sample ``target``.
    """
    rng = random.Random(seed)
    order = shards[:]
    rng.shuffle(order)
    needed = int(target * 1.3) if stop_early else None

    pool: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(0, len(order), max_workers):
            if needed is not None and len(pool) >= needed:
                break
            for rows in ex.map(read_shard, order[i : i + max_workers]):
                pool.extend(rows)

    if len(pool) <= target:
        logger.warning("split wanted %d docs but only %d available", target, len(pool))
        return pool
    return rng.sample(pool, target)


def _snapshot_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r["snapshot"]] = counts.get(r["snapshot"], 0) + 1
    return dict(sorted(counts.items()))


def write_jsonl_gz(path: str, records: list[dict]) -> None:
    with fsspec.open(path, "wt", compression="gzip") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build(
    output_dir: str, n_train: int, n_dev: int, n_test: int, seed: int, k_holdout: int, html_mode: HtmlMode
) -> None:
    shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    logger.info("found %d WARC shards under %s", len(shards), USEFUL_DIR)

    snapshots = _shard_snapshots(shards)
    val_idx, test_idx = held_out_indices(snapshots, k_holdout)
    held = val_idx | test_idx
    train_shards = [s for i, s in enumerate(shards) if i not in held]
    dev_shards = [shards[i] for i in sorted(val_idx)]
    test_shards = [shards[i] for i in sorted(test_idx)]
    logger.info("WARC split: train=%d dev=%d test=%d", len(train_shards), len(dev_shards), len(test_shards))

    # Distinct per-split seeds so the three shuffles/samples are independent yet reproducible.
    # Sampled rows carry the raw source columns; to_record applies body_strip + flags + split.
    raw_splits = {
        "train": collect_docs(train_shards, n_train, seed + 1, stop_early=True),
        "dev": collect_docs(dev_shards, n_dev, seed + 2, stop_early=False),
        "test": collect_docs(test_shards, n_test, seed + 3, stop_early=False),
    }
    splits = {name: [to_record(row, name, html_mode) for row in rows] for name, rows in raw_splits.items()}

    manifest = {
        "source": USEFUL_DIR,
        "scope": "actually-extracted pages only (data/, non-empty non-abstention final_output)",
        "html_field": HTML_FIELD_DESC[html_mode],
        "gold_field": "final_output",
        "seed": seed,
        "k_holdout_per_snapshot": k_holdout,
        "filtering": "none beyond non-empty final_output (no dedup, no length filter)",
        "warc_split": {"train": len(train_shards), "dev": len(dev_shards), "test": len(test_shards)},
        "counts": {name: len(recs) for name, recs in splits.items()},
        "snapshot_distribution": {name: _snapshot_counts(recs) for name, recs in splits.items()},
    }

    for name, records in splits.items():
        out = f"{output_dir}/{name}.jsonl.gz"
        write_jsonl_gz(out, records)
        logger.info("wrote %d records -> %s", len(records), out)
    with fsspec.open(f"{output_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote manifest -> %s/manifest.json", output_dir)


def collect_big(
    shards: list[str], target: int, per_warc_cap: int, seed: int, max_workers: int = 32
) -> tuple[list[dict], int]:
    """Sample ``target`` rows, taking at most ``per_warc_cap`` from each WARC so the
    set spans many WARCs (diversity) with bounded memory. Returns (rows, n_warcs_read)."""
    rng = random.Random(seed)
    order = shards[:]
    rng.shuffle(order)

    pool: list[dict] = []
    n_read = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i in range(0, len(order), max_workers):
            if len(pool) >= target:
                break
            chunk = order[i : i + max_workers]
            n_read += len(chunk)
            for rows in ex.map(read_shard, chunk):
                if len(rows) > per_warc_cap:
                    rows = rng.sample(rows, per_warc_cap)
                pool.extend(rows)

    if len(pool) <= target:
        logger.warning("big_train wanted %d but only %d available across %d WARCs", target, len(pool), n_read)
        return pool, n_read
    return rng.sample(pool, target), n_read


def _existing_warc_ids(output_dir: str, splits: tuple[str, ...]) -> set[str]:
    """All warc_record_ids already used by the given splits (for disjointness checks)."""
    ids: set[str] = set()
    for split in splits:
        with fsspec.open(f"{output_dir}/{split}.jsonl.gz", "rt", compression="gzip") as f:
            for line in f:
                ids.add(json.loads(line)["warc_record_id"])
    return ids


def build_big_train(
    output_dir: str, n: int, seed: int, reserve: int, per_warc_cap: int, k_holdout: int, html_mode: HtmlMode
) -> None:
    """Build big_train.jsonl.gz: ``n`` docs WARC-disjoint from the existing
    train/dev/test splits (same source, same body_strip'd schema).

    Disjointness: dev/test WARCs are the held-out set; the small train's WARCs are a
    prefix of the seed+1 shuffle, which we reserve. Then we assert no shared
    warc_record_id with the existing splits before writing.
    """
    shards = sorted(fsspec_glob(f"{USEFUL_DIR}/*.parquet"))
    snapshots = _shard_snapshots(shards)
    val_idx, test_idx = held_out_indices(snapshots, k_holdout)
    held = val_idx | test_idx
    train_pool = [s for i, s in enumerate(shards) if i not in held]

    # Reproduce the EXACT shuffled order the small train drew from (collect_docs used
    # seed+1), then skip its reserved prefix so big_train shares no WARC with it.
    shuffled = train_pool[:]
    random.Random(seed + 1).shuffle(shuffled)
    if reserve >= len(shuffled):
        raise ValueError(f"reserve {reserve} >= train pool {len(shuffled)}")
    big_candidates = shuffled[reserve:]
    logger.info("big_train: %d candidate WARCs (reserved first %d for the existing train)", len(big_candidates), reserve)

    rows, n_warcs = collect_big(big_candidates, n, per_warc_cap, seed + 4)
    records = [to_record(row, "big_train", html_mode) for row in rows]
    logger.info("big_train: %d records sampled from up to %d WARCs", len(records), n_warcs)

    # Bulletproof disjointness: no warc_record_id may appear in train/dev/test.
    existing = _existing_warc_ids(output_dir, ("train", "dev", "test"))
    overlap = {r["warc_record_id"] for r in records} & existing
    if overlap:
        raise RuntimeError(f"big_train overlaps existing splits on {len(overlap)} warc_record_ids; aborting")
    logger.info("verified disjoint: 0 shared ids vs %d existing train/dev/test ids", len(existing))

    manifest = {
        "source": USEFUL_DIR,
        "scope": "actually-extracted pages only; WARC-disjoint from train/dev/test",
        "html_field": HTML_FIELD_DESC[html_mode],
        "gold_field": "final_output",
        "seed": seed,
        "reserve_warcs": reserve,
        "per_warc_cap": per_warc_cap,
        "count": len(records),
        "warcs_sampled_up_to": n_warcs,
        "disjointness_verified": True,
        "snapshot_distribution": _snapshot_counts(records),
    }
    write_jsonl_gz(f"{output_dir}/big_train.jsonl.gz", records)
    logger.info("wrote %d records -> %s/big_train.jsonl.gz", len(records), output_dir)
    with fsspec.open(f"{output_dir}/big_train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote manifest -> %s/big_train_manifest.json", output_dir)


def build_new_dev(
    output_dir: str,
    n: int,
    seed: int,
    per_warc_cap: int,
    k_holdout: int,
    html_mode: HtmlMode,
    disjoint_from: tuple[str, ...],
    source_dir: str,
    out_name: str,
) -> None:
    """Build a brand-new dev set (``{out_name}.jsonl.gz``), WARC-disjoint from existing splits.

    Scans the candidate WARC shards under ``source_dir`` in a fresh seeded shuffle and SKIPS any
    whole WARC that shares a ``warc_record_id`` with an existing split in ``disjoint_from`` (read
    from ``output_dir``) — enforcing whole-WARC disjointness, not just per-doc, so there's no
    near-duplicate leakage. Takes up to ``per_warc_cap`` docs per clean WARC until ``n`` are
    collected, then asserts zero overlap before writing. Leaves existing splits untouched.

    ``source_dir`` may differ from the benchmark's own ``USEFUL_DIR`` — e.g. a parquet built over
    the uniform-random draw of the 10k pool — to make a dev set that spans WARCs outside the
    head-3k the rest of the benchmark is drawn from.
    """
    shards = sorted(fsspec_glob(f"{source_dir}/*.parquet"))
    # The held-out dev/test WARCs only exist (and only matter) for the benchmark's own source;
    # for any other source_dir the id-level disjoint_from filter already excludes dev/test WARCs.
    if source_dir == USEFUL_DIR:
        snapshots = _shard_snapshots(shards)
        val_idx, test_idx = held_out_indices(snapshots, k_holdout)
        held = val_idx | test_idx
        candidate_shards = [s for i, s in enumerate(shards) if i not in held]
    else:
        candidate_shards = shards

    existing = _existing_warc_ids(output_dir, disjoint_from)
    logger.info(
        "new_dev: %d shards under %s; %d existing ids to stay disjoint from (%s)",
        len(candidate_shards),
        source_dir,
        len(existing),
        ",".join(disjoint_from),
    )

    order = candidate_shards[:]
    random.Random(seed + 7).shuffle(order)
    rng = random.Random(seed + 7)

    pool: list[dict] = []
    used_warcs = skipped_warcs = 0
    for path in order:
        if len(pool) >= n:
            break
        rows = read_shard(path)
        if not rows:
            continue
        if {r.get("warc_record_id") for r in rows} & existing:  # WARC touches an existing split
            skipped_warcs += 1
            continue
        if len(rows) > per_warc_cap:
            rows = rng.sample(rows, per_warc_cap)
        pool.extend(rows)
        used_warcs += 1

    records = [to_record(row, out_name, html_mode) for row in (pool if len(pool) <= n else rng.sample(pool, n))]
    if len(records) < n:
        logger.warning("new_dev wanted %d but only %d available from %d clean WARCs", n, len(records), used_warcs)

    overlap = {r["warc_record_id"] for r in records} & existing
    if overlap:
        raise RuntimeError(f"new_dev overlaps existing splits on {len(overlap)} warc_record_ids; aborting")
    logger.info(
        "new_dev: %d records from %d clean WARCs (%d skipped as used); 0 overlap",
        len(records),
        used_warcs,
        skipped_warcs,
    )

    manifest = {
        "source": USEFUL_DIR,
        "scope": f"brand-new dev set; whole-WARC disjoint from {','.join(disjoint_from)}",
        "html_field": HTML_FIELD_DESC[html_mode],
        "gold_field": "final_output",
        "seed": seed,
        "per_warc_cap": per_warc_cap,
        "count": len(records),
        "clean_warcs_used": used_warcs,
        "used_warcs_skipped": skipped_warcs,
        "disjoint_from": list(disjoint_from),
        "disjointness_verified": True,
        "snapshot_distribution": _snapshot_counts(records),
    }
    manifest["source"] = source_dir
    write_jsonl_gz(f"{output_dir}/{out_name}.jsonl.gz", records)
    logger.info("wrote %d records -> %s/%s.jsonl.gz", len(records), output_dir, out_name)
    with fsspec.open(f"{output_dir}/{out_name}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("wrote manifest -> %s/%s_manifest.json", output_dir, out_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["splits", "big_train", "new_dev"], default="splits")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-big", type=int, default=100000)
    ap.add_argument("--n-newdev", type=int, default=1000, help="new_dev mode: size of the brand-new dev set.")
    ap.add_argument(
        "--newdev-name", default="dev2", help="new_dev mode: output basename + split label (e.g. dev2, dev3)."
    )
    ap.add_argument(
        "--source-dir",
        default=None,
        help="new_dev mode: parquet dir to sample WARCs from (default: the benchmark's own USEFUL_DIR). "
        "Point at a random-draw parquet to span WARCs outside the head-3k.",
    )
    ap.add_argument(
        "--disjoint-from",
        default="train,dev,test,big_train",
        help="new_dev mode: comma-separated existing split basenames (in --output-dir) to stay WARC-disjoint from.",
    )
    ap.add_argument("--reserve", type=int, default=DEFAULT_BIG_TRAIN_RESERVE)
    ap.add_argument("--per-warc-cap", type=int, default=DEFAULT_PER_WARC_CAP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k-holdout", type=int, default=1, help="WARCs per snapshot held out for EACH of dev and test")
    ap.add_argument(
        "--html-mode",
        type=HtmlMode,
        choices=list(HtmlMode),
        default=HtmlMode.BODY_STRIP,
        help="HTML stored in the `html` field: body_strip (teacher input) or raw (full page). Same rows either way.",
    )
    args = ap.parse_args()
    if args.mode == "splits":
        build(args.output_dir, args.n_train, args.n_dev, args.n_test, args.seed, args.k_holdout, args.html_mode)
    elif args.mode == "big_train":
        build_big_train(
            args.output_dir, args.n_big, args.seed, args.reserve, args.per_warc_cap, args.k_holdout, args.html_mode
        )
    else:
        build_new_dev(
            args.output_dir,
            args.n_newdev,
            args.seed,
            args.per_warc_cap,
            args.k_holdout,
            args.html_mode,
            disjoint_from=tuple(args.disjoint_from.split(",")),
            source_dir=args.source_dir or USEFUL_DIR,
            out_name=args.newdev_name,
        )


if __name__ == "__main__":
    main()
