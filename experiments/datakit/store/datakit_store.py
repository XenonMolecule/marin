# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit -> per-(cluster, quality) Levanter store via a shuffle.

The store routes each surviving doc through a Zephyr ``group_by`` keyed by
``(cluster, quality, sub)``, so a single reducer streams all of one bucket-shard's
documents into one materialized Levanter cache. No per-input-shard leaves are
created. The only intermediate is the scatter spill (~84K files for the full
store), and the final store is bounded by the configured number of subshards per
bucket. A measured 1%-stride benchmark put the full shuffle at ~17 TB compressed
scatter / ~2 h wall at ~2k workers, all in-region (no egress), with the cost
dominated by token I/O.

Pipeline:

1. **map** (per input shard): a 5-way positional join over tokenization,
   decontamination, domain assignment, quality, and dedup attributes, emitting
   ``{cluster, quality, sub, input_ids}`` per surviving doc. ``sub`` is a stable
   hash of the doc id mod that bucket's subshard count, so a hot bucket is split
   evenly across many reducers instead of one.
2. **group_by** ``(cluster, quality, sub)`` -> **reduce**: each reducer streams
   its group into one materialized cache at
   ``<output>/cluster=<C>/quality=<Q>/sub=<S>/<split>`` via ``SerialCacheWriter``.
   The trailing ``<split>`` level is load-bearing: Levanter resolves a
   ``DatasetComponent`` as ``<cache_dir>/<split>``
   (``levanter/data/text/datasets.py``, ``load_cache``/``train_sets``), so a cell
   written without it cannot be loaded by a mixture at all.
3. **driver finalize**: group reducer stats by ``(cluster, quality)`` and resolve
   each bucket to the single cache path a trainer loads (see
   :func:`_finalize_buckets`).

Without a prior store artifact, each bucket uses ``default_subshards`` (32 in
the production reference pipeline, one in smoke mode). Direct callers may pass
``bucket_token_hint`` from :func:`bucket_token_hint_from_artifact` to size each
bucket independently.

Zephyr retries individual map and reduce tasks, but the driver-side finalize is
not separately checkpointed. If the store driver dies after the shuffle finishes
and before the artifact is written, rerunning the store repeats the shuffle.

``resume=True`` makes a rerun reuse cells that already materialized: each cell
writes a ``_subshard_stat.json`` sidecar inside its atomic rename, so the sidecar
exists only beside a complete cache and a rerun skips that reducer. The map and
shuffle still repeat -- only the reduce is spared. This matters on preemptible
workers, where a multi-hour store otherwise loses every finished cell when the
worker group finally exhausts its retries.
"""

import dataclasses
import json
import logging
import math
import os
import re
from collections import defaultdict
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import (
    CacheLedger,
    CacheMetadata,
    SerialCacheWriter,
    consolidate_shard_caches,
)
from pydantic import BaseModel
from zephyr import counters
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.writers import atomic_rename

from experiments.datakit.store.store_compat import (
    AssignmentAttrData,
    DeconAttributes,
    FuzzyDupsAttrData,
    QualityScores,
    TokenizedAttrData,
    deterministic_hash,
    read_artifact,
    sp_exists,
    sp_glob,
    sp_open,
    write_artifact,
)

# ---------------------------------------------------------------------------
# FORK NOTE (2026-07-29). Copied from upstream marin @ experiments/datakit/store.
# The shuffle, subshard plan and artifact are UNCHANGED — those are the parts
# worth having. Five deliberate deviations, all in
# `store_compat.py` or marked inline:
#
#   1. `StoragePath` / `read_artifact` / `write_artifact` / `deterministic_hash`
#      do not exist in this checkout; `store_compat` supplies equivalents.
#   2. The upstream artifact classes (TokenizedAttrData, DeconAttributes,
#      AssignmentAttrData, QualityScores, FuzzyDupsAttrData) live in modules we
#      have not ported; `store_compat` defines the minimal shapes this file uses.
#   3. **decon is optional.** Upstream filters contaminated docs here because its
#      corpora arrive raw. Ours were already deconned upstream (high_quality) or
#      deliberately not (the rest), so passing `decontam=None` skips it and the
#      co-partitioning check anchors on cluster-vs-quality instead.
#   4. **dedup is optional** for the same reason; upstream already treats a
#      missing dedup table as "all singletons".
#   5. The per-bucket ledger merge is replaced by `_finalize_buckets` — this
#      levanter cannot read a ledger that points at child caches. See that
#      function's docstring; it is the one non-mechanical change.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


class BucketCacheStats(BaseModel):
    """Per-(cluster, quality) Levanter cache stats inside :class:`ClusteredStoreData`."""

    cluster_id: int
    quality_bucket: int
    path: str
    """The materialized cache directory -- what ``TreeCache.load`` takes. Ends in the
    split level, so use :attr:`component_cache_dir` to build a mixture component."""
    total_elements: int
    total_tokens: int
    n_shards: int

    @property
    def component_cache_dir(self) -> str:
        """What a Levanter ``DatasetComponent.cache_dir`` takes: :attr:`path` minus the
        split level, since Levanter itself appends ``/<split>`` when loading."""
        return os.path.dirname(self.path.rstrip("/"))


class ClusteredStoreData(BaseModel):
    """One Levanter cache per populated (cluster, quality) bucket.

    Persisted as ``<output_path>/artifact.json``. Load via
    ``read_artifact(output_path, ClusteredStoreData)``.
    """

    version: str = "v3"
    cache_path: str
    cluster_view: int
    bucket_edges: list[float]
    split: str
    buckets: list[BucketCacheStats]
    source_names: list[str]
    tokenizer: str
    counters: dict[str, int | float]


def _per_source_shard_tuples(
    *,
    source_name: str,
    tokenize: TokenizedAttrData,
    decontam: DeconAttributes | None,
    cluster_assign: AssignmentAttrData,
    quality: QualityScores,
    dedup_attr_dir: str,
    split: str,
) -> list[dict[str, str]]:
    """Align one source's co-partitioned attribute shards by basename."""
    tok_dir = tokenize.output_dirs.get(split)
    if tok_dir is None:
        raise FileNotFoundError(f"{source_name}: tokenize has no split={split!r}")
    tok_shards = sp_glob(f"{tok_dir.rstrip('/')}/*.parquet")
    if not tok_shards:
        raise FileNotFoundError(f"{source_name}: no tokenize shards under {tok_dir}")

    # FORK: decon and dedup are optional here. An empty string means "absent",
    # which the loaders below treat as no filtering.
    decon_dir = decontam.main_output_dir.rstrip("/") if decontam else ""
    cluster_dir = cluster_assign.output_dir.rstrip("/")
    quality_dir = quality.main_output_dir.rstrip("/")
    dedup_dir = dedup_attr_dir.rstrip("/") if dedup_attr_dir else ""
    return [
        {
            "tokenize": tok_path,
            "decontam": f"{decon_dir}/{os.path.basename(tok_path)}" if decon_dir else "",
            "cluster": f"{cluster_dir}/{os.path.basename(tok_path)}",
            "quality": f"{quality_dir}/{os.path.basename(tok_path)}",
            "dedup": f"{dedup_dir}/{os.path.basename(tok_path)}" if dedup_dir else "",
            "source_name": source_name,
            "basename": os.path.basename(tok_path),
        }
        for tok_path in tok_shards
    ]


def _read_columns(path: str, columns: list[str]) -> pa.Table:
    """Read Parquet through fsspec for compatibility with CoreWeave object storage."""
    with sp_open(path, "rb") as fh:
        return pq.read_table(fh, columns=columns)


def _id_column(table: pa.Table) -> pa.Array:
    """The ``id`` column, cast to string.

    A zero-row shard's ``id`` column can come back typed ``null`` rather than
    ``string`` -- pyarrow infers a column's type from its values, and an empty
    column has none to infer from. ``pc.equal`` has no kernel for
    ``(null, null)``, so two empty shards with two null-typed id columns crash
    the co-partitioning check instead of trivially matching. IDs are always
    strings regardless of what an empty column's inferred type says.
    """
    return table.column("id").combine_chunks().cast(pa.string())


def _load_decon_table(path: str) -> tuple[pa.Array | None, np.ndarray | None]:
    """FORK: an empty path means decon was applied upstream (or deliberately skipped),
    so there is nothing to filter and no anchor to align against."""
    if not path:
        return None, None
    table = _read_columns(path, ["id", "attributes"])
    ids = _id_column(table)
    contaminated = np.asarray(
        table.column("attributes").combine_chunks().field("contaminated"),
        dtype=bool,
    )
    return ids, contaminated


def _ids_equal(left: pa.Array, right: pa.Array) -> bool:
    """Whether two id arrays match element-for-element.

    ``min_count=0`` matters: ``pc.all`` over an EMPTY array returns null by
    default, so a plain truthiness check reads two zero-row shards as mismatched.
    13,541 of fineweb_cc's 21,531 shards are legitimately empty, so that reading
    would fail the store on the majority of the largest corpus.
    """
    return bool(pc.all(pc.equal(left, right), min_count=0).as_py())


def _load_cluster_table(path: str, cluster_col: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", cluster_col])
    return _id_column(table), np.asarray(table.column(cluster_col), dtype=np.int32)


def _load_quality_table(path: str) -> tuple[pa.Array, np.ndarray]:
    table = _read_columns(path, ["id", "quality_bucket"])
    return _id_column(table), np.asarray(table.column("quality_bucket"), dtype=np.int32)


def _load_dedup_canonical(path: str) -> dict[str, bool]:
    """Return sparse canonical flags; missing IDs are singleton documents."""
    if not path or not sp_exists(path):
        return {}
    with sp_open(path, "rb") as fh:
        parquet = pq.ParquetFile(fh)
        if parquet.metadata.num_rows == 0:
            return {}
        table = parquet.read(columns=["id", "attributes"])
    ids = table.column("id").to_pylist()
    canonical = table.column("attributes").combine_chunks().field("is_cluster_canonical").to_pylist()
    return dict(zip(ids, canonical, strict=True))


def _validate_cluster_view(cluster_assign: dict[str, AssignmentAttrData], cluster_view: int) -> str:
    """Check that every assignment artifact materialized the selected view."""
    for name, assignment in cluster_assign.items():
        valid_views = {assignment.k_train, *assignment.k_views}
        if cluster_view not in valid_views:
            raise ValueError(
                f"cluster_view={cluster_view} not in {name}'s views "
                f"(k_train={assignment.k_train}, k_views={assignment.k_views})"
            )
    return f"cluster_{cluster_view}"


def _resolve_dedup_attr_dir(
    *,
    source_name: str,
    main_output_dir: str,
    dedup: FuzzyDupsAttrData | None,
) -> str:
    """FORK: ``dedup=None`` means no fuzzy-dedup attributes exist, so nothing is
    dropped as a non-canonical duplicate."""
    if dedup is None:
        return ""
    entry = dedup.sources.get(main_output_dir)
    if entry is None:
        raise KeyError(
            f"{source_name}: dedup.sources has no entry for source_main_dir={main_output_dir!r}. "
            "Drop the source from the config or rebuild dedup with it included."
        )
    return entry.attr_dir


# Records flushed to the SerialCacheWriter at a time on the reduce side. Bounds
# reducer memory at ~_WRITE_FLUSH * avg-doc-bytes regardless of group size, so
# even the hottest bucket-shard (~10B tokens with adequate subshards) streams
# in constant memory.
_WRITE_FLUSH = 1024

# Default skew-splitting target: aim for this many tokens per reduce cache.
# ~20B keeps the hottest reducer to a few hundred GB of token I/O.
DEFAULT_TARGET_TOKENS_PER_SUBSHARD = 20_000_000_000

# Without a prior store artifact to size buckets from, split every bucket enough
# to keep the known ~651B-token hot bucket from becoming one multi-hour reducer.
DEFAULT_SUBSHARDS = 32

# Rows read from a tokenized shard at once during the positional join.
_TOKENIZE_BATCH_SIZE = 8192


@dataclasses.dataclass(frozen=True)
class _SubshardStat:
    """One reducer's materialized ``(cluster, quality, sub)`` cache summary, returned to the driver."""

    cluster: int
    quality: int
    sub: int
    path: str
    rows: int
    tokens: int
    # Numpy dtype name of the stored token arrays. Consolidation must open each
    # cache with a matching exemplar or tensorstore refuses the (unsafe) cast.
    token_dtype: str


# ---------------------------------------------------------------------------
# Map side: join + filter -> per-doc shuffle records.
# ---------------------------------------------------------------------------


def _iter_surviving_docs(spec: dict[str, str], cluster_col: str) -> Iterator[tuple[int, int, str, np.ndarray]]:
    """Join one shard's five datasets; yield ``(cluster, quality_bucket, doc_id, input_ids)`` per surviving doc.

    Reads decon/cluster/quality densely, dedup sparsely, streams tokenize in
    positional lockstep, and drops contaminated rows and dedup-cluster
    non-canonicals. Fails loud on missing or misaligned inputs.

    FORK: decon may be absent (see the module fork note). The co-partitioning
    check then anchors on cluster-vs-quality, which is the pair we actually
    produce shard-for-shard, and contamination filtering is a no-op.
    """
    where = f"{spec['source_name']}/{spec['basename']}"
    decon_ids, contaminated = _load_decon_table(spec["decontam"])
    cluster_ids, cluster_vals = _load_cluster_table(spec["cluster"], cluster_col)
    # Quality parquets carry a precomputed, calibrated ``quality_bucket`` column
    # (fast-transformer scorer), consumed as-is -- no score->bucket mapping here.
    quality_ids, quality_buckets = _load_quality_table(spec["quality"])
    n_anchor, n_cluster, n_quality = len(cluster_ids), len(cluster_ids), len(quality_ids)
    if n_cluster != n_quality:
        raise RuntimeError(
            f"{where}: dense-table row count mismatch "
            f"(cluster={n_cluster}, quality={n_quality}) -- co-partitioning broken"
        )
    # Equal row counts don't imply equal ID order. Verify the dense tables align
    # before routing positionally, then drop their ID arrays; the loop only needs
    # tokenization IDs for the dedup lookup.
    if not _ids_equal(cluster_ids, quality_ids):
        raise RuntimeError(f"{where}: cluster/quality id mismatch -- co-partitioning broken")
    if decon_ids is not None:
        if len(decon_ids) != n_cluster:
            raise RuntimeError(
                f"{where}: decon rows ({len(decon_ids)}) != cluster rows ({n_cluster}) " "-- co-partitioning broken"
            )
        if not _ids_equal(decon_ids, cluster_ids):
            raise RuntimeError(f"{where}: decon/cluster id mismatch -- co-partitioning broken")
    del decon_ids, cluster_ids, quality_ids
    dedup_canonical = _load_dedup_canonical(spec["dedup"])

    n_in = 0
    n_contaminated = 0
    n_dedup_dropped = 0
    n_out = 0
    with sp_open(spec["tokenize"], "rb") as fh:
        pf = pq.ParquetFile(fh)
        # Check the count from Parquet metadata before streaming. Without this the
        # positional slices run off the end of the attribute arrays mid-loop and
        # surface as a bare IndexError from the reduce side.
        if pf.metadata.num_rows != n_anchor:
            raise RuntimeError(
                f"{where}: tokenize rows ({pf.metadata.num_rows}) != attribute rows ({n_anchor}) "
                "-- co-partitioning broken"
            )
        row_idx = 0
        for batch in pf.iter_batches(batch_size=_TOKENIZE_BATCH_SIZE, columns=["id", "input_ids"]):
            tok_ids = batch.column("id").to_pylist()
            tok_input_ids = batch.column("input_ids")
            batch_len = len(tok_ids)
            contam_slice = None if contaminated is None else contaminated[row_idx : row_idx + batch_len]
            cluster_slice = cluster_vals[row_idx : row_idx + batch_len]
            bucket_slice = quality_buckets[row_idx : row_idx + batch_len]
            row_idx += batch_len
            for i, doc_id in enumerate(tok_ids):
                n_in += 1
                if contam_slice is not None and contam_slice[i]:
                    n_contaminated += 1
                    continue
                if dedup_canonical.get(doc_id) is False:
                    n_dedup_dropped += 1
                    continue
                ids = tok_input_ids[i].values.to_numpy()
                n_out += 1
                yield int(cluster_slice[i]), int(bucket_slice[i]), doc_id, ids
        if row_idx != n_anchor:
            raise RuntimeError(
                f"{where}: tokenize rows ({row_idx}) != attribute rows ({n_anchor}) " "-- co-partitioning broken"
            )
    counters.increment("datakit_store/records_in", n_in)
    counters.increment("datakit_store/contaminated_dropped", n_contaminated)
    counters.increment("datakit_store/dedup_noncanonical_dropped", n_dedup_dropped)
    counters.increment("datakit_store/records_out", n_out)


def _emit_for_shuffle(
    items: Iterator[list[dict[str, str]]],
    _shard_info: ShardInfo,
    *,
    cluster_col: str,
    subshards_for_bucket: dict[tuple[int, int], int],
    default_subshards: int,
    done_keys: frozenset[tuple[int, int, int]] = frozenset(),
) -> Iterator[dict[str, object]]:
    """Map one task (a batch of source shards) to per-doc shuffle records.

    ``done_keys`` are cells already materialized by an earlier run. Their records
    are dropped here rather than shuffled and discarded at the reducer, so each
    resumed attempt moves strictly less data than the last -- without this the
    map and shuffle cost the same on every retry no matter how much is finished,
    and a run that keeps dying inside the shuffle never converges.

    Yields ``{cluster, quality, sub, input_ids}``. ``sub`` is a stable hash of
    the doc id mod that bucket's subshard count (``subshards_for_bucket`` for
    hinted buckets, else ``default_subshards``), so a bucket's docs spread evenly
    across that many reducers regardless of how the docs are partitioned across
    map tasks (a per-task counter would pile every task's first doc onto ``sub=0``).
    """
    batch_specs = next(iter(items))
    n_tokens = 0
    for spec in batch_specs:
        for cluster, quality, doc_id, ids in _iter_surviving_docs(spec, cluster_col):
            k = subshards_for_bucket.get((cluster, quality), default_subshards)
            sub = deterministic_hash(doc_id) % k if k > 1 else 0
            if (cluster, quality, sub) in done_keys:
                continue
            n_tokens += len(ids)
            yield {"cluster": cluster, "quality": quality, "sub": sub, "input_ids": ids}
    counters.increment("datakit_store/tokens_out", n_tokens)


# ---------------------------------------------------------------------------
# Reduce side: one group -> one materialized Levanter cache.
# ---------------------------------------------------------------------------


STAT_SIDECAR = "_subshard_stat.json"
_CELL_RE = re.compile(r"cluster=(\d+)/quality=(\d+)/sub=(\d+)/")


def _cell_dir(output_path: str, key: tuple[int, int, int], split: str) -> str:
    cluster, quality, sub = key
    return f"{output_path.rstrip('/')}/cluster={cluster}/quality={quality}/sub={sub}/{split}"


def _materialized_keys(output_path: str, split: str) -> frozenset[tuple[int, int, int]]:
    """Cells a previous run finished, identified by their stat sidecar."""
    keys = set()
    for path in sp_glob(f"{output_path.rstrip('/')}/cluster=*/quality=*/sub=*/{split}/{STAT_SIDECAR}"):
        found = _CELL_RE.search(path)
        if found:
            keys.add(tuple(int(g) for g in found.groups()))
    return frozenset(keys)


def _read_subshard_stat(cache_dir: str, key: tuple[int, int, int]) -> _SubshardStat | None:
    """The stat for an already-materialized cell, or None if it is absent.

    The sidecar is written *inside* the atomic rename, so its presence means the
    cache directory is complete. It carries the token count because this
    checkout's ``SerialCacheWriter`` commits ``field_counts={}`` -- the ledger
    alone cannot report tokens, so a resumed cell would otherwise contribute 0
    and silently zero that cell's mixture weight.
    """
    cluster, quality, sub = key
    try:
        with sp_open(f"{cache_dir}/{STAT_SIDECAR}", "r") as fh:
            saved = json.load(fh)
    except (FileNotFoundError, OSError):
        return None
    return _SubshardStat(
        cluster=cluster,
        quality=quality,
        sub=sub,
        path=cache_dir,
        rows=saved["rows"],
        tokens=saved["tokens"],
        token_dtype=saved["token_dtype"],
    )


def _write_subshard_cache(
    key: tuple[int, int, int],
    group: Iterator[dict[str, object]],
    *,
    output_path: str,
    split: str,
    resume: bool = False,
) -> _SubshardStat:
    """Stream one ``(cluster, quality, sub)`` group into a materialized Levanter cache.

    Writes to ``<output>/cluster=<C>/quality=<Q>/sub=<S>/<split>`` via
    ``SerialCacheWriter`` in ``_WRITE_FLUSH``-record batches (constant memory).
    Returns the slim stat the driver needs to build the per-bucket sharded ledger.

    The trailing ``<split>`` is required, not cosmetic: Levanter loads a mixture
    component from ``<cache_dir>/<split>``, so a cell written one level up raises
    "No source and no cache found for component".
    """
    cluster, quality, sub = key
    cache_dir = _cell_dir(output_path, key, split)

    if resume:
        done = _read_subshard_stat(cache_dir, key)
        if done is not None:
            counters.increment("datakit_store/reduce_resumed", 1)
            logger.info("resume: %s already materialized (%d rows), skipping", cache_dir, done.rows)
            return done

    # group_by invokes reducers only for keys that received at least one record.
    it = iter(group)
    first = next(it)
    exemplar = {"input_ids": first["input_ids"]}

    # FORK: this checkout's SerialCacheWriter commits `field_counts={}`, so the
    # ledger cannot report token counts -- they are accumulated here instead.
    # Left to the ledger, every cell reports 0 tokens, which silently zeroes both
    # the mixture weights and the subshard sizing hint for the next build.
    n_tokens = 0
    with atomic_rename(cache_dir) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=cache_dir, metadata=CacheMetadata.empty()) as writer:
            buf: list[dict[str, object]] = [exemplar]
            n_tokens += len(first["input_ids"])
            for rec in it:
                buf.append({"input_ids": rec["input_ids"]})
                n_tokens += len(rec["input_ids"])
                if len(buf) >= _WRITE_FLUSH:
                    writer.write_batch(buf)
                    buf = []
            if buf:
                writer.write_batch(buf)
        # SerialCacheWriter has committed its ledger to tmp_path by now. Write the
        # sidecar inside the rename so it can never exist beside a partial cache:
        # its presence is what a resumed run treats as "this cell is done".
        with sp_open(f"{tmp_path}/{STAT_SIDECAR}", "w") as fh:
            json.dump(
                {
                    "rows": CacheLedger.load(tmp_path, CacheMetadata.empty()).total_num_rows,
                    "tokens": n_tokens,
                    "token_dtype": np.asarray(first["input_ids"]).dtype.name,
                },
                fh,
            )

    # SerialCacheWriter committed the ledger on clean exit; load it back so the
    # driver can merge without re-reading the tensorstore.
    ledger = CacheLedger.load(cache_dir, CacheMetadata.empty())
    counters.increment("datakit_store/reduce_rows", ledger.total_num_rows)
    counters.increment("datakit_store/reduce_tokens", n_tokens)
    return _SubshardStat(
        cluster=cluster,
        quality=quality,
        sub=sub,
        path=cache_dir,
        rows=ledger.total_num_rows,
        tokens=n_tokens,
        token_dtype=np.asarray(first["input_ids"]).dtype.name,
    )


# ---------------------------------------------------------------------------
# Subshard planning + driver-side per-bucket ledger merge.
# ---------------------------------------------------------------------------


def bucket_token_hint_from_artifact(artifact_path: str) -> dict[tuple[int, int], int]:
    """Load a prior :class:`ClusteredStoreData` and return ``{(cluster, quality): total_tokens}``.

    Use as ``bucket_token_hint`` for :func:`build_clustered_store` so the
    next build splits hot buckets proportionally to last build's token mass.
    """
    prior = read_artifact(artifact_path, ClusteredStoreData)
    return {(b.cluster_id, b.quality_bucket): b.total_tokens for b in prior.buckets}


def _plan_subshards(
    *,
    bucket_token_hint: dict[tuple[int, int], int] | None,
    target_tokens_per_subshard: int,
    max_subshards: int,
    default_subshards: int,
) -> dict[tuple[int, int], int]:
    """Map each bucket to a subshard count from its hinted token mass.

    ``ceil(tokens / target)`` clamped to ``[1, max_subshards]``. Buckets without
    a hint use ``default_subshards``; without any hint, every bucket uses that
    uniform count.
    """
    if not bucket_token_hint:
        logger.warning(
            "build_clustered_store: no bucket_token_hint; every bucket uses default_subshards=%d. "
            "Pass bucket_token_hint_from_artifact(<prior store>) to split hot buckets.",
            default_subshards,
        )
        return {}
    plan = {}
    for key, tokens in bucket_token_hint.items():
        plan[key] = max(1, min(max_subshards, math.ceil(tokens / target_tokens_per_subshard)))
    logger.info(
        "build_clustered_store: subshard plan over %d buckets, max=%d, total reduce caches=%d",
        len(plan),
        max(plan.values(), default=0),
        sum(plan.values()),
    )
    return plan


SUBSHARD_PLAN = "_subshard_plan.json"


def _load_or_persist_plan(output_path: str, plan: dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    """Pin a store's subshard plan on first build and reuse it verbatim after.

    ``sub`` is ``hash(doc_id) % k``, so k determines which subshard each doc
    belongs to. If k changes between runs, the docs already written under the old
    k live in subshards the new k never addresses: the resume filter stops
    matching them, they are written a second time under the new numbering, and
    the finalize sums both layouts into one bucket -- inflated tokens, in a store
    that still looks internally consistent.

    Recomputing the plan is therefore never safe once any data exists, no matter
    how the inputs are re-derived. The plan is written beside the store on the
    first build and is authoritative from then on.
    """
    path = f"{output_path.rstrip('/')}/{SUBSHARD_PLAN}"
    try:
        with sp_open(path, "r") as fh:
            stored = json.load(fh)
        loaded = {(int(c), int(q)): int(k) for c, q, k in stored["plan"]}
        logger.info("using the store's pinned subshard plan (%d buckets)", len(loaded))
        return loaded
    except (FileNotFoundError, OSError):
        pass
    with sp_open(path, "w") as fh:
        json.dump({"plan": [[c, q, k] for (c, q), k in sorted(plan.items())]}, fh)
    logger.info("pinned subshard plan for %d buckets", len(plan))
    return plan


def _existing_consolidation(cell_path: str, expected_rows: int) -> CacheLedger | None:
    """A previous run's finished concatenation for this bucket, or None.

    Consolidation is a driver-side copy of the whole bucket -- minutes per
    bucket, hours across a store -- and it is the last thing standing between a
    complete set of subshards and the artifact. Redoing every bucket on each
    retry means a store whose reduce has fully finished can still never
    complete, because no single run survives long enough to concatenate them
    all.

    Only a ledger whose row count matches is accepted: a concatenation killed
    partway leaves either no committed ledger or a short one, and reusing that
    would silently truncate the cell.
    """
    try:
        ledger = CacheLedger.load(cell_path, CacheMetadata.empty())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if ledger.total_num_rows != expected_rows:
        logger.info(
            "re-consolidating %s: ledger has %d rows, expected %d", cell_path, ledger.total_num_rows, expected_rows
        )
        return None
    logger.info("reusing consolidated cache %s (%d rows)", cell_path, ledger.total_num_rows)
    return ledger


def _finalize_buckets(
    *,
    subshard_stats: list[_SubshardStat],
    output_path: str,
    split: str,
) -> list[BucketCacheStats]:
    """Reduce each bucket's ``sub=*`` caches to the one cache path a trainer loads.

    FORK. Upstream writes a per-bucket ``shard_ledger.json`` *over* the sub caches
    and reads it back as a virtual concatenation. This checkout's
    :class:`~levanter.store.cache.TreeCache` has no such reader: ``TreeCache.load(d)``
    opens a single tensorstore at ``d``, so a ledger pointing at ``d/sub=*`` would
    load as an empty cache. Writing one anyway is the failure mode worth avoiding —
    it looks like a working cell and trains on nothing.

    So the bucket's cache is resolved concretely instead:

    * exactly one subshard (the normal case here — our hottest cell is ~2.3B tokens
      against the ~651B that forced upstream's 32-way split, so ``_plan_subshards``
      leaves k=1): the sub cache *is* the cell. No copy, no merge.
    * more than one: physically concatenate with levanter's own
      :func:`consolidate_shard_caches`, which appends each sub cache's tensorstore
      into ``bucket_root`` and commits a real ledger. Document boundaries are
      preserved — it shifts JaggedArray offsets, it does not touch token order.
    """
    by_bucket: dict[tuple[int, int], list[_SubshardStat]] = defaultdict(list)
    for s in subshard_stats:
        by_bucket[(s.cluster, s.quality)].append(s)

    base_path = output_path.rstrip("/")
    buckets: list[BucketCacheStats] = []
    for cluster, quality in sorted(by_bucket):
        subs = sorted(by_bucket[(cluster, quality)], key=lambda s: s.sub)
        rows = sum(s.rows for s in subs)
        tokens = sum(s.tokens for s in subs)
        if len(subs) == 1:
            cell_path = subs[0].path
        else:
            # Consolidate into the bucket root's own split level, so the merged cell
            # keeps the same `<cache_dir>/<split>` shape as the single-subshard case.
            cell_path = f"{base_path}/cluster={cluster}/quality={quality}/{split}"
            dtypes = {s.token_dtype for s in subs}
            if len(dtypes) != 1:
                raise RuntimeError(f"cluster={cluster} quality={quality}: mixed token dtypes {sorted(dtypes)}")
            exemplar = {"input_ids": np.zeros(0, dtype=dtypes.pop())}
            ledger = _existing_consolidation(cell_path, rows)
            if ledger is None:
                ledger = consolidate_shard_caches(
                    [s.path for s in subs], cell_path, exemplar, metadata=CacheMetadata.empty()
                )
            if ledger.total_num_rows != rows:
                raise RuntimeError(
                    f"cluster={cluster} quality={quality}: consolidated cache has "
                    f"{ledger.total_num_rows} rows, expected {rows}"
                )
        buckets.append(
            BucketCacheStats(
                cluster_id=cluster,
                quality_bucket=quality,
                path=cell_path,
                total_elements=rows,
                total_tokens=tokens,
                n_shards=len(subs),
            )
        )
        logger.info(
            "cluster=%d quality=%d: docs=%d tokens=%d subshards=%d -> %s",
            cluster,
            quality,
            rows,
            tokens,
            len(subs),
            cell_path,
        )
    return buckets


# ---------------------------------------------------------------------------
# Driver entry point.
# ---------------------------------------------------------------------------


def build_clustered_store(
    *,
    tokenize: dict[str, TokenizedAttrData],
    cluster_assign: dict[str, AssignmentAttrData],
    quality: dict[str, QualityScores],
    output_path: str,
    decontam: dict[str, DeconAttributes] | None = None,
    dedup: FuzzyDupsAttrData | None = None,
    cluster_view: int = 40,
    split: str = "train",
    worker_resources: ResourceConfig | None = None,
    max_workers: int = 4096,
    shards_per_task: int = 1,
    reduce_shards: int = 2048,
    bucket_token_hint: dict[tuple[int, int], int] | None = None,
    target_tokens_per_subshard: int = DEFAULT_TARGET_TOKENS_PER_SUBSHARD,
    max_subshards: int = 128,
    default_subshards: int = DEFAULT_SUBSHARDS,
    client=None,
    chunk_storage_prefix: str | None = None,
    resume: bool = False,
) -> ClusteredStoreData:
    """Shuffle 5-way join + filter into one materialized cache per ``(cluster, quality, sub)``.

    The store is born compact: reducers create the final materialized caches
    directly rather than producing per-input-shard leaf caches.

    Args:
        shards_per_task: Source shards per map task (batches reduce the map task
            count; does not affect the shuffle output).
        reduce_shards: ``num_output_shards`` for the ``group_by`` -- the number of
            reduce tasks the ~``sum(subshards)`` groups are spread across.
        bucket_token_hint: ``{(cluster, quality): tokens}`` used to size each
            bucket's subshard count (see :func:`bucket_token_hint_from_artifact`).
        target_tokens_per_subshard / max_subshards / default_subshards: subshard
            sizing knobs (see :func:`_plan_subshards`).
        client / chunk_storage_prefix: FORK -- passed through to ``ZephyrContext``
            so the store can run end-to-end against a ``LocalClient`` in a test
            instead of only against Iris.
    """
    if not tokenize:
        raise ValueError("build_clustered_store: tokenize is empty")
    optional_decontam = decontam or {}
    for label, d in (("cluster_assign", cluster_assign), ("quality", quality)):
        if set(d) != set(tokenize):
            missing = sorted(set(tokenize) - set(d))
            extra = sorted(set(d) - set(tokenize))
            raise ValueError(f"{label} source set must equal tokenize: missing={missing!r}, extra={extra!r}")
    # FORK: decon is optional, but a partial mapping is a config bug, not a
    # deliberate skip -- either every source has it or none does.
    if optional_decontam and set(optional_decontam) != set(tokenize):
        raise ValueError(f"decontam given for only some sources: {sorted(optional_decontam)!r}")
    if shards_per_task < 1:
        raise ValueError(f"shards_per_task must be >= 1, got {shards_per_task}")
    if reduce_shards < 1:
        raise ValueError(f"reduce_shards must be >= 1, got {reduce_shards}")
    if target_tokens_per_subshard < 1:
        raise ValueError(f"target_tokens_per_subshard must be >= 1, got {target_tokens_per_subshard}")
    if max_subshards < 1:
        raise ValueError(f"max_subshards must be >= 1, got {max_subshards}")
    if default_subshards < 1:
        raise ValueError(f"default_subshards must be >= 1, got {default_subshards}")

    # Every source must share one quality model so bucket IDs are comparable.
    models = {(q.model_dir, q.calib_file, tuple(q.bucket_edges)) for q in quality.values()}
    if len(models) != 1:
        raise ValueError(f"build_clustered_store: sources span multiple quality models: {sorted(models)}")
    bucket_edges = next(iter(quality.values())).bucket_edges

    cluster_col = _validate_cluster_view(cluster_assign, cluster_view)
    subshards_for_bucket = _plan_subshards(
        bucket_token_hint=bucket_token_hint,
        target_tokens_per_subshard=target_tokens_per_subshard,
        max_subshards=max_subshards,
        default_subshards=default_subshards,
    )

    # Resolve the flat per-source-shard spec list.
    shard_specs: list[dict[str, str]] = []
    for source_name in sorted(tokenize):
        tok = tokenize[source_name]
        main_dir = tok.source_main_dirs.get(split)
        if main_dir is None:
            raise ValueError(f"{source_name}: tokenize has no source_main_dir for split={split!r}")
        cluster_asg = cluster_assign[source_name]
        if cluster_asg.source_main_dir != main_dir:
            raise ValueError(
                f"{source_name}: cluster_assign.source_main_dir={cluster_asg.source_main_dir!r} "
                f"!= tokenize.source_main_dirs[{split!r}]={main_dir!r}"
            )
        dedup_attr_dir = _resolve_dedup_attr_dir(source_name=source_name, main_output_dir=main_dir, dedup=dedup)
        shard_specs.extend(
            _per_source_shard_tuples(
                source_name=source_name,
                tokenize=tok,
                decontam=optional_decontam.get(source_name),
                cluster_assign=cluster_asg,
                quality=quality[source_name],
                dedup_attr_dir=dedup_attr_dir,
                split=split,
            )
        )
    if not shard_specs:
        raise ValueError("No input shards resolved -- nothing to do")

    batched_specs = [shard_specs[i : i + shards_per_task] for i in range(0, len(shard_specs), shards_per_task)]
    logger.info(
        "build_clustered_store: %d sources, %d input shards -> %d map tasks, reduce_shards=%d -> %s",
        len(tokenize),
        len(shard_specs),
        len(batched_specs),
        reduce_shards,
        output_path,
    )

    if worker_resources is None:
        # 16g: the reduce side streams big groups through a tensorstore write
        # buffer (~512 MB write-chunk) and the map holds numpy token payloads.
        worker_resources = ResourceConfig(cpu=2, ram="16g", disk="16g")

    ctx = ZephyrContext(
        client=client,
        resources=worker_resources,
        coordinator_resources=ResourceConfig(cpu=1, ram="3g", preemptible=False),
        max_workers=min(max_workers, len(batched_specs)),
        chunk_storage_prefix=chunk_storage_prefix,
        name="datakit-clustered-store",
    )
    subshards_for_bucket = _load_or_persist_plan(output_path, subshards_for_bucket)
    done_keys = _materialized_keys(output_path, split) if resume else frozenset()
    if done_keys:
        logger.info(
            "build_clustered_store: resuming with %d cells already materialized; their docs are "
            "dropped at the map so the shuffle shrinks as the store fills",
            len(done_keys),
        )
    ds = (
        Dataset.from_list(batched_specs)
        .map_shard(
            lambda items, shard, cc=cluster_col, sfb=subshards_for_bucket, ds=default_subshards, dk=done_keys: (
                _emit_for_shuffle(
                    items, shard, cluster_col=cc, subshards_for_bucket=sfb, default_subshards=ds, done_keys=dk
                )
            )
        )
        .group_by(
            key=lambda r: (r["cluster"], r["quality"], r["sub"]),
            reducer=lambda key, group, op=output_path, sp=split, rs=resume: _write_subshard_cache(
                key, group, output_path=op, split=sp, resume=rs
            ),
            num_output_shards=reduce_shards,
        )
    )
    outcome = ctx.execute(ds, verbose=True)
    subshard_stats = [r for r in outcome.results if r is not None]
    # Cells filtered at the map never reach a reducer, so their stats must be
    # read back from the sidecars or the artifact would silently omit them and
    # the store would look like it lost those cells entirely.
    for key in sorted(done_keys):
        stat = _read_subshard_stat(_cell_dir(output_path, key, split), key)
        if stat is None:
            raise RuntimeError(f"resume: {key} was filtered from the shuffle but its stat is unreadable")
        subshard_stats.append(stat)
    logger.info(
        "build_clustered_store: wrote %d subshard caches (records_out=%d, tokens_out=%d)",
        len(subshard_stats),
        outcome.counters.get("datakit_store/records_out", 0),
        outcome.counters.get("datakit_store/tokens_out", 0),
    )

    buckets = _finalize_buckets(subshard_stats=subshard_stats, output_path=output_path, split=split)

    tokenizer = next(iter(tokenize.values())).tokenizer
    artifact = ClusteredStoreData(
        cache_path=output_path,
        cluster_view=cluster_view,
        bucket_edges=bucket_edges,
        split=split,
        buckets=buckets,
        source_names=sorted(tokenize),
        tokenizer=tokenizer,
        counters=dict(outcome.counters),
    )
    write_artifact(artifact, output_path)
    return artifact
