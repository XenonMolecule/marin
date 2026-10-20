# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The dedup pipeline as a sequence of claimable stages.

Stage by stage (``U`` units, ``R_*`` partitions; every arrow is a partitioned shuffle object)::

    pass1      (U)    read kept; normalize, hash, MinHash; collapse byte-identical docs
                      inside the unit      -> exact records (with copy counts) by content hash,
                                              per-unit stash
    exact      (R_e)  global exact dedup: the minimum doc key per content hash is kept
                                           -> dropped doc keys by unit; kept keys' copy counts
    lshmap     (U)    stash minus exact drops = unique docs -> (bucket, doc) by bucket; reps list
                      (with each unique doc's exact copy count)
    bucket     (R_b)  every non-singleton bucket -> (member, hub) by node partition
    ccNNN      (R_n)  connected components, one Large-Star / Small-Star round per stage, until
                      the edge set is a star forest (a seal reports zero violations)
    canon_a    (R_n)  member -> component minimum; canonical order key per member
    canon_b    (R_n)  canonical member per component (minimum fuzzy_dups order); component
                      sizes -> every node partition holding members of the component
    attrs      (R_n)  per-unit member table: (doc, component, is_canonical, component size)
    apply_cc   (U)    re-read kept, write unique docs minus non-canonical members
    c075_text  (U)    member text keyed by cluster (oversized clusters split as upstream does)
    c075_solve (R_c)  containment rule per cluster -> removed doc keys by node partition
    c075_marks (R_n)  per-unit removal lists
    apply_c075 (U)    re-read kept, write unique docs minus containment removals
    report     (1)    aggregate counters into ``{output_root}/stats/rdedup_stats.json``

Every output document carries ``exact_copies`` (input documents with its content hash) and
``fuzzy_cluster_size`` (unique documents in its connected component; 1 outside any component).

Every task is deterministic: the same inputs give byte-identical outputs, so a zombie writer
after a claim takeover is harmless. Every stage reads only paths derived from the plan and the
sealed index of the stage before it, never a listing of data.
"""

from __future__ import annotations

import dataclasses
import gzip
import logging
import multiprocessing
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor

import dupekit
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from marin.processing.classification.deduplication.cluster_dedup import ClusterDocument, find_duplicates
from marin.processing.classification.deduplication.cluster_verify import (
    MAXIMUM_CLUSTER_CHARS as SOLVE_MAX_CLUSTER_CHARS,
)
from marin.processing.classification.deduplication.cluster_verify import (
    MAXIMUM_DOCUMENT_CHARS as SOLVE_MAX_DOCUMENT_CHARS,
)

from experiments.baseline_collection.verified_dedup import C075_PARAMS, MAX_CLUSTER_SIZE
from experiments.datakit.scripts.fuzzy_cluster_text import MAXIMUM_DOCUMENT_CHARS as TEXT_MAX_DOCUMENT_CHARS
from experiments.datakit.scripts.fuzzy_cluster_text import SPLIT_SUBDIVISIONS, _split_hash
from experiments.fast_curation.rdedup import blocks, docs, graph
from experiments.fast_curation.rdedup.plan import Plan, load_inputs
from experiments.fast_curation.rdedup.store import ObjectStore, delete_prefix, dumps, read_json, store_for
from experiments.fast_curation.rdedup.tasks import TaskBoard

logger = logging.getLogger(__name__)

# ``n``: documents of the unit collapsed into this representative.
EXACT_DTYPE = np.dtype([("hi", "<u8"), ("lo", "<u8"), ("d", "<u8"), ("n", "<u4")])
# A kept doc key and the input documents sharing its content hash, for counts above 1.
COPIES_DTYPE = np.dtype([("d", "<u8"), ("n", "<u4")])
STASH_DTYPE = np.dtype(
    [("d", "<u8"), ("hi", "<u8"), ("lo", "<u8"), ("chars", "<i8"), ("sig", "u1"), ("bk", "<u8", (docs.NUM_BANDS,))]
)
REPS_DTYPE = np.dtype([("d", "<u8"), ("hi", "<u8"), ("lo", "<u8"), ("chars", "<i8"), ("n", "<u4")])
MEMBER_DTYPE = np.dtype([("d", "<u8"), ("root", "<u8")])
CANON_DTYPE = np.dtype([("root", "<u8"), ("k1", "<u8"), ("k2", "<u8"), ("k3", "<u8"), ("d", "<u8"), ("n", "<i8")])
# Component sizes are u8: the 8M run's largest component had 4,557,202,931 members (> u4).
SIZE_DTYPE = np.dtype([("root", "<u8"), ("size", "<u8")])
ATTR_DTYPE = np.dtype([("d", "<u8"), ("root", "<u8"), ("canon", "u1"), ("size", "<u8")])
KEY_DTYPE = np.dtype("<u8")
GZIP_LEVEL = 6
EXACT_COPIES_FIELD = "exact_copies"
CLUSTER_SIZE_FIELD = "fuzzy_cluster_size"
# Component sizes are binned by powers of two in the report.
_SIZE_BINS = 40
_MAX_COUNT = np.iinfo(np.uint32).max


@dataclasses.dataclass(frozen=True)
class Ctx:
    """What a task needs: the store, the plan, and the sealed results of earlier stages."""

    store: ObjectStore
    plan: Plan
    sealed: Callable[[str], dict]

    def data(self, name: str) -> str:
        return f"{self.plan.run_root}/data/{name}"

    def stage_dir(self, name: str) -> str:
        return f"{self.plan.run_root}/stages/{name}"

    def done(self, stage: str, task: int) -> dict:
        return read_json(self.store, f"{self.stage_dir(stage)}/done/{task:06d}.json")

    def inputs(self) -> list[str]:
        return _cached_inputs(self.store, self.plan.run_root)


_INPUTS_CACHE: dict[str, list[str]] = {}


def _cached_inputs(store: ObjectStore, run_root: str) -> list[str]:
    if run_root not in _INPUTS_CACHE:
        _INPUTS_CACHE[run_root] = load_inputs(store, run_root)
    return _INPUTS_CACHE[run_root]


@dataclasses.dataclass(frozen=True)
class StageSpec:
    """One stage: ``n_tasks`` claimable tasks, then a seal that sees every task's result."""

    name: str
    n_tasks: int
    run: Callable[[Ctx, int], dict]
    seal: Callable[[Ctx, list[dict]], dict]
    procs: int
    mem_gb: float
    deadline_hours: float

    def slots(self, cpus: int, mem_gb: float) -> int:
        """Concurrent tasks a worker of this shape can run for this stage."""
        return max(1, min(cpus // self.procs, int(mem_gb // self.mem_gb)))


def _pool(procs: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn"))


def _map_ordered(fn: Callable, args: list[tuple], procs: int) -> Iterator:
    """``fn(*a)`` for each ``a``, in order; in a spawn pool when ``procs > 1``."""
    if procs <= 1 or len(args) <= 1:
        for a in args:
            yield fn(*a)
        return
    with _pool(min(procs, len(args))) as pool:
        futures = [pool.submit(fn, *a) for a in args]
        for f in futures:
            yield f.result()


def _sum(results: Iterable[dict], key: str) -> int:
    return int(sum(r[key] for r in results))


def _as_count(values: np.ndarray, what: str) -> np.ndarray:
    """``values`` as u4 counts; a count beyond u4 fails loudly instead of wrapping."""
    if len(values) and int(values.max()) > _MAX_COUNT:
        raise RuntimeError(f"{what} {int(values.max())} does not fit a u4 count")
    return values.astype(np.uint32)


def _first_of_runs(*keys: np.ndarray) -> np.ndarray:
    """For arrays sorted by ``keys`` (primary last), mask of the first row of each equal run."""
    n = len(keys[0])
    mask = np.ones(n, dtype=bool)
    if n > 1:
        same = np.ones(n - 1, dtype=bool)
        for k in keys:
            same &= k[1:] == k[:-1]
        mask[1:] = ~same
    return mask


# ---------------------------------------------------------------------------------------- pass1


def _pass1_file(path: str, file_index: int) -> docs.FileDocs:
    return docs.pass1_file(store_for(path), path, file_index)


def run_pass1(ctx: Ctx, unit: int, procs: int) -> dict:
    plan = ctx.plan
    paths = ctx.inputs()
    files = list(plan.unit_files(unit))
    parts = list(_map_ordered(_pass1_file, [(paths[f], f) for f in files], procs))
    keys = np.concatenate([docs.doc_keys(p.file_index, p.rows) for p in parts])
    hi = np.concatenate([p.id_hi for p in parts])
    lo = np.concatenate([p.id_lo for p in parts])
    chars = np.concatenate([p.chars for p in parts])
    sig = np.concatenate([p.has_signature for p in parts])
    buckets = np.concatenate([p.buckets for p in parts]) if parts else np.zeros((0, docs.NUM_BANDS), np.uint64)

    # Keys increase in file-then-row order, so the first key of each content hash is the unit's
    # earliest copy -- and the global rule keeps the earliest copy overall.
    order = np.lexsort((keys, lo, hi))
    starts = np.flatnonzero(_first_of_runs(hi[order], lo[order]))
    by_key = np.argsort(order[starts])
    reps = order[starts][by_key]
    copies = np.diff(np.append(starts, len(order)))[by_key]

    exact = np.empty(len(reps), dtype=EXACT_DTYPE)
    exact["hi"], exact["lo"], exact["d"] = hi[reps], lo[reps], keys[reps]
    exact["n"] = _as_count(copies, "in-unit copies")
    writer = blocks.BlockObjectWriter(plan.exact_partitions)
    writer.add_array(exact, (exact["hi"] % np.uint64(plan.exact_partitions)).astype(np.int64))
    exact_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("pass1_exact"), unit))

    stash = np.empty(len(reps), dtype=STASH_DTYPE)
    stash["d"], stash["hi"], stash["lo"] = keys[reps], hi[reps], lo[reps]
    stash["chars"], stash["sig"], stash["bk"] = chars[reps], sig[reps], buckets[reps]
    stash_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("stash"), unit), stash)
    return {
        "exact": exact_meta,
        "stash": stash_meta,
        "files": len(files),
        "rows": sum(p.n_rows for p in parts),
        "kept": len(keys),
        "compacted": sum(p.compacted for p in parts),
        "unit_dups": len(keys) - len(reps),
        "unit_reps": len(reps),
        "no_signature": int((~sig[reps]).sum()),
        "chars_kept": int(chars.sum()),
    }


def seal_pass1(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("pass1_exact"), [r["exact"] for r in results], ctx.plan.exact_partitions
    )
    totals = {
        k: _sum(results, k)
        for k in ("files", "rows", "kept", "compacted", "unit_dups", "unit_reps", "no_signature", "chars_kept")
    }
    if totals["files"] != ctx.plan.n_files:
        raise RuntimeError(f"pass1 covered {totals['files']} files, plan has {ctx.plan.n_files}")
    return {
        **totals,
        "empty_text_filtered": totals["rows"] - totals["kept"],
        "exact_index": index,
        "stash_bytes": _sum((r["stash"] for r in results), "bytes"),
    }


# ---------------------------------------------------------------------------------------- exact


def run_exact(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    records = blocks.read_partition_array(
        ctx.store, ctx.data("pass1_exact"), partition, plan.n_units, plan.exact_partitions, EXACT_DTYPE
    )
    order = np.lexsort((records["d"], records["lo"], records["hi"]))
    first = _first_of_runs(records["hi"][order], records["lo"][order])
    dropped = np.sort(records["d"][order][~first])
    writer = blocks.BlockObjectWriter(plan.n_units)
    writer.add_array(dropped, plan.unit_of_keys(dropped))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("exact"), partition))

    starts = np.flatnonzero(first)
    n = records["n"][order].astype(np.int64)
    totals = np.add.reduceat(n, starts) if len(n) else np.zeros(0, dtype=np.int64)
    multi = totals > 1
    copies = np.empty(int(multi.sum()), dtype=COPIES_DTYPE)
    copies["d"], copies["n"] = records["d"][order][starts][multi], _as_count(totals[multi], "exact copies")
    copies = np.sort(copies, order="d")
    writer = blocks.BlockObjectWriter(plan.n_units)
    writer.add_array(copies, plan.unit_of_keys(copies["d"]))
    copies_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("exact_copies"), partition))
    return {
        "blocks": meta,
        "copies_blocks": copies_meta,
        "records": len(records),
        "unique": int(first.sum()),
        "cross_unit_dups": len(dropped),
        "exact_copies": int(n.sum()),
        "multi_copy_docs": len(copies),
    }


def seal_exact(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("exact"), [r["blocks"] for r in results], ctx.plan.n_units)
    copies_index = blocks.build_index(
        ctx.store, ctx.data("exact_copies"), [r["copies_blocks"] for r in results], ctx.plan.n_units
    )
    pass1 = ctx.sealed("pass1")
    unique = _sum(results, "unique")
    cross = _sum(results, "cross_unit_dups")
    if unique + cross != pass1["unit_reps"]:
        raise RuntimeError(f"exact: {unique} unique + {cross} dropped != {pass1['unit_reps']} unit reps")
    copies = _sum(results, "exact_copies")
    if copies != pass1["kept"]:
        raise RuntimeError(f"exact: copy counts sum to {copies}, pass1 kept {pass1['kept']} records")
    return {
        "unique": unique,
        "cross_unit_dups": cross,
        "exact_dups": cross + pass1["unit_dups"],
        "multi_copy_docs": _sum(results, "multi_copy_docs"),
        "index": index,
        "copies_index": copies_index,
    }


# --------------------------------------------------------------------------------------- lshmap


def _read_stash(ctx: Ctx, unit: int) -> np.ndarray:
    return blocks.read_single_block_array(ctx.store, ctx.done("pass1", unit)["stash"], STASH_DTYPE)


def run_lshmap(ctx: Ctx, unit: int) -> dict:
    plan = ctx.plan
    stash = _read_stash(ctx, unit)
    dropped = blocks.read_partition_array(
        ctx.store, ctx.data("exact"), unit, plan.exact_partitions, plan.n_units, KEY_DTYPE
    )
    unique = stash[~np.isin(stash["d"], dropped)]
    if len(unique) != len(stash) - len(dropped):
        raise RuntimeError(f"unit {unit}: {len(dropped)} exact drops but {len(stash) - len(unique)} matched")
    reps = np.empty(len(unique), dtype=REPS_DTYPE)
    for field in ("d", "hi", "lo", "chars"):
        reps[field] = unique[field]
    copies = blocks.read_partition_array(
        ctx.store, ctx.data("exact_copies"), unit, plan.exact_partitions, plan.n_units, COPIES_DTYPE
    )
    at = np.searchsorted(reps["d"], copies["d"])
    if len(copies) and (at.max() >= len(reps) or np.any(reps["d"][at] != copies["d"])):
        raise RuntimeError(f"unit {unit}: exact copy counts for doc keys that are not unique docs of the unit")
    reps["n"] = 1
    reps["n"][at] = copies["n"]
    reps_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("reps"), unit), reps)

    signed = unique[unique["sig"] == 1]
    lsh = np.empty(len(signed) * docs.NUM_BANDS, dtype=graph.LSH_DTYPE)
    lsh["b"] = signed["bk"].reshape(-1)
    lsh["d"] = np.repeat(signed["d"], docs.NUM_BANDS)
    writer = blocks.BlockObjectWriter(plan.lsh_partitions)
    writer.add_array(lsh, (lsh["b"] % np.uint64(plan.lsh_partitions)).astype(np.int64))
    lsh_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("lshmap"), unit))
    return {
        "lsh": lsh_meta,
        "reps": reps_meta,
        "unique": len(unique),
        "signed": len(signed),
        "unique_chars": int(unique["chars"].sum()),
        "exact_copies": int(reps["n"].sum(dtype=np.int64)),
    }


def seal_lshmap(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("lshmap"), [r["lsh"] for r in results], ctx.plan.lsh_partitions)
    unique = _sum(results, "unique")
    if unique != ctx.sealed("exact")["unique"]:
        raise RuntimeError(f"lshmap kept {unique} unique docs, exact stage counted {ctx.sealed('exact')['unique']}")
    copies = _sum(results, "exact_copies")
    if copies != ctx.sealed("pass1")["kept"]:
        raise RuntimeError(f"unique docs' exact copies sum to {copies}, pass1 kept {ctx.sealed('pass1')['kept']}")
    deleted = sum(
        delete_prefix(ctx.store, ctx.data(name) + "/") for name in ("stash", "pass1_exact", "exact", "exact_copies")
    )
    return {
        "unique": unique,
        "signed": _sum(results, "signed"),
        "unique_chars": _sum(results, "unique_chars"),
        "index": index,
        "deleted_objects": deleted,
    }


# --------------------------------------------------------------------------------------- bucket


def run_bucket(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    lsh = blocks.read_partition_array(
        ctx.store, ctx.data("lshmap"), partition, plan.n_units, plan.lsh_partitions, graph.LSH_DTYPE
    )
    pairs, stats = graph.bucket_hub_pairs(lsh)
    del lsh
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(pairs, plan.node_partition_of_keys(pairs["x"]))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("bucket"), partition))
    return {"blocks": meta, **dataclasses.asdict(stats)}


def seal_bucket(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(ctx.store, ctx.data("bucket"), [r["blocks"] for r in results], ctx.plan.node_partitions)
    totals = {k: _sum(results, k) for k in ("pairs", "buckets", "multi_buckets", "multi_members")}
    deleted = delete_prefix(ctx.store, ctx.data("lshmap") + "/")
    return {**totals, "largest_bucket": max(r["largest"] for r in results), "index": index, "deleted_objects": deleted}


# ------------------------------------------------------------------------------ components
#
# Round k runs three stages over the node partitions (see graph.py):
#   cc{k}a  apply the candidates of round k-1 (round 0: build edges and initial labels from the
#           bucket pairs); write labels; request the labels of every hub and current label
#   cc{k}b  answer the requests addressed to this partition's nodes
#   cc{k}c  from the answers, emit min-combined candidates to their nodes' partitions
# A round whose ``a`` stage changes no label has converged; its labels are final.

LABEL_DTYPE = np.dtype([("node", "<u8"), ("label", "<u8")])
CANDIDATE_DTYPE = np.dtype([("t", "<u8"), ("c", "<u8")])


def cc_name(round_index: int, step: str) -> str:
    return f"cc{round_index:03d}{step}"


def _labels_path(ctx: Ctx, round_index: int, partition: int) -> str:
    return blocks.map_object_path(ctx.data(f"cc_labels/{round_index:03d}"), partition)


def _load_edges(ctx: Ctx, partition: int) -> np.ndarray:
    meta = ctx.done(cc_name(0, "a"), partition)["edges"]
    return blocks.read_single_block_array(ctx.store, meta, graph.HUB_DTYPE)


def _load_labels(ctx: Ctx, round_index: int, partition: int) -> np.ndarray:
    meta = ctx.done(cc_name(round_index, "a"), partition)["labels"]
    return blocks.read_single_block_array(ctx.store, meta, LABEL_DTYPE)


def run_cc_a(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    out: dict = {}
    if round_index == 0:
        pairs = blocks.read_partition_array(
            ctx.store, ctx.data("bucket"), partition, plan.lsh_partitions, n, graph.HUB_DTYPE
        )
        g, labels = graph.initial_graph(pairs)
        edges = np.empty(len(g.ex), dtype=graph.HUB_DTYPE)
        edges["x"], edges["h"] = g.ex, g.eh
        out["edges"] = blocks.write_single_block(
            ctx.store, blocks.map_object_path(ctx.data("cc_edges"), partition), edges
        )
        out.update(in_records=len(pairs), edge_count=len(edges), changed=-1)
        nodes = g.nodes
    else:
        previous = _load_labels(ctx, round_index - 1, partition)
        nodes = previous["node"]
        labels = previous["label"].copy()
        received = 0
        stream = blocks.iter_partition_blocks(ctx.store, ctx.data(cc_name(round_index - 1, "c")), partition, n, n)
        for _m, block in stream:
            for frame in blocks.iter_frames(block):
                cand = np.frombuffer(frame, dtype=CANDIDATE_DTYPE)
                graph.apply_candidates(nodes, labels, cand["t"], cand["c"])
                received += len(cand)
        changed = int(np.count_nonzero(labels != previous["label"]))
        out.update(in_records=received, changed=changed)
        g = graph.PartitionGraph(ex=np.empty(0, np.uint64), eh=_load_edges(ctx, partition)["h"], nodes=nodes)
    state = np.empty(len(nodes), dtype=LABEL_DTYPE)
    state["node"], state["label"] = nodes, labels
    out["labels"] = blocks.write_single_block(ctx.store, _labels_path(ctx, round_index, partition), state)
    targets = graph.request_targets(g, labels)
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(targets, plan.node_partition_of_keys(targets))
    out["blocks"] = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "a")), partition))
    out.update(nodes=len(nodes), requests=len(targets))
    return out


def run_cc_b(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    state = _load_labels(ctx, round_index, partition)
    stream = blocks.iter_partition_blocks(ctx.store, ctx.data(cc_name(round_index, "a")), partition, n, n)
    writer = blocks.BlockObjectWriter(n)
    answered = 0
    for requester, block in stream:
        targets = blocks.concat_arrays([np.frombuffer(f, dtype=KEY_DTYPE) for f in blocks.iter_frames(block)], KEY_DTYPE)
        if len(targets) == 0:
            continue
        answers = np.empty(len(targets), dtype=LABEL_DTYPE)
        answers["node"], answers["label"] = targets, graph.lookup(state["node"], state["label"], targets)
        writer.add_array(answers, np.full(len(targets), requester, dtype=np.int64))
        answered += len(targets)
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "b")), partition))
    return {"blocks": meta, "answered": answered}


def run_cc_c(ctx: Ctx, partition: int, round_index: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    state = _load_labels(ctx, round_index, partition)
    edges = _load_edges(ctx, partition)
    g = graph.PartitionGraph(ex=edges["x"], eh=edges["h"], nodes=state["node"])
    labels = state["label"]
    answers = blocks.read_partition_array(ctx.store, ctx.data(cc_name(round_index, "b")), partition, n, n, LABEL_DTYPE)
    order = np.argsort(answers["node"])
    targets, values = graph.candidates(g, labels, answers["node"][order], answers["label"][order])
    cand = np.empty(len(targets), dtype=CANDIDATE_DTYPE)
    cand["t"], cand["c"] = targets, values
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(cand, plan.node_partition_of_keys(targets))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data(cc_name(round_index, "c")), partition))
    return {"blocks": meta, "candidates": len(cand)}


def seal_cc(ctx: Ctx, results: list[dict], round_index: int, step: str) -> dict:
    n = ctx.plan.node_partitions
    name = cc_name(round_index, step)
    index = blocks.build_index(ctx.store, ctx.data(name), [r["blocks"] for r in results], n)
    counts = {k: _sum(results, k) for k in results[0] if isinstance(results[0][k], int) and not k.startswith("_")}
    out = {"round": round_index, "step": step, **counts, "index": index}
    # Each round's inputs are dead once the next step has sealed.
    stale = {
        "a": [cc_name(round_index - 1, "c")] if round_index else ["bucket"],
        "b": [],
        "c": [name[:-1] + "a", name[:-1] + "b"],
    }
    out["deleted_objects"] = sum(delete_prefix(ctx.store, ctx.data(s) + "/") for s in stale[step])
    if step == "a" and round_index >= 2:
        out["deleted_objects"] += delete_prefix(ctx.store, ctx.data(f"cc_labels/{round_index - 2:03d}") + "/")
    if step == "a":
        out["converged"] = round_index > 0 and counts["changed"] == 0
    logger.info("components %s sealed: %s", name, {k: v for k, v in out.items() if k != "index"})
    return out


# ------------------------------------------------------------------------------- canonical


def _load_reps(ctx: Ctx, units: Iterable[int]) -> np.ndarray:
    pieces = [blocks.read_single_block_array(ctx.store, ctx.done("lshmap", u)["reps"], REPS_DTYPE) for u in units]
    return blocks.concat_arrays(pieces, REPS_DTYPE)


def run_canon_a(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    final = ctx.sealed("components")["final_round"]
    state = _load_labels(ctx, final, partition)
    nodes, root = state["node"], state["label"]
    members = np.empty(len(nodes), dtype=MEMBER_DTYPE)
    members["d"], members["root"] = nodes, root
    members_meta = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("members"), partition), members)

    reps = _load_reps(ctx, plan.node_partition_units(partition))
    at = np.searchsorted(reps["d"], nodes)
    if len(nodes) and (at.max() >= len(reps) or np.any(reps["d"][at] != nodes)):
        raise RuntimeError(f"partition {partition}: component members missing from the unique-doc lists")
    k1, k2, k3 = docs.canonical_order_keys(reps["hi"][at], reps["lo"][at])
    # Combine on the map side: only this partition's best member per component (and the member
    # count) crosses the shuffle, so a giant component costs one record per partition, not one
    # per member.
    order = np.lexsort((nodes, k3, k2, k1, root))
    first = _first_of_runs(root[order])
    starts = np.flatnonzero(first)
    best = order[starts]
    records = np.empty(len(best), dtype=CANON_DTYPE)
    records["root"], records["k1"], records["k2"], records["k3"] = root[best], k1[best], k2[best], k3[best]
    records["d"], records["n"] = nodes[best], np.diff(np.append(starts, len(order)))
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(records, plan.node_partition_of_keys(records["root"]))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_a"), partition))
    return {
        "blocks": meta,
        "members_meta": members_meta,
        "members": len(nodes),
        "member_chars": int(reps["chars"][at].sum()),
    }


def seal_canon_a(ctx: Ctx, results: list[dict]) -> dict:
    n = ctx.plan.node_partitions
    index = blocks.build_index(ctx.store, ctx.data("canon_a"), [r["blocks"] for r in results], n)
    deleted = sum(delete_prefix(ctx.store, ctx.data(name) + "/") for name in ("cc_edges", "cc_labels"))
    return {
        "members": _sum(results, "members"),
        "member_chars": _sum(results, "member_chars"),
        "index": index,
        "deleted_objects": deleted,
    }


def run_canon_b(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    pieces, sources = [], []
    for m, block in blocks.iter_partition_blocks(ctx.store, ctx.data("canon_a"), partition, n, n):
        for frame in blocks.iter_frames(block):
            pieces.append(np.frombuffer(frame, dtype=CANON_DTYPE))
            sources.append(np.full(len(pieces[-1]), m, dtype=np.int64))
    r = blocks.concat_arrays(pieces, CANON_DTYPE)
    # The node partition each record came from: it holds members of the record's component.
    source = np.concatenate(sources) if sources else np.zeros(0, dtype=np.int64)
    order = np.lexsort((r["d"], r["k3"], r["k2"], r["k1"], r["root"]))
    r, source = r[order], source[order]
    first = _first_of_runs(r["root"])
    starts = np.flatnonzero(first)
    sizes = np.add.reduceat(r["n"], starts) if len(r) else np.zeros(0, dtype=np.int64)
    canonical = np.sort(r["d"][starts])
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(canonical, plan.node_partition_of_keys(canonical))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_b"), partition))
    size_records = np.empty(len(r), dtype=SIZE_DTYPE)
    size_records["root"] = r["root"]
    size_records["size"] = np.repeat(sizes, np.diff(np.append(starts, len(r))))
    writer = blocks.BlockObjectWriter(n)
    writer.add_array(size_records, source)
    sizes_meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("canon_sizes"), partition))
    big = sizes > MAX_CLUSTER_SIZE
    histogram = (
        np.bincount(np.floor(np.log2(sizes)).astype(np.int64), minlength=_SIZE_BINS)
        if len(sizes)
        else np.zeros(_SIZE_BINS)
    )
    return {
        "blocks": meta,
        "sizes_blocks": sizes_meta,
        "components": len(starts),
        "members": int(sizes.sum()),
        "oversized": {str(k): int(s) for k, s in zip(r["root"][starts][big].tolist(), sizes[big].tolist(), strict=True)},
        "largest": int(sizes.max()) if len(sizes) else 0,
        "size_log2_histogram": [int(c) for c in histogram],
    }


def seal_canon_b(ctx: Ctx, results: list[dict]) -> dict:
    n = ctx.plan.node_partitions
    index = blocks.build_index(ctx.store, ctx.data("canon_b"), [r["blocks"] for r in results], n)
    sizes_index = blocks.build_index(ctx.store, ctx.data("canon_sizes"), [r["sizes_blocks"] for r in results], n)
    oversized: dict[str, int] = {}
    for r in results:
        oversized.update(r["oversized"])
    histogram = np.sum([r["size_log2_histogram"] for r in results], axis=0)
    ctx.store.write(f"{ctx.data('oversized')}.json", dumps(oversized))
    largest = sorted(oversized.values(), reverse=True)[:25]
    return {
        "components": _sum(results, "components"),
        "members": _sum(results, "members"),
        "largest": max(r["largest"] for r in results),
        "largest_sizes": largest,
        "oversized": len(oversized),
        "oversized_members": int(sum(oversized.values())),
        "size_log2_histogram": [int(c) for c in histogram],
        "index": index,
        "sizes_index": sizes_index,
    }


def run_attrs(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    n = plan.node_partitions
    members_meta = ctx.done("canon_a", partition)["members_meta"]
    members = blocks.read_single_block_array(ctx.store, members_meta, MEMBER_DTYPE)
    canonical = blocks.read_partition_array(ctx.store, ctx.data("canon_b"), partition, n, n, KEY_DTYPE)
    attrs = np.empty(len(members), dtype=ATTR_DTYPE)
    attrs["d"], attrs["root"] = members["d"], members["root"]
    attrs["canon"] = np.isin(members["d"], canonical)
    if int(attrs["canon"].sum()) != len(canonical):
        raise RuntimeError(f"partition {partition}: {len(canonical)} canonicals but {int(attrs['canon'].sum())} matched")
    sizes = np.sort(
        blocks.read_partition_array(ctx.store, ctx.data("canon_sizes"), partition, n, n, SIZE_DTYPE), order="root"
    )
    if not np.array_equal(sizes["root"], np.unique(members["root"])):
        raise RuntimeError(f"partition {partition}: component sizes do not cover each member component exactly once")
    attrs["size"] = sizes["size"][np.searchsorted(sizes["root"], members["root"])]
    units = plan.unit_of_keys(attrs["d"])
    written = {}
    for unit in plan.node_partition_units(partition):
        mine = np.sort(attrs[units == unit], order="d")
        written[str(unit)] = blocks.write_single_block(ctx.store, blocks.map_object_path(ctx.data("attrs"), unit), mine)
    return {
        "units": written,
        "members": len(attrs),
        "canonicals": len(canonical),
        "canonical_sizes": int(attrs["size"][attrs["canon"] == 1].sum(dtype=np.int64)),
    }


def seal_attrs(ctx: Ctx, results: list[dict]) -> dict:
    members = _sum(results, "members")
    if members != ctx.sealed("canon_b")["members"]:
        raise RuntimeError(f"attrs wrote {members} members, canon_b counted {ctx.sealed('canon_b')['members']}")
    # Each component has one canonical, so its recorded size is counted exactly once.
    canonical_sizes = _sum(results, "canonical_sizes")
    if canonical_sizes != members:
        raise RuntimeError(f"attrs: canonicals' component sizes sum to {canonical_sizes}, not {members} members")
    deleted = sum(
        delete_prefix(ctx.store, ctx.data(name) + "/") for name in ("canon_a", "canon_b", "canon_sizes", "members")
    )
    return {"members": members, "canonicals": _sum(results, "canonicals"), "deleted_objects": deleted}


def _attrs_of_unit(ctx: Ctx, unit: int) -> np.ndarray:
    partition = unit // ctx.plan.units_per_node_partition
    meta = ctx.done("attrs", partition)["units"][str(unit)]
    return blocks.read_single_block_array(ctx.store, meta, ATTR_DTYPE)


# ------------------------------------------------------------------------------------ apply


def _apply_file(
    path: str, rows: np.ndarray, hi: np.ndarray, lo: np.ndarray, copies: np.ndarray, cluster_sizes: np.ndarray
) -> tuple[bytes, int, int]:
    """Gzip member holding the surviving rows of one kept file: the reference's records plus
    ``exact_copies`` and ``fuzzy_cluster_size``."""
    table = docs.read_kept(store_for(path), path)
    records, texts = docs.output_records(table, rows)
    for record, text, h, l, n, size in zip(
        records, texts, hi.tolist(), lo.tolist(), copies.tolist(), cluster_sizes.tolist(), strict=True
    ):
        if docs.content_id(text) != (h << 64) | l:
            raise RuntimeError(f"{path}: a surviving row's text no longer hashes to its pass-1 id")
        record[EXACT_COPIES_FIELD] = n
        record[CLUSTER_SIZE_FIELD] = size
    payload = docs.encode_jsonl(records)
    return gzip.compress(payload, compresslevel=GZIP_LEVEL, mtime=0), len(records), sum(len(t) for t in texts)


def output_path(ctx: Ctx, rule: str, unit: int, chunk: int) -> str:
    folder = "deduped" if rule == "cc_canonical" else f"deduped_{rule}"
    return f"{ctx.plan.output_root}/{folder}/data-{unit:06d}-{chunk:03d}.jsonl.gz"


def _cluster_sizes(keys: np.ndarray, attrs: np.ndarray) -> np.ndarray:
    """Component size of each doc key of a unit; 1 for docs in no component."""
    sizes = np.ones(len(keys), dtype=np.uint64)
    at = np.searchsorted(attrs["d"], keys)
    member = at < len(attrs)
    member[member] = attrs["d"][at[member]] == keys[member]
    sizes[member] = attrs["size"][at[member]]
    return sizes


def _write_survivors(ctx: Ctx, unit: int, survivors: np.ndarray, attrs: np.ndarray, rule: str, procs: int) -> dict:
    """Write ``survivors`` (a REPS_DTYPE slice sorted by doc key) of ``unit`` as JSONL outputs."""
    plan = ctx.plan
    cluster_sizes = _cluster_sizes(survivors["d"], attrs)
    paths = ctx.inputs()
    files = list(plan.unit_files(unit))
    file_of = docs.key_file(survivors["d"]).astype(np.int64)
    bounds = np.searchsorted(file_of, np.asarray([*files, files[-1] + 1], dtype=np.int64))
    tasks, owners = [], []
    for i, f in enumerate(files):
        s = survivors[bounds[i] : bounds[i + 1]]
        if len(s):
            sizes = cluster_sizes[bounds[i] : bounds[i + 1]]
            tasks.append((paths[f], docs.key_row(s["d"]).astype(np.int64), s["hi"], s["lo"], s["n"], sizes))
            owners.append(i // plan.files_per_output)
    written = []
    docs_out = chars_out = 0
    chunks: dict[int, list[bytes]] = {}
    for owner, (member, n_docs, n_chars) in zip(owners, _map_ordered(_apply_file, tasks, procs), strict=True):
        chunks.setdefault(owner, []).append(member)
        docs_out += n_docs
        chars_out += n_chars
    for chunk, members in sorted(chunks.items()):
        path = output_path(ctx, rule, unit, chunk)
        ctx.store.write(path, b"".join(members))
        written.append(path)
    if docs_out != len(survivors):
        raise RuntimeError(f"unit {unit}: wrote {docs_out} docs for {len(survivors)} survivors")
    return {"docs": docs_out, "chars": chars_out, "outputs": written}


def _unit_reps(ctx: Ctx, unit: int) -> np.ndarray:
    return blocks.read_single_block_array(ctx.store, ctx.done("lshmap", unit)["reps"], REPS_DTYPE)


def run_apply_cc(ctx: Ctx, unit: int, procs: int) -> dict:
    reps = _unit_reps(ctx, unit)
    attrs = _attrs_of_unit(ctx, unit)
    dropped = attrs["d"][attrs["canon"] == 0]
    survivors = reps[~np.isin(reps["d"], dropped)]
    return _write_survivors(ctx, unit, survivors, attrs, "cc_canonical", procs)


def seal_apply(ctx: Ctx, results: list[dict], rule: str) -> dict:
    return {
        "rule": rule,
        "docs": _sum(results, "docs"),
        "chars": _sum(results, "chars"),
        "outputs": sum(len(r["outputs"]) for r in results),
    }


# ------------------------------------------------------------------------------------- c075


TEXT_SCHEMA = pa.schema(
    [
        pa.field("cluster_key", pa.string()),
        pa.field("id", pa.string()),
        pa.field("text", pa.large_string()),
        pa.field("d", pa.uint64()),
    ]
)


def cluster_key(root: int, text: str, doc_id: str, splits: int) -> str:
    """The upstream grouping key: the cluster, or for an oversized cluster a split of it.

    Oversized clusters split by one MinHash permutation of the document's word 5-grams, then
    uniformly by content id, exactly as ``fuzzy_cluster_text`` does.
    """
    if splits <= 1:
        return str(root)
    split_index = _split_hash(text) % splits
    sub_index = dupekit.hash_xxh3_64(doc_id.encode()) % SPLIT_SUBDIVISIONS
    return f"{root}:{split_index:04d}:{sub_index:02d}"


def _c075_text_file(
    path: str, rows: np.ndarray, keys: np.ndarray, roots: np.ndarray, splits: np.ndarray, n_partitions: int
) -> tuple[pa.Table, np.ndarray]:
    table = docs.read_kept(store_for(path), path, ["text"])
    texts = table.column("text").take(pa.array(rows, type=pa.int64())).to_pylist()
    out_keys, out_ids, out_texts = [], [], []
    for text, root, n_splits in zip(texts, roots.tolist(), splits.tolist(), strict=True):
        text = docs.compact_whitespace(text)
        doc_id = format(docs.content_id(text), "032x")
        text = text[:TEXT_MAX_DOCUMENT_CHARS]
        out_keys.append(cluster_key(root, text, doc_id, n_splits))
        out_ids.append(doc_id)
        out_texts.append(text)
    partitions = np.fromiter(
        (dupekit.hash_xxh3_64(k.encode()) % n_partitions for k in out_keys), dtype=np.int64, count=len(out_keys)
    )
    out = pa.table(
        {"cluster_key": out_keys, "id": out_ids, "text": pa.array(out_texts, type=pa.large_string()), "d": keys},
        schema=TEXT_SCHEMA,
    )
    return out, partitions


# Map-side text is flushed into partition frames in batches of this many characters, which keeps
# frames large enough that per-frame overhead stays negligible.
_C075_FLUSH_CHARS = 512 * 1024 * 1024


def run_c075_text(ctx: Ctx, unit: int, procs: int) -> dict:
    plan = ctx.plan
    paths = ctx.inputs()
    attrs = _attrs_of_unit(ctx, unit)
    oversized = read_json(ctx.store, f"{ctx.data('oversized')}.json")
    splits_by_root = {int(k): -(-size // MAX_CLUSTER_SIZE) for k, size in oversized.items()}
    splits = np.fromiter((splits_by_root.get(r, 1) for r in attrs["root"].tolist()), dtype=np.int64, count=len(attrs))
    file_of = docs.key_file(attrs["d"]).astype(np.int64)
    tasks = []
    for f in plan.unit_files(unit):
        mine = file_of == f
        if mine.any():
            a = attrs[mine]
            tasks.append(
                (paths[f], docs.key_row(a["d"]).astype(np.int64), a["d"], a["root"], splits[mine], plan.c075_partitions)
            )
    writer = blocks.BlockObjectWriter(plan.c075_partitions)
    pending: list[tuple[pa.Table, np.ndarray]] = []
    pending_chars = members = chars = 0

    def flush() -> None:
        nonlocal pending_chars
        if pending:
            writer.add_table(pa.concat_tables([t for t, _ in pending]), np.concatenate([p for _, p in pending]))
            pending.clear()
        pending_chars = 0

    for table, partitions in _map_ordered(_c075_text_file, tasks, procs):
        n_chars = int(pc.sum(pc.utf8_length(table.column("text"))).as_py() or 0)
        pending.append((table, partitions))
        pending_chars += n_chars
        members += table.num_rows
        chars += n_chars
        if pending_chars > _C075_FLUSH_CHARS:
            flush()
    flush()
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("c075_text"), unit))
    if members != len(attrs):
        raise RuntimeError(f"unit {unit}: emitted {members} member texts for {len(attrs)} members")
    return {"blocks": meta, "members": members, "chars": chars}


def seal_c075_text(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("c075_text"), [r["blocks"] for r in results], ctx.plan.c075_partitions
    )
    return {"members": _sum(results, "members"), "chars": _sum(results, "chars"), "index": index}


@dataclasses.dataclass
class SolveStats:
    documents: int = 0
    text_chars: int = 0
    clusters: int = 0
    singleton_groups: int = 0
    oversized_clusters: int = 0
    mid_cluster_flushes: int = 0
    truncated_documents: int = 0
    duplicates: int = 0


def solve_sorted_rows(rows: Iterable[dict], stats: SolveStats) -> Iterator[int]:
    """Doc keys the containment rule removes, from rows sorted by ``(cluster_key, id)``.

    A line-for-line port of ``cluster_verify.solve_text_shard``'s loop -- the per-document cap,
    the byte-bounded batches, and the mid-cluster flush -- so each cluster is solved in the same
    member order with the same batch boundaries.
    """
    params = C075_PARAMS

    def solve_batch(members: list[dict]) -> Iterator[int]:
        if len(members) < 2:
            return
        cluster = [
            ClusterDocument(id=m["id"], text=m["text"], file_idx=0, source_tag=docs.CC_SOURCE_TAG) for m in members
        ]
        for removal in find_duplicates(cluster, params):
            stats.duplicates += 1
            yield members[removal.member_index]["d"]

    def batches(members: list[dict]) -> Iterator[list[dict]]:
        if sum(len(m["text"]) for m in members) <= SOLVE_MAX_CLUSTER_CHARS:
            yield members
            return
        stats.oversized_clusters += 1
        ordered = sorted(members, key=lambda m: -len(m["text"]))
        batch: list[dict] = []
        size = 0
        for m in ordered:
            if batch and size + len(m["text"]) > SOLVE_MAX_CLUSTER_CHARS:
                yield batch
                batch, size = [], 0
            batch.append(m)
            size += len(m["text"])
        if batch:
            yield batch

    def solve(members: list[dict]) -> Iterator[int]:
        stats.clusters += 1
        if len(members) < 2:
            stats.singleton_groups += 1
            return
        for chunk in batches(members):
            yield from solve_batch(chunk)

    pending: list[dict] = []
    pending_chars = 0
    current: str | None = None
    for row in rows:
        stats.documents += 1
        stats.text_chars += len(row["text"])
        if len(row["text"]) > SOLVE_MAX_DOCUMENT_CHARS:
            stats.truncated_documents += 1
            row["text"] = row["text"][:SOLVE_MAX_DOCUMENT_CHARS]
        if row["cluster_key"] != current:
            if pending:
                yield from solve(pending)
            pending = []
            pending_chars = 0
            current = row["cluster_key"]
        elif pending_chars > SOLVE_MAX_CLUSTER_CHARS:
            stats.mid_cluster_flushes += 1
            yield from solve_batch(pending)
            pending = []
            pending_chars = 0
        pending.append(row)
        pending_chars += len(row["text"])
    if pending:
        yield from solve(pending)


def _sorted_rows(table: pa.Table) -> Iterator[dict]:
    """Rows in ``(cluster_key, id)`` order, materialized a slice at a time."""
    order = pc.sort_indices(
        table.select(["cluster_key", "id"]), sort_keys=[("cluster_key", "ascending"), ("id", "ascending")]
    )
    for start in range(0, len(order), 8192):
        yield from table.take(order.slice(start, 8192)).to_pylist()


# Compressed partition bytes one solve pass may load. Member text of near-duplicate clusters
# compresses far better than average (up to ~10x), and a pass holds the decompressed table plus
# the solver's n-gram sets, so this keeps a pass well inside its memory estimate.
_SOLVE_PASS_COMPRESSED_BYTES = 400 * 1024 * 1024


def _pass_of(cluster_keys: list[str], n_passes: int) -> np.ndarray:
    """Pass of each cluster key; salted so it is independent of the partition hash."""
    return np.fromiter(
        (dupekit.hash_xxh3_64(b"pass|" + k.encode()) % n_passes for k in cluster_keys),
        dtype=np.int64,
        count=len(cluster_keys),
    )


def _pass_table(ctx: Ctx, partition: int, pass_index: int, n_passes: int) -> pa.Table:
    """This partition's rows whose cluster key falls in ``pass_index``, streamed block by block."""
    plan = ctx.plan
    kept = []
    stream = blocks.iter_partition_blocks(
        ctx.store, ctx.data("c075_text"), partition, plan.n_units, plan.c075_partitions
    )
    for _m, block in stream:
        for frame in blocks.iter_frames(block):
            table = blocks.ipc_to_table(frame)
            if n_passes > 1:
                mask = _pass_of(table.column("cluster_key").to_pylist(), n_passes) == pass_index
                table = table.filter(pa.array(mask))
            if table.num_rows:
                kept.append(table)
    return pa.concat_tables(kept) if kept else TEXT_SCHEMA.empty_table()


def run_c075_solve(ctx: Ctx, partition: int) -> dict:
    """Solve one partition's clusters, in as many passes as its size needs.

    A pass holds only the clusters whose key hashes to it. Clusters are independent and each is
    solved whole, in ``(cluster_key, id)`` order, so the removals do not depend on the pass count.
    """
    plan = ctx.plan
    started = time.monotonic()
    compressed = blocks.partition_bytes(ctx.store, ctx.data("c075_text"), partition, plan.n_units, plan.c075_partitions)
    n_passes = max(1, -(-compressed // _SOLVE_PASS_COMPRESSED_BYTES))
    stats = SolveStats()
    removed_parts = []
    for pass_index in range(n_passes):
        table = _pass_table(ctx, partition, pass_index, n_passes)
        removed_parts.append(np.fromiter(solve_sorted_rows(_sorted_rows(table), stats), dtype=np.uint64))
        del table
    removed = np.sort(blocks.concat_arrays(removed_parts, KEY_DTYPE))
    writer = blocks.BlockObjectWriter(plan.node_partitions)
    writer.add_array(removed, plan.node_partition_of_keys(removed))
    meta = writer.finish(ctx.store, blocks.map_object_path(ctx.data("c075_solve"), partition))
    return {
        "blocks": meta,
        **dataclasses.asdict(stats),
        "passes": n_passes,
        "solve_seconds": time.monotonic() - started,
    }


def seal_c075_solve(ctx: Ctx, results: list[dict]) -> dict:
    index = blocks.build_index(
        ctx.store, ctx.data("c075_solve"), [r["blocks"] for r in results], ctx.plan.node_partitions
    )
    fields = [f.name for f in dataclasses.fields(SolveStats)]
    totals = {k: _sum(results, k) for k in fields}
    if totals["documents"] != ctx.sealed("c075_text")["members"]:
        raise RuntimeError(
            f"solve saw {totals['documents']} documents, text stage emitted {ctx.sealed('c075_text')['members']}"
        )
    deleted = delete_prefix(ctx.store, ctx.data("c075_text") + "/")
    return {
        **totals,
        "solve_seconds": sum(r["solve_seconds"] for r in results),
        "index": index,
        "deleted_objects": deleted,
    }


def run_c075_marks(ctx: Ctx, partition: int) -> dict:
    plan = ctx.plan
    removed = blocks.read_partition_array(
        ctx.store, ctx.data("c075_solve"), partition, plan.c075_partitions, plan.node_partitions, KEY_DTYPE
    )
    removed = np.sort(removed)
    units = plan.unit_of_keys(removed)
    written = {}
    for unit in plan.node_partition_units(partition):
        written[str(unit)] = blocks.write_single_block(
            ctx.store, blocks.map_object_path(ctx.data("c075_marks"), unit), removed[units == unit]
        )
    return {"units": written, "removed": len(removed)}


def seal_c075_marks(ctx: Ctx, results: list[dict]) -> dict:
    removed = _sum(results, "removed")
    if removed != ctx.sealed("c075_solve")["duplicates"]:
        raise RuntimeError(f"marks hold {removed} removals, solve found {ctx.sealed('c075_solve')['duplicates']}")
    deleted = delete_prefix(ctx.store, ctx.data("c075_solve") + "/")
    return {"removed": removed, "deleted_objects": deleted}


def run_apply_c075(ctx: Ctx, unit: int, procs: int) -> dict:
    reps = _unit_reps(ctx, unit)
    partition = unit // ctx.plan.units_per_node_partition
    meta = ctx.done("c075_marks", partition)["units"][str(unit)]
    removed = blocks.read_single_block_array(ctx.store, meta, KEY_DTYPE)
    survivors = reps[~np.isin(reps["d"], removed)]
    if len(reps) - len(survivors) != len(removed):
        raise RuntimeError(f"unit {unit}: {len(removed)} removals but {len(reps) - len(survivors)} matched")
    return _write_survivors(ctx, unit, survivors, _attrs_of_unit(ctx, unit), "c075", procs)


# ------------------------------------------------------------------------------------ report


def run_report(ctx: Ctx, _task: int) -> dict:
    names = ["pass1", "exact", "lshmap", "bucket", "components", "canon_a", "canon_b", "attrs"]
    if "cc_canonical" in ctx.plan.rules:
        names.append("apply_cc")
    if "c075" in ctx.plan.rules:
        names += ["c075_text", "c075_solve", "c075_marks", "apply_c075"]
    sealed = {name: {k: v for k, v in ctx.sealed(name).items() if k != "index"} for name in names}
    pass1, exact, canon_b = sealed["pass1"], sealed["exact"], sealed["canon_b"]
    unique = exact["unique"]
    report = {
        "run_root": ctx.plan.run_root,
        "output_root": ctx.plan.output_root,
        "n_warcs": ctx.plan.n_files,
        "fuzzy_params": {
            "num_perms": docs.NUM_PERMS,
            "num_bands": docs.NUM_BANDS,
            "ngram_size_chars": docs.NGRAM_SIZE,
            "seed": docs.SEED,
            "approx_jaccard_threshold": 0.75,
            "text_cap_chars": None,
        },
        "records_in": pass1["kept"],
        "empty_text_filtered": pass1["empty_text_filtered"],
        "exact_duplicates": exact["exact_dups"],
        "unique_docs": unique,
        "cluster_members": canon_b["members"],
        "components": canon_b["components"],
        "stages": sealed,
    }
    if "cc_canonical" in ctx.plan.rules:
        report["cc_canonical"] = {
            "docs": sealed["apply_cc"]["docs"],
            "chars": sealed["apply_cc"]["chars"],
            "expected_docs": unique - canon_b["members"] + canon_b["components"],
        }
        cc = report["cc_canonical"]
        if cc["docs"] != cc["expected_docs"]:
            raise RuntimeError(f"cc_canonical wrote {cc['docs']} docs, expected {cc['expected_docs']}")
    if "c075" in ctx.plan.rules:
        report["c075"] = {
            "docs": sealed["apply_c075"]["docs"],
            "chars": sealed["apply_c075"]["chars"],
            "removed": sealed["c075_marks"]["removed"],
            "rule": C075_PARAMS.model_dump(mode="json"),
            "max_cluster_size": MAX_CLUSTER_SIZE,
        }
    ctx.store.write(f"{ctx.plan.output_root}/stats/rdedup_stats.json", dumps(report))
    return {"report": f"{ctx.plan.output_root}/stats/rdedup_stats.json"}


def seal_report(ctx: Ctx, results: list[dict]) -> dict:
    return results[0]


# --------------------------------------------------------------------------------- pipeline


def _stage(name, n, run, seal, procs=1, mem_gb=4.0, deadline_hours=3.0) -> StageSpec:
    return StageSpec(name, n, run, seal, procs, mem_gb, deadline_hours)


def stage_sequence(plan: Plan, sealed: Callable[[str], dict], procs: int) -> Iterator[StageSpec]:
    """The stages in order. Component rounds continue until a round's seal reports convergence;
    ``sealed`` is only consulted for stages the caller has already found sealed."""
    u, rn = plan.n_units, plan.node_partitions
    yield _stage("pass1", u, lambda c, t: run_pass1(c, t, procs), seal_pass1, procs=procs, mem_gb=8, deadline_hours=4)
    yield _stage("exact", plan.exact_partitions, run_exact, seal_exact, mem_gb=6)
    yield _stage("lshmap", u, run_lshmap, seal_lshmap, mem_gb=10)
    yield _stage("bucket", plan.lsh_partitions, run_bucket, seal_bucket, mem_gb=12)
    round_index = 0
    while True:
        for step, run in (("a", run_cc_a), ("b", run_cc_b), ("c", run_cc_c)):
            yield _stage(
                cc_name(round_index, step),
                rn,
                lambda c, t, k=round_index, f=run: f(c, t, k),
                lambda c, rs, k=round_index, st=step: seal_cc(c, rs, k, st),
                mem_gb=10,
            )
            if step == "a" and sealed(cc_name(round_index, "a"))["converged"]:
                break
        else:
            round_index += 1
            continue
        break
    yield _stage("canon_a", rn, run_canon_a, seal_canon_a, mem_gb=6)
    yield _stage("canon_b", rn, run_canon_b, seal_canon_b, mem_gb=6)
    yield _stage("attrs", rn, run_attrs, seal_attrs, mem_gb=4)
    if "cc_canonical" in plan.rules:
        yield _stage(
            "apply_cc",
            u,
            lambda c, t: run_apply_cc(c, t, procs),
            lambda c, rs: seal_apply(c, rs, "cc_canonical"),
            procs=procs,
            mem_gb=8,
        )
    if "c075" in plan.rules:
        yield _stage("c075_text", u, lambda c, t: run_c075_text(c, t, procs), seal_c075_text, procs=procs, mem_gb=14)
        yield _stage("c075_solve", plan.c075_partitions, run_c075_solve, seal_c075_solve, mem_gb=15)
        yield _stage("c075_marks", rn, run_c075_marks, seal_c075_marks, mem_gb=3)
        yield _stage(
            "apply_c075",
            u,
            lambda c, t: run_apply_c075(c, t, procs),
            lambda c, rs: seal_apply(c, rs, "c075"),
            procs=procs,
            mem_gb=8,
        )
    yield _stage("report", 1, run_report, seal_report, mem_gb=2)


def component_rounds_result(ctx_sealed: Callable[[str], dict]) -> dict:
    """The ``components`` pseudo-stage: the converged round and per-round label changes."""
    rounds = []
    k = 0
    while True:
        result = ctx_sealed(cc_name(k, "a"))
        rounds.append({key: v for key, v in result.items() if key != "index" and not key.startswith("_")})
        if result["converged"]:
            return {"final_round": k, "rounds": rounds}
        k += 1


class Pipeline:
    """Finds the frontier stage of a run from sealed markers, caching what it has seen."""

    def __init__(self, store: ObjectStore, plan: Plan, procs: int) -> None:
        self.store = store
        self.plan = plan
        self.procs = procs
        self._sealed: dict[str, dict] = {}
        self.ctx = Ctx(store, plan, self.sealed)

    def sealed(self, name: str) -> dict:
        if name == "components":
            return component_rounds_result(self.sealed)
        if name not in self._sealed:
            self._sealed[name] = read_json(self.store, f"{self.plan.run_root}/stages/{name}/sealed.json")
        return self._sealed[name]

    def board(self, spec: StageSpec, claims) -> TaskBoard:
        return TaskBoard(self.store, self.ctx.stage_dir(spec.name), spec.n_tasks, claims)

    def frontier(self) -> StageSpec | None:
        """The first stage without a seal, or ``None`` once every stage is sealed."""
        for spec in stage_sequence(self.plan, self.sealed, self.procs):
            if spec.name in self._sealed:
                continue
            if self.store.stat(f"{self.plan.run_root}/stages/{spec.name}/sealed.json") is None:
                return spec
            self.sealed(spec.name)
        return None

    def stages_so_far(self) -> list[StageSpec]:
        """Every stage up to and including the frontier."""
        out = []
        for spec in stage_sequence(self.plan, self.sealed, self.procs):
            out.append(spec)
            if (
                spec.name not in self._sealed
                and self.store.stat(f"{self.plan.run_root}/stages/{spec.name}/sealed.json") is None
            ):
                break
            self.sealed(spec.name)
        return out
