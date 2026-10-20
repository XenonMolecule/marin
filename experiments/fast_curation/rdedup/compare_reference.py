# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare an rdedup run's fuzzy clusters with an existing Zephyr ``fuzzy/`` attribute tree.

Answers, for the same corpus:

* do both pipelines put exactly the same documents into clusters (the LSH graph and the exact
  dedup agree)?
* is every reference cluster inside one rdedup component (the reference clusters refine the
  true connected components)?
* how many reference cluster labels have no canonical member -- the fingerprint of
  connected components stopped at its iteration cap before converging?
* applying the REFERENCE canonical flags to this run's unique documents, how many documents
  and characters survive -- which must reproduce the reference ``deduped/`` counts exactly when
  normalization and exact dedup agree.

Runs in one process on a large-memory worker in the data region::

    python -m experiments.fast_curation.rdedup.compare_reference --run-root gs://.../rdedup/154k \\
        --reference-attrs gs://.../154117warcs/fuzzy/outputs/source_000 --output gs://.../compare.json
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from experiments.fast_curation.rdedup import blocks
from experiments.fast_curation.rdedup.plan import load_plan
from experiments.fast_curation.rdedup.stages import ATTR_DTYPE, REPS_DTYPE, Ctx, Pipeline
from experiments.fast_curation.rdedup.store import ObjectStore, dumps, store_for

logger = logging.getLogger(__name__)


def hex_ids_to_words(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """32-hex-digit ids -> (hi, lo) uint64 words."""
    n = len(ids)
    raw = np.frombuffer("".join(ids).encode(), dtype=np.uint8).reshape(n, 32)
    values = np.where(raw >= ord("a"), raw - ord("a") + 10, raw - ord("0")).astype(np.uint64)
    shifts = np.arange(60, -4, -4, dtype=np.uint64)
    return (values[:, :16] << shifts).sum(axis=1, dtype=np.uint64), (values[:, 16:] << shifts).sum(
        axis=1, dtype=np.uint64
    )


def _load_ours(ctx: Ctx, threads: int) -> dict[str, np.ndarray]:
    plan = ctx.plan

    def unit(u: int) -> tuple[np.ndarray, np.ndarray]:
        reps = blocks.read_single_block_array(ctx.store, ctx.done("lshmap", u)["reps"], REPS_DTYPE)
        partition = u // plan.units_per_node_partition
        attrs = blocks.read_single_block_array(ctx.store, ctx.done("attrs", partition)["units"][str(u)], ATTR_DTYPE)
        at = np.searchsorted(reps["d"], attrs["d"])
        return reps[at], attrs

    with ThreadPoolExecutor(threads) as pool:
        parts = list(pool.map(unit, range(plan.n_units)))
    reps = np.concatenate([p[0] for p in parts])
    attrs = np.concatenate([p[1] for p in parts])
    return {"hi": reps["hi"], "lo": reps["lo"], "chars": reps["chars"], "root": attrs["root"], "canon": attrs["canon"]}


def _load_reference(store: ObjectStore, attr_dir: str, threads: int) -> dict[str, np.ndarray]:
    paths = sorted(info.path for info in store.list(attr_dir.rstrip("/") + "/") if info.path.endswith(".parquet"))
    if not paths:
        raise FileNotFoundError(f"no reference attrs under {attr_dir}")

    def one(path: str):
        parquet = pq.ParquetFile(io.BytesIO(store.read(path)))
        if parquet.metadata.num_rows == 0:
            return None
        table = parquet.read(columns=["id", "dup_cluster_id", "is_cluster_canonical"])
        hi, lo = hex_ids_to_words(table.column("id").to_pylist())
        clusters = np.array(
            [hash(c) & 0xFFFFFFFFFFFFFFFF for c in table.column("dup_cluster_id").to_pylist()], dtype=np.uint64
        )
        canon = np.asarray(table.column("is_cluster_canonical").to_numpy(zero_copy_only=False), dtype=bool)
        return hi, lo, clusters, canon

    with ThreadPoolExecutor(threads) as pool:
        parts = [p for p in pool.map(one, paths) if p is not None]
    logger.info("read %d reference attr files", len(paths))
    return {
        "hi": np.concatenate([p[0] for p in parts]),
        "lo": np.concatenate([p[1] for p in parts]),
        "cluster": np.concatenate([p[2] for p in parts]),
        "canon": np.concatenate([p[3] for p in parts]),
    }


def compare(ours: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict:
    o = np.lexsort((ours["lo"], ours["hi"]))
    r = np.lexsort((ref["lo"], ref["hi"]))
    for k in ours:
        ours[k] = ours[k][o]
    for k in ref:
        ref[k] = ref[k][r]
    same_members = len(ours["hi"]) == len(ref["hi"]) and bool(
        np.array_equal(ours["hi"], ref["hi"]) and np.array_equal(ours["lo"], ref["lo"])
    )
    out: dict = {
        "our_members": len(ours["hi"]),
        "reference_members": len(ref["hi"]),
        "same_member_set": same_members,
        "our_components": int(ours["canon"].sum()),
        "reference_canonicals": int(ref["canon"].sum()),
        "reference_labels": len(np.unique(ref["cluster"])),
    }
    labels_with_canonical = np.unique(ref["cluster"][ref["canon"]])
    out["reference_labels_without_canonical"] = out["reference_labels"] - len(labels_with_canonical)
    if not same_members:
        # Compare on the members both runs share; report what differs.
        key = np.dtype([("hi", "<u8"), ("lo", "<u8")])
        ours_keys = np.stack([ours["hi"], ours["lo"]], axis=1).copy().view(key).ravel()
        ref_keys = np.stack([ref["hi"], ref["lo"]], axis=1).copy().view(key).ravel()
        _common, oi, ri = np.intersect1d(ours_keys, ref_keys, assume_unique=True, return_indices=True)
        out["our_members_not_in_reference"] = int(len(ours_keys) - len(oi))
        out["reference_members_not_in_ours"] = int(len(ref_keys) - len(ri))
        for k in ours:
            ours[k] = ours[k][oi]
        for k in ref:
            ref[k] = ref[k][ri]
    # Refinement: every reference label must sit inside one of our components.
    pairs = np.unique(np.stack([ref["cluster"], ours["root"]]), axis=1)
    out["reference_labels_spanning_components"] = int(pairs.shape[1] - len(np.unique(pairs[0])))
    roots, labels_per_root = np.unique(pairs[1], return_counts=True)
    out["our_components_split_by_reference"] = int((labels_per_root > 1).sum())
    out["members_in_split_components"] = int(np.isin(ours["root"], roots[labels_per_root > 1]).sum())
    out["reference_noncanonical_members"] = int((~ref["canon"]).sum())
    out["reference_noncanonical_chars"] = int(ours["chars"][~ref["canon"]].sum())
    out["our_noncanonical_members"] = int((ours["canon"] == 0).sum())
    out["our_noncanonical_chars"] = int(ours["chars"][ours["canon"] == 0].sum())
    out["same_canonical_flags"] = bool(np.array_equal(ours["canon"].astype(bool), ref["canon"]))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reference-attrs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    store = store_for(args.run_root)
    plan = load_plan(store, args.run_root)
    pipeline = Pipeline(store, plan, procs=1)
    ours = _load_ours(pipeline.ctx, args.threads)
    logger.info("loaded %d of our members", len(ours["hi"]))
    ref = _load_reference(store, args.reference_attrs, args.threads)
    result = compare(ours, ref)
    exact = pipeline.sealed("exact")
    lsh = pipeline.sealed("lshmap")
    result["unique_docs"] = exact["unique"]
    result["exact_duplicates"] = exact["exact_dups"]
    result["unique_chars"] = lsh["unique_chars"]
    if "reference_noncanonical_members" in result:
        result["reference_rule_survivor_docs"] = exact["unique"] - result["reference_noncanonical_members"]
        result["reference_rule_survivor_chars"] = lsh["unique_chars"] - result["reference_noncanonical_chars"]
        result["our_survivor_docs"] = exact["unique"] - result["our_noncanonical_members"]
        result["our_survivor_chars"] = lsh["unique_chars"] - result["our_noncanonical_chars"]
    store.write(args.output, dumps(result))
    logger.info("RESULT %s", json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
