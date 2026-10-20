# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vectorized graph steps for LSH bucketing and connected components.

Nodes are uint64 document keys. The fuzzy-duplicate graph links every pair of documents that
share an LSH bucket; its connected components are the duplicate clusters. The bucket stage
reduces each non-singleton bucket to a star: every member ``x`` linked to the bucket minimum
``h`` (its hub). Components are the same as for the all-pairs graph.

Components are computed with FastSV-style min-label propagation (Zhang, Azad and Hu, "FastSV:
a distributed-memory connected component algorithm with fast convergence", 2020): labels
start at each node's smallest hub, then each round every edge proposes the smaller of its two
labels to both endpoints and to both endpoints' labels (hooking), and every node takes its
label's label (shortcutting). Edges are stored at their member, never at the hub, and label
lookups are deduplicated per partition, so a giant component costs its hub nothing extra.
A round with no label change is a fixed point: every edge's endpoints share a label, which
is then the component minimum.
"""

from __future__ import annotations

import dataclasses

import numpy as np

LSH_DTYPE = np.dtype([("b", "<u8"), ("d", "<u8")])
HUB_DTYPE = np.dtype([("x", "<u8"), ("h", "<u8")])


def _sorted_unique_pairs(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((second, first))
    first, second = first[order], second[order]
    if len(first) == 0:
        return first, second
    keep = np.ones(len(first), dtype=bool)
    keep[1:] = (first[1:] != first[:-1]) | (second[1:] != second[:-1])
    return first[keep], second[keep]


def _group_starts(keys: np.ndarray) -> np.ndarray:
    """Start index of each run of equal values in sorted ``keys``."""
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([[0], np.flatnonzero(keys[1:] != keys[:-1]) + 1])


@dataclasses.dataclass(frozen=True)
class BucketStats:
    pairs: int
    buckets: int
    multi_buckets: int
    multi_members: int
    largest: int


def bucket_hub_pairs(lsh: np.ndarray) -> tuple[np.ndarray, BucketStats]:
    """``(member, hub)`` for every member of every bucket holding two or more documents.

    ``lsh`` holds ``(bucket, doc)`` records; a document listed twice in one bucket counts once,
    as ``connected_components`` dedups bucket members. The hub is the bucket's minimum document,
    and the hub is emitted as its own member so every node of the graph appears as a key.
    """
    b, d = _sorted_unique_pairs(lsh["b"], lsh["d"])
    starts = _group_starts(b)
    sizes = np.diff(np.append(starts, len(b)))
    multi = sizes >= 2
    member_mask = np.repeat(multi, sizes)
    hubs = np.repeat(d[starts], sizes)
    out = np.empty(int(member_mask.sum()), dtype=HUB_DTYPE)
    out["x"] = d[member_mask]
    out["h"] = hubs[member_mask]
    stats = BucketStats(
        pairs=len(b),
        buckets=len(starts),
        multi_buckets=int(multi.sum()),
        multi_members=len(out),
        largest=int(sizes.max()) if len(sizes) else 0,
    )
    return out, stats


# Labels are compared in a scrambled order, ``key * _MIX mod 2**64`` (a bijection for odd
# _MIX). Doc keys follow file order, so under plain key order every large component's label
# is a node from the earliest files, and one partition receives every hooking candidate and
# label request of the corpus. Scrambled, each component's final label is an arbitrary member,
# spread evenly over partitions. Any consistent total order yields the same components.
_MIX = np.uint64(0x9E3779B97F4A7C15)


def scramble(keys: np.ndarray) -> np.ndarray:
    return keys * _MIX


def order_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise minimum under the scrambled order."""
    return np.where(scramble(a) <= scramble(b), a, b)


@dataclasses.dataclass(frozen=True)
class PartitionGraph:
    """One node partition's static edges and its node set.

    ``ex``/``eh`` hold the edges ``(member, hub)`` whose member lives in this partition. Every
    member has at most 26 hubs, so no partition ever holds a hub's full member list, however
    large its component grows.
    """

    ex: np.ndarray
    eh: np.ndarray
    nodes: np.ndarray  # sorted


def initial_graph(pairs: np.ndarray) -> tuple[PartitionGraph, np.ndarray]:
    """Edges, nodes and initial labels from one partition's ``(member, hub)`` bucket pairs.

    A node's initial label is the smallest of itself and its hubs, all in its component.
    """
    x, h = _sorted_unique_pairs(pairs["x"], pairs["h"])
    starts = _group_starts(x)
    nodes = x[starts]
    _, hub_min = min_by_key(x, h)
    labels = order_min(nodes, hub_min)
    keep = x != h
    return PartitionGraph(ex=x[keep], eh=h[keep], nodes=nodes), labels


def lookup(nodes: np.ndarray, labels: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Labels of ``keys``, all of which must be in the sorted ``nodes``."""
    at = np.searchsorted(nodes, keys)
    if len(keys) and (at.max(initial=0) >= len(nodes) or np.any(nodes[np.minimum(at, len(nodes) - 1)] != keys)):
        raise KeyError("label lookup for a node this partition does not hold")
    return labels[at]


def request_targets(graph: PartitionGraph, labels: np.ndarray) -> np.ndarray:
    """Nodes whose labels this partition needs next: every hub and every current label."""
    return np.unique(np.concatenate([graph.eh, labels]))


def min_by_key(keys: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per distinct key, the minimum value under the scrambled order (keys returned sorted)."""
    order = np.lexsort((scramble(values), keys))
    keys, values = keys[order], values[order]
    starts = _group_starts(keys)
    return keys[starts], values[starts]


def candidates(
    graph: PartitionGraph, labels: np.ndarray, known: np.ndarray, known_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Label candidates ``(target, label)``, min-combined per target.

    ``known``/``known_labels`` are the answered labels of :func:`request_targets`. For each edge
    ``(x, h)`` with ``c = min(L(x), L(h))``: ``x`` and ``h`` get ``c`` (propagation) and the
    trees they point into, ``L(x)`` and ``L(h)``, get ``c`` (hooking). Every node ``v`` also
    gets ``L(L(v))`` (shortcutting). Together these are the FastSV update: labels only
    decrease, always name a node of the same component, and reach the component minimum in
    O(log n) rounds rather than O(diameter).
    """
    lx = lookup(graph.nodes, labels, graph.ex)
    lh = lookup(known, known_labels, graph.eh)
    c = order_min(lx, lh)
    shortcut = lookup(known, known_labels, labels)
    targets = np.concatenate([graph.ex, graph.eh, lx, lh, graph.nodes])
    values = np.concatenate([c, c, c, c, shortcut])
    return min_by_key(targets, values)


def apply_candidates(nodes: np.ndarray, labels: np.ndarray, targets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``labels`` lowered (scrambled order) by the minimum candidate per node, in place."""
    targets, values = min_by_key(targets, values)
    at = np.searchsorted(nodes, targets)
    if len(targets) and (at.max() >= len(nodes) or np.any(nodes[at] != targets)):
        raise KeyError("candidate for a node this partition does not hold")
    labels[at] = order_min(labels[at], values)
    return labels
