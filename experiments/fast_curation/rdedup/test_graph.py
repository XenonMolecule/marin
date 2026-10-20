# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.fast_curation.rdedup import graph


def _lsh(records: list[tuple[int, int]]) -> np.ndarray:
    out = np.empty(len(records), dtype=graph.LSH_DTYPE)
    out["b"] = [b for b, _ in records]
    out["d"] = [d for _, d in records]
    return out


def _components_by_union_find(lsh: np.ndarray) -> dict[int, int]:
    """Reference: component minimum of every document in a non-singleton bucket."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_bucket: dict[int, set[int]] = {}
    for b, d in zip(lsh["b"].tolist(), lsh["d"].tolist(), strict=True):
        by_bucket.setdefault(b, set()).add(d)
    for members in by_bucket.values():
        if len(members) < 2:
            continue
        for m in members:
            parent.setdefault(m, m)
        first = min(members)
        for m in members:
            ra, rb = find(first), find(m)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return {x: find(x) for x in parent}


def _as_partition(labels: dict[int, int]) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for node, label in labels.items():
        groups.setdefault(label, set()).add(node)
    return {frozenset(g) for g in groups.values()}


def _run_components(lsh: np.ndarray, n_partitions: int, max_rounds: int = 200) -> tuple[dict[int, int], int]:
    """Drive the partitioned rounds as the pipeline does: every step sees one partition only."""

    def partition_of(keys: np.ndarray) -> np.ndarray:
        return ((keys * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(40)) % np.uint64(n_partitions)

    pairs, _ = graph.bucket_hub_pairs(lsh)
    parts = partition_of(pairs["x"])
    state = [graph.initial_graph(pairs[parts == p]) for p in range(n_partitions)]
    graphs = [g for g, _ in state]
    labels = [lab for _, lab in state]
    for round_index in range(1, max_rounds + 1):
        # Requests and answers: each partition learns the labels it asked for, from their owners.
        known = []
        for p in range(n_partitions):
            targets = graph.request_targets(graphs[p], labels[p])
            owner = partition_of(targets)
            answers = np.concatenate(
                [graph.lookup(graphs[q].nodes, labels[q], targets[owner == q]) for q in range(n_partitions)]
            )
            ordered = np.concatenate([targets[owner == q] for q in range(n_partitions)])
            order = np.argsort(ordered)
            known.append((ordered[order], answers[order]))
        sent_t, sent_c = [], []
        for p in range(n_partitions):
            t, c = graph.candidates(graphs[p], labels[p], *known[p])
            sent_t.append(t)
            sent_c.append(c)
        all_t, all_c = np.concatenate(sent_t), np.concatenate(sent_c)
        owner = partition_of(all_t)
        changed = 0
        for p in range(n_partitions):
            before = labels[p].copy()
            graph.apply_candidates(graphs[p].nodes, labels[p], all_t[owner == p], all_c[owner == p])
            changed += int(np.count_nonzero(labels[p] != before))
        if changed == 0:
            out: dict[int, int] = {}
            for g, lab in zip(graphs, labels, strict=True):
                out.update(zip(g.nodes.tolist(), lab.tolist(), strict=True))
            return out, round_index
    raise AssertionError("did not converge")


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_components_match_union_find_on_random_lsh_graphs(seed):
    rng = np.random.default_rng(seed)
    n_docs, n_records = 3000, 9000
    # Few distinct buckets relative to records gives long chains and a few giant components.
    records = list(
        zip(rng.integers(0, 2500, n_records).tolist(), rng.integers(0, n_docs, n_records).tolist(), strict=True)
    )
    lsh = _lsh(records)
    got, rounds = _run_components(lsh, n_partitions=7)
    assert _as_partition(got) == _as_partition(_components_by_union_find(lsh))
    assert rounds < 40


def test_long_path_converges_to_its_minimum():
    # A path whose every edge lands in a different partition: the case plain min-label
    # propagation needs n rounds for.
    n = 2000
    rng = np.random.default_rng(5)
    labels = rng.permutation(n).astype(np.uint64) + np.uint64(10)
    records = [(i, int(labels[i])) for i in range(n - 1)] + [(i, int(labels[i + 1])) for i in range(n - 1)]
    got, rounds = _run_components(_lsh(records), n_partitions=13)
    assert len(set(got.values())) == 1
    assert len(got) == n
    assert rounds < 60


def test_singleton_buckets_and_repeated_members_do_not_create_nodes():
    # Doc 7 is alone in its buckets; doc 3 is listed twice in bucket 1 and alone otherwise.
    lsh = _lsh([(1, 3), (1, 3), (2, 7), (3, 7), (4, 5), (4, 9), (5, 9), (5, 11)])
    got, _ = _run_components(lsh, n_partitions=3)
    assert _as_partition(got) == {frozenset({5, 9, 11})}
