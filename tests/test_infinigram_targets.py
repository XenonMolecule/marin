# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pure logic of the infini-gram index registry and resolver.

Covers the parts that need no network or toolchain: registry invariants, index
output paths, the cross-region egress guard, confighash-prefix validation, and
shard planning.
"""

from __future__ import annotations

import pytest

from experiments.infinigram.build import DISK_FACTOR, SHARD_BYTES_THRESHOLD, plan_chunks, plan_shards
from experiments.infinigram.resolve import _assert_in_region, _bucket_of, resolve_prefix
from experiments.infinigram.targets import (
    DATASETS,
    Collection,
    DatasetSpec,
    IndexSource,
    all_targets,
    get_target,
)


def test_index_source_requires_exactly_one_of_globs_or_prefix():
    with pytest.raises(ValueError):
        IndexSource()
    with pytest.raises(ValueError):
        IndexSource(globs=("gs://b/x",), prefix="gs://b/y-*")
    assert IndexSource.at("gs://b/x").globs == ("gs://b/x",)
    assert IndexSource.under("gs://b/y-*/data-*.jsonl.gz").prefix is not None


def test_dataset_spec_rejects_unknown_region():
    with pytest.raises(ValueError):
        DatasetSpec(dataset="x", region="mars-west1", full=IndexSource.at("gs://b/x"))


def test_registry_targets_have_matching_region_bucket():
    for t in all_targets():
        assert t.index_dir.startswith(f"gs://marin-{t.region}/infinigram_indices/")
        assert t.index_dir.endswith(f"/{t.collection.value}/{t.dataset}")


def test_every_listed_dataset_present():
    # The datasets the user asked for must all be registered.
    expected = {
        "dclm",
        "nemotron_full",
        "fineweb",
        "fineweb_edu",
        "resiliparse",
        "high_quality",
        "llm_pipeline_v1",
        "med_quality",
        "low_quality",
    }
    assert expected <= set(DATASETS)


def test_get_target_unknown_dataset_and_missing_collection():
    with pytest.raises(ValueError):
        get_target("nope", Collection.FULL)


def test_bucket_parsing_and_region_guard():
    assert _bucket_of("gs://marin-us-central2/a/b.jsonl.gz") == "marin-us-central2"
    _assert_in_region(["gs://marin-us-central2/x", "gs://marin-us-central2/y"], "us-central2")
    with pytest.raises(ValueError):
        _assert_in_region(["gs://marin-us-central1/x"], "us-central2")


def test_resolve_prefix_rejects_wrong_star_count():
    with pytest.raises(ValueError):
        resolve_prefix("gs://b/no-star/data-*.jsonl.gz")  # zero confighash segments
    with pytest.raises(ValueError):
        resolve_prefix("gs://b/a-*/c-*/data-*.jsonl.gz")  # two confighash segments


@pytest.mark.parametrize(
    "byte_count,expected",
    [
        (0, 1),
        (10 * 1024**3, 1),
        (SHARD_BYTES_THRESHOLD, 1),
        (SHARD_BYTES_THRESHOLD + 1, 2),
        (3 * SHARD_BYTES_THRESHOLD, 3),
    ],
)
def test_plan_shards(byte_count, expected):
    assert plan_shards(byte_count) == expected


def test_plan_chunks_single_when_it_fits():
    gib = 1024**3
    # 13 GiB corpus, 90 GiB disk -> budget/DISK_FACTOR ~= 28 GiB -> one chunk
    assert plan_chunks([gib] * 13, 90 * gib) == [list(range(13))]


def test_plan_chunks_splits_large_corpus_within_budget():
    gib = 1024**3
    shard_bytes = [10 * gib] * 57  # ~570 GiB like resiliparse
    chunks = plan_chunks(shard_bytes, 90 * gib)
    # every chunk's corpus must be under the per-chunk budget
    budget = int(90 * gib / DISK_FACTOR)
    assert len(chunks) > 1
    for chunk in chunks:
        assert sum(shard_bytes[i] for i in chunk) <= budget
    # every shard assigned exactly once, in order
    assert [i for c in chunks for i in c] == list(range(57))


def test_plan_chunks_oversized_shard_gets_own_chunk():
    gib = 1024**3
    # a single shard bigger than the whole budget still becomes its own chunk
    chunks = plan_chunks([100 * gib], 90 * gib)
    assert chunks == [[0]]


def test_plan_chunks_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        plan_chunks([1, 2, 3], 0)
