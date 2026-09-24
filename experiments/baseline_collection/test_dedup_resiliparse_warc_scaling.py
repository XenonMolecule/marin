# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import pytest

from experiments.baseline_collection.dedup_resiliparse_warc_scaling import ShardOrder, shard_indices


def test_random_draws_nest_so_smaller_n_is_a_subset_of_larger_n():
    small, large = shard_indices(300, 10364, ShardOrder.RANDOM), shard_indices(3000, 10364, ShardOrder.RANDOM)

    assert set(small) < set(large) and len(large) == 3000
    assert small != shard_indices(300, 10364, ShardOrder.PREFIX)


def test_n_above_the_extraction_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        shard_indices(11, 10, ShardOrder.RANDOM)
