# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import pytest
from marin.execution.step_runner import StepRunner

from experiments.fast_curation.estimate_survival_from_partial_markers import estimate, select_marked_shards
from experiments.fast_curation.reapply_verified_dedup import ReapplyOutput, build_steps
from experiments.fast_curation.test_reapply_verified_dedup import RESOURCES, _built_tree, local_client  # noqa: F401


def test_sample_is_a_stable_subset_that_does_not_follow_shard_order():
    names = [f"part-{i:05d}.parquet" for i in range(200)]

    sample = select_marked_shards(names, 20)

    assert sample == select_marked_shards(list(reversed(names)), 20)
    assert len(set(sample)) == 20 and set(sample) <= set(names)
    assert sample != names[:20]


def test_estimate_counts_only_shards_whose_markers_exist(tmp_path, monkeypatch, local_client):  # noqa: F811
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    tree = _built_tree(tmp_path)
    StepRunner().run(build_steps(str(tree), ReapplyOutput.COUNTS, RESOURCES))
    # The reduce for part-00000 never finished: its marker file is absent.
    (tree / "dedup_c075" / "verified" / "outputs" / "source_000" / "part-00000.parquet").unlink()

    result = estimate(str(tree), sample_shards=10, max_workers=2)

    # part-00001 holds a_copy, b_full and y. cc_canonical drops all three; c075 drops only a_copy.
    assert (result["shards_with_markers"], result["shards_sampled"], result["docs"]) == (1, 1, 3)
    assert (result["cc_canonical_removed_docs"], result["c075_removed_docs"]) == (3, 1)
    assert result["cc_canonical_doc_survival_exact"] == pytest.approx(1 - 4 / 7)
